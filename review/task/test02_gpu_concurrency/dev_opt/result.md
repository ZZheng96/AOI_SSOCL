# AC-069 开发版小测：学习让行 + 锁等待可观测（2026-10-13，开发对话）

结论：引擎层单品类、并发 1 的场景下，学习并发时前台 P95 为 1359→255ms（预提取路径），特征已缓存时为 127ms，接近 idle 的 183ms。前台 lock_wait 只在学习线程让行超时、强行入锁时出现（30 次中 1 次，≤539ms）。**这不是 TEST-02 口径的复测，AC-069 仍待稳定版按 TEST-02 口径复测。**

## 环境与口径

- 开发版 `AOI_Core`。gold_finger 快照复制到 TEMP，current 切到 v1（生产 current=v2 是无主干降配版，不能即学），用完即删（已核实无残留）。配置为 `engine_fast.yaml`，GPU（cuda），`learning_parallel=False`。
- 图片取自 `storage\archive\gold_finger\20261011` 的前 20 张：10 张前台循环，10 张反馈（每档各复制一份）。
- 前台：同一线程闭环、零间隔连续调用 `predict_image_path` 30 次，每次记录墙钟和 `extra.lock_wait_ms`。
- 学习：后台线程依次 `submit_feedback(...,"correct")`，完成一张再提交下一张。
- 四档：
  - idle。
  - learn_before：yield 上界设为 0，build_item 锁外抛错后回落锁内，用来模拟改动前。
  - learn_after：当前实现。
  - learn_after_cached：反馈图事先检测过，特征已在缓存中。

## 结果（ms，n=30/档）

| 档 | P50 | P95 | max | lock_wait P95/max | 等锁次数(>1ms) | 学习完成数 | yield P50/max | hold P50/max |
|---|---|---|---|---|---|---|---|---|
| idle | 86.8 | 183.0 | 298.9 | 0/0 | 0 | 0 | - | - |
| learn_before | 101.0 | **1359.3** | 1672.5 | 898.5/1401.1 | 10 | 10 | 0/0 | 626.8/1501.2 |
| learn_after | 93.6 | **254.8** | 436.3 | 0/385.5 | 1 | 2 | 1281.5/1500.0 | 413.5/441.6 |
| learn_after_cached | 94.9 | **127.2** | 819.3 | 0/538.9 | 1 | 2 | 1367.0/1515.0 | 461.5/538.8 |

`lock_stats()["feedback"]`：n=14，max_wait=658ms，max_hold=1501ms（包含 learn_before 档）。逐请求数据见 `per_request.json`，汇总见 `summary.json`，终端输出见 `bench_stdout.txt`。

## 解读

- 改动前，学习在锁内读图并做 DINO 前向，hold 最长 1.5s，前台每隔一张就要等锁，P95≈1.36s。这与 TEST-02 c1 learning P95=1505ms 的量级一致。
- 改动后，学习先让行（前台闭环零间隔时总会等满 1.5s 上界），再在锁外预提取，锁内只做打分、状态更新和 WAL。前台只在上界到期、学习强行入锁时等一次锁。
- **吞吐代价**：在前台零间隔的持续负载下，学习被限速到约每 2s 一次（30 次请求只完成 2 次学习，改动前是 10 次）。上界保证学习不会饿死，但会被明显放慢。这是让行设计的预期结果。
- **锁内残余 hold 仍有约 400～540ms**（即使 cached 档不需要提特征也是如此）。这部分是打分、状态更新和回执/WAL 写入，是下一步压缩的方向。前台撞上这一次时，max 会到 436～819ms。
- learn_after 的 P95 比 cached 高：锁外 DINO 前向与前台推理争用同一 GPU，前台单次变慢（学习不持锁，所以不体现在 lock_wait 上）。

## 与 TEST-02 的口径差异

| 项 | TEST-02（V§62） | 本测 |
|---|---|---|
| 层级 | service.detect_image（persist=True、with_heatmap=True，含渲染和落库） | 引擎层 predict_image_path，不含 persist/heatmap |
| 版本/模型 | 稳定版，在 TEMP 中 prepare 两个品类（40/30 张，seed=777） | 开发版，gold_finger v1 快照副本，单品类 |
| 并发 | c1/c2/c4 多请求、多品类 | 前台单线程闭环 + 1 个学习线程 |
| 锁等待 | `t_lock_acquired` 实际取的是线程开始时间，queue≈0 是假象 | 引擎内部真实等品类锁时间 `extra.lock_wait_ms`（服务层经 service.py 透出） |
| 原值 | c1 idle 186/263，c1 learning 485/1505，c4 idle 301/406 | 见上表 |

所以本测的绝对值不能与 TEST-02 直接比较，只能说明改动前后的相对变化，以及等锁确实被消除。c4 idle P95=406ms 来自多请求互相等品类锁和渲染/持久化开销，本改动不涉及（前台之间仍串行）。

## 剩余缺口

- 稳定版按 TEST-02 口径（c1/c2/c4、两品类、service 层）复测没有做。
- 多前台请求之间的品类锁串行（c4 idle）未处理。
- 锁内残余 hold 约 0.4～0.5s 没有压缩。
- 学习吞吐在持续负载下被限速。
- 候选训练路径 `CandidateBudget.check` 的让行没有上界（可取消，让行时间会从墙钟预算中扣除）。
