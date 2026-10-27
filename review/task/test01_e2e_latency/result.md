# TEST-01 原生 2500×2500 端到端耗时（AC-066）

## 结论：FAIL（实测，非估算）

热态 P95 两种入口均超过 200ms 红线：

| 口径 | 入口 | n | P50 | P95 | P99 | max |
|---|---|---|---|---|---|---|
| wall（含持久化/落盘返回） | direct | 120 | 213.1 | **295.8** | 339.8 | 352.6 |
| wall（含 HTTP 传输+排队） | http | 120 | 203.3 | **274.7** | 292.4 | 296.5 |
| e2e（decode+infer+render） | direct | 120 | 206.8 | **288.4** | 319.3 | 327.9 |
| e2e | http | 120 | 192.6 | **258.7** | 264.5 | 275.2 |
| infer（引擎段） | direct | 120 | 70.3 | 134.7 | 151.2 | 163.6 |
| infer | http | 120 | 71.6 | 99.6 | 143.1 | 148.5 |

- 冷启动首请求：direct 458.3ms；http 467.6ms（均不含 prepare，模型已在隔离环境预激活）。
- 后台学习负载（submit_feedback ×2，生产默认 learning_parallel=False 串行让行）：
  与学习重叠的请求 wall ≈ 1026～1847ms（n=2/入口），前台被学习阻塞明显；
  该负载样本量小，仅作现象记录，正式并发口径见 TEST-02。
- 判定：是否满足 AC-066/G1「热态 P95 ≤ 200ms」——**不满足**（最低口径 http e2e P95 258.7ms 也超标约 29%）。

## 口径声明（按任务要求写明）

- `wall_ms`：调用方实测整请求墙钟。direct=decode+infer+render+persist+JSON 返回；
  http 另含 HTTP 传输与排队。
- `e2e_ms`：响应内 `latency_e2e_ms`，服务端口径=decode+infer+render，**不含** persist/HTTP。
- `decode/infer/render` 由响应字段拆分（total_latency_ms/latency_ms/latency_e2e_ms 差值），
  `persist_ms≈wall-e2e` 为近似（direct 含落盘+统计；http 含网络）。
- 样本：seed=20261013 从 197 张 2500×2500 归档图随机抽 40 张，不缩放，sha256 校验见 images.json。
- 每入口：冷启动 1 次 + 热身 5 次（剔除）+ 空闲 3 轮×40=120 次 + 学习负载若干。

## 重要附带发现（阻塞生产可用性）

- **稳定版生产快照 gold_finger v2 无法被当前代码加载**：`algo/persist.py:_restore_slot_state`
  抛 `ValueError: layout 快照缺少必要拟合统计: count_median, count_std, area_hi, ...`，
  要求用原训练数据重新 prepare。即当前稳定版直接启动后 gold_finger 检测不可用，
  必须重新 prepare。本次测量所用模型为隔离 TEMP 环境内以 40 张 train 域归档正常图
  （seed=123，与测试样本无交集）重新 prepare 的同架构同配置模型；耗时特性代表性成立，
  但该模型不是生产训练态，精度结论不适用。
- guard_bundle 红线工作正常：归档路径（非 train 域）被拒绝作为 init_normal，
  需复制到 train 目录后方可 prepare。

## 已知偏差

- GPU 为 GTX 1660 SUPER 6GB，非生产目标硬件；数值不代表验收硬件。
- 学习负载样本 n=2/入口（学习事件完成快），不作统计结论。
- OS 页缓存未清空。

## 复测（2026-10-13 第二轮，SYNC-15 OPT-DECODE 同步后，同 40 图同种子）

| 口径 | 入口 | 第一轮 P95 | 复测 P95 | 变化 |
|---|---|---|---|---|
| wall | direct | 295.8 | **235.3** | −60.5ms |
| e2e | direct | 288.4 | **216.3** | −72.1ms |
| infer | direct | 134.7 | 89.8 | −44.9ms |
| e2e | http | 258.7 | 245.3 | −13.4ms |
| wall | http | 274.7 | 283.4 | +8.7ms（HTTP 传输噪声） |

- 冷启动：direct 550.5ms / http 530.1ms。
- **判定仍为 FAIL**：最优口径 direct e2e P95 216.3ms > 200ms。decode 优化有效（infer/decode 段明显下降），
  剩余瓶颈为 PNG zlib inflate（约 90ms/张）与 render/persist 段；进一步须靠产线 JPEG/BMP 或 detect_frame 直送。
- 复测摘要覆盖写入 `raw\summary_direct.json` / `summary_http.json`（第一轮数值以本文档表格为准）。

## 第三轮复测（2026-10-13，SYNC-24/25/26 同步后，同 40 图同种子，direct）

| 口径 | 第一轮 | 第二轮 | 第三轮 | 累计变化 |
|---|---|---|---|---|
| wall P95 | 295.8 | 235.3 | **229.2** | −66.6ms |
| e2e P95 | 288.4 | 216.3 | **204.1** | −84.3ms |
| infer P95 | 134.7 | 89.8 | 86.7 | −48.0ms |
| 冷启动 | 458.3 | 550.5 | 532.7 | – |
| 学习负载 wall P95 | ~1800（n=2） | 1502（n=2） | **610.2**（n=12） | 让行使重叠样本显著增多且峰值大降 |

- **判定仍为 FAIL**：e2e P95 204.1ms > 200ms（差距收窄到约 2%）。渲染异步化（SYNC-24）
  与学习让行/锁粒度（SYNC-16/22/26）效果叠加显著；剩余瓶颈为 PNG zlib inflate 与锁外 persist。
- 注：第三轮学习负载 n=12（让行使更多请求与学习重叠），P95 610ms 较第一轮 ~1800ms 大幅改善。
