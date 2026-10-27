# §78 / SYNC-26：学习锁内持有缩短（check→recenter 锚定分复用）

环境：开发版 AOI_Core，TEMP 引擎和 TEMP sqlite。样本取自 stable archive gold_finger 的 2500² 图，seed 777，训练 40 张，反馈 12 张（ab 轮 20 张）。前台循环 6 张缓存命中图，间隔 20ms。`learning_parallel=False`。未触碰生产 DB 和快照。

## 锁内持有 lock_hold_ms

| 轮次 | accepted（含 recenter） | rolled_back 均值 | 全部均值 | 全部 P95 | gate reuse |
|---|---|---|---|---|---|
| before r1 | 908.5 | 586.8 | 613.6 | 783.0 | 0/1 重算 |
| before r2 | 1018.9 | 622.5 | 655.5 | 836.5 | 0/1 重算 |
| after r1 | 684.4 | 644.2 | 647.6 | 684.3 | 1 复用 |
| after r2 | 630.4 | 664.5 | 661.6 | 724.6 | 1 复用 |

- accepted 事件：均值 963.7→657.4ms，降低约 306ms（-32%），对应省掉一整轮 20 锚点 predict（约 450ms 中的大部分）。全部事件 P95 从 783～836 降到 684～725ms。
- rolled_back 事件：after 跨进程均值看起来高约 50ms。同进程逐事件交替关/开复用（bench_ab.json）后，off 为 658.6ms，on 为 662.7ms，差异属于噪声。探针 probe_overhead 测得：`_gate_anchor_pass` 与逐张调用耗时相当（433/430/461 vs 468/451/450ms）；`_gate_state_token` 为 0.09～0.15ms；分数逐张相等。

## 前台推理 wall（学习进行中，两轮合并）

| | P50 | P95 | max |
|---|---|---|---|
| before | 18.7 | 34.8 | 904～1018 |
| after | 17.7 | 29.8 | 701～743 |

空闲基线 P50 为 15～19ms，P95 为 17～24ms。前台受 §66 让行和缓存命中保护，P95 本来就低。本改动主要压低前台 max，也就是撞上 accepted 长锁的那一次。

## 精度

四轮 actions 序列完全相同，gate base_mean 相同（0.43490）。学习后 8 张 eval 的 fused 与 before_r1 的最大绝对差为 **0.0**。

## 结论与边界

- 本样本中 accepted 占比 1/12，所以整体均值改善有限。占多数的 rolled_back 路径锁内耗时仍为 check 的约 600ms，未缩短，原因见 V§78。
- 产物：
  - bench_learnlock.py（before/after/ab）
  - bench_{before,after}_r{1,2}.json、bench_ab.json
  - probe_overhead.py
  - run.log
  - smoke_out.log、smoke_err.log（全量 395 OK）
- 生产验收 0。
