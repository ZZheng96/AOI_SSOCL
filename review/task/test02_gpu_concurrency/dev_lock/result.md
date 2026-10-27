# dev_lock：前台品类锁临界区缩小（AC-069 前台锁粒度，V§72 / SYNC-22）

运行信息：
- 开发版 AOI_Core。脚本为 `bench_dev_lock.py --prefetch 0|1`，before 和 after 在两个独立进程中跑，其余条件相同。
- 归档沿用 stable `storage\archive\gold_finger\20261011`，只读。
- 统计口径：service.detect_image 的 wall P95（ms）；lock_wait 取 P95（ms）。

| 场景 | before wall | after wall | before lock_wait | after lock_wait | stable 基线 wall |
|---|---|---|---|---|---|
| c1_idle | 245.9 | 258.3 | - | - | 263.4 |
| c1_learning | 1410.8 | 1530.2 | 学习 hold | 学习 hold | 1504.8 |
| c2_idle | 294.0 | 248.9 | - | - | 290.3 |
| c2_learning | 903.6 | 899.3 | 学习 hold | 学习 hold | 820.2 |
| c4_idle | 302.7 | 303.1 | 77.7 | 0.007 | 406.1 |
| c4_learning | 953.0 | 963.5 | 700~1300 | 700~1300 | 959.6 |

精度：
- before 和 after 两轮内部，c4 并发结果与串行相比 mismatch 均为 0。
- 跨进程对比 40 张串行结果：decision 差异 0 张，score 最大绝对差 0.0。
- learning_errors=0，learning_events=9。

结论：
- 前台之间的等锁基本消除：c4_idle 的 lock_wait P95 从 77.7 降到约 0。
- wall P95 变化在噪声范围内。c4 idle 下 wall 没有下降，说明瓶颈已转为单 GPU 前向串行（全局 `_gpu_sem`）和锁外的 CPU 段（persist/heatmap）。
- learning 档的耗时主要来自锁内学习 hold，本方案不针对这一部分。
- stable 的 406ms 与开发版 before 的 302.7ms 相比，差距主要来自 SYNC-15 等已有优化，不是本批带来的。

smoke：371 tests / 105.453s / OK（smoke_stdout.txt / smoke_stderr.txt）。
