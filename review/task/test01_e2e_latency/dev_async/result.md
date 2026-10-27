# G1 dev_async：单图 direct e2e P95 继续压降（开发版）

结论：可视化渲染移出关键路径后，render 段 P95 确定性下降约 15ms（35→19ms）。机器空闲时 e2e P95 为 180.6ms，≤200。机器有负载时，三次同进程 A/B 的 new 侧 P95 为 225.3、214.5、190.4ms，不能稳定达标。剩余瓶颈是 decode（PNG inflate，约 93~107ms）和 infer 抖动（66~96ms），两者都在本批不可改的范围内（engine/offline）。精度逐图一致。

## 改动（仅开发版 AOI_Core）
- backend/pipeline/service.py
  - 异常图的 overlay（resize/cvt/着色/addWeighted/画框/编码）、热图 JPEG、mask PNG、raw npy 全部改为提交到 `_io_exec`（2 worker）。主线程只同步计算 `_heatmap_u8` 和 `_mask_and_boxes`，因为响应中的 heatmap_boxes 依赖它们。
  - 响应字段保持不变：路径、id、decision、框。DB id 仍在 `_persist` 中同步获得。
  - `_track`/`_pending`/`wait_artifact`：登记在途产物。`_safe_job` 捕获异常并写 `logger.exception`，任务失败会留下日志。
  - `finally` 中照旧 join io_futs，并且在 alarm 之前完成。
  - 两处逐位等价的微优化：
    - `_heatmap_u8` 改为原地 subtract/divide/multiply，耗时 7.3→3.4ms。
    - mask 二值化改用 `cv2.threshold(THRESH_BINARY)`，结果与 `(u8>t)*255` 相同。
    - 等价性检查：30 张随机图（含常数图）× 5 个阈值，结果全部相等。
- backend/api/routes_system.py（超出“只改 service.py”的范围，原因：异步写盘后，/api/file 可能先于文件落盘被访问）
  - `api_file` 在 `is_path_allowed` 之后调用 `wait_artifact`：超时返回 503 并带 Retry-After，任务失败返回 500。
- NEW tests/smoke/test_async_persist.py：5 项，覆盖字段与文件就绪、异步失败记日志、wait_artifact 超时与失败、等价性。

## 分段计时（bench_breakdown.py，最终代码，6 张 TEST-01 2500² PNG × 4 轮，n=24，breakdown.json）
| 分段 | p50 | p95 (ms) |
|---|---|---|
| decode | 92.6 | 98.6 |
| infer | 54.9 | 66.2 |
| render（同步剩余） | 0 | 18.9 |
| io_wait（不计入 e2e，n=4） | 0 | 15.8 |
| persist（不计入 e2e） | 4.9 | 20.5 |
| alarm | 0 | 0 |
| **e2e** | 149.4 | **180.6** |
| wall | 156.6 | 210.3 |

render 子步 p95（ms）：heatmap_u8 25.7→9.6，mask_boxes 9.1→5.0。改前 overlay 一项为 22~25，现已异步。

## 改前 / 改后 P95（ms）
| 口径 | 改前 | 改后 |
|---|---|---|
| 任务基线（稳定版 TEST-01 复测） | 216.3 | — |
| breakdown（不同时段） | 226.2 | 180.6 |
| A/B run2（同进程交替） | 225.3 | 225.3 |
| A/B run3 | 203.4 | 214.5 |
| A/B run4（ab.json） | 207.8 | 190.4 |
| render 段 P95（三次 A/B） | 34.0 / 35.9 / 33.5 | 18.9 / 19.2 / 18.1 |

e2e P95 主要受 infer 和 decode 的离群值影响。例如 run3 中 new 侧第 4 轮出现 infer 224 和 218 两个离群值。只看异常图逐对比较，new 普遍更低：201→187，204→185，220→194。

## 精度
三次 A/B 的 `consistency_all_identical` 都是 True，`ab_rows_decision_and_score_identical` 都是 True：
- 6 张图的 decision、score 逐位相同。
- defect_boxes 相同。
- heatmap、overlay、mask 文件 sha 相同。
- 响应键集合相同。

det881 的 localization_equal=False，唯一差异键是 `heatmap_postproc`。原因是其中的 mask_path、raw_heatmap_path 含有 `stem=原名_{毫秒时间戳}`，每次检测都不同，不是语义差异。heatmap_boxes 一致。

## smoke
`-m unittest discover -s tests\smoke`：**383 tests / 98.3s / OK**（smoke_out.log、smoke_err.log）。
注意：PowerShell 5 的 `1> 2>` 会截断原生进程的 stderr，这次改用 `Start-Process -RedirectStandardOutput/-RedirectStandardError` 来分别重定向。

## 瓶颈（未稳定达到 200 的原因）
1. decode：cv2 解 2500² PNG（zlib inflate），p95 98~107ms，约占 e2e 的一半。下采样已在 §69 判定不可行。换解码库或缓存属于 engine 或输入格式的改动，不在本批范围内。
2. infer：p50 55~70ms，但负载下 p95 达 82~96ms，并有 >200ms 的离群值。抖动来自 CPU 争用，需要 SS-A 算力隔离或 engine 侧改动，均不在本批可改范围内。
3. 同步 render 已降到 p95 约 19ms（heatmap_u8 + 连通域框）。这部分是响应框必需的，不能再移出。

## 产物
bench_breakdown.py / breakdown.json·log，bench_async_ab.py / ab.json·log（run4）、ab_run2.json、ab_run3.json·log，service_before.py（改前），smoke_out.log / smoke_err.log，progress.md。
