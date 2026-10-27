# final_perf 结果（性能 / 资源 / 显存 / 冷启动）

环境：GTX 1660 SUPER 6GiB，cv2 5.0.0，PIL 12.2.0，engine_fast.yaml（cuda，single 512）。
数据：稳定版快照只读复制到 TEMP（gold_finger v2 legacy 非 DINO；smt_solder v4 DINO），gold_finger/20261011 随机 20 张（seed 20261013，尺寸混杂 304×432 ~ 4096×3000），2500² 另用合成图。生产 DB/快照/stable_AOI_tree 未改。

定向测试：`tests\smoke\test_final_perf.py` 13 个 OK；回归 test_async_persist / test_isolated_candidate_path / test_candidate_release / test_snapshot_layout_compat 合计 88 个 OK（含 final_perf），test_cde_report / test_explain 17 个 OK。

---

## G1 / AC-066 / AC-007 e2e（200ms 目标）

建议状态：部分实现（解码已到 cv2 下限，上传重编码已消除；200ms 在本机 GPU 上不可达，瓶颈是推理本身）。

解码构成（中位数 ms，bench_decode.json `decode`）：

| 图 | PIL | cv2+BGR2RGB | cv2 RGB 直出 | `_decode_rgb` | REDUCED_2 | PIL PNG 重编码 |
|---|---|---|---|---|---|---|
| 真实 2500² png（9.8MB） | 120.2 | 107.6 | 97.5 | 100.5（maxdiff 0） | 98.5 | 557 |
| 同图转 jpg | 76.7 | 48.6 | 37.4 | 43.1 | 26.2 | — |
| 合成 2500² png | 245.1 | — | 112.2 | 114.1 | — | 640 |
| 合成 2500² jpg | 77.2 | — | 45.7 | 44.8 | 32.2 | — |

结论：
- PNG 解码约 100ms 几乎全是 libpng inflate；cv2 RGB 直出已是下限（另一 agent 的 OPT-DECODE 已上线，逐图 maxdiff=0）。
- IMREAD_REDUCED_2：PNG 无收益（98.5 vs 97.5，PNG 不支持 DCT 降采样）；JPEG 省约 11ms，但上采样后像素 maxdiff 28–56，不逐图一致 → 不采用。（精度门 6 张恰好都是缺陷图，score clip 到 2.0 使 abs_diff=0 无说服力，以像素门为准。）
- 解码/推理流水重叠：20 张串行 8315.8ms → 流水 6710.1ms，吞吐 ×1.239，scores_identical=True。单请求延迟不变，服务层不改，仅作吞吐结论（批量/视频场景可用）。
- 上传改 ndarray/JPEG：/detect/frame 已是 bytes→imdecode 零拷贝；JPEG 是有损，客户端改格式会改像素，不作为服务端改动。

改动（本组）：
- `backend\pipeline\service.py` `detect_pil` 新增可选 `raw: bytes`：上传本身是 cv2 安全 8 位 PNG（`_cv2_decodable`）时直接写原始 bytes，跳过 PIL 重编码；其它格式回退 `convert("RGB").save(compress_level=1)`（无损，4096×3000 编码 1131→650ms）。仅改上传落盘，不触及 SYNC-24/25 的 claim/持久化顺序与 detect_image 口径。
- `backend\api\routes_detect.py` /detect/upload 传 `raw=data`。

前后数字（bench_upload.json，TestClient 进程内 ASGI 往返，含 multipart+落盘+检测+落库，不含真实网络，同 seed 同 10 张）：

| 口径 | 改前 p50 / p95 | 改后 p50 / p95 / max |
|---|---|---|
| HTTP upload_png | 1086.7 / 1971.0 | 513.9 / 894.3 / 894.3 |
| HTTP frame_png（对照） | 472.3 / 702.6 | 519.5 / 802.1 / 802.1 |

upload 超 1s：改前多张 → 改后 0；upload 与 frame 逐图 score/is_anomaly 全一致（all_same=true）。

空闲 / 负载 e2e（bench_decode.json，20 张，未受本次改动影响）：

| 口径 | p50 | p95 | max |
|---|---|---|---|
| 空闲 svc_e2e（解码+推理+热图同步段，不含 HTTP/落库） | 376.3 | 609.0 | 624.1 |
| 空闲 svc_wall（含落库，不含 HTTP） | 393.0 | 628.3 | — |
| 空闲 纯推理 latency_ms | 278.0 | 428.6 | — |
| 空闲 HTTP frame_jpg（e2e 606.7 p95） | 334.4 | 721.7 | — |
| 合成 2500² png e2e（5 次） | 481.9 | 504.2 | — |
| 负载 svc_e2e（2 后台线程同品类，含排队） | 810.5 | 1163.6 | 1403.4 |
| 负载 lock_wait | 437.9 | 695.3 | — |
| 负载 纯推理 | — | 491.9 | — |

测试：`UploadRawBytesTest`（3）：RGB/L PNG 原始 bytes 落盘且像素与 PIL convert("RGB") 逐位一致；RGBA/调色板 PNG、JPEG 回退重编码且像素一致；无 raw 走旧路径。
剩余缺口：200ms 不可达（纯推理 p95 已 428ms）；负载下 6/43 超 1s，来自 GPU 品类锁排队，需排队上限/降载策略，不在本组改动范围。

## AC-069 rolled_back 路径锁内耗时

建议状态：部分实现（仅评估，未改）。
结论沿用 V§78：rolled_back 的 check 依赖锁内一致快照，无法拆到锁外；deepcopy 影子 370–1290ms 为主耗时。submit_feedback 已透出 `lock.{yield_ms,prefetch_ms,lock_wait_ms,lock_hold_ms}` 可观测。剩余缺口：锁内 deepcopy 未优化（需 copy-on-write 影子，改动面大）。

## AC-070 显存上界

建议状态：已实现（候选路径）；多槽/在线路径为观测。

改动：
- `algo\ssocl\candidate_budget.py`：新增 `vram_fraction`（默认 0.85，按设备总显存自适应）与 `max_vram_bytes`（覆盖）；cuda 下 reset_peak 并在每个 checkpoint 用 `max_memory_allocated` 校验，超限抛 `CandidateDeferred`（deferred、可重试）；非 cuda `enabled=False, skip_reason="no_cuda_device"`；snapshot 中 `vram_guaranteed="checkpoint"` + `vram` 明细；非法配置 ValueError。
- `backend\engine\__init__.py` `_learn_candidate_locked`：`torch.cuda.OutOfMemoryError` → empty_cache → deferred，工作态不破坏。

实测峰值（bench_vram.json，max_memory_allocated，PyTorch 分配器口径，不含 CUDA context）：

| 场景 | base MiB | peak MiB | delta MiB |
|---|---|---|---|
| idle gold_finger（legacy 非 DINO） | 0.0 | 0.0 | 0.0 |
| idle smt_solder（DINO，5 张） | 0.0 | 117.2 | 117.2 |
| 训练 submit_feedback(allow_legacy_online) | 93.8 | 125.3 | 31.5 |
| 回放（驱逐 pipe → 重载 + 重放 1 条 WAL） | 102.0 | 127.7 | 25.8 |
| 多槽（两品类常驻交替推理） | 103.5 | 126.9 | 23.4 |

最大 127.7 MiB，默认上限 5222.1 MiB（6GiB×0.85），余量充足。learn_candidate 在稳定版快照上报“legacy 快照缺必要学习态”（预期，需重新 prepare），因此候选训练峰值用单测覆盖。

测试（VramBudgetTest 5 个）：自适应上限与覆盖、非法配置；候选全链路超限 → deferred 且生产 pipe/版本/WAL 不变；OOM 模拟（patch feedback 抛 OutOfMemoryError）→ deferred、预测可用、新 event 重试成功；无 CUDA 降级跳过；真实 CUDA 分配 64/192MiB 触发超限。
剩余缺口：checkpoint 粒度校验（两 checkpoint 间瞬时峰值只能事后发现，靠 OOM 转 deferred 兜底）；在线即学/多槽常驻不受预算约束，仅观测。

## AC-071 冷启动预算

建议状态：已实现。

改动（`backend\engine\__init__.py`）：`_ensure_loaded` 分段计时 load/replay 并 `_cold_record`（失败不入缓存、不记录、不切版）；`COLD_START_BUDGET_DEFAULTS`（load 30s / replay 10s / first_infer 5s / warm 1s），可由 `ssocl.cold_start_budget` 覆盖，未知键 ValueError；`warmup(category, rgb, warm_runs=3)` 统计首图/热态；`cold_start_stats()` 只读视图。本轮修正：每次冷载开新记录、返回副本（原先返回内部 dict，重载后旧结果被覆盖）。

实测（bench_vram.json，ms）：

| 品类 | load | replay | first_infer | warm（中位） | within_budget |
|---|---|---|---|---|---|
| gold_finger v2 | 549.2 | 0.09 | 314.0 | 326.8 | True |
| smt_solder v4（首载含 DINO 权重） | 1547.8 | 0.13 | 216.7 | 112.9 | True |
| smt_solder 重载 + 重放 1 条 WAL | 301.3 | 477.9 | — | — | True |

测试（ColdStartTest 3 个）：分段统计与预热；预算可配置/超限进 violations/未知键报错；load_snapshot 抛错 → 不入缓存、无统计、current_version 不变。
剩余缺口：启动时未自动调用 warmup（需由服务启动钩子决定是否预热哪些品类）；超预算只记录不阻断。

## §65 遗留

1. 降级 slot 分数恒定：不是 bug。67 张“正常图”实为同一 MD5 文件；缺陷图被 CDF clip 到 2.0。`DegradedScoresVaryTest`：降级快照下不同图 raw 分数互不相同，同图恒定。诊断：diag_degraded_slots.py / .json。建议状态：已实现（结论 + 回归测试）。
2. decision_trace 为空：`service._finish_detect` 顶层透出 `decision_trace`，入库不再被 [] 覆盖（trace_len 1，DB 1）。`DecisionTraceTest` OK；test_async_persist EXPECTED_KEYS 同步。建议状态：已实现。

## 产物

bench_decode.py/.json、bench_upload.py/.json、bench_vram.py/.json、diag_degraded_slots.py/.json，均在本目录。
