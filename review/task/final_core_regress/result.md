# AOI_Core 最终回归结果（2026-10-14）

范围：Windows 本机，git head 7f8b655（工作区有大量未提交修改），未提交、未改生产 DB/快照/stable 树，未跑全量数据集。原始产物在 `raw/`。

## 汇总

| 项 | 当前值 | 原基线 | 结论 |
|---|---|---|---|
| smoke | 553 tests OK（smoke_stderr 154.956s；smoke_postfix_stderr 151.868s） | — | PASS |
| run_all | m01/m02/m10/m11/m14/m15/ui 全部通过，“✓ 全部冒烟通过”（runall_final_stdout.txt） | 修改前 m02 失败（runall_final_pre_m02fix_*） | PASS（m02 改了测试断言，见下文） |
| V1 加载链 | dino/trad vendor，dim 384（sha256_actual.txt） | — | PASS |
| E8 save/load | 10/10，max\|Δ\|=0.0 | — | PASS |
| AOI_feature | 15 组 200 维，AUROC 1.0 | — | PASS |
| 精度 bottle fused | 1.0 | 1.0 | PASS |
| 精度 screw fused | 0.9416 | 0.9452（stable acc_STABLE_screw_accuracy.json 也是 0.9416） | 比旧基线低 0.0036，与 stable 持平；判定：与 stable 持平，不算回退 |
| screw 分槽 | sem 0.9157 / disc 0.5867 / shead 0.9382 / inp 0.5292 / blob 0.7422 / trad 0.0975 / layout 0.4346 | — | 仅记录 |
| MVTec avg（2 类） | 0.9708 | — | 仅记录 |
| bottle_fast | 1.0 | — | PASS |
| btad01 | 0.9771 | 0.9676 | PASS（+0.0095） |
| 速度 A1 模型段 p95 | direct 181.7 / http 185.2 ms | 267.3 ms（生产 fullchain） | 数值更低，但口径不同，只作参考 |
| 速度 A2 热 engine_path | mean 169.5 / p95 192.6；复测 169.4 / 181.7 ms | mean 122.6（原记录 121.1）/ p95 129.9 ms | FAIL（约 +48ms，可复现） |
| 端到端 200ms 红线 | 热 wall p95：direct 285.0、http 286.8 ms | ≤200 ms | FAIL（未达标；此前也未达标） |
| E5 数据集适配 | 27/27 通过，68.7s | — | PASS |
| fullchain.py（生产后端） | 未跑 | — | 未跑（需要生产后端和生产 DB，违反隔离要求） |

环境：GTX 1660 SUPER（驱动 595.95），torch 2.5.1+cu121，CUDA 12.1，Python 3.11.9，sklearn 1.9.0（pickle 由 1.9.1 生成，运行时报 InconsistentVersionWarning）。

## run_all：m02 测试断言修改

原因：在线即学已经改成候选化（legacy_online_guard）。`/feedback` 和 `/review` 现在返回 `engine_update=None`，note 为“已排队候选；未应用、未激活，等待候选任务消费”，DB 中 `consumed=False`，引擎 feedback_log 不再增长。这属于期望值过时，不是功能缺陷。修改文件 `AOI_Core/tests/smoke/m02_e2e.py`：

1. §5 status：候选化时改为断言 `unconsumed_db >= 1`；非候选化分支保留原断言 `feedback_log_len >= 1`。
2. §15 M5a review：候选化时断言 note 含“候选”，并断言 `consumed is (not cand)`；非候选化分支保留 `engine_update is not None`。
3. §18 confirmed 反馈：断言改为 `engine_update is not None` 或 note 含“候选”。
4. §19 M6a 框选复核：与第 2 条相同的处理（note 含“候选”，`consumed is (not cand)`）；region 断言不变。
5. §35 M8a 反馈图：断言改为 `engine_update is not None` 或 note 含“候选”。

以下断言没有改，实测仍然通过：consolidate 生成 v2（`feedback_count=0`）、models v1/v2 登记、online_curve `total_feedback=3`、`correct=[0,0,1]`。
遗留：M5a/M6a 的 print 文案里写死了“consumed=True”，与实际不符，只影响日志文字，没有改。

## 速度明细（raw/speed_*）

输入：`aoi_hard01_2500_effattl5` 中的 8 张工程图，由真实图缩放到 2500，不是原生 2500。gold_finger，隔离 TEMP 快照和 SQLite。

| 脚本 / 模式 | 段 | n | mean | p50 | p95 | max |
|---|---|---|---|---|---|---|
| native_fullchain direct（warmup5，8×4） | reported_model_ms | 32 | 138.3 | 136.2 | 181.7 | 238.9 |
| | wall_ms（服务端到端） | 32 | 222.6 | 219.0 | 285.0 | 317.9 |
| native_fullchain http（uvicorn 回环） | reported_model_ms | 32 | 141.3 | 136.7 | 185.2 | 264.7 |
| | wall_ms（含 HTTP） | 32 | 238.5 | 236.0 | 286.8 | 360.8 |
| hard01_service | engine_path | 8 | 169.5 | 167.6 | 192.6 | 195.1 |
| | service_heatmap_persist | 8 | 212.6 | 206.4 | 230.5 | 230.7 |
| hard01_service 复测 | engine_path | 8 | 169.4 | 171.6 | 181.7 | 184.1 |
| | service_heatmap_persist | 8 | 196.8 | 196.1 | 213.0 | 216.1 |
| A2 原记录（10-12，aoi_hard01_service_zoivzjed） | engine_path | 8 | 121.1 | 120.5 | 129.8 | 132.9 |
| | service_heatmap_persist | 8 | — | 184.6 | 190.1 | 190.5 |

- 只统计 phase=hot，warmup 和 cold_first_request 已剔除。百分位用 numpy 线性插值，样本量小，仅作探索参考。native_fullchain 的 manifest 判定为 “G1 NOT ACCEPTED”，原因是工程输入不能用于正式验收。
- A2 回退的定位：同一命令、同一 prior 下做分段对比。`predict/_finalize/_hier/_attribute_types/future_wait` 的 p50 从 0.8ms 升到 49.9ms，约等于整体增量。原记录有 `predict/_get_item` 34.6ms，新结果没有这一段，改为引擎级 prefetch。config_sha256 也变了（6c4664bb→530525a2）。推测是缺陷分类特征（clf_imgfeat）的 future 在 finalize 时还没完成，可能是 CPU 预取被挪到其他位置或被别的任务挤占；这一点没有验证。按任务要求，没有改代码。
- A1 的 267.3 来自生产 fullchain（生产后端、真实负载）。本次用的是隔离 native_fullchain 的模型段，两者口径不同，不能直接判定改善。

## E5

`tests/_archive/_smoke_dataset_adapter.py`，用临时 AOI_CONFIG 隔离运行（DB 和 storage 都在 `%TEMP%\aoi_e5_20261014145221`，端口 8017），PYTHONPATH 设为 AOI_Core。共 27 项，全部 ✅：probe 识别 mvtec/BTAD/GYU-DET/data_local/solder/shift/jump/误报；solder dir_rules 导入，normal 41 / anomaly 22；component 导入模板 8 张进 L3，anomaly 14。

## 遗留问题

1. HARD-01 速度：A2 engine_path 热 mean 从 121→169ms（+48ms，可复现），集中在 `_attribute_types/future_wait`。原因未查明。
2. 端到端 200ms 红线仍未达标：服务 wall p95 约 285ms，HTTP 约 287ms。没有原生 2500 样本，并发未测。
3. screw fused 0.9416，比旧基线 0.9452 低 0.0036。与 stable 持平，所以不是本轮引入的；trad 0.0975、inp 0.5292、disc 0.5867 偏弱。
4. sklearn 版本不一致：pickle 用 1.9.1 生成，运行环境是 1.9.0，有 InconsistentVersionWarning。建议统一版本或重新导出 pickle。
5. m02 中 M5a/M6a 的 print 文案写死了 “consumed=True”，与候选化后的实际状态不符。
6. fullchain.py（生产后端口径）本次未跑；A1 没有完全同口径的复测。
7. 工作区相对 7f8b655 有大量未提交修改，本次结果对应的是当前工作区，不是 HEAD。

## 收尾调度修复复测

### 结论

- A2（在 `predict` / `predict_frame` 中把 clf/align 预取提前到 `_prefetch_cpu` 之前）对 HARD-01 和 direct **无效**。这两条路径走的是 engine `_predict_prefetch` → `prepare_input`（锁外）+ `_predict_prepared`（锁内），没有经过 A2 修改的代码。A2 两次 future_wait 仍约 40ms。
- 根因已定位：`_predict_prepared` 在锁内、DINO 前向已经完成之后才提交 `_prefetch_clf_imgfeat`。`_img_features` 在 2500² 下单测 p50 62.6ms，而锁内能和它重叠的工作只有约 25ms（_record ~10.6 + build_heatmap_mask ~13–15 + _hier 自身 ~5），所以 `_attribute_types` 空等约 40ms。
- 修复是纯调度改动，修完后 HARD-01 engine_path 回到基线水平，future_wait 从约 40ms 降到约 1–4ms。E8 10/10，max|Δ|=0；bottle fused 1.0；smoke 553 OK。遗留问题第 1 条已定位并修复。

### 代码改动（仅 `algo/eval/offline.py`）

1. `prepare_input`：`t_io` 之后、切块和 DINO 前向之前，先调用 `clf_fut = self._prefetch_clf_imgfeat(img)`，并在返回 dict 中带上 `clf_fut`。这段运行在锁外，与前向重叠。
2. `_predict_prepared`：在锁内判断，只有 `defect_clf` 已 fitted 且 `item is prep["item"]`（同一 img 对象）时才复用 `prep["clf_fut"]`，否则按原逻辑在锁内重新提交。`_align_fut` 没有改，因为它依赖可变的 `_align_ref`，而且只要约 15ms。
3. 数值不变的依据：`_img_features` 是无状态纯函数，输入是同一个 img 对象，采用与否在锁内判定，与原逻辑一致。E8 和 bottle 的实测结果见下文。

### HARD-01 速度（ms，热态，hard01_service）

| 运行 | engine_path mean / p50 / p95 / max | engine future_wait mean / p50 / p95 | service_heatmap_persist mean / p50 / p95 / max | service 内 future_wait mean / p50 / p95 |
|---|---|---|---|---|
| 基线（A2 前） | 122.6（原记录 121.1）/ – / 129.9 / – | – | – | – |
| A2 早先记录 | 169.5 | 49.7 / 49.9 / 66.9 | 192.6 | – |
| A2 早先复测 | 169.4 | – | 181.7 | – |
| A2 postfix | 161.2 / 157.8 / 175.0 / 175.3 | 41.1 / 42.4 / 45.0 | 196.1 / 197.5 / 206.0 / 206.0 | 39.5 / 38.7 / 47.3 |
| A2 postfix2 | 157.7 / 158.6 / 161.9 / 162.1 | 40.0 / 40.3 / 43.3 | 196.2 / 193.8 / 209.5 / 212.0 | 44.4 / 42.3 / 58.4 |
| 修复后 sched | 126.3 / 127.0 / 131.1 / 131.2 | 1.4 / 0.7 / 4.0 | 162.7 / 162.4 / 172.0 / 174.5 | 3.9 / 2.4 / 12.1 |
| 修复后 sched2 | 130.1 / 130.5 / 139.5 / 141.4 | 3.7 / 2.0 / 8.9 | 162.3 / 160.8 / 174.2 / 180.3 | 4.0 / 1.5 / 14.3 |

原始数据：`raw/speed_hard01_postfix*`、`raw/speed_hard01_sched*`。sched2 比基线 mean 高约 4–8ms，在两次运行的波动范围内，但没有完全回到 121–123。

### native_fullchain direct（ms，hot n=32）

| 运行 | reported_model_ms mean / p50 / p95 / max | reported_partial_e2e_ms mean / p50 / p95 / max | wall_ms mean / p50 / p95 / max |
|---|---|---|---|
| 旧（本文前述） | 138.3 / 136.2 / 181.7 / 238.9 | 195.9 / 193.1 / 243.2 / 292.4 | 222.6 / 219.0 / 285.0 / 317.9 |
| 修复后（`raw/speed_direct_postfix_*`） | 84.7 / 83.5 / 92.3 / 96.4 | 141.6 / 141.8 / 150.7 / 157.6 | 162.8 / 162.4 / 171.5 / 186.1 |

- direct 只在修复后的代码上跑过，没有单独跑 A2-only 的 direct，所以无法拆分 A2 和本次修复各自的贡献。按路径分析，A2 不会影响 direct。
- manifest 判定仍为 “G1 NOT ACCEPTED”，与旧运行相同（工程输入不算正式验收）。wall p95 171.5 低于 200ms 红线，但输入是缩放后的工程图，没有原生 2500 样本，也没有测并发，只能作为参考，不能据此关闭遗留问题第 2 条。

### 数值一致性

- E8 save/load：10/10，max|Δ|=0.0，PASS（`raw/e8_postfix_accuracy_saveload.json`）。
- bottle fast：fused_auroc 1.0；sem 0.9947、disc 0.9909、shead 1.0；ms_per_image 76.3（`raw/acc_bottle_fast_postfix.json`）。

### smoke

`-m unittest discover -s tests\smoke`：Ran 553 tests，OK（154.415s）（`raw/smoke_postfix2_stdout.txt` / `smoke_postfix2_stderr.txt`）。

## 三项补全后回归（2026-10-15）

背景：本轮新增了按类型孵育（`algo/ssocl/incubate.py`、`banks.py`、`feedback.py`）、伪标签（`backend/self_learning/label_audit.py`、`service.py`）、E_open 自动注册（`algo/feature_revision.py`、`backend/api/open_alerts.py`），并在 PCB_Dual 中新增了 FeaturePage。环境与上一轮相同，`QT_QPA_PLATFORM=offscreen`。本轮未跑全量数据集，未改生产 DB、快照或 stable 树，未提交。产物统一用 `raw/r3_*` 前缀。

| 项 | 本次 | 上次 | 结论 |
|---|---|---|---|
| smoke 首跑 | Ran 572，FAILED 1（`r3_smoke_out/err.txt`） | 553 OK | 失败 1 项，原因见下文 |
| smoke 修复后 | Ran 572，OK（152.7s，`r3_smoke_postfix_*`） | 553 OK | PASS（新增 19 个用例） |
| run_all 首跑 | m01 失败（`r3_runall_*`） | 全部通过 | 不可复现，见下文 |
| run_all 复跑 | m01/m02/m10/m11/m14/m15/ui 全部通过，“✓ 全部冒烟通过”，UI 含特征工程页（`r3_runall_rerun_*`） | 全部通过 | PASS |
| E8 save/load | 10/10，max\|Δ\|=0.0（`r3_e8_accuracy_saveload.json`） | 10/10，Δ=0 | PASS |
| bottle_fast | fused 1.0；sem 0.9947 / disc 0.9909 / shead 1.0（`r3_acc_bottle_fast.json`） | 1.0，分槽相同 | PASS，无变化 |
| HARD-01 engine_path（mean / p50 / p95 / max） | 107.3 / 104.0 / 120.1 / 122.1；复测 109.5 / 105.5 / 117.8 / 118.2 | sched 126.3 / 127.0 / 131.1；sched2 130.1 / 130.5 / 139.5 | PASS，未变差（低约 17–20ms） |
| HARD-01 service_heatmap_persist（mean / p50 / p95 / max） | 142.8 / 139.3 / 153.5 / 153.8；复测 140.1 / 137.0 / 151.8 / 152.4 | 162.7 / 162.4 / 172.0；162.3 / 160.8 / 174.2 | PASS，未变差 |

命令（cwd AOI_Core）：
- run_all：`python -u tests\run_all.py`
- E8：`%TEMP%\fcr_e8_r3.py`。这是上一轮 `fcr_e8.py` 的副本，只改了输出文件名。
- bottle_fast：`python -u -m tests.bench.static_accuracy --dataset mvtec --categories bottle --profile fast --out raw\r3_acc_bottle_fast.json`
- HARD-01：`python -u -m tests.bench.hard01_service --prior-result %TEMP%\aoi_hard01_2500_effattl5\result.json`

HARD-01 的结果 JSON 已复制为 `raw/r3_speed_hard01_{xqds2rt3,qtgh9gsd}_result.json`。完整编排见 `raw/r3_chain.log`。

### smoke 失败与最小修复

- 失败用例：`test_ui_stats_caliber.BackendToUiTest.test_real_overview_to_dashboard`，第 140 行断言 `range="7d"` 时口径行含“近 7 天（起点 ”。该测试文件是新增的，尚未纳入 git 跟踪。
- 原因：后端 `/stats/overview` 的 scope_stats 已经返回 `range` 和 `window_start`，但 `ui/pages/dashboard_page.py` 的 `scope_caliber_texts` 生成口径行时没有使用这两个字段。这是 UI 文案缺失，与三项补全无关。
- 修复：在 `scope_caliber_texts` 中，当 `range` 不是 `all` 且 `window_start` 存在时，在口径行末尾追加 `｜近 7 天（起点 YYYY-MM-DD HH:MM:SS）`。全局（all）口径的文案不变。改动 3 行，`test_ui_stats*caliber.py` 共 14 项全部通过，全量 smoke 572 项 OK。

### run_all 首跑 m01 失败

- 现象：m01 子进程打印 `[storage]` 后直接以非 0 退出，stdout 和 stderr 中都没有 traceback。
- 复跑：单独跑 `-m tests.smoke.m01_engine` 通过；`run_all.py m01` 通过；全量 run_all 复跑也全部通过。判定为偶发，不可复现。推测是 CUDA 或原生层的瞬时问题，未能定位。没有改代码。如果后续再次出现，建议在 run_all 中捕获子进程退出码对应的信号。
