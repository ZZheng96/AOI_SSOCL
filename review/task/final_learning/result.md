# final_learning（学习 / 反馈 / 证据链）结果

开发版：`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`。未提交 git，未改生产 DB、快照或 stable_AOI_tree，未跑全量数据集。learning_parallel 保持不变。推理热路径没有改动，p95 不受影响。

**`backend\pipeline\service.py` 本轮未修改**，只读取了 `recompute_from_postproc` / `load_raw_heatmap`。

## 模拟方式
- DB：TEMP sqlite，通过 patch `database._engine` / `_SessionLocal` 接入。
- 引擎：用 Mock 替代 `learn_candidate`；`DetectionEngine.__new__` 绕过加载；patch `__init__` 后验证 `get_engine` 单例。
- 存储：TEMP 目录，用 `os.utime` 把文件时间改成 40、200、120 天前，模拟过期文件。
- 热图：用 `DetectionService._finish_detect` 的合成热图生成真实的 float16 raw `.npy`，与 test_raw_heatmap_retention 相同。
- API：FastAPI TestClient + ApiKeyMiddleware，用 Mock key→role 校验 operator/engineer/admin 权限。

## 逐项状态

| AC | 建议状态 | 改动 |
|---|---|---|
| AC-038 | 已实现 | `backend\engine\__init__.py`：新增 `LegacyOnlineLearningBlocked(PermissionError)`；新增类属性 `legacy_online_guard=False`。`submit_feedback(..., allow_legacy_online=False)` 在守卫开启且未授权时抛异常，抛出点在 `_ensure_loaded` 之前。`get_engine()` 返回的生产单例设置 `legacy_online_guard=True`；直接构造的引擎（测试/bench）不受影响。`backend\self_learning\service.py`：legacy 分支透传 `allow_legacy_online=allow_legacy_online`。入口已经保证这里为 True，同时也符合 test_work_candidate_isolation_ac038 的静态守卫，即生产代码不写字面量 True。 |
| AC-072 剩余 | 已实现 | ① 迟到标签：`service.label_lateness()` / `late_label_stats()`。`build_feedback_candidate` 遇到超过 `business_budget.label_late_tolerance_days` 的标签时，写入 `learning_status=late_excluded` 和 `label_lateness`，返回 `{"status":"late_excluded","trained":False}`，不调用 `learn_candidate`，也不置 consumed。② 保留期：新建 `backend\core\retention.py`（`preview_retention` 只读扫描；`execute_retention` 默认 dry-run）。真删要求同时满足 `approval.confirm=True`、approver 非空、`plan_id` 与当前预览一致；执行时逐项复核仍过期、在存储根内、不是 symlink，并追加 `retention_audit.jsonl`。snapshots 和 WAL 永不进入计划。③ API（`backend\api\routes_system.py`）：`GET /system/retention/preview`（engineer）、`POST /system/retention/execute`（admin，默认 dry-run，审批失败返回 403，真删写 log_action）、`GET /system/retention/late_labels`。 |
| AC-063 | 部分实现（API 已完成，UI 未接） | `backend\api\region_review.py`：新增 `HeatmapRecompute`（rel_thresh 取值 (0,1)、min_area、morph_kernel，均可选）和 `POST /detections/{id}/heatmap/recompute`。接口只读，不落库，返回 `persisted=False`。错误码：404 无检测、409 未留存 raw、403 路径不在热图目录、410 raw 已被清理。缺口：`ui\...\region_review_dialog.py` 还没有复算按钮，可在 `/api/detections/{id}/regions` 调用旁边接入。 |
| AC-019 | 部分实现（沿用已有开关，默认未切换） | `algo\eval\offline.py` 已经有 `calibration.isolation=holdout`：槽位只在 fit 子集拟合，CDF、阈值、open 用留出正常图；`cde_report.calibration_contract` 会报告 `self_match_risk`。默认仍为 `self`，本轮没有改默认值，也没有实现 K 折交叉拟合。原因是切换默认值会改变所有品类的 CDF 和阈值，需要在全量或分品类数据集上验证不退化，而这超出了本轮授权（禁止跑全量、禁止精度退化）。 |
| AC-041 / 042 | 未实现 | L0/L1/L2 时序和 review_stage 需要横跨 pipeline\service.py（待同步批次）、DB 字段和 UI。按约束本轮不碰。 |
| AC-044～050 | 未实现 | 争议修订链、撤错重建、参考标签重算、六臂同预算都是跨候选发布和巩固的大改动，与并行 agent 的改动面重叠，本轮时间预算内无法闭环。现有基础可复用：region consumption 和 stale_after_use、`PARENT_REBUILD_REQUIRED_*` fail-closed。 |
| AC-052～054 | 未实现 | 需要定义未知信号分类口径、准入阈值和效果关联指标，属于需求待定项，本轮未动。 |
| AC-058 | 未实现 | 接收→落库率需要接入数据源采集计数，涉及 pipeline\service.py，本轮未动。 |
| AC-059 | 部分（间接） | 临时图本身的生命周期本轮未改。新增的保留期清理覆盖了 `uploads/` 过期文件，规则为 raw_hot_days + 30×raw_cold_months 天，需审批后执行。 |
| AC-068 | 未实现 | 学习资源上界的模拟测量本轮未做。 |

## 测试
新增 `tests\smoke\test_final_learning.py`，共 13 个用例，带进度打印：
- `EngineGuardTest`：test_guard_blocks_unauthorized_legacy、test_get_engine_sets_guard、test_service_legacy_branch_authorizes
- `LateLabelTest`：test_label_lateness_pure、test_late_excluded_not_trained_and_counted、test_on_time_label_trains、test_late_label_api
- `RetentionTest`：test_preview_and_dry_run_delete_nothing、test_execute_requires_approval_and_matching_plan、test_execute_with_approval_deletes_and_audits、test_api_roles_and_default_dry_run
- `HeatmapRecomputeApiTest`：test_recompute_paths（默认结果与在线一致、改阈值、404/409/403/410、只读）、test_route_validation（422/409/200，默认 base_dir 取 storage("heatmaps")）

运行环境：`QT_QPA_PLATFORM=offscreen`，Python311。范围是新增测试加上相关已有测试：test_feedback_evidence、test_feedback_idempotency、test_region_used_status、test_stats_null_and_consume、test_raw_heatmap_retention、test_business_budget、test_work_candidate_isolation_ac038、test_cde_report。结果：**Ran 74 tests, OK**。另外单独跑了 test_async_persist：5 个用例，OK。

## 改动文件
- `backend\engine\__init__.py`（AC-038）
- `backend\self_learning\service.py`（AC-038 透传、AC-072 迟到标签）
- `backend\api\region_review.py`（AC-063 API）
- `backend\api\routes_system.py`（AC-072 保留期和迟到统计 API）
- `backend\core\retention.py`（新建，AC-072）
- `tests\smoke\test_final_learning.py`（新建）
