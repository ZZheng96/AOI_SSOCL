# round2_learning 结果

开发版 `d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`；未碰 stable_AOI_tree、生产 DB/快照、禁改文件（dino.py / pipeline\service.py / candidate_release.py / parity.py / engine\__init__.py 均未改）；未提交 git；learning_parallel 默认未动。

## 改动文件

| 文件 | AC | 内容 |
|---|---|---|
| `ui\widgets\region_review_dialog.py` | 063 | 新增 `build_recompute_payload`（与后端约束一致，None=沿用原参数）、`RECOMPUTE_ERRORS`；画布只读预览层 `set_preview_boxes/preview_items`（青色虚线，不入区域）；对话框新增 rel_thresh/min_area/morph_kernel（0=原值）+「热图复算(只读预览)」「清除预览」按钮，调 `POST /api/detections/{id}/heatmap/recompute`，失败（409/403/410/404/None）提示并清空预览；不改区域与修订号 |
| `algo\eval\offline.py` | 019 | 抽出纯函数 `resolve_isolation`；默认 `isolation` 由 `self` 改为 `auto`：留出后 fit ≥ `holdout_min_fit`(16) 走 holdout（ratio 0.2，min 4，seed 固定），否则回退 self 并记原因；显式 self/holdout 行为不变；决策写入 CDE 报告 `calibration.isolation_resolution` |
| `backend\self_learning\label_audit.py`（新） | 042/044 | `audit_label_order`（ok/late/before_detection/out_of_order/unknown_time + first/re_detect 分报，trainable 标记）；`LabelRecord` + `resolve_labels`（人工/伪标签分存，人工优先取最高修订，同修订冲突=disputed，伪标签过期或 teacher 不符=stale，伪标签缺 teacher_version 拒绝） |
| `backend\self_learning\service.py` | 042 | 追加只读 `label_order_audit(category)`：join Feedback/Detection、排除作废、容差取 business_budget.label_late_tolerance_days |
| `backend\api\open_alerts.py` | 052 | 新增只读 `GET /api/review/open_alerts/summary?category=`：by_status / by_cause / by_learning_status（关联反馈 learning_trace）/ admission_candidates |

推理路径：上述改动均不在单图推理调用链上（offline.fit 仅建库/重建；UI 按钮按需触发；审计/汇总为只读接口），单图耗时不变（估算，未实测 GPU）。

## 测试

新增 `tests\smoke\test_round2_learning.py`（9 用例，带 `[AC-xxx i/n]` 进度）：全部通过。
- AC-063：payload 校验；离屏对话框点击按钮 → 请求路径/体正确、预览框数、标签文本、区域与修订号不变、清除预览；409/None 失败；参数无效不发请求
- AC-019：auto 默认 n=100→holdout 80/20、互斥且覆盖、seed 确定；n=10 回退 self；显式 self/holdout；未知值；边界 20/19
- AC-042：合成时间序列六种情况；TEMP sqlite 服务层（作废排除、只读）
- AC-044：resolve_labels 各分支 + ValueError
- AC-052：TEMP sqlite + TestClient 汇总计数，且不遮挡 `/{detection_id}` 路由

回归（定向）：test_final_learning / region_review_edit / region_review_draw / open_alert_queue / business_budget / cde_report / final_model_protocol / final_sim_review / stats_null_and_consume / ac042_review_stage 共 88 用例 OK；snapshot_layout_compat / localization / candidate_learning_state / feature_registry / isolated_candidate_path / detection_version / dataset_isolation 共 60 用例 OK。改动文件 py_compile 通过。未跑全量数据集。

## 各 AC 建议状态

| AC | 建议状态 | 证据 / 缺口 |
|---|---|---|
| 063 | 已实现（待真机确认） | 后端复算 API + UI 按钮/预览层已接，离屏测试覆盖；缺真实桌面与真实留存 raw 热图端到端验证 |
| 019 | 部分实现 | 默认改 auto（够样本即 holdout，可回 `isolation: self`），决策入报告；缺真实数据集上 holdout vs self 的精度/阈值对比（auto 下 fit 集少约 20%，需在真实数据确认不退化后才能定为已实现）；已有快照不受影响，需重建才生效 |
| 042 | 部分实现 | 时序审计纯函数 + 只读服务；尚未接入候选构建前的自动过滤（`build_feedback_candidate` 仍只用迟到判定）与 UI/API 暴露 |
| 044 | 部分实现 | 标签身份数据结构与解析规则已有并测试；未落 DB 表、未接入训练样本组装 |
| 052 | 部分实现 | 处置→反馈→学习状态汇总接口；缺 UI 面板与与候选发布结果的回链 |
| 041 | 部分实现（维持） | RNG 契约已有；缺真实 GPU 与完整回放 |
| 045 / 046 | 未实现（维持） | 本轮未动；需改禁改文件或真实数据 |
| 047 / 048 | 部分实现（维持） | candidate_guard / reason_code 已有；缺完整重建流程（涉及 candidate_release，禁改） |
| 049 | 部分实现（维持） | paired.py 已有；缺六臂对照与 CI |
| 050 | 部分实现（维持） | status / reason_code 已有；缺时间线、水位、取消 |
| 053 | 部分实现（维持） | 实际版本已记录；全血缘未完成 |
| 054 | 部分实现（维持） | learning_trace 透传；缺独立 evidence |
| 058 | 部分实现（维持） | 游标与口径已补；本轮无新增 |
| 068 | 部分实现（维持） | 预算与 deferred 已有；缺真实墙钟与显存实测 |
