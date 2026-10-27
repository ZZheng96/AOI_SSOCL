# AC-072 业务预算与保留期：开发版可配置化（2026-10-13，开发对话）

结论：开发版实现完成；定向 10/10 OK，全量 smoke **405 tests / 109.1s / OK**（基线 395 + 新增 10）。生产验收 0。

## 配置项（`config.yaml` → `business_budget`，默认值 = 用户定值）

| key | 类型 | 默认 | 允许范围 |
|---|---|---|---|
| g2_max_fpr_pct | float | 5.0 | 0～100 |
| g2_min_recall_pct | float | 95.0 | 0～100 |
| g6_max_old_recall_drop_pt | float | 2.0 | 0～100 |
| max_gray_ratio_pct | float | 15.0 | 0～100 |
| raw_hot_days | int | 90 | 1～36500 |
| raw_cold_months | int | 12 | 1～1200 |
| heatmap_overlay_days | int | 30 | 1～36500 |
| snapshot_keep_recent | int | 10 | 1～10000 |
| snapshot_keep_activated | bool | true | — |
| evalrun_audit_months | int | 12 | 1～1200 |
| label_late_tolerance_days | int | 7 | 1～3650 |
| 反馈/事件/WAL 保留 | 固定 | 永久 | 不可改，只展示 |

校验：类型错误（含 bool 混入数值、int 字段传 float）、越界、NaN 一律拒绝，并返回原因；拒绝时内存值不变。POST /system/config 需要 admin。手改 yaml 写入的非法存量值在读取时回落默认。

## 消费点

- 统计页：误报率 KPI 显示“上限 X%”，超限时标红并标“（超限）”；灰区率 KPI 同理。
- 评估页：选中 EvalRun 后，显示 FPR/缺陷召回与达标线的对照（达标/超限/低于下限/无定义）。
- 候选门禁：`Settings.candidate_release_policy(max_predict_ms, valid_seconds)` 把 pct/100 映射为 `candidate_release.evaluate` 所需的 policy（max_fpr/min_recall/max_gray/old_recall_drop），可通过 ready 判定。candidate_release.py 未修改，调用方需改用这个 helper 才会生效。
- 未接入：保留期只登记和展示，系统不自动删除；标签迟到容忍只存配置，入训路径未接入（后端没有迟到判定概念，接入会改变入训行为）。

## 文件

- DIFF `AOI_Core/backend/core/config.py`
- DIFF `AOI_Core/backend/api/routes_system.py`
- DIFF `AOI_Core/ui/pages/settings_page.py`
- DIFF `AOI_Core/ui/pages/stats_page.py`
- DIFF `AOI_Core/ui/pages/eval_page.py`
- DIFF `AOI_Core/ui/main_window.py`
- NEW `AOI_Core/tests/smoke/test_business_budget.py`（10 项）

## 命令（cwd=AOI_Core，`$env:QT_QPA_PLATFORM="offscreen"`）

- 定向：`"C:\Program Files\Python311\python.exe" -m unittest tests.smoke.test_business_budget -v`，结果 10 tests / 0.35s / OK
- 全量：通过 Start-Process 后台运行 `-m unittest discover -s tests\smoke`，结果见 `smoke_out.log`、`smoke_err.log`：405 tests / 109.143s / OK

测试全部使用 TEMP yaml，未触碰生产 DB 或快照。
