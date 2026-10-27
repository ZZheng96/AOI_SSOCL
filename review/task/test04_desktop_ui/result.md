# TEST-04 真实桌面 UI 联调（AC-051/056/057，兼证 AC-052/053/058/067/072）

## 结论：PASS（所覆盖场景），部分场景未覆盖

真实桌面（非 offscreen）启动稳定版 `main.py`（后端 8017 + PySide UI），逐项核对 UI 与 API 一致。

## 已验证场景（截图 + API JSON 配对）

| 场景 | UI 表现 | API 配对 | 判定 |
|---|---|---|---|
| Gauge 空态 | 实时页 Gauge 显示 "0.000 / 未知·未判定"，阈值/版本/raw-cal 分等全部 "-" | 无检测时不补 0 不猜测 | PASS（AC-056/057 空态契约） |
| 工作台口径卡 | 推理耗时 110ms（注明"仅模型推理"）、e2e "156/P95 317ms"、接收→落库率"不可用（暂无相机接收计数数据源，不以 0 代替）"、口径行完整 | /api/stats/overview scope_stats | PASS（AC-058，SYNC-13 真实桌面首验） |
| open 告警横幅 | 红色"不完备预警：smt_solder 近 6 帧未知模式 3 次(50%)…未知不等于缺陷…处置不自动训练/发布" | /api/review/open_alerts total=3 | PASS（AC-052/057 open≠缺陷） |
| 开放告警样本列表 | 3 条（检测 918/916/914），模型版本"未知"、open 分"None"、状态"待处置"、证据"旧证据未知" | items 字段逐一对齐（model_version=null→未知、legacy_unknown→旧证据未知、is_defect=false） | PASS（AC-053 legacy 未知保留） |
| 开放告警详情 | 原图 + 原始 JSON 全文 + 处置表单（原因必填），标题"未知不等于缺陷" | 对话框内 JSON 与 API 响应逐字段一致 | PASS |
| 评估看板 + 业务预算线 | gold_finger test 评估：AUROC 0.500/FPR 100%/"✗ 精度未达标"；**业务预算达标线：FPR ≤5%（超限）、召回 ≥95%（达标）**；延迟柱图红色超预算柱 + 200ms 红色虚线 | EvalRun id=1 绑定 model_version=2 + snapshot_meta_sha256 + split=test；metrics 与 UI 一致 | PASS（AC-067 门禁展示 + AC-072 用户定值生效） |

## 未覆盖/边界

- **Gauge 有值态**（真实 score+decision 的 Gauge）未在桌面验证：需要数据源→工单→试检链路（多次 DB 写入），本轮未建工单。空态契约已验证，有值态由离屏契约测试（8 项）覆盖。
- 告警优先于 warn 的聚合排序：本轮未构造同时含 alert+warn 的场景。
- 评估运行向稳定版 DB 写入 EvalRun id=1 与 task 18（正常应用记录，可在评估看板"删除所选"移除）。
- 与 TEST-07 交叉确认：生产 v2（降级模式）test 集 FPR=100%、AUROC=0.5，与 TEMP 新 prepare 模型结果一致——精度问题为系统性，非单模型问题。

## 产物

- raw/eval_board_screenshot.png（全屏 1920×1080，评估看板最终状态）
- raw/api_snapshots.json（open_alerts/review_queue/review_stats）
- raw/api_eval_pair.json（eval_runs + task18 全量 JSON）
