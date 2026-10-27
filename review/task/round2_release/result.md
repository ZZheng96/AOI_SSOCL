# round2_release 结果

范围：AOI_tree\AOI_Core（开发版）。未动 stable_AOI_tree / offline.py / region_review_dialog.py / backend engine·service；未提交 git；未改生产 DB/快照；仅合成数据/fake。learning_parallel 默认值未改，推理路径未改（新增代码均为离线/只读运维函数）。

## 改动文件
- `algo\release_ops.py`（未跟踪新文件，此前无引用、无测试）
  - 修复：`SNAPSHOT_REQUIRED` 中 `python/torch/numpy` → `python_version/torch_version/numpy_version`，与 `persist.write_snapshot_state` 实际写入键一致（原先真实快照恒判 incomplete）。
  - 新增：`revision_binding(store, model_snapshot)`（AC-061），基于 `FeatureRevisionStore.detection_stamp` + `lineage` 只读给出 bound / 未绑定 / 快照不匹配。
- `tests\smoke\test_round2_release.py`（新增，12 例，终端打印 `[i/12]` 进度）

## 测试结果
- `tests\smoke\test_round2_release.py`：12 passed（7.9s）
- 相关已有：test_final_release_chain / test_feature_revision / test_feedback_routes / test_bank_lineage_incubate / test_explain / test_final_sim_slots / test_hard_admission / test_snapshot_layout_compat：55 passed，36 subtests passed（22s），无回归。

## AC 建议状态
| AC | 建议状态 | 证据 | 缺口 |
|---|---|---|---|
| AC-011 组级独立选组 | 部分实现→基本实现 | `group_selection_plan`；测试 #1（局部组不被整图否定、样本不足/反向不选、不自动生效） | 未接入 UI/API |
| AC-012 档案血缘/完整性 | 基本实现 | `stamp_profile_lineage` / `profile_completeness` / `report_status`；测试 #2 | 贡献档案生成流程尚未自动调用 stamp |
| AC-013 trad 打分协议描述 | 已实现（文档化审计） | `trad_protocol_audit`；测试 #3 | — |
| AC-014 逐框归因 | 基本实现 | `box_evidence` + `box_attribution_audit`（dominant_slot 空→image_fallback）；测试 #4 | 未在 UI 展示审计结果 |
| AC-015 表示变化计划 | 基本实现 | `representation_change_plan`；测试 #6 | 计划只生成不执行（按设计需人工） |
| AC-016 角色只读/启停 | 基本实现 | `feature_ops_view` / `submit_toggle`（仅 engineer/admin，结果为 draft+parent）；测试 #7 | 未接入 API/UI 权限体系 |
| AC-028 记忆资产清单 | 已实现 | `memory_asset_inventory`；测试 #8；test_bank_lineage_incubate | — |
| AC-029 库血缘/孵育 | 已实现 | test_bank_lineage_incubate（NormalBank/DefectBank lineage、incubation_pool） | — |
| AC-030 反馈路由 | 已实现 | `feedback.route_of`；test_feedback_routes | — |
| AC-031 缺陷种子初始化 | 已实现 | `feedback.seed_init_defect`（独立门控、失败整批撤销）；test_feedback_routes | — |
| AC-032 逐条血缘 | 已实现 | 测试 #8 + test_bank_lineage_incubate | — |
| AC-033 统一能力卡 | 基本实现 | `capability_card`（槽+融合后通道，缺省显式 unknown）；测试 #9 | 未接入 UI |
| AC-034 难例准入登记门禁 | 基本实现 | `admission_registration_gate`（gain/redundancy/regression/g1/g2/version，齐全仅登记 draft，不动生产指针）；测试 #10；test_hard_admission | 未接入 E_open 实际流程 |
| AC-035 快照盘点 | 基本实现 | `snapshot_completeness` 与 persist 实写字段对齐（inspect 校验）；测试 #5 | `feature_revision_binding` 仍恒为 unknown |
| AC-061 revision 绑定 | 部分实现 | `revision_binding` + store bind/rollback/lineage/detection_stamp；测试 #7 | persist 未调用 `revision_binding`，生产 trad 仍非 revision 管理，快照 meta 如实写 unknown |
| AC-060 编排总表 | 部分实现 | `orchestration_table`（如实 is_seven、missing_base、候选槽、diff）；测试 #11/#12 | 仅数据层，无统一 UI 编排页 |

## 共性缺口
- `release_ops` 全部为离线只读/登记函数，尚未接入 backend API 与 UI。
- AC-061 / AC-035 闭环需生产 trad 纳入 FeatureSetRevision 管理后，在 `persist.write_snapshot_state` 中调用 `revision_binding`（涉及快照写入，本轮按约束未做）。
