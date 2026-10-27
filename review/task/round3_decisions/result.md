# 第三轮：待决策事项前端化（开发版 AOI_Core）

把需要人工拍板的事项做成 `config.yaml` 的 `governance_decisions` 段。用户在设置页「待决策事项」分组中修改并保存（需要 admin 权限），后端负责校验、执行 409 门禁并按选择执行。agent 没有替用户做任何决策，所有默认值都是保守值。

## 改动文件

| 文件 | 改动 |
|---|---|
| `backend/core/config.py` | 新增 `GOVERNANCE_DECISIONS_SPEC`、`FROZEN_THRESHOLD_KEYS`、`_SCHEMA["governance_decisions"]`；`_check_range` 加枚举校验；新增访问器 `Settings.governance_decisions()`，存量值非法时回落默认 |
| `backend/core/deplock.py`（新） | `lock_status()` 对比 `requirements-lock.txt` 与实装版本，忽略 `+cu126` 等本地后缀；`dependency_guard(policy)` 在 block 且不一致时抛 `CandidateReleaseError(DEPENDENCY_LOCK_MISMATCH)` |
| `backend/api/routes_system.py` | `/system/config` 白名单新增 11 项，GET 附带 value/default/impact/readonly/options/locked；POST 增加冻结 409、只读项拒绝、服务端写冻结时间；`/system/info` 新增 `governance_decisions` 和 `dependency_lock` |
| `backend/api/routes_candidates.py` | decision 在 `auto_on_approve` 下批准成功后自动激活，激活失败不回滚批准，失败原因放在 `auto_activate`；activate 前先做依赖锁检查，block 返回 409，warn 附带 `dependency_lock_warning` |
| `backend/engine/__init__.py`（追加） | 新增 `apply_governance_isolation(cfg)`；`prepare` 中在 `apply_scenario_and_profile` 之后追加一行调用 |
| `ui/pages/settings_page.py` | 新增「待决策事项」分组：choice 用下拉框，只读项用标签，每项下方显示说明、当前值、默认值和影响范围；冻结时禁用阈值编辑器并给出 tooltip；「恢复默认」不改决策项，也跳过已锁定项；冻结状态变化后重新拉取表单 |
| `tests/smoke/test_round3_decisions.py`（新） | 12 个用例，打印进度 |

未修改：`algo/eval/offline.py`、`algo/backbone`、`backend/pipeline/service.py`、stable_AOI_tree、生产 DB、快照、共享 review 文档。

## 决策项清单（`governance_decisions.*`）

| 键 | 默认 | 取值 | 影响范围 |
|---|---|---|---|
| `gt_threshold_frozen`（AC-067/072） | false | bool | 冻结后，修改 `business_budget` 的 g2_max_fpr_pct / g2_min_recall_pct / g6_max_old_recall_drop_pt / max_gray_ratio_pct 或 `gt_dataset_ref` 时，POST /system/config 整批返回 409 `THRESHOLD_FROZEN`（detail.keys 列出冲突键），不做部分写入 |
| `gt_threshold_frozen_at` | "" | 只读 | 服务端维护：开启冻结时写入 ISO 秒级时间，关闭时清空；客户端提交其他值会被拒绝 |
| `gt_dataset_ref` | "" | str | 只做登记，冻结期间锁定 |
| `ac045_l1_real_training_enabled` | false | bool | 只登记意图，无运行时影响 |
| `ac046_l1_pollution_exit_enabled` | false | bool | 只登记意图，无运行时影响 |
| `candidate_release_mode` | manual | manual / auto_on_approve | auto_on_approve：候选批准成功后自动激活（同样经过依赖锁检查） |
| `calibration_isolation` | auto | auto / self / holdout | 非 auto 时注入 `cfg["calibration"]["isolation"]`，由 offline `resolve_isolation` 读取。auto 不覆盖 yaml，保持原行为。只影响之后新建的快照 |
| `dependency_lock_policy` | warn | warn / block | 候选激活（手动或自动）前检查依赖锁：block 返回 409 `DEPENDENCY_LOCK_MISMATCH` 且不切换版本；warn 放行并附告警。`/system/info.dependency_lock` 显示当前状态 |
| `ac062_legacy_artifact_dir`（§7） | "" | str | 只做登记 |
| `sync28_29_accepted`（§7） | false | bool | 只做登记 |
| `production_acceptance_signed`（§7） | false | bool | 只做登记 |

冻结规则：只有"当前已冻结，且本次请求没有解冻"时才拦截被锁值的实际变化。因此以下三种情况都能正常保存：
- UI 全量回传且值未变；
- 同一批提交"冻结 + 定值"，本次提交的值即成为冻结值；
- 同一批提交"解冻 + 改值"。

AC-070（显存上限）不在 §7 中，本轮未处理。

## 测试结果

命令（cwd=AOI_Core，`QT_QPA_PLATFORM=offscreen`）：

```
python -m pytest tests\smoke\test_round3_decisions.py tests\smoke\test_business_budget.py tests\smoke\test_round2_learning.py tests\smoke\test_consolidate_auto_activate.py -q
```

结果为 33 passed（新增 12 个，已有 21 个），未跑全量测试。新用例覆盖：
- 默认值与校验，包括非法存量值回落；
- GET 返回的字段；
- 冻结后 409、同值不触发 409；
- 冻结时间的写入与清空，只读项被拒；
- `/system/info` 的依赖锁信息；
- deplock 的后缀忽略和 warn/block 两种策略；
- 隔离模式经 `resolve_isolation` 生效；
- auto_on_approve 自动激活，以及 block 下激活返回 409、批准不回滚；
- 设置页的控件、保存、锁定和恢复默认。

## 边界

- 生产验收为 0。AC-045/046、AC-062、SYNC-28/29、生产验收签字这几项只记录用户的选择，没有接入任何执行路径。
- UI 侧的 409 沿用 `ApiClient` 已有的 `error_occurred` 提示（"请求失败（HTTP 409）：…"）。UI 已禁用锁定项，正常操作不会触发 409。
