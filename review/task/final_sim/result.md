# final_sim 结果（模拟验证 / 部署 / 定位 / UI 工作区）

环境：Python 3.11（`C:\Program Files\Python311\python.exe`），cwd 为 `AOI_Core`，`QT_QPA_PLATFORM=offscreen`，日期 2026-10-13。
所有证据都是 TEMP、合成数据或离屏模拟，不能替代生产验收，生产勾选保持 `[ ]`。未改生产 DB、生产快照、stable_AOI_tree 和共享文档，未提交 git。learning_parallel 未改动，推理路径未改动，p95 不受影响。

## 本组新增测试（共 31 项，全部 OK）

| 文件 | 项数 | 结果 |
|---|---|---|
| tests/smoke/test_final_sim_ui.py | 4 | OK 0.67s，另有 tests.ui.test_gauge_contract 回归 8 项 OK |
| tests/smoke/test_final_sim_roles.py | 3 | OK 4.96s |
| tests/smoke/test_final_sim_slots.py | 3 | OK |
| tests/smoke/test_final_sim_review.py | 7 | OK 0.88s，另有 region_review 相关回归 29 项 OK |
| tests/smoke/test_final_sim_loc.py | 4 | OK 3.0s |
| tests/smoke/test_final_sim_video.py | 3 | OK 1.1s |
| tests/smoke/test_final_sim_deploy.py | 4 | OK 15.5s |
| tests/smoke/test_final_sim_cleanup.py | 3 | OK 0.06s |

已有定向测试复跑（本组没有改这些测试）：test_regress_gate_ac036 4 项、test_cde_report 11 项、test_feedback_idempotency 11 项、test_candidate_release 47 项、test_final_model_parity 5 项、test_router_finite 5 项，全部 OK。

## 改动的代码文件

- `algo/backbone/dino.py`（AC-075）：权重按 vendor → hub 缓存 → 在线顺序回退。vendor 分支改为 `pretrained=False` 加 `torch.load`，不再经 file:// 把权重复制进 checkpoints。离线且缺权重时抛出明确的 RuntimeError。
- `tools/cleanup_data.py`（AC-079）：新增 `_protected_versions`，dry-run 时逐项打印保留原因（current、回滚链、候选 parent 等）。
- `algo/decision/localization.py`、`backend/api/region_review.py`、`ui/widgets/region_review_dialog.py`（AC-064）：
  - op 增加 `redo`，Literal 同步加入 redo。
  - 对话框支持拖边（4 条边）、整框平移、重做按钮。
  - 所有修订仍要求 reason 和 expected_revision。

## 逐项结论

### AC-036 发布/漂移回滚：建议保持部分实现
- 模拟：复跑 test_regress_gate_ac036（固定初始尾部、累计上限、连续违规冻结、恢复）4 项 OK。
- 缺口：独立缺陷锚、域外锚、G1 锚，以及按更新类型的容忍度都还没有实现，慢漂移的真实数据也没有覆盖。本组没有改代码。

### AC-037 分机制收益：建议保持部分实现
- 模拟：test_cde_report.MechanismGainTest（四口径、消融、披露、事实标记）在 11 项中通过。
- 缺口：真实取分开关和 API 缺失，结果不作为生产门禁。

### AC-039 事件与水位：建议保持部分实现
- 模拟：
  - test_feedback_idempotency 11 项：重启重放、重复 WAL、坏 JSON 行审计、WAL 失败不完成。
  - test_candidate_release 中的 prepared/rollback/崩溃重启恢复用例。
- 缺口：跨文件事务不是 exactly-once，完整 G3 和审计持久化仍缺。

### AC-043 独立回放门控：建议保持部分实现
- 模拟：test_candidate_release 中的 `test_replay_evidence_is_traceable_and_memory_only` 和 `test_replay_evidence_rejects_parent_or_wal_mismatch` 通过。
- 缺口：跨域分层独立训练回放、最差旧域报告，以及 G1/G2/G3/G6 联合退出都没有实现。

### AC-051 alert/warn 契约：建议已实现（离屏）
- 模拟：test_final_sim_ui 的 DashboardIncompleteChainTest，用 TEMP SQLite 加真实 DashboardPage 走 `_fill_incomplete`：
  - 品类 A 为 2/8，不告警；品类 B 为 3/4，告警。
  - 旧 warn 行被忽略，计数为 1，tooltip 为“预警品类：B”。
- 缺口：真实桌面（TEST-04）和产线联调。

### AC-056 Gauge 唯一判定：建议已实现（离屏）
- 模拟：test_final_sim_ui 的 `test_live_strip_colors_and_badges` 和 `test_strip_click_review_and_incremental`，用真实 MonitorPage 加 FakeClient，并对 viewer 打桩。
  - gauge 的 `_score` 和 `_threshold` 使用原始值，`_decision` 来自后端。
  - 缺少 decision 时显示“未知 / 未判定”，不做二次阈值。
- 另有 gauge_contract 8 项回归 OK。
- 缺口：真实桌面。

### AC-057 三态与 open 分离：建议已实现（离屏）
- 模拟：
  - 条带颜色依次为 SUCCESS / WARNING / WARNING（孤立 pulse）/ SUCCESS / DANGER / TEXT_SUB（旧行无 decision）。
  - KPI 显示“异常 3 · 灰区 1”。
  - open 只控制 badge_open，不改变 verdict。灰区提示只在 gray 时显示。
  - 点击条带会取消跟随。after_id 增量更新、重复投递幂等、越界时 reset_required 后重新订阅都已验证。
- 缺口：真实桌面闭环。

### AC-060 七槽编排：建议保持未实现
理由：功能本身没有实现，测试只能证明清单如实。
- 模拟：test_final_sim_slots 3 项，用合成 pipe 调 `slot_health`：
  - fast 三槽（sem/disc/inp）报告为 3 项，缺 blob/trad/layout/tpl，`is_seven=False`。fast 三槽不能称为七槽落地。
  - 七槽加 open 时，open 标为 `in_fusion=False`。反向体检槽显示 reverse，未体检槽显示 unverified，降级可见。
- 缺口：统一编排页、启停、plugin/config/hash/diff，以及按角色发布都没有实现。

### AC-062 Trad 接入前等价：建议保持未实现
- 模拟：用合成工件跑 test_final_model_parity 5 项 OK。
  - 生产自对照八项全部通过，可以激活。
  - 未播种 RNG 的候选和不跳过自身的 kNN 候选都被判为 new_candidate，被 fail-closed 拒绝。
  - 生产哈希不变。
- 缺口：冻结的真实旧生产协议工件和 AOI_feature 真实适配器都还没有跑过等价验证，所以不能改为已实现。

### AC-064 多框复核：建议已实现（离屏）
- 模拟：test_final_sim_review 7 项：
  - 4 条边拖动，修订带 reason 和 expected_revision。
  - 整框平移。
  - undo 后 redo，事件只追加，revision 递增。
  - redo 栈在新操作后清空。
  - 他人改动后返回 409。
  - 幂等在 API 层处理。
- 另有回归 29 项 OK。
- 缺口：真实桌面（TEST-04）。大框完全覆盖小框时，小框只能从表格选中。

### AC-065 后处理与定位：建议已实现（合成）
- 模拟：test_final_sim_loc 4 项，使用合成标注定位集，分 dev 和 test 两个切分。
  - 冻结版本只在 dev 上网格选出：mask_thresh 0.4、min_area 4、work_side 512，配置为 `pp-2f54c2e670`。
  - dev 指标：recall 0.923、precision 0.970、misloc 0.056。
  - test 只评估一次，不回调参数：small recall 0.833，后处理 6.35ms。
  - 已验证 top-k、相邻分离、边界裁剪、原图坐标。
- 产物：`raw/ac065_loc_frozen.json`。
- 缺口：真实标注的定位集。如果要求真实数据，这一项回退为部分实现。

### AC-073 角色工作区与证据面板：建议保持部分实现
- 模拟：test_final_sim_roles 3 项。子进程中用 TEMP cfg，设置三把 key（operator/engineer/admin）。
  - 四个阶段及其入口、接口：
    - 检测：GET live
    - 复核：POST regions
    - 评估：GET 候选和 POST acceptance
    - 发布：decision、activate、rollback、recover
  - 无 key 返回 401。低于该阶段最低角色返回 403，且发布逻辑调用次数为 0。达到角色时不是 401/403。admin 发布返回 409（FakeRelease），写入 `*_rejected` 审计，user 为 admin。
  - UI 侧：operator 只能看到检测和复核入口。`_apply_role` 切换时，隐藏页会退回到可见页，badge 显示“操作员”。
- 缺口：领域层内部调用没有审计绑定的恢复授权；没有逐 region 原图、热图、框的统一证据面板；现有页面不等于完整工作区；没有真实桌面联调。

### AC-075 权重加载回退：建议已实现（模拟干净机）
- 模拟：test_final_sim_deploy，在 TEMP 下设置 TORCH_HOME 为空、无缓存、socket 断网：
  - 存在 vendor 权重时加载成功，且 checkpoints 中不产生副本。
  - 缺权重时抛出明确的 RuntimeError。
- 产物：`raw/ac075_offline_deploy.json`。
- 缺口：真实物理干净机复验。

### AC-076 部署与兼容资产：建议保持部分实现
- 模拟：19 个 pkl 经 safe_pickle 清点，0 个加载错误。只有 defect_clf 出现 sklearn 版本警告（pickle 为 1.9.1，环境为 1.9.0）。
- 产物：`raw/ac076_sklearn_inventory.json`。
- 缺口：锁文件和环境不一致：

  | 项 | 环境 | 锁文件 |
  |---|---|---|
  | torch | 2.5.1+cu121 | 2.14.0+cu126 |
  | numpy | 2.4.6 | 2.5.2 |
  | Python | 3.11 | 3.12 |

  另外 layout7 往返、缺量 fail closed、PCB 服务超时提示都没有覆盖。

### AC-077 历史双轨纠偏：建议保持部分实现
只列清单，没有改代码，也没有运行。

| 轨 | 位置 | 性质 |
|---|---|---|
| 旧研究协议（含泄漏风险） | `review/old/repro/research/src/` 下的 datalocal.py、isolation.py、tpl.py、disc.py、metrics.py、feedback.py | 研究引用，不是项目性能 |
| 旧同步备份 | `review/sync_backup/SYNC-26/algo/ssocl/feedback.py` | 历史快照 |
| 当前诚实协议 | `AOI_Core/algo/...` 加上 RegressGate、CandidateRelease | 当前口径 |

- 规则：两轨要分别标注时间、数据和边界，禁止对异协议结果逐位相减。
- 缺口：同流量化对照没有跑（依赖 AC-079 的同流泄漏实验）。

### AC-078 实机视频多域：建议部分实现（原为未实现）
- 模拟：test_final_sim_video 3 项，用合成视频文件（3 个域 × 3 个种子）走现有的 `iter_video_detections`，按帧记录资产和时序，并报告最差域。
  - pcb 是最差域：强缺陷 recall 0.889，弱缺陷 0.333。结果如实记录，没有调阈值。
- 产物：`raw/ac078_video_domains.json`。
- 缺口：实机或真实视频，以及独立的 G1/G2/G6 门禁。

### AC-079 泄漏与残留治理：建议保持部分实现
- 模拟：test_final_sim_cleanup 3 项。dry-run 清单共 136 项，运行前后文件指纹不变，没有删除任何文件。current、回滚链、候选 parent 都标为保留。
- 产物：`raw/ac079_dryrun_inventory.txt`。
- 缺口：同流泄漏量化实验没有跑；删除审批流程没有做。

### AC-080 夹具与严格输出：建议保持部分实现
- 模拟：test_router_finite 5 项 OK：
  - 单正样本时 std 为 None，并带 availability。
  - 训练出现非有限值时 fail-closed。
  - 检测在持久化前拒绝非法值。
- m14_template 需要真实训练（最长等待 900s），按约束没有跑。
- 缺口：真实数据重训和历史污染专项验证。

## 状态建议汇总

| AC | 建议 |
|---|---|
| 051、056、057、064、065、075 | 已实现（模拟或离屏，生产勾选 0） |
| 078 | 由未实现改为部分实现 |
| 036、037、039、043、073、076、077、079、080 | 部分实现 |
| 060、062 | 未实现（模拟只证明清单和门禁如实，功能或真实工件仍缺） |
