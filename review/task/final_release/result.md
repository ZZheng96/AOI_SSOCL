# 发布、门禁、血缘、审计链 —— 子任务结果

开发目录：`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`。全部验证都在 TEMP 目录中完成，使用合成标注集，`Pipeline.predict` 用替身，业务库是 TEMP SQLite。没有改生产 DB、快照或 stable_AOI_tree，没有提交 git。`learning_parallel` 保持原值。推理路径没有改动，因此 p95 不受影响。

## 改动文件

- `backend/self_learning/candidate_release.py`
  - 新增模块级函数：`business_policy()`、`wilson()`、`looser_than_business()`、`check_recovery_authorization()`。
  - 新增 `release_audit` 哈希链表，以及 `_audit` / `audit_chain` 方法。
  - 新增 `_check_eval_run`，在 evaluate、decide、switch 中接入。
  - 新增 `snapshot_prepare` / `snapshot_commit`。
  - `recover` 支持 `snapshot_activate`，`sync_model` 也纳入 `snapshot_activate`。
- `backend/engine/__init__.py`
  - `activate` / `rollback` 增加 `recovery_authorization` 参数。`allow_stale=True` 时必须带授权。
  - 普通快照激活改为走发布账本，流程为 prepared → 切指针 → committed，每一步写审计。
- `backend/self_learning/service.py`：`activate_model` 透传 `recovery_authorization`。API 层仍固定传 `allow_stale=False`，未改。
- `tests/smoke/test_representation_rebuild.py`：fake engine 补上 `storage_dir`。原来的 stale 回滚用例改为两步：不带授权时被拒，带授权后成功。
- 新增 `tests/smoke/test_final_release_chain.py`，共 9 个用例，带 `[i/N]` 进度输出。

## 测试结果

- `tests/smoke/test_final_release_chain.py`：9 passed，用时 8.5s。
- 定向回归共 97 passed、38 subtests passed，用时 42.4s。覆盖的文件：
  - test_final_release_chain
  - test_representation_rebuild
  - test_candidate_release
  - test_isolated_candidate_path
  - test_stale_override_forbidden
  - test_candidate_audit_log
  - test_business_budget
- `test_final_perf.py::test_load_failure_does_not_switch` 通过（普通激活路径冷载失败时不切换）。

## 逐项说明

### AC-067 独立精度强门禁，建议状态：已实现（模拟验收）

- 业务冻结阈值
  - 冻结阈值通过 `Settings.candidate_release_policy()` 读取，配置来源是 business_budget 的 fpr、recall、gray、old drop 四项。
  - evaluate 时，如果传入的 policy 比冻结阈值宽松，直接拒绝，错误码为 `POLICY_LOOSER_THAN_BUSINESS`。
- Wilson 95% 置信区间
  - 每类样本数 ≥80 时强制门禁，看 FPR 上界和召回下界。违规码分别为 `POLICY_MAX_FPR_CI` / `POLICY_MIN_RECALL_CI`。
  - 样本数不足时，`ci_gate` 标为 `insufficient_samples`，只用点估计门禁。这样做是为了兼容现有的小样本夹具。
- EvalRun 绑定
  - 绑定字段：候选/父版本、候选/父快照文件 hash、WAL、数据 hash、config hash、业务阈值及其 hash。
  - decide（批准）和 switch（激活，锁内）都会强制校验，错误码如下：
    - `EVAL_RUN_MISSING`：缺少 EvalRun。
    - `EVAL_RUN_BINDING_MISMATCH`：绑定不一致，`fields` 会标出 version/snapshot/data/config 中哪项不符。
    - `EVAL_RUN_PARENT_STALE`：父版本已过期。
    - `EVAL_RUN_POLICY_CHANGED`：业务阈值变了。
- 对应测试：
  - test_policy_looser_than_business_rejected
  - test_business_policy_changed_after_eval（批准和激活两条路径都覆盖）
  - test_eval_run_binding_mismatch_and_missing
  - test_parent_stale_rejected
  - test_wilson_ci_gate_passes_clean_synthetic（每类 80 张、0 错，结果为 passed 且 ci_gate=enforced）
  - test_wilson_ci_gate_rejects_point_pass（1 个 FP：点估计 0.0125 ≤ 0.05，但 CI 上界约 0.067，结果为 rejected）
- 缺口：没有用真实标注集和真实模型做精度验收。合成集只能证明门禁逻辑正确。

### AC-073 领域层 allow_stale 恢复授权，建议状态：该子缺口已闭环，整项仍为部分实现

- `engine.activate` / `rollback` / `service.activate_model` 在 `allow_stale=True` 时必须带 `recovery_authorization`，包含 actor、reason、ticket、category、version、expires_at。
- 校验顺序：缺授权 → 字段缺失 → 范围不符 → 已过期，对应错误码依次为 `STALE_AUTHORIZATION_REQUIRED` / `INVALID` / `SCOPE` / `EXPIRED`。
- 授权通过后会计算 `authorization_hash`，和 actor 一起写入 `snapshot_activate_prepared` 审计。
- API 层仍是 409 STALE_OVERRIDE_FORBIDDEN，未变。
- 对应测试：test_allow_stale_requires_bound_authorization、test_representation_rebuild::test_activate_rollback。
- 缺口：权威文档中 AC-073 的主体是角色工作区和证据面板（UI 四阶段），不属于本组。

### AC-074 对比与回滚审计，建议状态：部分实现（事务批准链和普通快照审计已闭环）

- 事务批准链：`release_audit` 表的每条记录都带 prev_hash/hash，形成哈希链，按以下顺序写入：
  1. evaluate
  2. approve / reject（带 actor、理由、report_hash、eval_run_id）
  3. activate / rollback 的 `_prepared` 和 `_committed`
  4. recover 的 `_recovered_committed` 或 `_recovered_aborted`

  每条审计都和账本的业务变更在同一个 SQLite 事务（BEGIN IMMEDIATE）中提交。`audit_chain()` 能检出篡改或断链，并给出 `broken_at`。
- 普通快照激活现在也要先写 `snapshot_activate` prepared 事件，提交后写 committed，两步都写审计。
  - 如果切完指针后进程崩溃，事件会停在 prepared，此后再次激活会被拒。
  - `recover` 补记为 committed 后，再同步业务 Model 镜像。
- 对应测试：test_audit_chain_same_transaction_and_tamper、test_plain_snapshot_activation_ledger_crash_recover。
- 关于日志与业务同事务的取舍：OperationLog 在业务 DB，账本在 `feedback_receipts.sqlite3`，两边是不同的库，无法共享一个事务。因此以账本内同事务的 `release_audit` 作为权威审计链，OperationLog 保留为结构化镜像，原有写入逻辑没动。
- 缺口：
  - 模型、config、feature、data 的 diff 对照视图还没做。
  - 完整发布故障的 G3 验收还没做。
  - OperationLog 与账本之间仍不是原子写入。

### AC-040 可恢复短锁发布，建议状态：部分实现（有增量）

- 增量：普通快照激活纳入 prepared/committed 协议和 `recover`，`sync_model` 能跟随普通快照指针。
- 缺口与原记录一致：跨介质非原子、启动时自动恢复、多进程、真实数据与鉴权。

### AC-055 四时点追溯，建议状态：部分实现（有增量）

- 增量：批准和激活这两个时点进入审计链，带 actor、eval_run_id、版本和 event_id，记录与账本同事务。
- 缺口：检测、反馈、学习三个时点与候选的关联，以及 was_consumed=True 的情况，都还没测。

### AC-061 FeatureSetRevision，建议状态：部分实现（未改）

EvalRun 已绑定快照和 config 的 hash，但没有接入 FeatureSetRevision manifest。该 AC 与 AC-011～016 的特征注册表交叉，本轮优先做能闭环的 AC-067/073/074，所以没动。

### AC-011～016、AC-028～035，建议状态：维持原状态（未改）

这些属于特征选择/贡献/归一/解释/血缘、记忆库分层/成簇/反馈分路/孵育头/E_open/完整快照，涉及算法与训练路径。要做必须跑训练或数据集，且容易造成精度退化，超出本轮“发布门禁/审计链”能闭环的范围。没有改动，也没有声明进展。
