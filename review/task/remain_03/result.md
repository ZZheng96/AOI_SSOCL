# REMAIN-03 验收结果：生产 Trad 特征血缘绑定闭环

- 任务：REMAIN-03（P0，关联 AC-073/074/076 发布链）——核实生产 Trad 特征血缘绑定闭环
- 验收对象（只读）：`d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core`
- 日期：2026-10-14；方式：只读代码审计（实现不存在，按规约跳过 TEMP 实证）
- **总体结论：FAIL —— 生产绑定闭环不存在。FeatureSetRevision 仅实现为「前门门禁」侧车（front-door gate），生产推理路径按设计契约明确不读取/不绑定 revision；Detection 记录无任何特征血缘字段。**

## 权威要求逐项判定

> 要求原文：「核实生产 Trad 实际读取并绑定 FeatureSetRevision；Detection 记录真实 revision、实现/配置 hash、维度、seed、模型/快照血缘；激活/回滚/重启恢复后与 pointer 一致。无 pointer 必须 unknown/unbound，不允许从 registry 猜测」

### a. 生产 Trad 推理是否真实读取并绑定 FeatureSetRevision —— FAIL（未实现，且为有意设计）

1. `algo/feature_revision.py:1` 模块 docstring 自述：「不可变特征 manifest 的前门禁；**不切换生产 extractor 或模型指针**」。
2. `tests/smoke/test_feature_revision.py:151-156` 存在**静态设计契约测试** `test_production_and_frontdoor_contract_static`，断言 `algo/slots/trad.py` 中**不得出现** `feature_revision`（`assertNotIn("feature_revision", trad)`，line 155）。即「生产 trad 不引用 revision」是被测试固化的设计。
3. 全库 grep：生产推理路径（`algo/slots/trad.py`、`algo/vendor/traditional.py`、`algo/decide.py`、`backend/engine/__init__.py` 的 predict/activate/rollback 路径）**无任何** `feature_revision` / `production_pointer` / `detection_stamp` 引用；TradSlot 的特征来源是 `algo/features/registry.py` 的常量注册表 `_SPEC`（15 组 200 维，registry.py:19-50）+ 快照内 `cfg.yaml`，不读生产指针。
4. API 层自述未绑定：
   - `backend/api/routes_models.py:281-282`：`GET /models/feature-revisions` 固定返回 `"production_binding": "unknown", "changes_production_extractor": False`。
   - `backend/api/routes_models.py:313-314`：revision 状态流转返回 `"production_model_activated": False, "仅更新特征门禁状态，不切换生产提取器或模型"`。
   - `backend/api/routes_models.py:452-454`：draft 提交返回 `"changes_production": False`。
5. 快照 meta 落盘时硬编码不绑定：`algo/persist.py:652-653`：`"feature_revision_binding": {"status": "unknown", "revision_id": None, "reason": "production_trad_not_revision_managed"}` —— reason 字面承认「生产 trad 不受 revision 管理」。

### b. Detection 记录是否写入血缘字段（真实计算值）—— FAIL（未实现）

1. `backend/db/models.py:214-245` `Detection` 表 schema：仅有 `model_id`（FK）、`final_score`、各级分数、`latency_ms/e2e`、`n_tiles`(JSON)、热图路径。**无** feature_revision、实现 hash、配置 hash、特征维度、seed、模型/快照血缘列。models.py:351 的 `extra` 列属于 `OperationLog` 而非 `Detection`。
2. 唯一写入点 `backend/pipeline/service.py:719-745` `_persist()`：落库字段仅 `model_id`（由 `model_version` 反查 Model 行，service.py:705-718）+ `n_tiles` JSON（内含 model_version/threshold/boxes/types/decision_trace 等，**无特征血缘键**）。
3. 推理结果契约 `backend/engine/__init__.py:39-66` `DetectionResult`：无 revision/hash/seed/维度字段；`extra` 为「预留扩展」（line 66）且生产路径无人写入。
4. `feature_revision.py:298-307` 的 `detection_stamp()`（docstring 自述「写入 Detection.extra 的特征集关联」）在生产代码中**零调用方**：唯一调用者 `algo/release_ops.py:191-198 revision_binding()` 仅被测试（`tests/smoke/test_round2_release.py:195-207`）引用，属于 dead code。

### c. 激活/回滚/重启后绑定与 pointer 一致 —— FAIL（对账链路不存在）

1. 模型激活：`backend/api/routes_models.py:602-693` → `backend/engine/__init__.py:883-921`，指针机制为 `storage/snapshots/{category}/current.json`（engine/__init__.py:4），发布账本 prepared→committed（engine/__init__.py:915-921）。其中 routes_models.py:629-641 有「快照 meta 带 revision_id 时 check_activation」的门禁，**但因 persist.py:652 永远写 `revision_id=None`，该分支在生产中永不触发（门禁死代码）**。
2. 回滚：`engine/__init__.py:923-957` 仅切模型版本指针，**无任何特征生产指针（registry.production.json）对账**。
3. 重启恢复：engine/__init__.py:1-5 docstring —— 启动仅按 `current.json` 激活快照；**无特征指针与模型指针的一致性校验**。
4. 特征生产指针本身（`feature_revision.py:234-287` 的 `bind_production`/`rollback_production`，pointer 文件 `registry.production.json`）**没有 HTTP API 入口**：全 backend grep 无调用，仅 `tests/smoke/test_round2_release.py:198-206` 单测调用。即运维上根本无法把任何 revision 绑成生产。

### d. 无 pointer 时行为是否 unknown/unbound、不从 registry 猜测 —— PASS（代码层面符合）

1. `feature_revision.py:298-302 detection_stamp`：无 pointer → `{"feature_revision": None, "status": "unbound"}`，不回退 registry。
2. `persist.py:652-653`：快照 meta 如实写 `unknown` + 原因 `production_trad_not_revision_managed`。
3. `routes_models.py:237-238`：模型列表 meta 缺省时回退 `{"status": "unknown", "revision_id": None}`。
4. `routes_models.py:622` 注释明记：「仅显式绑定 revision 的模型受新门禁管理；**旧生产 trad 不猜测绑定**」。
5. `algo/release_ops.py:143-151 snapshot_completeness`：bind 缺失 → `feature_revision: unknown`、`reproducible: False`，如实降级。

## 结论汇总

| 判定点 | 结论 |
|---|---|
| a. 生产 Trad 读取并绑定 FeatureSetRevision | **FAIL（未实现）** |
| b. Detection 记录真实血缘字段 | **FAIL（未实现）** |
| c. 激活/回滚/重启与 pointer 一致 | **FAIL（未实现）** |
| d. 无 pointer → unknown/unbound，不猜测 | **PASS** |

**总体：FAIL，生产绑定闭环不存在。** 已实现的部分是「候选特征 revision 的前门门禁/审计侧车」（注册→评估→审批→激活门禁、难例准入、parity 等价、生产指针文件机制），但与生产推理、Detection 落库、模型激活/回滚/重启主链路完全断开。按规约未建 TEMP 环境（无实现可实证），不虚构绑定证据。

## 开发阻塞清单（实现闭环所需最小工作项）

1. **设计决策先行**：当前 `tests/smoke/test_feature_revision.py:151-156` 把「trad.py 不含 feature_revision」固化为契约。需在引擎层（而非槽层）接线绑定——推理时在 `engine.predict_*` 读取 `FeatureSetRevision.production_pointer()` 并写入结果；同步修订该静态契约测试。
2. **Detection schema 扩展**：新增列或结构化 JSON 字段承载 `feature_revision_id`、`implementation_hashes`、`config_hash`、`feature_dimension`、`seed`、`model_snapshot`（category/version/artifact_hashes/backbone_sha256/code_sha256），并在 `pipeline/service.py:_persist` 写入**真实计算值**（非占位）。
3. **接入 detection_stamp**：在推理路径调用 `detection_stamp(model_snapshot)`（现零生产调用），未绑定如实落 `unbound`。
4. **bind/rollback HTTP 入口**：为 `bind_production`/`rollback_production` 增加受角色管控的 API 路由 + 审计（现仅测试可达）。
5. **persist.py:652 硬编码替换**：快照落盘时调用 `release_ops.revision_binding(store, model_snapshot)`（现仅测试引用）写真实绑定态。
6. **激活/回滚/重启对账**：`engine.activate`/`rollback` 与启动恢复（current.json 路径）增加「快照 meta.revision_id ↔ 特征生产指针 current」一致性校验；`routes_models.py:629-641` 的 check_activation 分支当前因 revision_id 恒 None 成为死代码，接线后需覆盖回滚与崩溃恢复场景。

## 证据文件

见 `raw/`：`feature_revision_excerpt.txt`、`detection_schema_excerpt.txt`、`persist_excerpt.txt`、`routes_models_excerpt.txt`、`engine_excerpt.txt`、`grep_hits.txt`。
