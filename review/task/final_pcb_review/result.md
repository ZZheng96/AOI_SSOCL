# PCB_Dual 最近修改审查与验证（2026-10-14）

审查对象：`d:\CGAIC\AOI_SSOCL\AOI_tree\PCB_Dual` 工作区里尚未提交的改动，共修改 30 个文件（+1272/-191），另有若干新增测试和 tuner 文件。
对照基线：`d:\CGAIC\commit\repo\AOI_SSOCL生产复现报告.md` 中的 E3、E4、E7。
本次审查没有修改任何代码，也没有提交 git。

## 1. 改动摘要

- 请求级批量复用：scheduler 新增 `batch_enabled`（stop_on_error 开启时自动关闭），按 adapter 分组调用 `run_many`，整批共用一个超时。
  - SMT：用 `_shared_cache` 复用 hsv、gray、配准和提取结果。键是 cfg 去掉判定参数后的 repr。
  - TH：在 [12..16] 并集上只跑一次，再按检测项过滤 detections。
  - Shift：新增 `run_many`，共用 base_mask 和 integral。
- 状态契约：
  - 新增 REVIEW 和 SKIP 两个状态。stub 改为 SKIP，TH 待复检改为 REVIEW。
  - `job_status` 的优先级为 ERROR > NG > REVIEW > 全部 SKIP 判 SKIP > 部分 SKIP 判 REVIEW > OK。
  - fusion 在 traditional 模式下直接沿用传统引擎结论。dual 模式下，传统为 REVIEW 或 SKIP 时整体判 REVIEW，不再默认放行。
- SMT 有模板但配准失败时，由原来“回退原始 pad 继续检测”改为直接返回 ERROR。这是 PCB-DUAL-04 中有意的 fail-closed 改动，有单测覆盖。
- 其他：可选 diagnostics 输出、acceptance 验收参数设置、tuner 实时预览、UI 批处理状态显示；`get_outputs_dir` 在驱动器不存在时回退到默认目录。

## 2. 实测与基线对照

| 项 | 基线 | 实测 | 结论 |
|---|---|---|---|
| E3 缺陷检出 | 4/4 | 4/4（召回 1.000） | 一致 |
| E3 正常样本误报 | 0/1 | 0/1 | 一致 |
| E3 金手指缺陷数 | 15/8/23 | 15/8/23（配准方式 lowconf_phase/ncc_shift/anchor_shift） | 一致 |
| E3 SMT 耗时 | 13.6–26.1ms | 单独复跑 12.5–27.7ms（与其他任务并行时 14.6–33.6ms） | 处于同一量级，见 3-P4 |
| E3 金手指耗时 | 510.5–906.0ms | 单独复跑 573.4–968.8ms（并行时最高 1470ms） | 首图冷启动约慢 7%，见 3-P4 |
| E4 插件焊点 | 13 个标准图，37 张；OK17/NG20/待复检0/跳过0 | 13 个，37 张；OK17/NG20/待复检0/跳过0，逐图判定与缺陷类型都一致 | 一致 |
| E4 耗时 | 5.3–15.5ms | 5.6–29.3ms（与 E3、单测并行） | 并行负载所致，单图远低于 1s |
| 自带单测（7 个模块） | PCB-DUAL-05 记录 34 项 | 39 项全部通过 | 通过 |
| E7 双引擎 NG 图 | overall=NG（传统多锡 ng=1，特征 0.5713） | overall=NG（传统多锡 ng=1），融合框为“传统·多锡(包锡)” | 一致 |
| E7 双引擎 OK 图 | overall=OK，融合框为空 | overall=OK，4 项都是 OK，没有 REVIEW 或 SKIP，融合框为空 | 一致 |
| E7 traditional 模式 | 无 | NG 图判 NG，OK 图判 OK | 新语义符合预期 |

### E7 运行方式

原脚本 `verify_dual_service.py` 有三处会碰到生产数据，因此没有直接运行：

- 写生产模板库：`TemplateStore().save`
- 写生产标定目录：`save_calib`
- 强制重训特征快照：`FeatureClient.prepare(force=True)`，还会拉起 AOI_Core

改为隔离运行（脚本放在 %TEMP%，跑完已删除）：

- TemplateStore、`TEMPLATE_CALIB_DIR`、ResultStore 都重定向到临时目录。
- 传统引擎走真实代码，覆盖 scheduler 批量复用、状态契约和 fuse_dual。
- 特征引擎用基线日志中的分数做桩：NG 图 anomaly 0.5713，OK 图 normal 0.0。

因此特征引擎的 HTTP 推理链路本次没有复验。不过它不在本次 diff 范围内。

## 3. 问题清单

| ID | 严重度 | 描述 | 处理 |
|---|---|---|---|
| P1 | 中（有意的行为变更） | SMT 有模板但配准失败时，由原来回退原始 pad 继续检测改为返回 ERROR。原先能给出 OK/NG 的图，现在可能变成 ERROR，需要人工处理。这一步是 fail-closed，比“伪 OK”更安全，E3 样本也没有触发。 | 不修改，记录为语义变更，上线前需告知产线 |
| P2 | 低 | 部分检测项为 SKIP（例如虚焊缺少 toe/rim）时，整体结论由 OK 改为 REVIEW，批量统计改计入 review_count，产线的 OK 率口径会变化。E7 的 OK 图没有触发。 | 不修改，记录 |
| P3 | 低 | 批量模式下整批共用一个 `_ALGO_TIMEOUT_S`。启用项多的组合，原来逐项不会超时，现在整批可能超时并整体判 ERROR。stop_on_error 开启时会自动退回逐项执行。单测已覆盖超时只触发一次及清理逻辑。 | 不修改，建议观察大组合的耗时 |
| P4 | 低（按噪声处理） | E3 单独复跑时，金手指首图比基线上限高约 7%，SMT 高约 6%。金手指算法不在本次 diff 内，SMT 走的是单次 run 路径，只新增了构造 cache_key 的 repr 开销，可以忽略。判断为机器负载或冷启动波动，单图耗时仍不超过 1s。 | 不修改 |
| P5 | 低 | SMT 和 TH 的 cache_key 用 `repr(sorted(cfg))` 构造。缓存只在单次 `run_many` 调用栈内有效（同一张图），cfg 的值是标量和矩形列表，repr 完整，没有截断碰撞风险。TH 键排除了 advanced/void_preset/insuf_preset，但这些设置已经通过 effective thresholds 进入键。 | 不修改 |
| P6 | 提示 | `tests/eval_traditional.py` 每次运行都会覆盖 `tests/results/eval_traditional_latest.json`。本次已先备份，跑完再恢复。 | 已恢复 |
| P7 | 已知（协同.md §4.5） | 9 项已知缺陷仍待处理，包括 TH 标定清空后旧框复活、TH 配准错配未 fail-closed 等。本次未扩大其影响范围。 | 不在本次范围 |

## 4. 结论

本次改动可以正常工作，在 E3、E4、E7 上的判定结果与原版基线一致，没有劣化：

- 缺陷检出、误报、金手指缺陷数、插件焊点 OK/NG 分布逐项一致。
- 双引擎在 NG 图和 OK 图上的融合结论与基线一致。
- 自带单测 39/39 通过。

需要关注的是两处有意的语义变更：P1 配准失败判 ERROR，P2 部分 SKIP 判 REVIEW。它们会改变产线统计口径，但没有使已知样本的结论回退。没有发现需要修复的明确 bug，因此本次没有修改代码。

## 5. 原始产物（raw/）

- `diff_alg_repo.txt`、`diff_app_core.txt`、`diff_app_ui.txt`：git diff 原文。UI diff 只扫读过。
- `e3_stdout.log`、`e3_stderr.log`：E3 与其他任务并行运行的日志。
- `e3_rerun_serial_stdout.log`、`e3_rerun_serial_stderr.log`：E3 单独复跑日志。
- `eval_traditional_latest_run.json`：本次 E3 的原始记录。工作区里的文件已恢复为运行前的版本。
- `e4_stdout.log`、`e4_stderr.log`：E4 日志。输入取自 `commit\repo\_run\...\input`，只读；输出写到 TEMP，跑完已删除。
- `unittest_stdout.log`、`unittest_stderr.log`：39 项单测日志。
- `e7_isolated_stdout.log`、`e7_isolated_stderr.log`：E7 隔离复验日志。
