# final_model：专家协议与模型、特征类 AC 最终验收（合成数据）

范围：AC-001～006、008～010、017、018、020～023、026、061、062。
约束：全部使用合成数据或 TEMP 注册表；不改生产 DB/快照/stable_AOI_tree；生产推理路径未改（新增模块仅离线/评价使用），learning_parallel 不涉及，纯推理 p95 不受影响。

## 改动（仅新增文件，未改已有文件）

| 文件 | 作用 |
|---|---|
| `AOI_Core/algo/features/parity.py` | AC-062 等价 harness：`ProductionCandidate`（生产 TradSlot 适配器）+ `run_parity()` 八项逐项比对，输出 `FeatureSetRevision._parity` 可消费的 fail-closed 证据（status/verdict/checks/diffs/baseline_implementation_hashes/protocol_hash/numeric） |
| `AOI_Core/algo/decision/scheme_compare.py` | AC-008/009 三方案对照纯函数 `compare_schemes()`：整图 / 全局+全量局部 / 全局+预算细判，报块数、缺陷块覆盖、recall/FPR、救回漏检、新增误报、否决新增漏检、显存（未传入时 `not_measured`）及副作用四分类 |
| `AOI_Core/tests/smoke/test_final_model_parity.py` | 5 个测试（AC-062/010） |
| `AOI_Core/tests/smoke/test_final_model_protocol.py` | 9 个测试（AC-003/004/005/006/008/009/018/020/021/022/023） |

## 测试

命令（cwd=AOI_Core，`QT_QPA_PLATFORM=offscreen`）：
`python -m unittest tests.smoke.test_final_model_protocol tests.smoke.test_final_model_parity tests.smoke.test_feature_revision tests.smoke.test_cde_report tests.smoke.test_local_branch`
结果：Ran 54 tests，OK（新增 14 个 + 既有 40 个定向回归），约 18 s。

## 逐条结论

| AC | 类型 | 测试 / 模拟 | 建议状态 | 剩余缺口 |
|---|---|---|---|---|
| AC-001 分辨率选择 | 仅模拟 | `tile_size_bounds` 无档案→`profile_missing`、冲突→`global_only`（test_ac003_roi_and_ac004_bounds） | 部分实现 | 无按品类冻结的最小缺陷像素/对比度/漏检/域差档案；G2/G5 真实验收未做 |
| AC-002 全局上下文 | 仅模拟 | 三方案中 global_only 独立保留并作为对照基线（test_ac008_ac009_three_schemes） | 部分实现 | 结构/布局/大范围异常的独立验证集与联合专家协议缺 |
| AC-003 单级局部覆盖 | 代码+合成 | ROI 内块不越界、并集覆盖 ROI；覆盖与全局分无关（compute_local_tiles 无全局分入参） | 部分实现 | ROI 覆盖真实目标缺陷需真实标注验证 |
| AC-004 块与坐标 | 代码+合成 | 60 组随机尺寸/块长/步长：无空块、块尺寸恒定、末行末列贴边、跨块随机缺陷并集覆盖、`tile_box_to_image`/`tile_to_image_coords` 反解；上下界与冲突交全局 | 已实现 | 业务 min_footprint 由品类档案给出（同 001） |
| AC-005 粒度与标签 | 代码+合成 | 面积占比 0.5/0.75→正、0.075→丢弃(-1)、0→负；`context_crop` 尺寸≥框+2·context、贴边回移 | 部分实现 | 训练/建库/校准/推理视图全链一致性及伪异常独立消融未在真实链路验证；阈值仍为候选 |
| AC-006 块图误报预算 | 代码+合成 | 4000 张合成正常图、9 块、块独立(ρ=0)/强相关(ρ=0.8)：Šidák α_blk 正确，校准集与独立测试集并集 FPR ≤ α=0.05（+抽样容差）；未配置时来源标 `unfrozen_default_from_tau_quantile`；小样本标注 | 部分实现 | `resolve_budget` 未接 `Settings.business_budget()`（g2_max_fpr_pct），图级总 FPR 与分支分配未冻结；“报警规则选择”尚未开发 |
| AC-008 细判副作用 | 代码+合成 | added_by_local/confirmed/global_only/vetoed 分报；否决全局阳性单列 `veto_new_misses` | 已实现（离线评价口径） | blob 等高分辨率补充的覆盖/FPR 真实验证缺 |
| AC-009 三方案验收 | 代码+合成 | 同数据三方案：块数 0/24/6、救回 2/1、新增误报 1/0、否决新增漏检、缺陷块覆盖 1.0/0.5、deltas；显存未传入时 `not_measured` | 部分实现 | 显存需实测传入；真实数据开发选参 + 独立 G1/G2 验收未做 |
| AC-010 统一注册表 | 代码+合成 | 等价证据→FeatureSetRevision 门禁：生产自对照可激活，候选被拒，生产哈希不变（test_activation_gate_consumes_evidence） | 部分实现 | 生产 trad 与评价侧仍非同一提取实现；15 组 200 维组级配置未统一进生产 |
| AC-017 难例特征准入 | 仅模拟 | 无新增；准入闭环依赖 062 证据 + 010 门禁（已测） | 部分实现（维持） | 漏检/误报/冲突/漂移/open 池真实试算与独立增益/冗余评估缺 |
| AC-018 CDF 契约 | 代码+合成 | n=15 标 small_sample；n=3000 的 p95/p99 bootstrap std 低于 n=15；holdout 自匹配侧偏低 → `self_match_bias<0`；method/clip/“非缺陷概率”语义齐 | 已实现 | 真实品类尾部下限数值待真实正常集 |
| AC-020 固定融合基线 | 代码+合成 | sanity 缺失/≤0.5 的槽不进 equal/sanity 基线，hard_rule_max 单列 | 已实现 | 独立集（非 in_sample）准入结果待真实数据 |
| AC-021 软路由准入 | 代码+合成 | 分层三集 36/12/12 + 18/6/6；固定权重已最优时 router 不胜 margin → `trace_only`；n_params=k²+k | 已实现（准入逻辑） | 真实数据上未有 permitted 证据，主融合不接管（符合要求） |
| AC-022 域差能力 | 代码+合成 | 同分布→normal_drift；中位不动但尾部变宽→FPR 膨胀判 shift；中位平移→shift；dynamic_routing=not_integrated | 已实现（指标与定义） | 阈值为候选；真实新域验证缺 |
| AC-023 路由训练修正 | 代码+合成 | stratified_split 正负各自完整划分无重叠；报告 norm_entropy∈[0,1]/overfit_gap/collapsed | 部分实现 | U15 在真实数据上的 mil−λH 重评未做 |
| AC-026 更新准入 | 仅模拟 | 本组未改；由 learning/release 组覆盖 | 部分实现（维持） | 每轮前后独立 FPR/召回/域外误报/G1 真实闭环缺 |
| AC-061 FeatureSetRevision | 代码+合成 | 五态+fail-closed 激活门禁消费真实 harness 证据（TEMP） | 部分实现 | parent/工作区/Detection 关联、发布回滚、active 切生产仍缺 |
| AC-062 Trad 接入前等价 | 代码+合成 | 生产自对照八项全过（equivalent）；未播种 RNG 候选（模拟 feature_lib）→ rng/extraction/preprocess 失败、全局 RNG 被污染；skip_self=False 候选（模拟 GroupKNN）→ knn/scoring/report 失败，其余五项通过；harness 不污染全局 RNG | 部分实现 → 机制已实现 | 缺 AOI_feature 侧真实适配器及在真实图上的通过工件；差异候选的 G1/G2 验收未做 |

## 关键发现

- 生产 trad：`_vec` 以 `np.random.seed(0)` 包裹并恢复，vendor `extract` 内部 `default_rng(42)`，kNN 取 `sort(d)[1:4]`（跳过自身）。AOI_feature 侧若用未播种 RNG 或 skip_self=False，harness 均判为 `new_candidate`，激活门禁拒绝。
- AC-006 两层校准在块强相关时 Šidák 偏保守，`max(sidak, empirical)` 仍保证并集 FPR 不超预算；但预算本身尚未与业务预算绑定。
