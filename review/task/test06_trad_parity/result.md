# TEST-06 结果：AC-062 旧生产 trad 协议八项等价比对（AOI_feature 候选）

- **结论：FAIL / verdict = new_candidate**（8 项中 2 过 6 败，max_abs_diff = 14.665 ≫ atol=1e-6）
- 运行时间：2026-10-14，用时约 5.5s，环境见 env.json，复跑见 cmd.txt
- 参照：生产 `algo/vendor/traditional.py` + `algo/slots/trad.py`（哈希门禁值与 parity_result.json 内 baseline_implementation_hashes 一致）
- 候选：AOI_feature `FeatureSet(trunk_preset())` + `GroupKNN(k=3)`，适配器 `raw/aoi_feature_candidate.py`（U27 播种包裹、显式 skip_self=True）
- 样本：合成图 train=6（64×64, seed 0-5），probe=3（300×280 缩小 / 20×40 放大 / 64×64 原尺寸，seed 100-102），造法同 `AOI_Core/tests/smoke/test_final_model_parity.py`

## 八项结果

| # | check | 结果 | 关键数值 |
|---|-------|------|----------|
| 1 | extraction | FAIL | max_abs_diff = 14.665（全部来自 selfsim 组） |
| 2 | preprocess | FAIL | spec_equal=true，n_resize_probes=2，数值差 = 14.665（同一 selfsim 差异传导） |
| 3 | dimension_order_slice | PASS | dims={200}，trace 通过 registry.check_trace |
| 4 | rng | PASS | 重复提取逐位一致（repeat diff=0），不污染全局 RNG |
| 5 | bank_compatibility | FAIL | max_abs_diff = 7.203（selfsim 5 维 z 化后传导进 norm/bank） |
| 6 | knn_neighbors_self_match | FAIL | 6 探针中 1 个近邻次序变化（[5,0,4]→[5,4,0]，第 2/3 近邻交换） |
| 7 | scoring | FAIL | max_abs_diff = 0.376 |
| 8 | report | FAIL | 3/3 探针 score 不一致（邻居索引与 enabled_groups 一致） |

## 差异归因（逐组比对见 raw/group_diffs.json）

**唯一根因：② selfsim（区域自相似）组的子采样 RNG 口径不同。** 三个探针 15 组逐组
max|diff| 中，除 selfsim 外 **其余 14 组全部为精确 0.0**——即 AOI_feature 声称的
"默认参数与主干逐位一致"对 14/15 组成立。

- 生产 `vendor/traditional.py` extract()：`rng = rng or np.random.default_rng(42)`，
  selfsim 超过 100 块时用 `default_rng(42).choice` 子采样（Generator，每次调用新建）。
- 候选 `feature_lib.py` selfsim：`np.random.choice`（全局 RandomState），依赖调用方
  U27 包裹 `np.random.seed(0)`（scoring.extract_features / 本适配器均如此）。
- RandomState(seed=0) 与 default_rng(42) 的 choice 算法与种子均不同 → 块数 >100
  （本测试三种尺寸在 scale=4 下均触发：64×64→256 块）时采样到的块集合不同 →
  selfsim 5 维特征不同（probe 最大 14.67）→ z-score 后传导至 bank/kNN/scoring/report。
- preprocess 项：规格声明一致（spec_equal=true），失败 purely 是数值项被同一
  selfsim 差异击穿，不是缩放实现差异（缩放代码两侧逐行相同，且 14 组全 0 可证）。

## 次要观察（不影响本次判定，记录在案）

1. **nan_policy 未实现**：生产 extract() 末尾 `np.nan_to_num(..., nan=0, posinf=0, neginf=0)`；
   FeatureSet.extract_one 无此步骤。本测试合成图未产生 NaN/inf 故未触发，但属真实口径差。
2. **dtype 路径**：生产拟合链路 float32（arr_w float32 → norm/bank float32）；候选
   GroupKNN 按 scoring.extract_features 惯例 float64。本次差异被 selfsim 掩盖，
   若 selfsim 修复需复核此项是否引入 >1e-6 残差。
3. **适配器自身 bug（已修，非语义差异）**：首轮运行 `_dist` 对广播后的 (N,) 距离
   多取了一次 `[0]` 导致近邻列表为空；修正后重跑，本结果为修正后结果。首轮日志
   已被覆盖，最终日志见 raw/run_log.txt。
4. skimage 0.26 对 `morphology.remove_small_objects(min_size=...)` 弃用告警，
   两侧同用 min_size=50，行为一致，无数值影响。

## 判定依据与后续

harness fail-closed 设计即如此：候选未证明与生产等价 → verdict=new_candidate，
差异进入新候选评价/准入流程，不得冒充生产等价。若后续要让 AOI_feature 达到等价，
最小修复方向是把 feature_lib selfsim 的子采样改为 `np.random.default_rng(42)`
（或主干接受外部注入 rng），本次未对任一侧源码做任何改动。

## 产物清单

- `result.md`（本文件）
- `env.json`（机器/python/依赖版本/种子/两侧代码 sha256）
- `cmd.txt`（复跑命令）
- `raw/aoi_feature_candidate.py`（适配器）
- `raw/run_test06.py`（运行脚本）
- `raw/parity_result.json`（harness 证据：八项 checks + diffs + numeric + 哈希门禁）
- `raw/group_diffs.json`（15 组×3 探针逐组 max|diff| 归因）
- `raw/run_log.txt`（运行日志，含进度输出）
