# Round3 精度定位与修复（gold_finger，fast profile：sem/disc/shead）

## 结论
- 业务目标（FPR≤5% 且 recall≥95%）在当前协议下**未达标**；校准修复后 baseline 可分性达到 AUROC 0.97，但固定阈值 FPR 仍 100%。
- 主根因 = **数据/阈值域覆盖**：tau 只由 train/good 拟合，train 正常（0598~0621 png）与 test 正常（Image_2026、t_fov）是不同采集批次，test 正常全部落在 train 上沿之外 → 任何"仅用 train/good"的分位阈值都会把它们判 anomaly。这不是评估口径问题（EvalRun 与 Pipeline.evaluate 均用 `rec["fused"]` + sklearn AUROC，口径正确）。
- 放大因素（已修）= **校准硬限幅**：`CDFCalibrator/TplCalibrator.transform` 末尾 `np.clip(q,-1,2)`，末 bin 很窄导致外推斜率极大，域外分数大量平局于 2.0，丢失排序（baseline shead raw AUROC 1.0 → cal 0.5；sem/shead test 侧 100% 饱和）。
- 次要发现（未改）：sem 槽在本品类方向相反（raw AUROC 0.30/0.31）；domain 模式下 sanity 加权按 init_defect（与 repaired 成对、只差缺陷区域）算出 shead≤0.5 而置零，权重全给 disc；t_fov 是训练集未覆盖的第二个域（disc 对其全部误报）。

## 改动文件
- `AOI_Core/algo/fusion/calibrate.py`：新增 `_soft_bound`，CDF 与 Tpl 的 `np.clip(q,-1,2)` 改为严格单调软限幅（[0,1] 内恒等；q>1→1+e/(1+e)；q<0→−a/(1+a)）。值域契约 [-1,2] 不变，快照只存 edges，向后兼容。
- `AOI_Core/tests/smoke/test_round3_accuracy.py`（新增，4 例）。
- 复现/分析脚本：`review/task/round3_accuracy/repro_accuracy.py`、`analyze.py`；原始数据 `raw/`（修复前在 `raw/before/`）。
- 未改：stable_AOI_tree、blob.py、offline.py、生产 DB/快照（全部在 TEMP 副本，已清理）。

## 数字（小子集；单图 ≤368ms）
| 场景 | 指标 | 修复前 | 修复后 |
|---|---|---|---|
| baseline（TEST-07 切分，10 正常 / 10 缺陷） | fused AUROC | 0.935 | **0.970** |
| | 固定阈值 FPR / recall（tau_high=0.828） | 1.00 / 1.00 | 1.00 / 1.00 |
| | 最优点：FPR≤5% 时最大 recall | 0.00 | **0.80**（阈 1.989） |
| | 最优点：recall≥95% 时最小 FPR | 0.20 | 0.20 |
| | shead cal AUROC / 顶部平局数 | 0.50 / 全饱和 | 1.00 / 1 |
| domain（36 正常训练含 Image_2026/fov；17 正常含 5 t_fov / 20 缺陷） | fused AUROC | 0.784 | 0.703 |
| | 固定阈值 FPR / recall | 0.294 / 0.70 | 0.294 / 0.70 |
| | recall≥95% 时最小 FPR | 0.353 | 0.353 |

domain 下 AUROC 下降是因为权重=disc 单槽，修复后 cal 排序与 disc raw 排序完全一致（0.703）；修复前 0.784 来自硬截断平局的偶然收益，不是真实可分性。

单槽可分性上限（raw，test 侧事后统计，仅诊断，不可用于调参）：
- baseline shead raw AUROC 1.0（阈 4.81 时 FPR 0 / recall 1.0）；disc 0.97。
- domain shead 0.956（FPR≤5% 时 recall 0.80）；**去掉 t_fov 后 shead 1.0、disc 0.975**。

## 判断
- 可分性没到上限：shead 在已覆盖的域里能完全分离。卡点在阈值与权重，它们只能在训练域内拟合，外推不到新批次。
- 特征层面：sem 在本品类是反向特征；disc 对未覆盖域（t_fov）不稳。
- 要达标，需要用协议内的方法把 test 批次的正常样本纳入 train/good（覆盖 Image_2026、t_fov），或者给每个批次做正常样本标定。光改校准、阈值算法达不到 FPR≤5%。

## 建议状态
**部分修复 / 未达标（blocked on data）**：校准排序缺陷已修，并有测试守护。精度达标的前提是补齐 train/good 的域覆盖，这一步需要数据/协议层决策。可选的后续项（未做，需另行验证防泄漏）：sanity 加权改用留出的同域 init_defect，或剔除反向的 sem 槽。

## 测试
`tests/smoke/test_round3_accuracy.py` 4/4 通过。相关已有测试 test_final_model_protocol、test_snapshot_layout_compat、test_local_branch、test_cde_report、test_round3_decisions 共 57 例 + 64 subtests 全部通过。
