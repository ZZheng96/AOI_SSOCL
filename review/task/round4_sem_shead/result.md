# Round4：sem 方向反 / domain 下 shead 权重置零 根因

## 结论
- sem：代码没有反向。AUROC 0.30 来自数据和协议本身，不修。
- shead：权重置零由 `weight_mode=sanity` 的 `w=max(0, auc-0.5)` 规则触发，是有意逻辑，不是 bug，不修。sanity 偏低同样来自数据配对混淆。
- 没有改算法代码，所以不涉及修复前后对比，精度和 p95 与 round3 相同。下表数字是本轮在相同代码上复测得到的，与 round3 一致。

## 改动文件
- 新增 `AOI_Core/tests/smoke/test_round4_sem_shead.py`：4 个合成特征回归用例，跑在 CPU 上，不加载 DINO。
- 新增 `review/task/round4_sem_shead/diag.py`（诊断脚本）和 `raw/{baseline,domain}{,_pairs}.json`。
- 生产 DB、快照、stable、git 都没有动，也没有改另一个 agent 负责的 coverage/baseline/rebuild 相关文件。

## 根因
1. sem 打分链路是正确的距离语义：`algo/slots/sem.py` 中 `dist = 1 - max(cos(patch, bank))`，取 topk 均值，分越大越异常。bank 由 train/good 的 L2 归一化特征做 coreset 得到。`_slot_image_score` 的 multiscale 聚合和 `CDFCalibrator` 都是单调递增的，不会翻转方向。用例 r4-1 和 r4-2 覆盖了这两点。
2. 数据结构：gold_finger 的每张正常图都是某张缺陷图修复后的 `*_repaired` 孪生图，二者只在缺陷区域有差别。init_defect（manifest train_anomaly，即 0598-0621 和 1289-1307）的孪生图都在 train/good 里。
   - 如果孪生图在 fit 库中，缺陷图约等于库内成员，sem 分数就低。domain 模式下这类缺陷的 sem 均值为 0.400，低于留出的校准正常图 0.509；孪生图不在库中的 5 张为 0.484。所以 sem 的 sanity 只有 0.095。
   - 在 test 集上，正常图来自 Image_2026 批，离训练库更远；缺陷图 1308-1334 与库内的 1289-1307 是同一采集批。因此 sem 实际衡量的是批间距离，而不是缺陷本身，AUROC 为 0.30。
   - 同库配对检验（test 缺陷原图对比它自己的 repaired 孪生图，消除批间差异）：domain 模式下 sem 在 75% 的配对中给缺陷图更高分，baseline 模式下是 60%。说明方向正确，只是信号很弱，均值差只有 0.0125 和 0.0003。
3. shead sanity 偏低的原因：
   - domain 模式下 36 张正常图走 holdout，7 张用于校准、29 张用于拟合。sanity 用的是留出的正常图，这一步本身是诚实的。
   - 但 30 张 init_defect 中有 25 张的孪生图位于负样本训练集中。正负样本特征几乎相同，导致 DeviationLoss 无法收敛（loss_n≈5.9，loss_a≈6.1）。
   - 结果是 init_defect 均值 2.57，校准正常均值 2.56，几乎分不开，sanity 只有 0.38，于是权重被置为 0。实际上 shead 在 test 上的 AUROC 是 0.956，样本内 sanity 低估了它。这正是代码注释 U26 已经警示过的情况（域差品类上 sanity 不可外推），属于设计边界。
   - baseline 模式下 14 张正常图回退到 self 模式，sanity 包含自匹配，因此偏乐观（0.819）。

## 数字（seed 42 / 引擎默认，固定切分，不挑样本；baseline test=20，domain test=37）
| 模式 | 权重 sem/disc/shead | sanity sem/disc/shead | fused AUROC | FPR≤5% recall | 等权对照 AUROC / recall | 逐槽 test AUROC sem/disc/shead | p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| baseline | .361/.278/.361 | .819/.745/.819 | 0.970 | 0.80 | 0.970 / 0.80 | 0.300/0.970/1.000 | 167 / 307 |
| domain | 0/1/0 | .095/.643/.383 | 0.703 | 0.00 | 0.682 / 0.20 | 0.309/0.703/0.956 | 172 / 293 |

我评估过一个修复方向：domain 模式下改为等权。结果 AUROC 从 0.703 降到 0.682，recall 从 0 升到 0.2，有升有降，不能算精度不退化，所以没有引入开关。真正有效的改法是在 sanity 和 shead 训练中剔除孪生图已在正常库中的 init_defect（按 stem 去配对）。这属于数据协议的改动，超出本轮"代码反向才修"的范围，留作后续建议。

## 命令（在 AOI_Core 下执行，先设置 `$env:QT_QPA_PLATFORM="offscreen"`）
```
& "C:\Program Files\Python311\python.exe" ..\..\review\task\round4_sem_shead\diag.py domain     # ~46s
& "C:\Program Files\Python311\python.exe" ..\..\review\task\round4_sem_shead\diag.py baseline   # ~24s
& "C:\Program Files\Python311\python.exe" -m unittest tests.smoke.test_round4_sem_shead tests.smoke.test_round3_accuracy   # 8 OK
```
diag.py 在 TEMP 下建隔离目录，用 sqlite 临时 DB，跑完自动删除。结果写到 `%TEMP%\r4_diag\out`，本目录的 raw/ 是从那里拷贝过来的。
