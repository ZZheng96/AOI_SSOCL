# Round4：新批次纳入 good 重建基线（TEMP 模拟）

## 结论
- 重建后，新批次（Image_2026 / t_fov）在固定阈值下的 FPR 有所下降，但离达标还远。单靠“补批次重建”解决不了精度问题。
- B 组（init_defect=[]）：tau_high 从 0.895 降到 0.663。新批次 eval 的 FPR 从 0.556 降到 0.333，但 recall 也从 0.79 降到 0.26，FPR≤5% 工作点的 recall 从 0.158 降到 0.105。整体 AUROC 从 0.655 降到 0.585。
- A 组（init_defect=train_anomaly 30 张，与生产口径一致）：E1+E2 的 AUROC 从 0.597 升到 0.801，但提升几乎都来自 E1。E1 的缺陷图和原批次正常图是同名的“原图/修复图”配对，分数几乎相同；E1 AUROC=1.0 反映的是批次被分开了，而不是模型学会了判别缺陷。E2 是成对难例，AUROC 只有 0.54~0.58，接近随机。由此判断，模型本身对缺陷的敏感度不足，这才是真正的瓶颈。
- t_fov 批次在重建后仍有 67%~100% 被误报，原批次正常图的 FPR 也没有改善。A 组维持 1.0；B 组从 0.94 降到 0.5，但代价是 recall 大幅下降。
- 纯推理 p95 没有变差：n=240 时 B 为 256→255ms，A 为 401→257ms。n=60 那次 +4~5% 的差异属于噪声，详见下文。
- 实现过程中发现生产路径有一个真实缺陷：test/good 的路径直接并入 init_normal 会被 `guard_bundle` 拒绝，因为路径中必须含 train 段。原来的 fake-engine 单测覆盖不到这种情况。现已通过 staging 修复。

## 改动文件（未提交）
| 文件 | 改动 |
|---|---|
| `AOI_Core/algo/eval/coverage.py` | 新增 `split_extra_good(paths, approved, seed=0, fit_ratio=0.5)`：先排序去重，再按 `infer_batch` 分组，每组用 `default_rng(seed).permutation` 打乱，取 floor(n·0.5) 张作 fit，其余作 eval。同时更新了 `batch_coverage` 的 note |
| `AOI_Core/backend/api/routes_models.py` | 新增 `_approved_extra_fit_paths` 和 `_stage_approved_extra`。后者把 fit 子集复制到 `storage/images/{cat}/approved_extra/train/good/{stem}_x{sha1[:8]}{suffix}`，已存在的文件不重复复制，副本的 `infer_batch` 结果不变。`_prepare_engine` 的 DB 分支会把这些副本追加进 init_normal，并从 val 中剔除对应的 fit 原图。登记为空时行为完全不变 |
| `AOI_Core/backend/evaluation/service.py` | `run_accuracy_eval`：登记非空时，以该品类全部 test/normal 作为划分池（不受 limit 截断），并剔除 fit 子集，防止泄漏 |
| `AOI_Core/backend/api/routes_system.py` | 更新 `approved_extra_good_batches` 的配置说明和 `_GOV_IMPACT` 文案，明确会生效于下次重建，但不会自动触发重建 |
| `AOI_Core/tests/smoke/test_round4_batch_rebuild.py`（新增） | 共 6 个用例：split 的确定性、互不相交和顺序无关；默认值下行为不变；登记后 staged 副本、val 剔除和 guard_bundle 都能通过，且重复执行幂等；评估剔除 fit 子集，且 limit 不影响划分 |

## 划分（seed=0，未做任何调整）
- fit（8 张）：Image_20260116194303443、193850558、194241339、194217730、194228683、194458486（均为 _repaired），以及 t_fov_26、t_fov_6
- eval（9 张）：Image_20260116193703828、193859871、194235228、194441909、194259739、193843246，以及 t_fov_3、t_fov_13、t_fov_14
- base_normal 14 张（train/good 0598~0621）。E1 缺陷 10 张（1308~1320 原图）。E2 缺陷 9 张（与 eval 正常图同名的缺陷原图）。orig 正常 32 张（1308~1334、2444~2465 的 repaired 图）。
- 缺陷图和 eval 正常图都没有进入拟合。完整清单见 `raw/split.json`。

## 数字表
eval 正常图只有 9 张，所以 FPR≤5% 实际等价于 FP=0。wp_recall 指该工作点下的 recall。

B 组（init_defect=[]，主结果），tau_high 从 0.895 降到 0.663：
| 评估集 | AUROC 前→后 | 固定阈值 FPR 前→后 | 固定阈值 recall 前→后 | wp_recall 前→后 |
|---|---|---|---|---|
| E1 | 0.7556→0.6222 | 0.556→0.333 | 0.90→0.20 | 0.20→0.10 |
| E2 | 0.5432→0.5432 | 0.556→0.333 | 0.667→0.333 | 0.111→0.111 |
| E1+E2 | 0.6550→0.5848 | 0.556→0.333 | 0.79→0.26 | 0.158→0.105 |
| 原批次正常 FPR | — | 0.9375→0.50 | — | — |
| 新批次 eval FPR：Image_2026 / t_fov | — | 0.333→0.167 / 1.0→0.667 | — | — |

A 组（init_defect=30 张，对应生产口径），tau_high 从 0.828 降到 0.749。权重从 sem/disc/shead = 0.36/0.28/0.36 变为 0.25/0/0.75：
| 评估集 | AUROC 前→后 | 固定阈值 FPR 前→后 | 固定阈值 recall 前→后 | wp_recall 前→后 |
|---|---|---|---|---|
| E1 | 0.6556→1.0 | 1.0→0.333 | 1.0→1.0 | 0→1.0 |
| E2 | 0.5309→0.5802 | 1.0→0.333 | 1.0→0.333 | 0→0 |
| E1+E2 | 0.5965→0.8012 | 1.0→0.333 | — | 0→0.526 |
| 原批次正常 FPR | — | 1.0→1.0 | — | — |
| 新批次 eval FPR：Image_2026 / t_fov | — | 1.0→0.0 / 1.0→1.0 | — | — |

两次运行的指标和阈值完全一致，说明结果是确定性的。

纯推理延迟（ms）：
| 运行 | B p50 | B p95 | A p50 | A p95 |
|---|---|---|---|---|
| 第 1 次（n=60） | — | 432.8→454.5 | — | 473.5→492.1 |
| 第 2 次（n=240，额外 3 轮轮换计时） | 56.5→58.4 | 256.2→254.6 | 113.4→94.0 | 400.7→257.3 |

单张 max 偶尔达到 2.0~2.6s，推测是冷缓存或 GC 抖动，两组都出现过，与重建无关。

## 命令
```powershell
cd d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core
$env:QT_QPA_PLATFORM="offscreen"
# 新增 smoke 测试（6/6 通过）
& "C:\Program Files\Python311\python.exe" -m pytest tests\smoke\test_round4_batch_rebuild.py -q
# 回归测试（47/47 通过；改完 routes_system 文案后又重跑 round4/round3_coverage/round3_decisions，26/26 通过）
& "C:\Program Files\Python311\python.exe" -m pytest tests\smoke\test_round4_batch_rebuild.py tests\smoke\test_round3_coverage.py tests\smoke\test_round3_decisions.py tests\smoke\test_representation_rebuild.py tests\smoke\test_round3_accuracy.py tests\smoke\test_business_budget.py tests\smoke\test_dataset_isolation.py -q
# TEMP 模拟（脚本已在运行后删除；它只读 DB，所有拟合和 staging 都在 TEMP 下的临时目录中完成）
& "C:\Program Files\Python311\python.exe" $env:TEMP\aoi_r4\sim_batch_rebuild.py
$env:R4_LAT_REPS="3"; $env:R4_SUFFIX="_rep2"; & "C:\Program Files\Python311\python.exe" $env:TEMP\aoi_r4\sim_batch_rebuild.py
```
原始输出见 `raw/`：split.json、per_image(.json/_rep2)、summary(.json/_rep2)、sim_log(.txt/_rep2)。

## 未验证 / 局限
- 生产 DB 分支的 `_prepare_engine` 没有用真实 engine 端到端跑过。模拟脚本复现了相同的 bundle 组装逻辑，单测用的是 fake engine 加真实小文件。
- `datasource_id` 分支没有接入 approved_extra，未做改动。
- eval 样本量很小（正常 9 张，缺陷 19 张），各项指标的置信区间很宽。
- 本次没有修改生产 DB、快照或 stable，也没有提交 git。
