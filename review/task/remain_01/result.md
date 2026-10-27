# REMAIN-01 验收结论：AOI_feature 与 AOI_Core Trad 全链路特征 parity 复核

- 任务：REMAIN-01（P0，关联 AC-062 / TEST-06）
- 验收对象（固定版本）：`d:\CGAIC\AOI_SSOCL\stable_AOI_tree` 当前源码（只读执行，未修改任何文件）
- 参照：`d:\CGAIC\AOI_SSOCL\AOI_tree`（开发版，仅诊断用）；生产参照=两侧 AOI_Core `TradSlot(legacy)`（5 个参照文件哈希两侧一致，见 env.json）
- 协议：TEST-06 旧协议原样（同 harness `algo/features/parity.py`、同合成图造法、同种子、atol=1e-6）
- 日期：2026-10-14；执行环境：Windows + py -3.11（详见 env.json）

## 总判定

**FAIL（稳定版，8 项过 2 败 6）。历史 TEST-06 FAIL 结论维持，不可改写。**

两层原因：

1. **修复未同步进稳定版**（与任务背景"该修复已同步进稳定版"不符）。稳定版 AOI_feature
   仍是修复前实现：selfsim 用全局 `np.random.choice`（RandomState）、`extract_one`
   无 rng 参数、无 `nan_to_num`、manifest `extract_seed=0`、scoring 仍 U27 全局种子
   包裹。共 **4 个文件与开发版存在差异**（feature_lib.py / feature_set.py /
   presets.py（仅 docstring）/ scoring.py，SHA256 见 env.json），其余 9 个文件一致。
2. **即使把当前修复同步进稳定版，全链路 parity 仍不能 PASS**（协议 B 诊断，见下）：
   开发版修复版跑同一协议，特征层面 15 组逐位一致（max|diff|=0.0），但 fit/scoring
   路径仍有 3 项超 atol——生产 `TradSlot.fit` 对 float32 特征栈求 mean/std/bank
   （float32 累加），候选 `GroupKNN.fit` 用 float64 → bank z 分数差 3.255e-03、
   score 差 6.30e-05。即历史 TEST-06 的"dtype 路径差异"次要观察在 selfsim 修复后
   浮现为主阻塞。

## 协议设计

- **协议 A（权威验收）**：稳定版树 = 稳定版 AOI_Core harness + 稳定版 AOI_feature 候选
  （legacy_u27 口径：extract 前 `np.random.seed(0)` 包裹、后恢复，即历史 TEST-06 FAIL
  时候选口径）。预期复现 FAIL → 已复现。
- **协议 B（诊断参照）**：开发版树 = 开发版 AOI_Core harness（与稳定版哈希一致）+
  开发版 AOI_feature 候选（explicit_rng42 口径：`extract_one(img, rng=np.random.default_rng(42))`，
  与修复后 `scoring.extract_features` 同口径）。验证修复在全链路的实际效果。
- 固定输入（与 TEST-06 完全一致，造法 `default_rng(seed)` 正弦/余弦基 + N(0,8) 噪声 +
  3 通道 roll）：train=6 张 64×64（seed 0-5）；probe=3 张（300×280 seed100 /
  20×40 seed101 / 64×64 seed102）；另加常量图 `np.full((64,64,3),128)` 做
  dtype/非有限值专项。全局 `np.random.seed(0)`；atol=1e-6。

## 八项逐项结论（协议 A，稳定版 = 验收对象）

| # | 验收项 | 结论 | 关键数值 / 证据 |
|---|--------|------|----------------|
| 1 | 特征提取 | **FAIL** | 全向量 max abs diff = **14.665**（probe0 300×280）/ 2.901（probe1 20×40）/ 9.513（probe2 64×64）≫ atol=1e-6；逐组归因：差异 100% 位于 selfsim 组，其余 14 组逐位一致（0.0）。raw/01_extraction.json、raw/group_diffs.json |
| 2 | 预处理 | **FAIL** | spec 声明一致（spec_equal=true，max_side=256/min_side=32）；但越界图（300×280、20×40，n=2）提取差 14.665——resize 逻辑同源，失败系 selfsim 差异经预处理项传导（harness 该项同时检查越界图数值）。raw/02_preprocess.json |
| 3 | 15组/200维顺序与slice | **PASS** | trace 检查 0 错误；候选 15 组 slice 与 registry.TRAD_GROUPS 完全一致；维度=200。raw/03_dimension_order_slice.json |
| 4 | 显式 default_rng(42) 行为 | **FAIL**（口径判定） | harness rng 布尔=True（重复提取差 0.0、无全局泄漏 marker 检查通过、全局 RNG 前后 4 次快照 sha256 均为 `012a3189dd9a9348`）；**但候选口径=全局 RandomState + seed(0) 包裹，非生产的显式 `default_rng(42)`**（semantics_match_production=false），selfsim 子采样块集合不同即项 1 的 14.665 根因。raw/04_rng.json、raw/rng_snapshots.json |
| 5 | dtype/非有限值处理 | **FAIL** | 常量图（128 填充）：候选原始输出含 **3 个 NaN**（selfsim 空块均值 + 相关除零），生产 0 个（`nan_to_num` 归零）；非有限位置不匹配；dtype 两侧均 float32；两侧均有限位置 max abs diff=0.0。另注意候选 preprocess_spec 声称 `nan_policy: "zero"` 但实现未兑现（声明-行为不一致）。raw/05_dtype_nonfinite.json、raw/constant_image.json |
| 6 | kNN 邻居/自匹配 | **FAIL** | 邻居精确一致率 **5/6 = 83.33%**（train1 邻居顺序 [5,4,0] vs 生产 [5,0,4]），元素重叠率 100%（6/6 图邻居集合相同）；harness knn_neighbors_self_match=False。raw/06_knn_neighbors_self_match.json、raw/neighbors_agreement.json |
| 7 | 评分 | **FAIL** | 6 图（train[:3]+probe）score max abs diff = **0.3759** ≫ atol（train0 差 8.5e-3、train1 差 9.7e-2 等，selfsim 差异经 z-score 放大传导）。raw/07_scoring.json |
| 8 | 报告 | **FAIL** | harness report=False（探针 report 的 score/neighbors 不等价，mismatch 随 scoring/knn 传导）。raw/08_report.json、raw/parity_result.json |

harness 汇总（协议 A）：status=**failed**，verdict=new_candidate，
checks：extraction=F, preprocess=F, dimension_order_slice=T, rng=T,
knn_neighbors_self_match=F, scoring=F, report=F, bank_compatibility=F；
numeric.max_abs_diff=**1.467e+01**。生产参照哈希与历史 TEST-06 基线一致
（vendor/traditional.py `8b74188c…`、slots/trad.py `d3caf552…`），排除参照漂移。

## 协议 B（开发版，诊断参照）——修复的实际效果

- harness：status=failed；**5/8 PASS**（extraction/preprocess/dimension_order_slice/rng/
  knn_neighbors_self_match），**3 FAIL**：bank_compatibility（max abs diff=**3.255e-03**）、
  scoring（**6.30e-05**，probe1）、report（mismatch probes [0,1]）。
- 任务八项：1-6 PASS，7 评分 FAIL、8 报告 FAIL。
- 特征层面：15 组 × 3 探针**逐位一致**（max|diff|=0.0，selfsim 修复在提取层完全有效）；
  邻居一致率 **100%**（6/6 精确一致）；常量图 0 NaN 双侧一致、diff=0.0
  （nan_to_num 修复有效）；RNG 快照前后一致无泄漏。
- 残余 FAIL 根因（**新阻塞**）：生产 `TradSlot.fit`（algo/slots/trad.py L126-128）对
  **float32** 特征栈求 mean/std/bank（float32 累加 + float32 z-clip），候选
  `GroupKNN.fit`（AOI_feature/scoring.py L34-38）对 **float64** 栈计算 → bank z 分数
  差 3.255e-03 → score 差 6.3e-05 > atol=1e-6。特征逐位一致时此差异成为唯一失配源。
- 证据：raw/dev/ 下同名文件（parity_result.json、01-08 逐项、neighbors_agreement.json、
  group_diffs.json、constant_image.json、rng_snapshots.json、env_snapshot.json）。

## 与历史 FAIL 结论的关系

- 历史 TEST-06 FAIL（max_abs_diff=14.665，selfsim RNG 根因）对稳定版**完全复现**
  （同值 14.665191650390625），结论维持、**不可改写**。
- 修复同步进稳定版后仍**不可直接改写**：须先解决 fit 路径 dtype 口径差异
  （GroupKNN.fit 对齐生产 float32 路径，或经评审调整 atol/口径），再按本协议复验
  通过后方可改写。

## 后续动作（建议，交开发方）

1. 将开发版 4 文件（feature_lib.py / feature_set.py / scoring.py 的功能性改动；
   presets.py 仅 docstring 差异）同步进稳定版 AOI_feature。
2. 对齐 `GroupKNN.fit` 与生产 `TradSlot.fit` 的 dtype 路径（生产为 float32 栈上
   mean/std + float32 z-clip bank），使 bank_compatibility/scoring/report 过 atol=1e-6。
3. 同步+修复后重跑本协议（cmd.txt 两条命令）复验，全 PASS 方可改写 TEST-06 结论。

## 缺口与边界

- 稳定版判定基于当前源码（修复未同步状态）；未对稳定版做任何修改。
- 协议 B 为诊断参照，不构成对稳定版的验收证据。
- 常量图专项（任务第 5 项）为 harness 八项之外的补充检查，判定口径：非有限位置
  一致 + 均有限位置 max abs diff ≤ atol + dtype 一致。
- 生产 DB/快照/模板/参数库未触碰；无 git 操作；估算未写成实测；生产验收框未勾。

## 证据索引（raw\）

- `parity_result.json` / `dev\parity_result.json`：harness 八项 checks + diffs + numeric 原始证据
- `01_extraction.json` … `08_report.json`（+ `dev\` 同名）：任务八项逐项对比（每项一个文件）
- `group_diffs.json` / `dev\group_diffs.json`：15 组 × 3 探针逐组 max abs diff
- `neighbors_agreement.json` / `dev\neighbors_agreement.json`：邻居一致率明细（6 图）
- `rng_snapshots.json` / `dev\rng_snapshots.json`：全局 RNG 前后状态快照（4 点，sha256 摘要）
- `constant_image.json` / `05_dtype_nonfinite.json`：常量图 dtype/非有限值专项
- `run_log_stable.txt` / `run_log_dev.txt`：完整运行日志（含进度与警告）
- `run_remain01.py` / `remain01_candidate.py`：本次执行脚本副本
- `env_snapshot.json` / `dev\env_snapshot.json`：运行环境快照
