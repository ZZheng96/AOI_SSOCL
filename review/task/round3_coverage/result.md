# round3_coverage 结果

工作目录为开发版 `AOI_tree\AOI_Core`。未修改 stable_AOI_tree、生产 DB、生产快照和共享 review 文档，未提交 git，也没有跑全量数据集。

## 1. blob_gpu_enabled（默认 false）
- `backend\core\config.py`：在 `GOVERNANCE_DECISIONS_SPEC` 中新增 `blob_gpu_enabled: (bool, False)`。
- `backend\api\routes_system.py`：在 `/system/config` 中新增说明和影响范围，说明文字为“实测单图 P95 从 316.8ms 降到 181.8ms；开启前需先复核 blob 和融合的 AUROC”。
- `algo\slots\blob.py`：只改了开关的读取，blob 算法没有动。algo 不反向依赖 backend，所以由 `set_gpu_override(fn)` 注入读取函数，`backend\engine\__init__.py` 在导入时注册 `governance_blob_gpu_enabled`。新建和 load_snapshot 两条构造路径都会生效。
- 开关语义：AOI_BLOB_GPU 未设置时，`cfg.gpu` 或治理配置任一为真即启用；AOI_BLOB_GPU 设置时以它为准，"0"、""、"false" 表示强制关闭。和需求原文“任一为真”相比，有一处例外：显式设 0 时强制关闭。这样做是为了兼容已有用例 `test_gpu_off_by_default`。设备不是 cuda 或 CUDA 不可用时，不会启用 GPU。

## 2. 批次覆盖检查（只读）
- 新增 `algo\eval\coverage.py`：
  - `infer_batch`：父目录不是通用目录（good/train/test…）时取父目录名，否则取文件名的字母前缀。日期型数字取前 4 位，例如 Image_20260301_001 → `Image_2026`、t_fov_03 → `t_fov`，纯数字文件名记为 `numeric`。
  - `batch_coverage`：输出 `uncovered_batches`、`out_of_domain_ratio`（正常样本分数 > train 上沿的比例）、per_batch 统计和 `suggestions`（“将批次 X 的正常样本纳入 train/good 后重建”）。已登记批次标为“已登记”，不再生成建议。
  - train 上沿：优先取 train 分数最大值。运行时 Pipeline 没有保存 train 的 fused 分数，所以评估中退回到 `decider.tau_high`（train/good 的 q99），结果里用 `train_ref_source` 标明来源。train 路径取 `pipe._train_normal_paths`；如果缺失，`covered` 记为 None，页面显示“来源未知”。
- 新增登记项 `approved_extra_good_batches`（str，逗号分隔，默认 ""），只做登记，不自动重建。
- `backend\evaluation\service.py::run_accuracy_eval`：metrics 中新增 `batch_coverage`，用 try/except 包住，失败不会阻断评估。
- `ui\pages\eval_page.py`：在业务预算达标线之后显示“⚠ 批次覆盖告警：未覆盖批次 …　域外比例 …”和建议。

## 3. FPR≤预算时的最大 recall 工作点
- `coverage.max_recall_at_fpr`：判定口径为 `score>=thr`，和 F1 扫描一致；recall 相同时取更高阈值（FPR 更低）。
- 评估 metrics 中新增 `budget_working_point`，包含 recall、fpr、threshold、tp/fp/fn/tn 和 max_fpr，预算取 `business_budget.g2_max_fpr_pct/100`。标注为“参考工作点，不修改生产阈值”，decider 不变。
- 评估页显示：“参考工作点（FPR≤5%）：最大缺陷召回 …　阈值 …（仅供参考，不修改生产阈值）”。

## 测试
- 新增 `tests\smoke\test_round3_coverage.py`，共 8 项，带 [i/8] 进度输出。覆盖：治理项默认值和校验、/system/config 文案、GPU 开关的 OR 逻辑和 env=0 强关（patch 了 cuda 可用性）、engine 注册读取配置、批次推断、覆盖检查、工作点（与暴力枚举对照）、service 参考报告、评估页文案。
- `tests\smoke\test_round3_decisions.py`：在 GOV_DEFAULTS 中同步加入两项新默认值。
- 命令：`python -m unittest tests.smoke.test_round3_coverage tests.smoke.test_round3_decisions tests.smoke.test_business_budget tests.smoke.test_round3_speed tests.smoke.test_round3_accuracy`，结果为 Ran 40 tests，OK。speed 中的 GPU 用例自带 skipUnless(cuda)，本次没有单独确认它是实际运行还是被跳过。
- 未验证项：真实快照上的评估链路（需要生产 DB 和快照，按约束没有跑）；真实数据的批次命名是否都符合推断规则，需要在实际数据上复核一次。
