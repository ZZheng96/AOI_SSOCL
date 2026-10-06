---
name: mm-solve
description: 数学建模端到端 AI 工作流编排主 skill。拿到赛题后把 读题拆解→analysis.md→数据→模型→结果→图表→论文→提交 串成一条可执行流水线：每阶段给输入/动作/产物/验收标准/下一跳，调用 mm-* 环节 skill 与 mmkit 代码库，并做跨问数据流与产物对账。
whenToUse: 拿到赛题准备开工、需要按部就班推进某道题、复盘时按流水线重建旧题
---

# mm-solve：数学建模端到端编排（执行引擎）

> 定位：`mm-solve-runbook`（四天日程）的执行引擎。runbook 管"哪天做什么"，本 skill 管"每步怎么做、产出什么、怎么验收"。
> ★ 铁律：**开工前必须先读经验库（阶段 0）**——工具箱沉淀的规则/经验是做题的默认约束，不许跳过。

## 0. 工作区约定

- 每道题一个工作区 `contests/<年>-<赛>/problem-<X>/`：`analysis.md` + `code/` + `data/{raw,processed,results}/` + `figures/eda/` + `paper/`（含 `paper/figures/`）+ `models/`；
- 初始化：`python code/scripts/new_contest.py <年> <赛> --problems A B C`；环境用 `.venv\Scripts\python.exe`；
- 全库固定 `mmkit.set_seed(42)`；代码头注明 AI 辅助来源（`code/scripts/annotate_ai.py` 批量补，附件4 合规）；
- ★ 数据落盘红线（`code/meta/04-data-layout.md`）：**数据/图/结果一律存赛题本地**
  `contests/<年>-<赛>/problem-<X>/data/` 与 `.../figures/`，禁止落全局 `data/`；
  mmkit 落盘传 `base=problem_dir("<年>-<赛>/problem-<X>")`；全局 `data/` 只放跨赛题共享素材。
- 落盘约定：结果 JSON/CSV → `data/results/`，论文图 → `paper/figures/`（300dpi PNG+PDF），可复现命令写进 analysis.md §复现命令。analysis.md 可无 YAML front-matter（参考 `contests/2024-huawei/problem-C/analysis.md`；runner 支持可选 front-matter 存题号/数据源/种子元数据）。

## 0.5 先读经验库（开工必做，读 3 个 meta 目录 + 官方模板）

按本题题型读以下文件，**把命中规则写进 analysis.md 头部"经验清单"小节**（避免边做边忘）：

| 经验源 | 文件 | 必读要点 |
|---|---|---|
| 数据/代码 | `code/meta/01-experiment-workflow.md`、`04-data-layout.md` | 数据赛题本地、结果 JSON 命名、图与求解分离、可复现种子 |
| 模型选型 | `models/meta/01-model-selection-patterns.md`、`02-experiment-as-evidence.md`、`03-rapid-method-and-ai.md` | 题型→模型匹配、实验即证据、AI 流水线纪律 |
| 论文写作 | `paper/meta/01~07`（结构/图表/段落/摘要/误差/评审/篇幅）、`15-official-template-2026-gap.md` | 官方模板细节（见下）、摘要总分总、每问流程图 |
| 图表设计 | `paper/diagram-design-guide.md`（38 种视觉类型/语义模式/密度 4/10/焦点色，源自 Claude editorial-diagrams 2.5）+ `mm-plotting` | 何时画图、类型选择、标注/配色/密度规范 |
| AI 提示词 | `code/prompts/`（数据/建模/论文撰写/摘要优化/自查清单/题型 A-F 提示词库） | 按当前环节取用可复制提示词模板 |
| 官方模板 | `paper/official-template-2026/`（docx）+ `15-official-template-2026-gap.md` | 封面/摘要"针对问题X"句式/假设开头句/符号三栏表/4.2 数据处理框架图/页脚页码无页眉 |
| 获奖结构 | `paper/meta/08~10`（2023A/2024D/2024C 拆解）、`paper/award-patterns-30.md` | 题型对应结构骨架与 30 条可复用套路 |
| 题型指南 | `paper/meta/11-signal-pipeline-guide.md`（信号）、`12-rl-dlr-writing-guide.md`（RL） | 按题型选读 |
| 竞赛规则 | `mm-contest-intel`（格式/AI 合规/时间节点） | 提交流程与硬性红线 |

官方模板三铁律（每条必做，对应 `paper/meta/15` 反思）：
1. **摘要"总分总"**：总起 → `针对问题X，…`（X 加粗）逐段（每问方法+硬数字）→ `总结：…`；
2. **问题分析叙事**：每问 `针对问题X，首先分析本质难点…据此确定切入点（初衷）…备选方案为何放弃…`；
3. **每问必带流程图**：问题重述 1.2 下放问题方法流程图、4.2 数据处理下放"数据处理框架图"、每问算法/求解配流程图。

## 1. 流水线总览（8 阶段契约）

| # | 阶段 | 主 skill | 核心产物 | 验收一句话 |
|---|---|---|---|---|
| 0 | 读经验库 | （本 skill §0.5） | analysis.md 头部"经验清单" | 命中规则已写入 |
| 1 | 读题拆解 | mm-contest-intel / mm-model-select | analysis.md | 每问有候选模型+数据来源 |
| 2 | 数据准备 | mm-data-prep / mmkit.data | data/processed/*.pkl + figures/eda/ | 无未处理缺失、种子固定、数据在赛题本地 |
| 3 | 模型选型 | mm-model-select / mm-models-* | models/README.md | 每问主+对比+灵敏度，有理由表 |
| 4 | 求解执行 | solve_all.py / mmkit.runner（逐问 solve_pN.py 备选） | data/results/*.json/csv + 图 + manifest.json + experiment_log.csv | 每问有结果+图，manifest 生成，结果在赛题本地 |
| 4.5 | Notebook 实验（可选） | mmkit.notebook（py_to_notebook / export_solve_notebook / export_pipeline_notebook） | problem-X/notebooks/*.ipynb | 脚本可转 notebook 迭代，AI 改 cell、人类可审查 |
| 5 | 结果验证 | mmkit.models.evaluate / mm-error-sensitivity | 误差表+灵敏度表 | CV 口径指标、误差长尾有说明 |
| 6 | 绘图定稿 | mm-plotting / mmkit.viz | paper/figures/ 终稿 | 每图有 caption 与 \ref |
| 7 | 论文组装 | mm-latex-paper / mm-flowchart / mm-abstract | paper/main.tex + main.pdf | 编译通过、摘要"针对问题X"总分总、每问流程图、数据处理框架图 |
| 8 | 提交检查 | mm-submit-check / mm-review-sim / **mmkit.audit+rework** | 审核报告 + 返修清单（全 VERIFIED）+ PDF + MD5 + 附件包 | **audit 零 FAIL、返修闭环全部 VERIFIED**、无个人信息 |


## 2. 阶段契约（输入→动作→产物→验收→下一跳）

### 阶段 1：读题拆解
- **输入**：官方赛题 PDF + 附件 + 赛制信息
- **动作**：`mm-contest-intel` 查赛制/时间窗口 → 通读题目，每问转成"问题特征"一句话（`mm-model-select` 速查表）→ 写 analysis.md，章节固定：问题重述/数据概况/问题拆解/模型选择与理由/求解与结果摘要/论文写作要点/复盘记录/复现命令 → 数据概况表列清每个附件（文件/说明/行数/列数/缺失风险）
- **产物**：analysis.md（每问：问题类型+隐含目标+输出要求+模型候选+数据来源）
- **验收**：每问 ≥1 候选模型且指向具体数据文件；附件清单齐全；问题重述无歧义
- **下一跳**：→ 阶段 2

### 阶段 2：数据准备
- **输入**：`data/raw/` 官方附件
- **动作**：`mm-data-prep` 全流程 + `mmkit.data`：`read_table`/`read_dir_tables` 读取 → `clean_df`/`fill_missing`/`remove_outliers` 清洗 → 变换与特征工程（`scale`/`onehot`/`lag_features`/`fft_features`…）→ `eda_report` 出图到 `figures/eda/` → 落盘 `data/processed/*.pkl`（或 csv）→ 数据要点写回 analysis.md §数据概况
- **产物**：`data/processed/` 清洗数据 + `figures/eda/` 探索图 + 清洗前后说明
- **验收**：无未处理缺失/异常（或已说明处理方式）；`set_seed(42)` 固定；关键分布与相关性已出图；时序先切分后做滑窗特征（防泄漏）
- **下一跳**：→ 阶段 3

### 阶段 3：模型选型
- **输入**：analysis.md 分问思路 + 数据概况
- **动作**：`mm-model-select` 按问题特征查表 → 每问定主模型+1 对比模型+1 灵敏度方法 → 到模型族 skill（`mm-models-evaluation/optimization/prediction/statistics/ml-dl/mechanism/graph-sim`）或单模型 skill（`mm-model-ahp/topsis/entropy-weight/milp/metaheuristics/arima/grey-gm11`）取三件套（README/model.tex/model.py，可用 `new_model.py` 生成）→ 汇总 `models/README.md`
- **产物**：`models/README.md`（每问：主模型/对比模型/灵敏度方法/选型理由）
- **验收**：有选型理由表；主+对比+灵敏度三者齐；理由含数据量匹配、假设可验
- **下一跳**：→ 阶段 4

### 阶段 4：求解执行
- **输入**：`data/processed/` + `models/README.md` + `analysis.md`（可选 YAML front-matter）
- **动作**：优先用编排器 `python code/scripts/solve_all.py <problem_dir> [--skip-prep]`（基于 `mmkit.runner`：parse_analysis → 依序执行 solve_pN.py → collect_experiment 追加 `data/results/experiment_log.csv` → build_manifest 生成 `manifest.json`，单步失败不中断并汇总成功/失败清单）；也可逐问写 `code/solve_pN.py`（参考 problem-C 的 solve_p1..p5.py）。脚本末尾 `save_json` 指标到 `data/results/problemX_*.json`、`save_df` 关键表、`save_fig` 落图
- **产物**：`data/results/problemX_*.json/csv` + `figures/*.png` + `data/results/experiment_log.csv` + `manifest.json`
- **验收**：每问有结果文件与图；指标可复现（seed 固定）；manifest/实验日志已登记
- **下一跳**：→ 阶段 5

### 阶段 5：结果验证
- **输入**：results JSON + 求解脚本
- **动作**：`mm-error-sensitivity` + `mmkit.models.evaluate`：`cv_evaluate`（5 折 CV，报告泛化口径）→ `sensitivity_analysis`（±10%/±20% 扰动表）→ 误差三件套（指标表 / 预测 vs 真实图 / 相对误差分布与长尾定位，训练 vs CV 分开标注）→ 对比模型同口径重跑
- **产物**：每问误差表（CV 指标）+ 灵敏度表 + 误差分布说明
- **验收**：指标均为 CV/留出口径（禁止训练集指标冒充泛化指标）；误差长尾有说明（如对数域建模）；对比模型同口径
- **下一跳**：→ 阶段 6

### 阶段 6：绘图定稿
- **输入**：results JSON
- **动作**：`mm-plotting` + `mmkit.viz`：`apply_style()` 设中文字体/300dpi → 从 results 读数据出图（图=实验产物，禁止凭空画）→ `save_fig` 300dpi PNG+PDF 到 `paper/figures/` → 每图配 caption 与 `\ref`；用 `PALETTE`/`COLORBLIND_SAFE` 统一配色（同一指标全文同色）
- **产物**：`paper/figures/` 终稿（fig<N>_<语义>.png/pdf）
- **验收**：每图在 main.tex 有 `\includegraphics`+caption+正文 `\ref`；无占位框、无 72dpi 图
- **下一跳**：→ 阶段 7

### 阶段 7：论文组装
- **输入**：analysis.md + paper/figures/ + data/results/ + models/ 三件套
- **动作**：`mm-latex-paper`：复制 `paper/latex-template-official/` → `\makecover` 填题号/队号 → 按题型套 `paper/latex-template-types/skeleton-*.tex` → `\input models/<类别>/<模型>/model.tex` 公式段 → results 表转 LaTeX 表格（pd.to_latex 或模板手写）→ 每问画流程图/技术路线图（mm-flowchart：tikz 或 `mmkit.viz.flowchart`）→ 摘要五要素（mm-abstract：数字回填自 experiment_log.csv，硬数字 `\textbf`）→ AI 声明（ai-disclosure.tex）→ `latexmk -xelatex` 编译 → `check_figures.py` + `check_tex_fragments.py`
- **产物**：`paper/main.tex` + `main.pdf`
- **验收**：编译通过；摘要五要素齐全且 ≤2 页；每问有流程图；正文 `\ref` 全部解析；无悬空引用/缺图
- **下一跳**：→ 阶段 8

### 阶段 8：提交检查（审核 → 返修闭环 → 打包）
- **输入**：main.pdf + code/ + 附件
- **动作**：
  1. **完篇审核**：`python code/scripts/audit_paper.py contests/<年>-<赛>/problem-<X>`（提交模式，**不带** --allow-cover-placeholder）→ audit_report.md；
  2. **返修闭环**：`python code/scripts/rework_paper.py <problem_dir>` 生成返修清单 → 逐项按 fix_hint 修复 → `--mark-fixed <CID>` → `--verify` 复验（**致命项全 VERIFIED 且零 FAIL 才能提交**）；
  3. `mm-submit-check`：`paper/checklists/paper-checklist.md` 逐项 → `check_figures.py` → `package_submit.py --problem X --team ... --pdf paper/main.pdf --attach-dir code`（命名 A<队号>.pdf + MD5 + 附件 zip ≤50MB）→ `mm-review-sim` 七维打分 + `mm-vision` 读图自查
- **产物**：审核报告 + rework_tracker.json（全 VERIFIED）+ 命名 PDF + MD5 文件 + 附件包
- **验收**：audit 零 FAIL、返修闭环全部 VERIFIED；checklist 全过；除封面外无个人信息；MD5 后不再改 PDF；提前 2h 完成提交
- **下一跳**：结束 → 复盘（analysis.md §复盘记录 + 获奖论文萃取脚本对标）

## 3. 跨问数据流与产物对账

- **跨问数据流**：上一问输出可作为下一问输入（例 2024-C：P1 波形分类结果回填 P4 特征、P4 预测损耗供 P5 双目标）。在 analysis.md §问题拆解标注每问输入依赖（原始数据 or 前问产物），按依赖拓扑顺序实现（先做被依赖问）。
- **产物对账（manifest）**：`data/results/manifest.json` 登记 图→数据源（results JSON/key）→论文 `\ref`；提交前跑 `code/scripts/build_manifest.py` 生成 `experiment_log.csv` 与 `paper/figures/manifest.md` 对账：figures/ 每张 png 均被引用（无孤儿图/占位框）、每个 results JSON 至少被一图或一表使用；表↔图↔正文数字以 JSON 为唯一事实源，禁止手抄。

## 4. AI 协作纪律

- 每步 AI 产物标注来源：代码头/AI 声明注明工具与日期（`code/scripts/annotate_ai.py` 批量补，附件4 合规）；AI 生成占比 ≤30%（03-ai-workflow 红线）；
- 随机种子：`set_seed(42)`，一切划分/CV/采样固定；
- CV 口径优先：论文泛化指标一律 K 折 CV 或留出集，训练指标只作过程参考，禁止冒充（口径纪律同 mm-abstract，红线）；
- AI 不代写摘要、不查文献、不编参数；代码逐行看懂 + 假数据冒烟验证；
- 图即实验产物：论文图全部来自求解/验证脚本落盘文件，禁止手绘占位框。

## 5. 资源

### 依赖 skill

| skill | 用途 |
|---|---|
| mm-solve-runbook | 四天日程/里程碑（本 skill 的时间表） |
| mm-contest-intel | 赛制/时间窗口/格式硬性要求 |
| mm-model-select | 问题特征→模型候选 |
| mm-data-prep | 清洗/特征/EDA |
| mm-models-*（7 族）+ mm-model-*（单模型） | 模型三件套 README/model.tex/model.py |
| mm-plotting | 论文级绘图总规范（图表类型/配色/分辨率） |
| mm-latex-paper | 官方模板/章节骨架/编译/查图 |
| mm-flowchart | 流程图/技术路线图（tikz 或 mmkit.viz.flowchart） |
| mm-abstract | 摘要五要素 + 数字回填 + 口径纪律 |
| mm-error-sensitivity | 误差/灵敏度分析章节（指标表、预测 vs 真实图、扰动表） |
| mm-submit-check | 打包/MD5/时间窗口 |
| mm-review-sim | 评审模拟七维打分 + 自动化检查命令链 |
| mm-3d-visual / mm-vision | 三维题配图 / 读图审图 |

### 编排器与自动化工具（P1/P2 已落地）

- 求解编排：`mmkit.runner`（parse_analysis / run_question / run_all / collect_experiment / build_manifest / make_run_id）+ `code/scripts/solve_all.py <problem_dir> [--skip-prep]`——依序执行 prep_eda/solve_pN、失败不中断、追加 experiment_log.csv、生成 manifest.json。
- 结果→论文：`mmkit.paper.csv2latex`（CSV/DataFrame → 三线表 tex 片段）、`mmkit.paper.fig_catalog` + `check_figures.py --report`（图表↔\ref 对账）。
- 附件/提交：`mmkit.io.xlsx_fill`（fill_xlsx_template / assert_consistency / verify_filled_xlsx）+ `code/scripts/fill_attachment.py`。
- 复现评分：`code/scripts/score_reproducibility.py reproduction/`（方法/工作流/写作覆盖评分卡，见 `reproduction/README.md`）。
- 模型桥接：`mmkit.models.registry`（scan_models / get_model / import_api / run_script，79 个模型三件套可程序化调用）。

### 目录与脚本

- 工作区：`contests/<年>-<赛>/problem-<X>/`（analysis.md、code/、data/{raw,processed,results}/、figures/eda/、paper/{figures,latex-template-official,latex-template-types,sections,checklists}/、models/）
- 代码：`code/mmkit/`（data / models / viz / utils / paper / io / runner / geo3d）、`code/scripts/`（new_contest / new_model / check_figures / check_tex_fragments / verify_models / package_submit / check_paper_length / build_manifest / annotate_ai / solve_all / fill_attachment / score_reproducibility）
- 规范：`code/meta/01-experiment-workflow.md`（实验记录与落盘）、`code/meta/03-ai-workflow-and-figures.md`（六阶段 + 图表纪律）、`paper/plotting-guide.md`、`paper/checklists/*.md`
- 范例：`contests/2024-huawei/problem-C/`（analysis.md + solve_p1..p5.py + paper/main.tex 全链路样例）
