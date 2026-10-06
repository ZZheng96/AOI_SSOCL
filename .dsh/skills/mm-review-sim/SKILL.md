---
name: mm-review-sim
description: 评审模拟与赛前自查。当需要在上交前用评审视角给论文/成果打分（格式合规、篇幅、图表密度、摘要质量、方法深度、代码可复现、AI 合规七维度，含获奖线与一票否决项）、运行自动化检查命令链、或输出评审模拟打分卡时使用本 skill。
whenToUse: 赛前自查、评审模拟、论文打分、提交前质量评估时
---

# 评审模拟（赛前自查 = 用评审视角打分）

## 为什么做

- A/B 题评估教训：格式违规一票否决 + 篇幅/图表远低于获奖线 = **不能获奖**（见 `docs/quality-evaluation-AB.md`）；
- 与其交卷后才知道差距，不如提前用评审视角逐维度打分、按 P0/P1/P2 修复闭环。

## 打分维度表（获奖线 vs 一票否决项）

| 维度 | 获奖线 | 一票否决项 |
|---|---|---|
| 格式合规 | 页码页脚中部阿拉伯数字、无页眉、封面官方 logo、字体字号合规 | 格式违规（页眉/页码错）；**除封面外出现个人信息**（单位/姓名/队号） |
| 篇幅 | 正文 ≥30 页、45 页左右最佳；每问 6~8 页 | 篇幅远低于获奖线（如 <15 页）且无扩充迹象 |
| 图表密度 | 25~40 张；每图有 caption 且正文引用 | 占位框当图（\fbox 未替换）；大量悬空引用 |
| 摘要质量 | 五要素 + 每问 2~4 个定量结果 + 对比基准 + 硬数字加粗 | 无数字只有"效果良好"；**训练指标冒充泛化指标** |
| 方法深度 | 模型+算法+最优性/对比验证；灵敏度/误差分析齐全 | 只有套话无求解细节；结果无法复现 |
| 代码可复现 | set_seed(42)、结果落盘 data/results/、README/运行文档 | 随机未固定、结果手编、无运行说明 |
| AI 合规 | 附件 4 声明齐全（代码头注释 + 论文 AI 使用声明 + 引用来源标注） | **AI 未披露**（按规则取消评奖资格） |

参考权重（AB 评估用）：格式合规 15% / 论文内容与图表 40% / 代码质量 20% / 模型质量 25%；获奖 ≈ 8+/10。

## 自动化检查命令链（按序执行）

```powershell
# 0. ★完篇审核（AI 全自动自查表，6 维度/匿名/AI 合规/摘要/检验/图表/数据落盘，零 FAIL 才算过）
python code/scripts/audit_paper.py contests/<年>-<赛>/problem-<X>

# 1. 图表与引用（缺图/低分辨率/悬空引用）
python code/scripts/check_figures.py paper/main.tex

# 2. 篇幅/结构/图表密度/灵敏度/文献自检（零 FAIL 才算过）
python code/scripts/check_paper_length.py paper/main.tex --pdf paper/main.pdf --bib paper/ref.bib

# 3. 模型片段静态检查（label 前缀/环境配对/禁 documentclass）
python code/scripts/check_tex_fragments.py

# 4. 提交打包（命名/MD5/附件 ≤50MB）
python code/scripts/package_submit.py --problem A --team 25000010001 `
    --pdf contests/2025-huawei/problem-A/paper/main.pdf `
    --attach-dir contests/2025-huawei/problem-A/code --outdir dist
```
> 0 号（audit_paper.py）输出 `data/results/audit_report.md`：汇总表 + 整体判定 + 问题分级（致命/一般/轻微）+ 优先级整改清单，对应 2026 国赛 AI 全自动自查表（标准版）6 大维度 26 子项；一票否决项（匿名泄漏/AI 未披露/摘要训练指标冒充）自动 FAIL。

再逐条对照 `paper/checklists/paper-checklist.md` 手工项（封面 logo、个人信息、AI 声明、关键词个数、PDF 命名）。

## 模拟流程（五步）

1. **跑命令链**：执行上文 4 条检查命令，记录每个 FAIL/WARN；
2. **逐维打分**：按维度表给 ★ 并附"证据/问题"（引用具体页、图、章节）；
3. **定级**：命中任何一票否决项 → 本轮直接判不获奖，先修 P0；
4. **列改进清单**：P0（一票否决，必须清）→ P1（大幅影响评分）→ P2（提升上限）；
5. **修复后复评**：P0 清零、P1 完成后重跑一轮打分卡，直到加权 ≥8/10。

## 评审模拟打分卡模板

```markdown
# 评审模拟打分卡（<题号>，<日期>）

| 维度 | 权重 | ★ 评分 | 证据/问题 | 改进项（P0/P1/P2） |
|---|---|---|---|---|
| 格式合规 | 15% | ★★☆☆☆ | 页眉未删、页码在右上 | P0：改 huawei-cup.sty 页脚页码+删页眉 |
| 论文内容与图表 | 40% | ★★★☆☆ | 图 8 张 <25，正文 12 页 | P1：接真实结果图、扩写问题分析章 |
| 代码质量 | 20% | ★★★★☆ | set_seed(42)、结果落盘 ✓ | P2：补 README |
| 模型质量 | 25% | ★★★☆☆ | 无最优性验证 | P1：小图精确解算 gap |
| **加权总分** | | **5.3/10** | 距获奖（8+）差 2.7 | 先清 P0，再攻 P1 |

# 改进项清单（按优先级）
- [ ] P0 ……
- [ ] P1 ……
- [ ] P2 ……
```

评分口径：每维 0~10 分（★ = 2 分），加权求和；任何一票否决项命中 → 本轮**直接判不获奖**，先修复再评。

## 常见坑

- 只自查格式不查内容（格式过了、内容空 → 照样拿不到奖）；
- 用训练集指标自我表扬（评审视角第一眼就穿帮）；
- 打分卡写了不改：P0 项必须清零后才进入下一轮评估；
- 忽略 AI 合规（附件 4 是取消资格级风险，不是普通扣分项）。

## 资源

- 方法论：`docs/quality-evaluation-AB.md`（A/B 题完整评估示例 + P0/P1/P2 修复闭环）
- 评审视角：`paper/meta/06-reviewer-perspective.md`；自查清单：`paper/checklists/paper-checklist.md`
- 检查脚本：`code/scripts/check_figures.py`、`check_paper_length.py`、`check_tex_fragments.py`、`package_submit.py`
- 规则情报：`mm-contest-intel`；提交打包：`mm-submit-check`；摘要：`mm-abstract`
