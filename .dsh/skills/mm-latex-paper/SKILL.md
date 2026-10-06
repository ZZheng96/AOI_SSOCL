---
name: mm-latex-paper
description: 论文生成与 LaTeX 排版。当需要按华为杯官方格式生成论文（封面+统一摘要页+正文）、按题型套用章节骨架、引用模型公式片段、管理参考文献、编译与检查图表引用时使用本 skill。
whenToUse: 写论文、论文排版、摘要撰写、编译与图表检查时
---

# 论文生成（LaTeX）

## 模板结构（paper/latex-template-official/）

| 文件 | 用途 |
|---|---|
| `main.tex` | 主文件：封面 `\makecover` + 摘要 `mcmabstract` + 正文骨架 |
| `huawei-cup.sty` | 样式（页面/字体/页眉/封面/摘要/算法/代码环境） |
| `ref.bib` | gbt7714 数值式参考文献 |
| `latexmkrc` | latexmk 配置（XeLaTeX + biber） |

## 使用步骤

1. 复制 `latex-template-official/` 到赛题 `paper/` 目录；
2. `\makecover` 填题号/队号/题目/队员；**logo 占位换成官方文件**（figures/logo1~4.png）；
3. 按题型选骨架 `paper/latex-template-types/skeleton-*.tex` 的章节套路；
4. 模型公式：`\input{models/<类别>/<模型>/model.tex}` 或复制公式段（label 前缀 eq:/tab:/alg: 全局唯一即可）；
5. 摘要按五要素写（`paper/sections/abstract.md`），≤ 2 页；
6. AI 使用声明：`\input{paper/sections/ai-disclosure.tex}` 并填写；
7. 编译：`latexmk -xelatex main.tex`（需可用 TeX 发行版；本机 MiKTeX 若损坏需先修复）；
8. 检查：`python code/scripts/check_figures.py main.tex`（缺图/悬空引用/低分辨率）。

## 合规红线

- 除封面外不得出现单位/姓名/队号（页眉、附录、代码注释都查）；
- 摘要 ≤ 2 页；五要素齐全；
- 引用程序与 AI 产品注明来源；查重前自查。

## 资源

- 写作指南：`paper/sections/writing-guide.md`；自查：`paper/checklists/paper-checklist.md`
- 绘图：`mm-plotting`；提交：`mm-submit-check`
