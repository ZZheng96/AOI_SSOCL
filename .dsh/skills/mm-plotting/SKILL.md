---
name: mm-plotting
description: 论文级绘图方案。当需要生成论文配图（折线/柱状/散点/热力/箱线/雷达/等值线/网络/甘特/帕累托/流程图）、设置中文字体、控制分辨率与配色、检查图表规范性时使用本 skill。
whenToUse: 任何要进论文的图、EDA 图转论文图、图表规范性检查
---

# 论文级绘图

## 规范速查

- 入口：`apply_style()` 设置中文字体（微软雅黑/SimHei 自动探测）、字号 10.5pt、300dpi；
- 保存：`save_fig(fig, name, outdir="figures", formats=("png","pdf"))` —— PNG 300dpi 进论文，PDF 留矢量；
- 尺寸：`set_size()`（通栏 14cm / 单栏 7cm，黄金比例）；
- 配色：`PALETTE`/`COLORBLIND_SAFE`，同一指标全文同色；
- 图题在下、表题在上；正文必须 `\ref` 引用每张图。

## 图表类型（mmkit.viz.charts）

| 需求 | 函数 |
|---|---|
| 趋势/多方案对比 | `line_chart`（支持误差带） |
| 分类比较/占比 | `bar_chart`、`pie_chart` |
| 相关/分布 | `scatter_plot`（含 R²）、`hist_plot`、`box_plot` |
| 相关矩阵/混淆矩阵 | `heatmap` |
| 综合评价对比 | `radar_chart` |
| 二维函数/区域 | `contour_plot` |
| 网络结构 | `network_plot` |
| 调度排程 | `gantt_chart` |
| 多目标优化 | `pareto_front` |
| 建模/算法流程 | `flowchart` |

## 使用步骤

1. 读 `paper/plotting-guide.md`（完整规范与图密度建议）；图表设计规范（38 类型/语义模式/密度/焦点色）见 `paper/diagram-design-guide.md`；
2. 用 charts 函数生成 fig → `save_fig` 落盘 `figures/`；
3. 论文中 `\includegraphics` + caption + 结论性文字；
4. 提交前 `python code/scripts/check_figures.py main.tex` 检查缺图/分辨率/悬空引用。

## 常见坑

- PNG 默认 72dpi → 必须 `save_fig`（300dpi）；
- 中文乱码/负号方块 → `apply_style()` 已处理 `unicode_minus`；
- 坐标轴无单位、图例缺失；
- 三维图只有一个视角（至少正视+俯视，见 `mm-3d-visual`）。

## 资源

- 规范：`paper/plotting-guide.md`；设计规范：`paper/diagram-design-guide.md`（38 种视觉类型/语义模式/密度 4/10/焦点色，源自 Claude editorial-diagrams 2.5）
- 代码：`code/mmkit/viz/style.py`、`code/mmkit/viz/charts.py`
- 检查脚本：`code/scripts/check_figures.py`
