---
name: mm-flowchart
description: 流程图与技术路线图生成。当需要为问题分析画总体技术路线图、为每个算法/模型配流程图（tikz 或 mmkit.viz.charts.flowchart 两套实现）、或检查流程图画法规范与引用时使用本 skill。
whenToUse: 画流程图、技术路线图、算法流程示意图、流程图规范检查时
---

# 流程图 / 技术路线图

## 什么时候放流程图

- **总体技术路线图**：放"问题重述与分析"末尾（评审 30 秒看全貌，一图看懂解题思路）；
- **每个算法/模型配一张流程图**：评审铁律"复杂过程画流程图"——模型建立、求解算法、迭代优化都值得配图；
- 流程复杂到文字说不清时，一律用图代替大段文字（图文并茂是获奖论文共性）。

## 要素规范

| 要素 | 画法 |
|---|---|
| 开始/结束 | 圆角框（rounded rectangle） |
| 处理步骤 | 矩形框（rectangle） |
| 判定 | 菱形（diamond），分支必须标"是/否" |
| 流向 | 箭头（Stealth），单向清晰 |
| 模块归属 | 泳道式/虚线分组标注（哪个环节、哪个模型） |

- 图题在**下方**，正文必须 `\ref` 引用；图内中文（XeLaTeX + ctex 或 Python 中文字体）；
- 同一篇论文内所有流程图风格统一（同一配色/字体/箭头样式）。

## 方案 A：tikz 流程图（论文内直接排版）

前置：`\usepackage{tikz}\usetikzlibrary{shapes.geometric,arrows.meta,positioning}`（缺库会编译报错）。

完整可复制模板（开始→预处理→建模→判定→输出）：

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=1.2cm and 1.6cm,
  box/.style={rectangle, draw, fill=blue!8, minimum width=2.6cm, minimum height=0.9cm, align=center, font=\small},
  dec/.style={diamond, draw, fill=yellow!15, aspect=2, align=center, font=\small},
  term/.style={rounded rectangle, draw, fill=green!10, minimum width=2.2cm, minimum height=0.8cm, align=center, font=\small},
  arr/.style={-{Stealth[length=2.5mm]}, thick}
]
  \node[term] (s) {开始};
  \node[box, below=of s] (d1) {数据预处理\\（缺失/异常/标准化）};
  \node[box, below=of d1] (d2) {模型建立\\（公式推导）};
  \node[dec, below=of d2] (c1) {满足约束？};
  \node[box, right=of c1] (d3) {调整参数/惩罚项};
  \node[box, below=of c1] (d4) {求解与结果分析};
  \node[term, below=of d4] (e) {输出结论};
  \draw[arr] (s) -- (d1);
  \draw[arr] (d1) -- (d2);
  \draw[arr] (d2) -- (c1);
  \draw[arr] (c1) -- node[left] {否} (d4);
  \draw[arr] (c1) -- node[above] {是} (d3);
  \draw[arr] (d3) |- (d2);
  \draw[arr] (d4) -- (e);
\end{tikzpicture}
\caption{总体技术路线图}\label{fig:route}
\end{figure}
```

## 方案 B：Python 出图（推荐，与实验数据风格一致）

```python
from mmkit.viz.charts import flowchart
from mmkit.viz.style import apply_style, save_fig

apply_style()   # 中文字体 + 300dpi
fig = flowchart(
    nodes=["开始", "数据预处理", "模型建立", "约束检查", "求解与分析", "输出结论"],
    edges=[(0, 1), (1, 2), (2, 3), (3, 4), (3, 2), (4, 5)],   # 3→2 为"否"回环
    title="总体技术路线图",
)
save_fig(fig, "route", outdir="figures")   # PNG(300dpi) + PDF 双格式
```

## 验收标准

- 每问（含技术路线图）**至少 1 张流程图**，全篇风格统一；
- 每张图有 caption、图题下方、正文 `\ref` 引用；
- PNG 300dpi 或矢量 PDF；`python code/scripts/check_figures.py main.tex` 通过（无缺图/悬空引用）。

## 常见坑

- tikz 库未加载（`shapes.geometric/arrows.meta/positioning`）→ 编译报错；
- 节点文字溢出 → 设 `minimum width` + `align=center`，长文字用 `\\` 换行，字号 `\small`；
- 判定菱形分支不标"是/否"→ 评审看不懂回环逻辑；
- 图未 `\ref`（悬空图）→ `check_figures.py` 报 FAIL；
- 中文乱码 → 用 XeLaTeX 编译；Python 出图必须 `apply_style()`。

## 资源

- 模板：`paper/sections/template-flowchart.tex`（tikz 方案 A + Python 方案 B 注释版）
- 代码：`code/mmkit/viz/charts.py`（`flowchart`）、`code/mmkit/viz/style.py`（`apply_style/save_fig`）
- 检查：`code/scripts/check_figures.py`；绘图总规范：`mm-plotting`
