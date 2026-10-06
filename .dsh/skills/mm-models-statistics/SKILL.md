---
name: mm-models-statistics
description: 统计类模型族。当需要假设检验、方差分析、相关性分析、聚类、判别分析、蒙特卡洛/Bootstrap、回归诊断等统计推断与分析时使用本 skill。
whenToUse: 问题含"显著性/差异/相关/分组/聚类/检验/置信区间"等关键词，或需要严谨统计依据时
---

# 统计类模型族（models/04-statistics）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 假设检验 | t/F/卡方/正态性/非参 | `hypothesis-testing` |
| 方差分析 ANOVA | 单/双因素 + Tukey | `anova` |
| 相关分析 | Pearson/Spearman/偏相关 | `correlation-analysis` |
| 聚类分析 | K-means/层次/DBSCAN/GMM | `clustering` |
| 判别分析 | LDA/QDA/Fisher | `discriminant-analysis` |
| 蒙特卡洛/Bootstrap | 模拟+置信区间 | `monte-carlo-bootstrap` |
| 回归诊断 | 残差/异方差/共线性/影响点 | `regression-diagnostics` |

## 推荐套路

1. 差异/显著性 → 检验或 ANOVA（先正态性/方差齐性检验）；
2. 变量关系 → 相关分析（+偏相关剔除混淆）；
3. 分群 → K-means（肘部/轮廓系数定 k）+ 降维可视化（`mm-models-ml-dl` 的 PCA/t-SNE）；
4. 分类判别 → LDA/QDA（可解释）+ 分类器对比（`mm-models-ml-dl`）；
5. 不确定度 → Bootstrap 置信区间；模拟 → 蒙特卡洛；
6. 回归建模前 → `regression-diagnostics`（VIF 共线性、残差正态、异方差）。

## 论文要点

- 检验结果写全：统计量、p 值、显著性结论（表格式）；
- 聚类给轮廓系数/CH/DB 指标 + 可视化；
- 回归诊断结果是"为什么选这个模型"的证据，评审加分项。

## 资源

- `models/README.md`；`paper/latex-template-types/skeleton-statistics.tex`
- 2025 B/E 题的数据分析部分主用本族
