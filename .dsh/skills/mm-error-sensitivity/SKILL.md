---
name: mm-error-sensitivity
description: 误差分析与灵敏度分析章节撰写。当需要写"模型评价/误差与精度分析/灵敏度分析"小节（指标表 RMSE/MAE/MAPE/R²、预测 vs 真实图、相对误差分布与长尾定位、参数 ±10%/±20% 扰动表、训练 vs CV 口径分开标注），或调用 code/mmkit/models/evaluate.py 生成指标、CV 与灵敏度数据时使用本 skill。
whenToUse: 写误差分析、灵敏度分析、模型评价章节、生成指标表与扰动表时
---

# 误差分析与灵敏度分析（获奖论文标配）

## 为什么必备

- 评审铁律第 3 条：**每个问题求解后必给误差分析**（残差/方差/概率分布均可）；
- 获奖论文共性：误差指标表 + 预测 vs 真实图 + 灵敏度/稳健性结论，缺一项都是失分点。

## 误差分析三件套（每问至少 1 图 1 表）

**① 指标表**：回归 RMSE/MAE/MAPE/R²；分类 acc/F1/AUC；聚类轮廓/CH/DB。多模型对比时**训练 vs CV 必须分开标注**。

```latex
\begin{table}[htbp]
\centering
\caption{模型精度指标（5 折 CV 均值）}\label{tab:err-metrics}
\begin{tabular}{lcccc}
\toprule
模型/方法 & RMSE & MAE & MAPE(\%) & $R^2$ \\
\midrule
本文模型 & $e_1$ & $e_2$ & $e_3$ & $r_1$ \\
基准模型 1 & $e_4$ & $e_5$ & $e_6$ & $r_2$ \\
\bottomrule
\end{tabular}
\end{table}
```

**② 预测 vs 真实散点图**（R² 标注在图上）；**③ 相对误差分布**（直方/分场景箱线）——用于定位误差集中在哪些样本（长尾/高波动）。

## 指标与 CV 数据生成（evaluate.py）

```python
from mmkit.models.evaluate import regression_metrics, cv_evaluate

# 5 折 CV：返回 {folds: {fold_i: metrics}, mean: metrics}
cv = cv_evaluate(model, X, y, task="regression", cv=5, random_state=42)
print(cv["mean"])          # {'r2':..., 'mae':..., 'rmse':..., 'mape_pct':...}
print(cv["folds"])         # 逐折指标（可画逐折柱状图）

# 单次评估（测试集/验证集，非训练集！）
m = regression_metrics(y_test, pred)
```

分类/聚类：`classification_metrics(y_true, y_pred, y_prob)`（acc/F1/AUC）、`clustering_metrics(X, labels)`（轮廓/CH/DB）。

## 灵敏度分析

- 形式：对核心参数施加 ±10%/±20% 扰动 → 指标变化% + 最优方案/排序是否保持不变；
- 结论句式（直接套用）：
> 对参数 X 施加 ±20% 扰动，<指标>变化 <数字>%，<排序/最优方案>保持不变，
> 表明模型在该参数上<稳健/需谨慎选取>；原因在于<机理解释：目标函数平缓/约束起主导作用>。

```python
from mmkit.models.evaluate import sensitivity_analysis

# 超参数网格灵敏度 → DataFrame（论文"参数敏感性"图表数据源）
df = sensitivity_analysis(model, X, y,
                          param_grid={"n_estimators": [50, 100, 200],
                                      "max_depth": [3, 5, 7]},
                          metric="rmse", task="regression", cv=3)
df.to_csv("data/results/sensitivity.csv", index=False, encoding="utf-8-sig")
```

问题级参数（阈值/容量等）的 ±20% 扰动：写循环改参数重跑求解，结果落盘 `data/results/`，再画扫描曲线/扰动表（模板见 `paper/sections/template-sensitivity.tex`）。稳健性结论建议配多次随机种子重复（均值±方差）。

## 多模型对比口径

- 表格中**"训练"与"CV/测试"两套指标分列**，严禁混用或只报训练集；
- 摘要/结论只引用 CV 或测试集指标（训练集 MAPE 当结果 = 评审红线）；
- 理论 vs 仿真 vs 实测三列对比（误差 %）是获奖级加分写法。

## 验收标准

- 每问 ≥1 张误差图（预测 vs 真实 / 误差分布）+ 1 张灵敏度表（扰动表或扫描表）；
- 指标表含 2 个以上模型/基准对比，训练与 CV 分开标注；
- 灵敏度有结论句（稳健/敏感 + 原因）；`check_figures.py` 与 `check_paper_length.py` 相关项通过。

## 常见坑

- 只报训练集指标（红线）；多模型只比最优不比基线；
- 灵敏度只给表不给结论句，或结论与数据矛盾；
- 误差图无 R²/无坐标单位；误差分布不解释（系统性偏差 vs 随机误差）；
- 扰动表数字手编，与代码运行结果不一致。

## 资源

- 模板：`paper/sections/template-error-analysis.tex`、`paper/sections/template-sensitivity.tex`
- 经验：`paper/meta/05-sensitivity-error-ai.md`；评审视角：`paper/meta/06-reviewer-perspective.md`
- 代码：`code/mmkit/models/evaluate.py`（`regression_metrics/cv_evaluate/sensitivity_analysis`）
- 绘图：`mm-plotting`；摘要口径：`mm-abstract`
