---
name: mm-models-prediction
description: 预测类模型族。当需要为时间序列外推、回归预测、趋势预测问题选择或调用线性回归、ARIMA、灰色预测 GM(1,1)、插值拟合、SVR、梯度提升树、LSTM/GRU、马尔可夫、小波分析等模型时使用本 skill。
whenToUse: 问题含"预测/外推/趋势/回归/未来/估计"等关键词时
---

# 预测类模型族（models/03-prediction）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 线性回归（岭/Lasso） | 多元回归+显著性 | `linear-regression` |
| ARIMA | 时序外推（自动选阶） | `time-series-arima` |
| 灰色预测 GM(1,1) | 小样本外推 | `grey-prediction-gm11` |
| 插值与拟合 | 样条/参数拟合 | `interpolation-fitting` |
| SVR | 小样本非线性回归 | `support-vector-regression` |
| 梯度提升树 | XGBoost/LightGBM/GBDT | `gradient-boosting-trees` |
| LSTM/GRU | 深度学习时序 | `lstm-gru` |
| 马尔可夫 | 状态转移预测 | `markov-prediction` |
| 小波分析 | 去噪+时频+趋势外推 | `wavelet-analysis` |

## 推荐套路

1. 数据量小（<30 样本）→ GM(1,1) / SVR / 插值；
2. 数据量大且平稳性可处理 → ARIMA（AIC 选阶）→ 对比 XGBoost / LSTM；
3. **多模型对比是论文标配**：简单基准（均值/线性）→ 统计模型 → ML → DL，报 RMSE/MAE/MAPE；
4. 预测必须给**预测区间**（残差 Bootstrap/分位数）；
5. 信号类（振动/速率）→ 先小波/FFT 特征，再回归。

## 使用步骤

1. 读包 README；2. 跑 model.py 演示；3. 套用数据；4. 结果进论文（误差表+拟合曲线图）。

## 论文要点

- 训练/验证/测试划分写明（滚动窗口防泄漏）；
- 误差表（RMSE/MAE/MAPE）+ 预测曲线对比图（真实 vs 各模型）；
- 外推结果给区间；说明模型适用边界（如"GM(1,1) 适合单调趋势，波动大时失效"）。

## 资源

- `models/README.md`；`paper/latex-template-types/skeleton-prediction.tex`
- 2025 B 题（链路速率）主用本族 + `mm-models-ml-dl`
