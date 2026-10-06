---
name: mm-model-arima
description: ARIMA 时间序列单模型精讲。当用 ARIMA/SARIMA 做时序外推、自动选阶（AIC）、残差诊断与多步预测时使用本 skill。
whenToUse: 已选 ARIMA（时间序列预测场景）且需要直接调用实现时
---

# ARIMA（单模型）

## 三件套
- 代码：`models/03-prediction/time-series-arima/model.py`
- 论文片段：`models/03-prediction/time-series-arima/model.tex`
- 介绍：`models/03-prediction/time-series-arima/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/03-prediction/time-series-arima")
from model import TimeSeriesARIMA

m = TimeSeriesARIMA(seasonal=False)
m.fit(y)                 # AIC 自动选阶 (p,d,q)
pred = m.predict(steps=30)
print(m.order, m.metrics)
```

## 竞赛要点
1. 流程：平稳性检验（ADF）→ 差分 → 定阶（AIC）→ 残差白噪声检验（Ljung-Box）；
2. 论文给：ACF/PACF 图、选阶表（AIC 对比）、预测误差（RMSE/MAPE）；
3. 与 GM(1,1)/XGBoost/LSTM 对比是标准配置；
4. 多步预测给预测区间（±1.96σ）；
5. 有季节性 → SARIMA（seasonal=True）。

## 常见坑
- 未差分就拟合（非平稳 → 伪回归）；
- 过拟合阶数（p,q 过大）→ AIC 会惩罚但需交叉验证；
- 预测值漂移（d>0 无 trend 时）→ 注意 trend 参数；
- 数据量 < 30 时 ARIMA 不稳，改用 GM(1,1)。
