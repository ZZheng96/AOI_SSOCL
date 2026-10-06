---
name: mm-model-grey-gm11
description: 灰色预测 GM(1,1) 单模型精讲。当小样本（数据点少）趋势外推、无法用统计模型时使用本 skill。
whenToUse: 已选 GM(1,1)（小样本预测场景，n<30 或 n 很小）且需要直接调用实现时
---

# 灰色预测 GM(1,1)（单模型）

## 三件套
- 代码：`models/03-prediction/grey-prediction-gm11/model.py`
- 论文片段：`models/03-prediction/grey-prediction-gm11/model.tex`
- 介绍：`models/03-prediction/grey-prediction-gm11/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/03-prediction/grey-prediction-gm11")
from model import GreyPredictionGM11

g = GreyPredictionGM11()
g.fit(x0)              # 原始序列（≥4 个点）
pred = g.predict(steps=5)
print(g.params, g.accuracy)   # a,b 参数 + 后验差比 C / 小误差概率 p
```

## 竞赛要点
1. 适用：**单调趋势 + 小样本**；波动大时先做数据平滑或分组；
2. 精度检验必写：后验差比 C（<0.35 好）与小误差概率 p（>0.95 好）；
3. 相对残差检验 + 级比检验（建模前验证可行性）；
4. 论文给：累加序列、白化方程、参数表、拟合+外推图；
5. 组合：GM(1,1) 做趋势 + 残差修正（如马尔可夫修正）提升精度。

## 常见坑
- 数据含负值/零 → 先平移（级比检验要求全正）；
- 外推步数过多（>数据长度）误差放大；
- 序列波动大时 GM(1,1) 失效，改 ARIMA/回归。
