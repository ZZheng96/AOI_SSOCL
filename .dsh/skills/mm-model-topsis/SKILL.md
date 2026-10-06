---
name: mm-model-topsis
description: TOPSIS 单模型精讲。当确定用 TOPSIS 做逼近理想解排序、结合熵权/CRITIC 权重、贴近度计算与排序时使用本 skill。
whenToUse: 已选 TOPSIS（评价排序场景）且需要直接调用实现时
---

# TOPSIS（单模型）

## 三件套
- 代码：`models/01-evaluation/topsis/model.py`（含熵权组合权重演示）
- 论文片段：`models/01-evaluation/topsis/model.tex`
- 介绍：`models/01-evaluation/topsis/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/01-evaluation/topsis")
from model import TOPSIS

t = TOPSIS(direction=[1, -1, 1])   # 各指标方向：1 越大越优，-1 越小越优
t.fit(X, weights=None)             # weights=None 时内部用熵权法；或传入 [w1,w2,...]
print(t.scores, t.rank)            # 贴近度与排序
```

## 竞赛要点
1. **先归一化再加权**：正向指标 `x'=(x-min)/(max-min)`，负向指标取倒数或反转；
2. 权重来源写清楚：熵权（客观）/AHP（主观）/组合，论文给权重表；
3. 输出：正负理想解、贴近度 C_k、排序表 + 雷达图/柱状图；
4. 与 AHP/熵权组合是最高频组合套路（"熵权-TOPSIS"）；
5. 灵敏度：权重扰动下排序稳定性。

## 常见坑
- 忘记指标同向化（负向指标直接参与计算会倒排）；
- 标准化用 z-score 会破坏 TOPSIS 的 0-1 框架，用 min-max；
- 方案太少（<3）时区分度弱。
