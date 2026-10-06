---
name: mm-model-ahp
description: 层次分析法 AHP 单模型精讲。当确定用 AHP 计算主观权重、做一致性检验（CR<0.1）、方案排序或灵敏度分析时使用本 skill（含三件套调用细节）。
whenToUse: 已选 AHP（评价/权重场景）且需要直接调用实现时
---

# 层次分析法 AHP（单模型）

## 三件套
- 代码：`models/01-evaluation/ahp/model.py`（`AHP` 类：fit/rank/sensitivity/evaluate/plot）
- 论文片段：`models/01-evaluation/ahp/model.tex`（权重/一致性/综合得分公式 + 结果表）
- 介绍：`models/01-evaluation/ahp/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/01-evaluation/ahp")
from model import AHP

ahp = AHP(["经济性", "技术性", "环保性"])
ahp.fit(judgment_matrix)          # 1-9 标度判断矩阵（互反矩阵）
print(ahp.weights, ahp.cr)        # 权重向量 + 一致性比率
order = ahp.rank(score_matrix)    # 方案排序（越大越优）
ahp.sensitivity(score_matrix)     # 权重 ±20% 扰动稳健性
```

## 竞赛要点
1. **一致性检验必须写进论文**：`CR=CI/RI<0.1`，RI 查表（n=3→0.58, 4→0.90, 5→1.12）；
2. 准则数 ≤ 7±2，否则一致性难通过；
3. 主客观组合：AHP 权重 × 熵权权重（`mm-models-evaluation` 的 entropy-weight）→ 组合 → TOPSIS；
4. 稳健性：权重扰动 ±20% 排序不变 → 论文"模型评价"节加分；
5. 判断矩阵打分建议取专家几何平均。

## 常见坑
- 判断矩阵不互反（a_ij ≠ 1/a_ji）→ 代码会报错；
- n=1,2 时 CR 无意义（RI=0），代码自动置 0；
- 方案间得分差异小时 AHP 区分度不足，可换熵权或加指标。
