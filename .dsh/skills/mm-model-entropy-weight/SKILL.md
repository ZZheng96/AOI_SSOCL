---
name: mm-model-entropy-weight
description: 熵权法单模型精讲。当需要用信息熵计算客观权重、与 AHP 组合赋权或作为 TOPSIS 权重来源时使用本 skill。
whenToUse: 已选熵权法（客观权重场景）且需要直接调用实现时
---

# 熵权法（单模型）

## 三件套
- 代码：`models/01-evaluation/entropy-weight/model.py`
- 论文片段：`models/01-evaluation/entropy-weight/model.tex`
- 介绍：`models/01-evaluation/entropy-weight/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/01-evaluation/entropy-weight")
from model import EntropyWeight

ew = EntropyWeight(direction=[1, -1])
weights = ew.fit(X)     # 返回权重向量
print(weights, ew.info_entropy)
```

## 竞赛要点
1. 公式：信息熵 `e_j = -1/ln(m) Σ p_ij ln p_ij`（p 为归一化占比），权重 `w_j=(1-e_j)/Σ(1-e_j)`；
2. 指标差异越大 → 熵越小 → 权重越大（客观性卖点）；
3. 组合赋权：`w = α·w_AHP + (1-α)·w_entropy`，α 取 0.5 或离差最大化；
4. 论文给熵值表 + 权重表；与 TOPSIS 组合使用；
5. 指标全相等（p 均匀）时熵=1、权重=0，注意去重。

## 常见坑
- p_ij=0 时 ln(0) 无定义 → 代码已处理（0·ln0=0）；
- 需先同向化 + min-max 归一化（0 值处理）；
- 样本太少时权重波动大。
