---
name: mm-models-evaluation
description: 评价类模型族。当需要为打分/排序/比选/综合评价问题选择或调用 AHP、熵权法、TOPSIS、灰色关联、模糊综合评价、PCA、因子分析、DEA、CRITIC 等模型（含三件套：README/model.tex/model.py）时使用本 skill。
whenToUse: 问题含"评价/排序/权重/效率/综合得分"等关键词时
---

# 评价类模型族（models/01-evaluation）

## 模型速查

| 模型 | 用途 | 依赖 | 目录 |
|---|---|---|---|
| AHP | 主观权重 + 一致性检验 + 排序 | 判断矩阵 | `ahp` |
| 熵权法 | 客观权重（信息熵） | 决策矩阵 | `entropy-weight` |
| TOPSIS | 逼近理想解排序 | 决策矩阵+权重 | `topsis` |
| 灰色关联 GRA | 序列关联度评价 | 决策矩阵 | `grey-relational-analysis` |
| 模糊综合评价 | 隶属度综合评判 | 决策矩阵+评语集 | `fuzzy-comprehensive-evaluation` |
| PCA | 降维 + 综合得分 | 数值矩阵 | `pca` |
| 因子分析 | 潜因子提取 | 数值矩阵 | `factor-analysis` |
| DEA | 多投入产出效率 | 投入/产出矩阵 | `dea` |
| CRITIC | 对比强度+冲突性权重 | 决策矩阵 | `critic` |

## 推荐套路

1. 数据齐全 → 熵权或 CRITIC（客观）；
2. 有主观经验 → AHP（配合一致性检验 CR<0.1）；
3. 主客观结合 → AHP×熵权组合赋权 + TOPSIS 排序（最常用组合，论文易出彩）；
4. 多投入产出 → DEA（CCR/BCC + 规模收益）；
5. 评价后必做灵敏度分析（权重 ±20% 扰动排序稳定性）。

## 使用步骤

1. `mm-model-select` 确认选型；
2. 读对应包 `README.md`（适用场景/原理/步骤/论文写法）；
3. 运行 `python models/01-evaluation/<slug>/model.py` 看演示；
4. 把 `model.py` 复制进赛题 code/ 改造数据输入；
5. 把 `model.tex` 公式段复制进论文（`\input` 亦可），图用 `mm-plotting` 生成。

## 论文要点

- 摘要写"XX 法确定权重（CR=0.0x<0.1）+ TOPSIS 排序"；
- 权重表、排序表、雷达图/柱状图齐全；
- 灵敏度分析放"模型评价"节。

## 资源

- 索引与选题速查：`models/README.md`
- 建设规范：`models/MODEL_BUILD_SPEC.md`
- 论文骨架：`paper/latex-template-types/skeleton-evaluation.tex`
