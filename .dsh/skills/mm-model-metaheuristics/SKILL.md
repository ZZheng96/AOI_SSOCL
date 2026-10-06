---
name: mm-model-metaheuristics
description: 元启发式算法单模型精讲。当用遗传算法 GA、粒子群 PSO、模拟退火 SA 求解大规模/非线性优化、调度、路径问题时使用本 skill。
whenToUse: 已选元启发式（大规模优化/组合优化场景）且需要直接调用实现时
---

# 元启发式（GA/PSO/SA，单模型）

## 三件套
- 代码：`models/02-optimization/metaheuristics/model.py`
- 论文片段：`models/02-optimization/metaheuristics/model.tex`
- 介绍：`models/02-optimization/metaheuristics/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/02-optimization/metaheuristics")
from model import Metaheuristics

m = Metaheuristics(method="pso", pop_size=50, max_iter=200, seed=42)
best_x, best_f = m.solve(fitness_func, bounds)
print(best_x, best_f)
```

## 竞赛要点
1. **收敛曲线必画**（适应度 vs 迭代），多算法对比（GA vs PSO vs SA）；
2. 报多次运行统计（最好/平均/方差），避免单次偶然；
3. 与精确解/理论下界对比报 gap（如"GA 相对关键路径下界 gap=2.1%"）；
4. 参数敏感性：种群规模/迭代次数对解质量的影响（论文加分）；
5. 编码设计写清楚（二进制/实数/排列编码）——评审关注点。

## 常见坑
- 未固定随机种子 → 结果不可复现；
- 罚函数处理约束不当导致不可行解；
- 收敛过早（早熟）→ 增大变异率/多样性保持；
- 只在测试函数验证，不验证真实数据规模。
