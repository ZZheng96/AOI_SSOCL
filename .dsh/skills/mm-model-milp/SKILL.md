---
name: mm-model-milp
description: 整数/混合整数规划 MILP 单模型精讲。当建模含整数/0-1 变量（选址、排产、调度、分配、启动成本）并用分支定界求解时使用本 skill。
whenToUse: 已选 MILP（离散优化场景）且需要直接调用实现时
---

# 整数/混合整数规划 MILP（单模型）

## 三件套
- 代码：`models/02-optimization/integer-programming/model.py`
- 论文片段：`models/02-optimization/integer-programming/model.tex`
- 介绍：`models/02-optimization/integer-programming/README.md`

## 快速调用
```python
import sys
sys.path.insert(0, "code")
sys.path.insert(0, "models/02-optimization/integer-programming")
from model import IntegerProgramming

m = IntegerProgramming()
res = m.solve(c, A_ub, b_ub, integer_vars=[0, 1, 2], bounds=...)
print(res.x, res.fun)   # 最优解与目标值
```

## 竞赛要点
1. 0-1 变量处理"是否/选址/启动"；大 M 法线性化 `if-then` 约束；
2. **小规模用精确解，大规模用启发式**：先报 MILP 最优（验证），再报元启发式 gap；
3. 论文给：变量/约束/目标公式 + 分支定界伪代码 + 松弛间隙（LP relaxation gap）；
4. 求解器：scipy.optimize.milp（内置）/pulp/CBC；大规模可提 ortools；
5. 2025 A 题（NPU 调度）可先建 MILP 小规模基准，再上启发式。

## 常见坑
- 整数变量未声明 → 解出小数被当作"可行"；
- 大 M 取值过大会数值不稳定；
- 规模 > 几百变量时 MILP 求解时间失控 → 提前设 time limit 并报告间隙。
