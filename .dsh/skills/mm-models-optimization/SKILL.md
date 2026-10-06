---
name: mm-models-optimization
description: 优化类模型族。当需要为求最优解/调度/路径/分配/多目标/组合优化问题选择或调用 LP、MILP、NLP、动态规划、图论算法、多目标 NSGA-II、元启发式、DAG 调度等模型时使用本 skill。
whenToUse: 问题含"最小/最大/最优/调度/路径/分配/排程/约束"等关键词时
---

# 优化类模型族（models/02-optimization）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 线性规划 LP | 连续线性目标/约束 | `linear-programming` |
| 整数/混合整数 MILP | 离散决策（含固定成本） | `integer-programming` |
| 非线性规划 NLP | 非线性目标/约束（SLSQP） | `nonlinear-programming` |
| 动态规划 DP | 多阶段决策/最短路/背包 | `dynamic-programming` |
| 最短路 | Dijkstra/Bellman-Ford/Floyd | `shortest-path` |
| 最小生成树 | 网络连通最小成本 | `minimum-spanning-tree` |
| 最大流/最小割 | 网络容量问题 | `max-flow` |
| 指派问题 | 匈牙利算法 | `assignment-problem` |
| 多目标优化 | NSGA-II + 帕累托（HV/IGD） | `multi-objective-optimization` |
| 元启发式 | PSO/GA/SA 全局搜索 | `metaheuristics` |
| 背包/装箱 | 0-1 背包 DP、FFD 装箱 | `knapsack-bin-packing` |
| DAG 调度 | 拓扑排序+关键路径+资源调度 | `dag-scheduling` |

## 选型指南

- 变量连续 → LP/NLP；含整数/0-1 → MILP；
- 规模小（<100 变量）→ 精确求解（MILP/DP）；规模大 → 元启发式（GA/PSO/SA）对比精确解；
- 多目标（成本 vs 时间 vs 资源）→ NSGA-II 帕累托前沿 + 折中解；
- 调度类（任务依赖资源）→ `dag-scheduling`（关键路径下界 + 启发式）+ 甘特图；
- 路径类 → 最短路/MST/最大流 + networkx 交叉验证。

## 使用步骤

1. 读包 `README.md`；2. 跑 `model.py` 演示；3. 改造输入数据；4. 公式段进论文。

## 论文要点

- 给出**形式化**（变量/目标/约束公式）+ 算法伪代码 + 结果表/甘特图；
- 大规模实例必须对比：精确解（小规模验证）vs 启发式（大规模），报 gap；
- 复杂度分析一句话（如"MILP 在 n>200 时不可行，改用 GA，间隙 2.1%"）。

## 资源

- `models/README.md` 索引；`paper/latex-template-types/skeleton-optimization.tex`
- 三维/网络图用 `mm-plotting`；2025 A 题（NPU 调度）主用本族 + DAG 调度
