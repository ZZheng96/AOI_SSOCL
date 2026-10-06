---
name: mm-models-mechanism
description: 机理建模模型族。当需要为物理/生物/工程机理问题建立常微分方程、偏微分方程、差分方程模型，做无量纲化、稳定性分析、Sobol 灵敏度分析时使用本 skill。
whenToUse: 问题含"机理/动力学/传播/演化/守恒/微分方程/稳定性"等关键词时
---

# 机理建模族（models/06-mechanism）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 常微分方程 ODE | Logistic/SIR/SEIR + 参数估计 | `ode-models` |
| 偏微分方程数值解 | 热传导 FTCS/Crank-Nicolson + 参数反演 | `pde-numerical` |
| 差分方程 | 离散动力系统稳定性 | `difference-equation` |
| 无量纲化与稳定性 | 模型化简、相图、平衡点分析 | `nondimensionalization-stability` |
| 灵敏度分析 | Sobol 全局/局部灵敏度 | `sensitivity-analysis` |

## 推荐套路

1. 有物理/生化机理 → 守恒律/动力学定律建模 → ODE/PDE；
2. 数据驱动 vs 机理 → 参数估计（最小二乘+多起点全局搜索）→ 拟合验证；
3. 必做：平衡点稳定性分析 + 无量纲化（论文"模型简化"加分项）；
4. 关键参数 → Sobol 全局灵敏度（一阶/总效应指标表）。

## 论文要点

- 模型推导完整（假设→方程→边界/初始条件）；
- 参数估计表（估计值/标准误/拟合 R²）；
- 数值解 vs 观测对比图；稳定性结论一句话；
- 适用：2025 B 题（信道机理/SINR 缺口）、E 题（振动机理）等。

## 资源

- `models/README.md`；`paper/latex-template-types/skeleton-mechanism.tex`
