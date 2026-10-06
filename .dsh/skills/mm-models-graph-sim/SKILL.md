---
name: mm-models-graph-sim
description: 图论/网络/仿真模型族。当需要复杂网络分析、网络传播、排队论、离散事件仿真、元胞自动机等图论与仿真建模时使用本 skill。
whenToUse: 问题含"网络/传播/排队/仿真/元胞自动机/复杂系统"等关键词时
---

# 图论/网络/仿真族（models/07-graph-network + 08-simulation）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 复杂网络分析 | 度分布/聚类系数/社区发现 | `07-graph-network/complex-network-analysis` |
| 网络传播 | SIR 在网络上传播 | `07-graph-network/network-propagation` |
| 排队论 | M/M/c 指标 + 模拟校验 | `07-graph-network/queueing-theory` |
| 离散事件仿真 DES | 事件驱动仿真 | `08-simulation/discrete-event-simulation` |
| 元胞自动机 CA | 生命游戏/SIR/交通 | `08-simulation/cellular-automaton` |

## 推荐套路

1. 系统关系网 → 复杂网络指标（度/介数/PageRank/社区）+ 可视化；
2. 传播/扩散 → 网络 SIR 或元胞自动机，标定参数（与 ODE 对照，报 RMSE/R²）；
3. 服务系统 → 排队论（M/M/c 解析解 + DES 模拟交叉验证，误差 <5%）；
4. 复杂随机系统 → DES（多次重复运行，报均值±标准差，Little 定律校验）。

## 论文要点

- 仿真类：写明运行次数、预热期、随机种子、与解析解/理论值对比表；
- 网络类：可视化图（`mm-plotting.network_plot`）+ 指标表；
- 传播类：演化曲线（峰值/终局规模）+ 参数敏感性。

## 资源

- `models/README.md`；2025 D 题（路径/融合）可配合 `mm-models-optimization` 图论工具
