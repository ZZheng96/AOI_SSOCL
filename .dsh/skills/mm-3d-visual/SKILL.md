---
name: mm-3d-visual
description: 三维可视化与重建。当需要三维散点/表面/体素可视化、点云处理（下采样/法线/ICP 配准）、网格生成（Marching Cubes/Poisson）、几何量计算（体积/表面积/粗糙度）、三维重建（如围岩裂隙）或多视角渲染图时使用本 skill。
whenToUse: 2025 C 题及任何含"三维/点云/网格/重建/体素"的题目；论文三维展示图
---

# 三维可视化与重建

## 依赖与说明

- 已安装：`pyvista`、`trimesh`、`plotly`、`opencv-python`、`scikit-image`（code/requirements-3d.txt）；
- **open3d 无 Python 3.13 wheel**（环境约束）：点云/网格用 pyvista + trimesh 等效实现；
- 实现位置：`code/mmkit/geo3d/`（M5 随 C 题复盘填充具体管线）。

## 三维重建标准管线（C 题围岩裂隙）

1. **图像预处理与分割**：OpenCV 滤波 → U-Net/阈值分割裂隙（`mm-models-ml-dl`）→ 连通域提取；
2. **点云生成**：由分割掩码/钻孔数据生成三维点集（含坐标尺度）；
3. **点云处理**：下采样、法线估计、离群点剔除；多视角/多孔数据 **ICP 配准**；
4. **曲面重建**：Marching Cubes 等值面 / Poisson 重建生成三角网格（trimesh/pyvista）；
5. **几何量计算**：体积、表面积、截面、剖切；裂隙粗糙度 JRC（`Z2` 法）；
6. **渲染出图**：正视 + 俯视两视角、坐标轴标注、300dpi PNG（论文）+ 交互 HTML（附件，plotly）。

## 论文要点

- 三维图至少两个视角；真实 vs 重建并排同视角对比；
- 精度评价：点到面 RMSE、几何量相对误差表；
- 附件放交互版三维模型（plotly HTML / glTF）。

## 资源

- 骨架：`paper/latex-template-types/skeleton-image3d.tex`
- 绘图规范：`mm-plotting`；分割模型：`mm-models-ml-dl`（unet-segmentation）
