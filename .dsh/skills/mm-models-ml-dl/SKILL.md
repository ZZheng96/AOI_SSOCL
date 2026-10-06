---
name: mm-models-ml-dl
description: 机器学习/深度学习模型族。当需要分类、集成学习、降维、CNN、U-Net 分割、Transformer、迁移学习、特征选择、不平衡处理等 ML/DL 建模时使用本 skill（torch 完整版 + 无 torch 降级路径）。
whenToUse: 问题含"分类/识别/分割/图像/特征/深度学习/准确率"等关键词，或传统模型效果不足时
---

# 机器学习/深度学习族（models/05-ml-dl）

## 模型速查

| 模型 | 用途 | 目录 |
|---|---|---|
| 分类器全家桶 | LR/SVM/RF/KNN/MLP 对比 | `classification-family` |
| 集成学习 | Bagging/Boosting/Stacking | `ensemble-learning` |
| 降维可视化 | PCA/t-SNE/UMAP/LDA | `dimensionality-reduction` |
| CNN | 图像分类/特征提取 | `cnn` |
| U-Net 分割 | 图像语义分割（C 题裂隙） | `unet-segmentation` |
| Transformer | 时序/特征建模 | `transformer-basics` |
| 迁移学习 | 源域→目标域小样本 | `transfer-learning` |
| 特征选择 | RFE/重要性/过滤法 | `feature-selection` |
| 不平衡学习 | SMOTE 等重采样 | `imbalanced-learning` |

## 使用须知

- **依赖**：torch/xgboost/lightgbm 已装入 `.venv`（见 code/requirements-ml.txt）；所有 model.py 在无 torch 时自动降级（sklearn/skimage 演示），完整版需 torch；
- GPU：本机无 CUDA（torch CPU 版），训练规模控制在 CPU 可承受范围（小图/小网络/少轮次）；
- 分类必报 accuracy/F1/AUC + 混淆矩阵热力图；
- 分割必报 Dice/IoU（U-Net 训练约 10 轮即可演示）。

## 推荐套路

1. 分类/回归 → `classification-family` 先多模型对比，再 `ensemble-learning` 提升；
2. 图像分类 → `cnn`；图像分割 → `unet-segmentation`（配合 OpenCV/skimage 预处理）；
3. 小样本目标域 → `transfer-learning`；
4. 特征过多 → `feature-selection`（RFE 常比全特征好）；
5. 类别不平衡 → `imbalanced-learning`（SMOTE 提升 G-mean）。

## 论文要点

- 写明训练配置（轮数/批量/学习率/优化器/随机种子），保证可复现；
- 消融/对比实验（如"U-Net vs 传统阈值法 Dice 提升 X"）；
- 数据增强与预处理写清楚；图表用 `mm-plotting`。

## 资源

- `models/README.md`；2025 C 题（裂隙分割）主用 U-Net；B 题（速率）用 CNN/Transformer/迁移学习
