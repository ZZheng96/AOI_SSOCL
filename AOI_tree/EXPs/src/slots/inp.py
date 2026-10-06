"""inp 槽位：图内内在原型偏离（INP，零样本、域差免疫，§U4/L0 兜底）

核心假设：同一张图内，大多数局部区域是正常的，异常区域会偏离图内
"自找原型"。因此不需要外部参考库——train/test 域差对它不构成攻击，
这正是 sem 在域差下反转而 inp 免疫的原因。
实现（demo4 §5.4 的 DINO patch 级版本）：
  - (G,G) patch 特征归一化
  - 空间 3x3 网格：每网格取 patch 中位数向量为"该位置正常原型"
  - patch 偏离 = 1 - cosine(patch, 网格原型)
  - 网格分数 = 网格内 patch 偏离 top-10% 均值；图像分数 = 网格 top-2 均值
  - 热力图 = 每 patch 偏离度 (G,G)
fit 无参（逐图自适应）。
"""
import numpy as np
import torch
import torch.nn.functional as F
from .base import Slot


class InpSlot(Slot):
    name = "inp"
    needs_dino = True

    def __init__(self, cfg, device="cuda"):
        self.device = device
        self.grid = cfg.get("spatial_grid", 3)     # 3x3 空间网格
        self.topk_ratio = cfg.get("grid_topk_ratio", 0.1)   # 网格内偏离 top-10%
        self.image_topk = cfg.get("image_topk", 2)          # 图像分数=网格 top-2

    def fit(self, ctx):
        return self  # 无参：逐图自适应，无需训练

    def score_tiles(self, tile_feats, tile_imgs=None):
        """tile_feats: (T,384,G,G) -> tile_scores (T,), heatmaps"""
        T, D, G, _ = tile_feats.shape
        flat = tile_feats.flatten(2).permute(0, 2, 1).float().to(self.device)  # (T,G*G,D)
        flat = F.normalize(flat, dim=-1)
        return self._score(flat, G, self.grid)

    def _score(self, flat, G, g):
        """flat: (T,G*G,D) 归一化 -> tile_scores, heatmaps"""
        # 空间网格：每 patch 的 (row, col)，映射到 g 网格
        row = torch.arange(G, device=self.device).repeat_interleave(G) // (G // g)  # (G*G,)
        col = torch.arange(G, device=self.device).repeat(G) // (G // g)
        cell = row * g + col                                                       # (G*G,)
        tile_scores, heatmaps = [], []
        for t in range(flat.shape[0]):
            f = flat[t]                                   # (G*G, D)
            dev = torch.zeros(G * G, device=self.device)
            cell_proto = {}
            for c in range(g * g):
                idx = (cell == c)
                if idx.sum() == 0:
                    continue
                proto = f[idx].median(dim=0).values       # 网格中位数原型
                cell_proto[c] = proto
                dev[idx] = 1 - (f[idx] * proto).sum(-1)   # cosine 偏离
            # 网格分数 = 网格内 top-10% 偏离均值；图像分数 = 网格 top-2 均值
            cell_scores = []
            for c, proto in cell_proto.items():
                idx = (cell == c)
                d = dev[idx]
                k = max(1, int(d.numel() * self.topk_ratio))
                cell_scores.append(float(d.topk(k).values.mean()))
            cell_scores = torch.tensor(cell_scores, device=self.device)
            img_score = float(cell_scores.topk(min(self.image_topk, len(cell_scores))).values.mean())
            tile_scores.append(img_score)
            heatmaps.append(dev.reshape(G, G).cpu().numpy())
        return np.asarray(tile_scores), heatmaps
