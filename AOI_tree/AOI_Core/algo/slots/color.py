"""color 槽位：色彩变化检测（赛题五类缺陷之一，§0.5 color 组，U93/U94 补齐）

HSV 直方图距离（无网络、CPU 原生、可解释）：
  fit：train/good 图的 HSV（H×S 2D）直方图求均值 → 正常色彩分布
  score：测试图 HSV 直方图与正常分布的卡方距离（chi2），色彩偏移越大分数越高
  校准：CDF 在 train/good 上统一（与其他槽位同机制）

U94（2026-08-24）：U93 覆盖矩阵暴露"色彩变化归因 0%"——根因是 引擎槽位体系
无色彩通道（sem/disc/shead/blob/trad/layout/inp/tpl 全是灰度/结构特征），carpet/
leather/metal_nut/pill/wood 的 color 缺陷被 blob/sem 触发后误归"外观缺陷"。本槽位
补齐色彩维度，SLOT_TYPE_MAP 增 "color": ["色彩变化"]，使类型归因能区分色彩类。

只对色相+饱和度敏感（V 明度默认不参与，光照/明度偏移不算色彩变化，域差免疫更强）。
"""
import numpy as np
import cv2
from .base import Slot


class ColorSlot(Slot):
    name = "color"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        self.h_bins = cfg.get("h_bins", 18)          # 色相 bin
        self.s_bins = cfg.get("s_bins", 16)          # 饱和度 bin
        self.max_side = cfg.get("max_side", 1024)    # 工作分辨率（直方图统计，降采样无损）

    def _hist(self, img):
        """RGB -> HSV(H×S 2D) 归一化直方图（V 明度不参与，光照鲁棒）。"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32) / 180.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        hist, _, _ = np.histogram2d(h.ravel(), s.ravel(),
                                    bins=[self.h_bins, self.s_bins],
                                    range=[[0.0, 1.0], [0.0, 1.0]])
        hist = hist / (hist.sum() + 1e-6)
        return hist.flatten()

    def fit(self, ctx):
        hists = []
        for tiles in ctx["train_tile_imgs"]:
            img = tiles[0]                       # single 模式：tile 即整图
            if self.max_side and max(img.shape[:2]) > self.max_side:
                s = self.max_side / max(img.shape[:2])
                img = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                                       max(1, int(img.shape[0] * s))))
            hists.append(self._hist(img))
        self.hist_mean = np.mean(hists, axis=0)
        self.hist_std = np.std(hists, axis=0) + 1e-6
        return self

    def score_tiles(self, tile_feats=None, tile_imgs=None):
        scores = []
        for t in tile_imgs:
            if self.max_side and max(t.shape[:2]) > self.max_side:
                s = self.max_side / max(t.shape[:2])
                t = cv2.resize(t, (max(1, int(t.shape[1] * s)), max(1, int(t.shape[0] * s))))
            h = self._hist(t)
            # 卡方距离（直方图分布偏移标准度量）
            d = float(np.sum((h - self.hist_mean) ** 2 / (self.hist_mean + 1e-6)))
            scores.append(d)
        return np.asarray(scores), [None]
