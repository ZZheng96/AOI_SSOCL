"""trad 槽位：传统手工特征（demo4 已验证的 200 维体系，§3.3 特征注册表）

needs_dino=False。200 维特征 + kNN 距离，双打分路径（U27，2026-08-16 诊断）：
  整图路径（主）：整图特征 vs 整图正常库 -> 最近 3 邻居距离均值（demo4 t 分支
    原版打分；诊断 diag_trad_score：整图 3NN 均值 >> tile 级任何聚合，
    solder 0.993 vs 旧 tile 路径 0.47 反转的根因即在此）
  tile 路径（fallback）：无整图库时逐 tile min-kNN + top-3 聚合
保留 15 组分组信息供追溯（trad_groups），但打分用全向量（AOI_dev 证明全特征
略优于单组，组间有交互）。
"""
import os
import sys
import numpy as np
import torch
from .base import Slot

# 复用 algo/vendor/traditional.py 传统特征提取器（随 AOI_sys 打包，无外部 demo4 依赖）
from ..vendor.traditional import TraditionalFeatureExtractor as TraditionalExtractor


class TradSlot(Slot):
    name = "trad"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        self.ext = TraditionalExtractor()
        self.bank = None     # (N,200) tile 级 fp32 numpy（fallback 路径）
        self.topk = cfg.get("topk", 5)          # 每 tile 的 top-k patch 距离
        self.norm_ref = None
        self.bank_w = None     # (N,200) 整图库（U27 主路径）
        self.norm_w = None
        # U36（2026-08-16）：稀疏维 z-score 爆炸修复——训练 std≈0 的维度
        # 测试轻微偏离即 z 爆到 1e6（screw dim72 材质反光组），单样本击穿 AUROC。
        # 归一化后 z clip 到 [-zclip, zclip]（保序，10σ 外视为等价极端异常）。
        self.zclip = float(cfg.get("zclip", 10.0))

    def _z(self, feat, norm):
        """z-score 归一化 + clip（U36）。norm=(mean, std)。"""
        z = (feat - norm[0]) / norm[1]
        if self.zclip > 0:
            np.clip(z, -self.zclip, self.zclip, out=z)
        return z

    def _vec(self, tile):
        """U27：提取前固定全局 RNG（demo4 区域自相似用未播种 np.random.choice
        采样 100 块，特征随调用次数抖动 ~0.1，落在 CDF 分位边缘即翻转校准分，
        component 实测 trad AUROC 0.4286/0.6429 不可复现）。提取后恢复状态。"""
        state = np.random.get_state()
        np.random.seed(0)
        try:
            return self.ext.extract(tile).astype(np.float32)   # (200,)
        finally:
            np.random.set_state(state)

    def fit(self, ctx):
        # 整图库（U27 主路径：demo4 t 分支式打分）
        imgs = ctx.get("train_imgs")
        if imgs:
            arr_w = np.stack([self._vec(im) for im in imgs])   # (N,200)
            self.norm_w = arr_w.mean(axis=0), arr_w.std(axis=0) + 1e-6
            self.bank_w = self._z(arr_w, self.norm_w)
            return self                    # 主路径激活，跳过 tile 库（省 3600 tile×30ms）
        # tile 库（fallback：ctx 无整图时）
        vecs = []
        for tiles in ctx["train_tile_imgs"]:
            for t in tiles:
                vecs.append(self._vec(t))
        arr = np.stack(vecs)                       # (N,200)
        self.norm_ref = arr.mean(axis=0), arr.std(axis=0) + 1e-6
        self.bank = self._z(arr, self.norm_ref)
        return self

    def score_image(self, img):
        """整图打分（U27）：kNN 最近 3 邻居 L2 距离均值（U99：跳过最近邻 [1:4] 消除
        train/good 的"自己距离 0"偏差——train 最近邻是自己，test 无自己可跳，口径
        不一致导致 test 分数系统性偏高 → CDF 外推 → fused 右移 → 决策阈值失效 U95）。
        保留 L2 判别力（AUROC 0.864 不变），只统一口径。"""
        if self.bank_w is None:
            return None
        v = self._z(self._vec(img), self.norm_w)
        d = np.linalg.norm(self.bank_w - v, axis=1)
        s = float(np.sort(d)[1:4].mean())          # 跳过最近邻（train 为自己，test 为次近邻）
        return s, np.asarray([s]), [None]

    def score_tiles(self, tile_feats=None, tile_imgs=None):
        """fallback：逐 tile kNN 距离（L2），图像分数=tile top-3。返回 (T,) + None 热力图"""
        scores = []
        for t in tile_imgs:
            v = self._z(self._vec(t), self.norm_ref)
            d = np.linalg.norm(self.bank - v, axis=1).min()
            scores.append(float(d))
        return np.asarray(scores), [None] * len(tile_imgs)
