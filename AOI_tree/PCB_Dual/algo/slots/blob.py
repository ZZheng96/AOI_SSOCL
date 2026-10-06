"""blob 槽位：DoG 斑点检测（data_local 微缺陷的域不变槽位，§8 机理防线）

v0.2 修正：弃用 skimage blob_log（打分取 sigma² 导致常数饱和，且每 tile 数百毫秒）。
改用 cv2 向量化高斯差分（DoG）：
  - 两个尺度带（σ 1→2、2→4）覆盖细小斑点；
  - 双极性（亮点/暗点都算异物）；
  - 分数 = 响应图 top-10 均值（对强小斑点敏感，对大尺度纹理不敏感）。
无网络、CPU 原生。

U22（2026-08-15 实测）：blob 保持 CPU cv2 全分辨率不动——
  * 降采样（max_side=2048）伤检出：anomaly 前20张 3/20 vs 全分辨率 6/20；
  * GPU 化（conv2d 手工高斯核）伤精度：blob AUROC 0.7075→0.6112、fused→0.3529、
    检出 5/20→3/20，且无提速收益。cv2 DoG 是 gold_finger 微缺陷的关键特征，
    任何实现变更（降采样/fp16/手工核）都改变微小响应排序。320ms CPU 成本是精度代价。
"""
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from .base import Slot


class BlobSlot(Slot):
    name = "blob"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        self.topk = cfg.get("topk", 10)
        self.with_heatmap = cfg.get("with_heatmap", False)
        self.max_side = cfg.get("max_side", 0)   # 默认 0=不降采样（U22 实测伤精度）
        # U56（2026-08-16）：两个 DoG 尺度带相互独立，并行计算提速
        # （大图 3000×4000 单带 ~155ms，并行后 blob 整体 ~160ms），
        # np.maximum 组合顺序无关，数值与串行完全一致。
        self._exec = ThreadPoolExecutor(max_workers=2)

    def fit(self, ctx):
        return self  # 无训练，尺度由 CDF 校准统一

    def _dog_response(self, g, sigma_scale=1.0):
        """g: float32 [0,1] 灰度 -> 响应图（双极性 DoG max）
        sigma_scale：降采样后 σ 按比例缩放，保持斑点尺度一致。"""

        def _band(s1, s2):
            b1 = cv2.GaussianBlur(g, (0, 0), s1 * sigma_scale)
            b2 = cv2.GaussianBlur(g, (0, 0), s2 * sigma_scale)
            return np.abs(b1 - b2)

        f1 = self._exec.submit(_band, 1, 2)
        f2 = self._exec.submit(_band, 2, 4)
        return np.maximum(f1.result(), f2.result())

    def score_tiles(self, tile_feats=None, tile_imgs=None):
        scores, heatmaps = [], []
        for t in tile_imgs:
            # tile 已是 512² 局部块：全分辨率计算，微斑点不被降采样抹掉
            g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 \
                if t.ndim == 3 else t.astype(np.float32) / 255.0
            h, w = g.shape
            scale = 1.0
            if self.max_side > 0 and max(h, w) > self.max_side:
                scale = self.max_side / max(h, w)
                g = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))))
            r = self._dog_response(g, sigma_scale=scale)
            k = min(self.topk, r.size)
            scores.append(float(np.partition(r.ravel(), -k)[-k:].mean()))
            heatmaps.append(r if self.with_heatmap else None)
        return np.asarray(scores), heatmaps
