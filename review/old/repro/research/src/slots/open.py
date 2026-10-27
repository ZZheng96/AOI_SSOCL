"""E_open 开放通道（§3.3 完备性机制）：PCA 重构误差 = 未解释信号哨兵

设计语义（§3.3）：完备性无法形式化证明，但可运行时发现并报告不完备。
E_open 对 DINO tile 特征做低秩重构建模——重构误差高 = 特征不在正常流形上，
即"存在无法被任何命名特征组解释的信号"。它不用于解释（黑盒），只用于：
1. 不完备报告：open 校准分高且命名槽位判定 normal → 疑似体系外缺陷信号；
2. 兜底捕获：判错难例若命名组平坦但 open 高 → 特征孵化/新增槽位的工作流入口。

无梯度（PCA 纯统计）、CPU 快、与 sem 互补（sem=局部最近邻距离，open=全局流形偏移）。
"""

import numpy as np
import torch


class OpenDetector:
    def __init__(self, n_components=0.95, topk_ratio=0.05):
        self.n_components = n_components   # PCA 保留方差比例（int 则为分量数）
        self.topk_ratio = topk_ratio
        self.pca = None

    def _patches(self, tile_feats):
        """(T,384,G,G) -> (T*G*G,384) fp32 numpy"""
        f = torch.as_tensor(np.asarray(tile_feats, dtype=np.float32))
        if f.ndim == 3:
            f = f.unsqueeze(0)
        T, D, G, _ = f.shape
        return f.flatten(2).permute(0, 2, 1).reshape(-1, D).numpy(), G

    def fit(self, tile_feats_list):
        """train/good tile 特征 -> PCA（正常流形）。tile_feats_list: list[(T,384,G,G)]"""
        from sklearn.decomposition import PCA
        chunks, G = [], 32
        for feats in tile_feats_list:
            x, G = self._patches(feats)
            chunks.append(x)
        X = np.concatenate(chunks)
        self.pca = PCA(n_components=self.n_components).fit(X)
        self.mean_err = float(np.mean(self._recon_err(X)))
        return self

    def _recon_err(self, X):
        """重构误差 per patch：||X - X_hat||² 的 sqrt"""
        Xh = self.pca.inverse_transform(self.pca.transform(X))
        return np.linalg.norm(X - Xh, axis=1)

    def score(self, tile_feats):
        """-> (tile_scores (T,), heatmaps list[(G,G)])"""
        if self.pca is None:
            return np.array([0.0]), [None]
        X, G = self._patches(tile_feats)
        err = self._recon_err(X)                       # (T*G*G,)
        T = X.shape[0] // (G * G)
        maps = err.reshape(T, G, G)
        k = max(1, int(G * G * self.topk_ratio))
        flat = maps.reshape(T, -1)
        idx = np.argsort(-flat, axis=1)[:, :k]
        scores = flat[np.arange(T)[:, None], idx].mean(axis=1)
        return scores.astype(np.float64), [maps[i] for i in range(T)]
