"""组级打分：主干 trad 槽位同口径（U27 播种 / U36 z-clip）

只复用口径，不复用实现的原因：TradSlot 输出"全向量一个分"，
本模块需要"任意维度子集一个分"（独立 AUROC / 留一消融 / 排列重要性），
因此把 kNN 打分参数化为维度子集。AUROC 是秩指标，打分不做通道级标准化
（主干的标准化交给融合层 CDF，本模块不涉及融合）。
"""
import numpy as np

Z_CLIP = 10.0   # U36：稀疏维 z 爆炸截断（与 trad 槽位 zclip 同值）


def extract_features(extractor, paths, load_image):
    """逐图提取特征，U27 口径：每张图提取前固定全局 RNG（extractor 内②区域
    自相似子采样用未播种 np.random.choice），提取后恢复状态。"""
    feats, ok_paths = [], []
    for p in paths:
        img = load_image(p)
        if img is None:                      # 解码失败显式剔除，下游按 ok_paths 对齐
            print(f"[warn] 解码失败已剔除: {p}")
            continue
        state = np.random.get_state()
        np.random.seed(0)
        try:
            feats.append(extractor.extract(img).astype(np.float64))
        finally:
            np.random.set_state(state)
        ok_paths.append(p)
    return np.stack(feats), ok_paths


class GroupKNN:
    """维度子集 kNN 打分器：fit 一次全向量正常库，score 可按任意维度子集打分。"""

    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, ref_vecs: np.ndarray):
        """ref_vecs: [N, D] 训练正常特征（仅 train 域，isolation 红线）"""
        self.mean = ref_vecs.mean(axis=0)
        self.std = ref_vecs.std(axis=0) + 1e-6          # 与 trad 槽位同取 1e-6
        self.bank = np.clip((ref_vecs - self.mean) / self.std, -Z_CLIP, Z_CLIP)
        return self

    def score(self, vecs: np.ndarray, dims: slice | np.ndarray | None = None,
              skip_self: bool = False) -> np.ndarray:
        """vecs: [M, D] -> [M]；仅查询训练库自身时传 skip_self=True。"""
        z = np.clip((vecs - self.mean) / self.std, -Z_CLIP, Z_CLIP)
        if dims is not None:
            z, bank = z[:, dims], self.bank[:, dims]
        else:
            bank = self.bank
        d = np.sqrt(((z[:, None, :] - bank[None, :, :]) ** 2).sum(axis=-1))
        k = min(self.k, len(self.bank) - int(skip_self))
        if k < 1:
            return d.min(axis=1)
        start = int(skip_self)
        return np.sort(d, axis=1)[:, start:start + k].mean(axis=1)

    def score_permuted(self, vecs: np.ndarray, dims: slice,
                       rng: np.random.Generator) -> np.ndarray:
        """排列重要性：仅在测试集内部打乱 dims 维度（切断该组信号），全向量打分"""
        zp = vecs.copy()
        zp[:, dims] = zp[rng.permutation(len(zp))][:, dims]
        return self.score(zp)
