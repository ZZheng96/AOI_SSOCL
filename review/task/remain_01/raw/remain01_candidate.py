"""REMAIN-01 适配器：AOI_feature 的 FeatureSet + GroupKNN 包装成 parity.Candidate 接口。

改编自 TEST-06 适配器（raw/aoi_feature_candidate.py @ test06_trad_parity），sys.path 由
run_remain01.py 按 --tree 注入（stable/dev 两树模块同名，必须单进程单树）。

口径（不篡改任一侧语义，只做包装）：
- mode="legacy_u27"（稳定版）：extract_one 无 rng 参数，U27 包裹（提取前
  np.random.seed(0)、后恢复全局状态），与稳定版 scoring.extract_features 同口径——
  即历史 TEST-06 FAIL 时的候选口径。
- mode="explicit_rng42"（开发版）：extract_one(img, rng=np.random.default_rng(42))，
  与开发版修复后 scoring.extract_features 同口径（显式 Generator，不碰全局 RNG）。
- 特征转 float64（scoring.extract_features 的 .astype(np.float64) 同款）；extract()
  返回 float32 + 分组 trace（key, end, ms）。
- 打分：GroupKNN(k=3)，neighbors/report 显式 skip_self=True（对应生产 sort(d)[1:4]
  跳自身口径）。
"""
import numpy as np

from algo.features import registry as freg          # noqa: E402
from AOI_feature.feature_set import FeatureSet      # noqa: E402
from AOI_feature.presets import trunk_preset        # noqa: E402
from AOI_feature.scoring import GroupKNN, Z_CLIP    # noqa: E402


class AOIFeatureCandidate:
    """AOI_feature 候选实现（trunk_200 预设，声称与生产 trad 逐位一致）。"""

    def __init__(self, mode):
        assert mode in ("legacy_u27", "explicit_rng42"), mode
        self.mode = mode
        self.fs = FeatureSet(trunk_preset())
        self.knn = GroupKNN(k=3)
        # 稳定版 probe 内 selfsim 用全局 np.random.choice，包状态保护（dev 版用
        # default_rng 不碰全局，包一层同样无害）
        st = np.random.get_state()
        try:
            self._meta = self.fs.probe()
        finally:
            np.random.set_state(st)

    # ---------------- 提取 ----------------
    def preprocess_spec(self):
        spec = dict(freg.PREPROCESS)
        spec["max_side"] = int(self.fs.preprocess_cfg["max_side"])
        spec["min_side"] = int(self.fs.preprocess_cfg["min_side"])
        return spec

    def extract_raw(self, img):
        """未做 float32 化的原始 extract_one 输出（dtype/非有限值专项用）。"""
        if self.mode == "legacy_u27":
            state = np.random.get_state()
            np.random.seed(0)
            try:
                return self.fs.extract_one(img)
            finally:
                np.random.set_state(state)
        return self.fs.extract_one(img, rng=np.random.default_rng(42))

    def extract(self, img):
        """→ (200,) float32 原始特征 + 分组 trace（key, end, ms）。"""
        v = np.asarray(self.extract_raw(img), dtype=np.float32)
        trace = [(m["key"], int(m["slice"].stop), 0.0) for m in self._meta]
        return v, trace

    # ---------------- 拟合 ----------------
    def fit(self, train_imgs):
        vecs = np.stack([np.asarray(self.extract_raw(im), dtype=np.float64)
                         for im in train_imgs])
        self.knn.fit(vecs)
        return self

    def bank(self):
        return self.knn.bank, (self.knn.mean, self.knn.std)

    # ---------------- 打分 ----------------
    def _dist(self, img):
        """与 GroupKNN.score 同公式的全维距离向量（float64）。"""
        v = np.asarray(self.extract_raw(img), dtype=np.float64)
        z = np.clip((v - self.knn.mean) / self.knn.std, -Z_CLIP, Z_CLIP)
        return np.sqrt(((z[None, :] - self.knn.bank) ** 2).sum(axis=-1))

    def neighbors(self, img):
        """→ (score, 近邻索引)；显式跳自身 [1:4]（生产 U99 口径）。"""
        d = self._dist(img)
        order = np.argsort(d)
        score = float(self.knn.score(
            np.asarray(self.extract_raw(img), dtype=np.float64)[None, :],
            skip_self=True)[0])
        return score, [int(i) for i in order[1:4]]

    def report(self, img):
        score, nb = self.neighbors(img)
        return {"score": score, "neighbors": nb,
                "enabled_groups": [g["key"] for g in self.fs.enabled_groups()]}
