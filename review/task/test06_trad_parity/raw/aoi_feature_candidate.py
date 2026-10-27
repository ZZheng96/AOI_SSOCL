"""TEST-06 适配器：AOI_feature 的 FeatureSet + GroupKNN 包装成 parity.Candidate 接口。

口径说明（不篡改任一侧语义，只做包装）：
- 提取：FeatureSet(trunk_preset()).extract_one，U27 包裹（提取前 np.random.seed(0)，
  提取后恢复全局状态），与 scoring.extract_features 同口径；特征转 float64
  （scoring.extract_features 的 .astype(np.float64) 同款）。
- trace：由 FeatureSet.probe() 的分组 slice 构造 [(key, end, ms)]；probe 内 selfsim
  会消耗全局 RNG（np.random.choice），故 probe 只在初始化时执行一次并包状态保护。
- 打分：GroupKNN(k=3)，neighbors/report 显式 skip_self=True（对应生产 sort(d)[1:4]
  跳自身口径；GroupKNN 默认 skip_self=False，必须显式覆盖）。
- bank：GroupKNN.bank（z-clip 后正常库）+ (mean, std)（std 已含 +1e-6）。
- preprocess_spec：声明候选声称实现的主干规格（registry.PREPROCESS），max_side/
  min_side 取自 FeatureSet 实际配置；数值是否兑现由 harness 逐项验证。
"""
import sys
from pathlib import Path

import numpy as np

_AOI_TREE = Path(__file__).resolve().parents[4] / "AOI_tree"
for _p in (str(_AOI_TREE / "AOI_Core"), str(_AOI_TREE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from algo.features import registry as freg          # noqa: E402
from AOI_feature.feature_set import FeatureSet      # noqa: E402
from AOI_feature.presets import trunk_preset        # noqa: E402
from AOI_feature.scoring import GroupKNN, Z_CLIP    # noqa: E402


class AOIFeatureCandidate:
    """AOI_feature 候选实现（trunk_200 预设，声称与生产 trad 逐位一致）。"""

    def __init__(self):
        self.fs = FeatureSet(trunk_preset())
        self.knn = GroupKNN(k=3)
        # probe 内 selfsim 用全局 np.random.choice，包状态保护且只跑一次
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

    def _extract_raw(self, img):
        """U27 口径：提取前固定全局 RNG，提取后恢复状态。"""
        state = np.random.get_state()
        np.random.seed(0)
        try:
            return self.fs.extract_one(img)
        finally:
            np.random.set_state(state)

    def extract(self, img):
        """→ (200,) float32 原始特征 + 分组 trace（key, end, ms）。"""
        v = self._extract_raw(img).astype(np.float32)
        trace = [(m["key"], int(m["slice"].stop), 0.0) for m in self._meta]
        return v, trace

    # ---------------- 拟合 ----------------
    def fit(self, train_imgs):
        # scoring.extract_features 同款：U27 包裹 + float64
        vecs = np.stack([self._extract_raw(im).astype(np.float64)
                         for im in train_imgs])
        self.knn.fit(vecs)
        return self

    def bank(self):
        return self.knn.bank, (self.knn.mean, self.knn.std)

    # ---------------- 打分 ----------------
    def _dist(self, img):
        """与 GroupKNN.score 同公式的全维距离向量（float64）。"""
        v = self._extract_raw(img).astype(np.float64)
        z = np.clip((v - self.knn.mean) / self.knn.std, -Z_CLIP, Z_CLIP)
        return np.sqrt(((z[None, :] - self.knn.bank) ** 2).sum(axis=-1))

    def neighbors(self, img):
        """→ (score, 近邻索引)；显式跳自身 [1:4]（生产 U99 口径）。"""
        d = self._dist(img)
        order = np.argsort(d)
        score = float(self.knn.score(
            self._extract_raw(img).astype(np.float64)[None, :],
            skip_self=True)[0])
        return score, [int(i) for i in order[1:4]]

    def report(self, img):
        score, nb = self.neighbors(img)
        return {"score": score, "neighbors": nb,
                "enabled_groups": [g["key"] for g in self.fs.enabled_groups()]}
