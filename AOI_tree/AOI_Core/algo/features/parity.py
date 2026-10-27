"""AC-062：候选特征实现 vs 生产 TradSlot 的八项数值等价 harness（只读，不切换生产指针）。

候选（如 AOI_feature 侧 feature_lib/GroupKNN 适配器）实现 Candidate 接口；harness 以全新
生产 TradSlot 为参照逐项比对 PARITY_CHECKS，输出 FeatureSetRevision._parity 可消费的
fail-closed 证据。任一项不等价 → status=failed、verdict=new_candidate（差异进入新候选
的评价/准入流程，不得冒充生产等价）。
"""
import numpy as np

from ..feature_revision import PARITY_CHECKS, digest, production_hashes
from ..slots.trad import TradSlot
from . import registry as freg

PROTOCOL = {"name": "trad_parity", "version": 1, "checks": list(PARITY_CHECKS),
            "reference": "algo.slots.trad.TradSlot(legacy)", "knn": "sort(d)[1:4].mean()",
            "rng": "explicit default_rng(42); no global RNG mutation"}


class ProductionCandidate:
    """生产实现自身包装成候选（自对照基线；也是适配器写法示例）。"""

    def __init__(self, cfg=None):
        self.slot = TradSlot(cfg or {})

    def preprocess_spec(self):
        return dict(freg.PREPROCESS)

    def extract(self, img):
        """→ (200,) float32 原始特征 + 分组 trace。"""
        v = self.slot._vec(img)
        return v, list(self.slot.ext.last_trace)

    def fit(self, train_imgs):
        self.slot.fit({"train_imgs": list(train_imgs)})
        return self

    def bank(self):
        return self.slot.bank_w, self.slot.norm_w

    def neighbors(self, img):
        """→ (score, 近邻索引)；score 为整图异常分。"""
        v = self.slot._z(self.slot._vec(img), self.slot.norm_w)
        d = self.slot._dist(self.slot.bank_w, v)
        order = np.argsort(d)
        return float(np.sort(d)[1:4].mean()), [int(i) for i in order[1:4]]

    def report(self, img):
        e = self.slot.explain(img)
        return {"score": e["score"], "neighbors": [r["index"] for r in e["neighbors"]],
                "enabled_groups": e["enabled_groups"]}


def _maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def run_parity(candidate, train_imgs, probe_imgs, atol=1e-6, lineage=None):
    """candidate 与生产参照逐项比对。lineage={revision_id, seed, dataset_split, model_snapshot}。
    返回证据 dict（checks 八项布尔 + diffs 明细 + numeric），fail-closed。"""
    ref = ProductionCandidate()
    train_imgs, probe_imgs = list(train_imgs), list(probe_imgs)
    checks, diffs, numeric_diffs = {}, {}, []

    def _set(key, ok, detail):
        checks[key] = bool(ok)
        if not ok:
            diffs[key] = detail

    # preprocess：规格一致 + 越界尺寸（需缩放的图）提取一致
    ps_c, ps_r = candidate.preprocess_spec(), ref.preprocess_spec()
    odd = [im for im in probe_imgs if max(im.shape[:2]) > ps_r["max_side"]
           or min(im.shape[:2]) < ps_r["min_side"]]
    pd = max([_maxdiff(candidate.extract(im)[0], ref.extract(im)[0]) for im in odd] or [0.0])
    _set("preprocess", ps_c == ps_r and pd <= atol and bool(odd),
         {"spec_equal": ps_c == ps_r, "max_abs_diff": pd, "n_resize_probes": len(odd)})

    # extraction + dimension_order_slice
    ext_d, trace_errs, dims = [], [], set()
    for im in probe_imgs:
        vc, tc = candidate.extract(im)
        vr, _ = ref.extract(im)
        ext_d.append(_maxdiff(vc, vr))
        dims.add(int(np.asarray(vc).shape[0]))
        trace_errs.extend(freg.check_trace(tc))
    numeric_diffs += ext_d
    _set("extraction", max(ext_d) <= atol, {"max_abs_diff": max(ext_d)})
    _set("dimension_order_slice", not trace_errs and dims == {freg.TRAD_DIM},
         {"trace_errors": trace_errs[:5], "dims": sorted(dims)})

    # rng：重复调用确定 + 不污染全局 RNG + 与生产同值
    st = np.random.get_state()
    a = candidate.extract(probe_imgs[0])[0]
    np.random.seed(12345)
    b = candidate.extract(probe_imgs[0])[0]
    np.random.seed(12345)
    marker = np.random.random()
    np.random.seed(12345)
    candidate.extract(probe_imgs[0])
    leak = np.random.random() != marker
    np.random.set_state(st)
    rd = _maxdiff(a, b)
    _set("rng", rd <= atol and not leak, {"repeat_max_abs_diff": rd, "global_rng_mutated": leak})

    # 拟合后：bank / kNN 自匹配 / scoring / report
    candidate.fit(train_imgs)
    ref.fit(train_imgs)
    (bc, nc), (br, nr) = candidate.bank(), ref.bank()
    bd = max(_maxdiff(bc, br), _maxdiff(nc[0], nr[0]), _maxdiff(nc[1], nr[1]))
    numeric_diffs.append(bd)
    _set("bank_compatibility", bd <= atol, {"max_abs_diff": bd})

    nb_bad, sc_d = [], []
    for i, im in enumerate(train_imgs[:3] + probe_imgs):
        sc, ic = candidate.neighbors(im)
        sr, ir = ref.neighbors(im)
        sc_d.append(abs(sc - sr))
        if ic != ir:
            nb_bad.append({"probe": i, "candidate": ic, "production": ir})
    numeric_diffs += sc_d
    _set("knn_neighbors_self_match", not nb_bad,
         {"mismatch": nb_bad[:5], "production_rule": "skip nearest (self) → [1:4]"})
    _set("scoring", max(sc_d) <= atol, {"max_abs_diff": max(sc_d)})

    rp_bad = []
    for i, im in enumerate(probe_imgs):
        rc, rr = candidate.report(im), ref.report(im)
        if (rc["neighbors"] != rr["neighbors"] or rc["enabled_groups"] != rr["enabled_groups"]
                or abs(rc["score"] - rr["score"]) > atol):
            rp_bad.append(i)
    _set("report", not rp_bad, {"mismatch_probes": rp_bad})

    ok = all(checks.get(k) is True for k in PARITY_CHECKS)
    max_d = max(numeric_diffs) if numeric_diffs else float("inf")
    return {**(lineage or {}),
            "status": "passed" if ok else "failed",
            "verdict": "equivalent" if ok else "new_candidate",
            "checks": {k: checks.get(k, False) for k in PARITY_CHECKS},
            "diffs": diffs,
            "failed_checks": [k for k in PARITY_CHECKS if checks.get(k) is not True],
            "baseline_implementation_hashes": production_hashes(),
            "protocol_hash": digest(PROTOCOL),
            "numeric": {"n_samples": len(probe_imgs) + len(train_imgs),
                        "max_abs_diff": max_d if np.isfinite(max_d) else -1.0,   # 形状不符→无效值，门禁拒绝
                        "atol": float(atol)}}
