"""trad 槽位：传统手工特征（demo4 已验证的 200 维体系，§3.3 特征注册表）

needs_dino=False。200 维特征 + kNN 距离，双打分路径（U27，2026-08-16 诊断）：
  整图路径（主）：整图特征 vs 整图正常库 -> 最近 3 邻居距离均值（demo4 t 分支
    原版打分；诊断 diag_trad_score：整图 3NN 均值 >> tile 级任何聚合，
    solder 0.993 vs 旧 tile 路径 0.47 反转的根因即在此）
  tile 路径（fallback）：无整图库时逐 tile min-kNN + top-3 聚合

组级配置（AC-010/011/013，组元数据见 algo/features/registry.py）：
  groups: "all"（默认，等价旧版全向量）| [组 key 列表] | "auto"（仅用 fit 期
    train/good + init_defect 开发集前向选组，禁止触碰 test）
  scoring: "legacy"（默认，启用维度上的 z-score L2）| "grouped"（每组距离除以
    该组正常留一近邻距离中位数，剔除近零方差维；反向/常数组权重置零）
"""
import numpy as np
from .base import Slot
from ..features import registry as freg

# 复用 algo/vendor/traditional.py 传统特征提取器（随 AOI_sys 打包，无外部 demo4 依赖）
from ..vendor.traditional import TraditionalFeatureExtractor as TraditionalExtractor


def _auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return None
    allv = np.concatenate([neg, pos])
    if np.ptp(allv) <= 0:
        return 0.5
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


class TradSlot(Slot):
    name = "trad"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        cfg = dict(cfg or {})
        self.cfg = cfg
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
        self.group_spec = cfg.get("groups", "all")
        self.scoring = cfg.get("scoring", "legacy")
        if self.scoring not in ("legacy", "grouped"):
            raise ValueError(f"trad.scoring 仅支持 legacy/grouped: {self.scoring}")
        self.auto_min_auroc = float(cfg.get("auto_min_auroc", 0.55))
        self.auto_min_gain = float(cfg.get("auto_min_gain", 0.005))
        self.auto_min_defects = int(cfg.get("auto_min_defects", 3))
        self.groups = (list(freg.GROUP_KEYS) if self.group_spec == "auto"
                       else freg.resolve_groups(self.group_spec))
        self.group_scale = {}      # grouped：组 -> 正常留一近邻距离中位数
        self.group_weight = {}     # grouped：组 -> 权重（反向/常数=0）
        self.group_valid_dims = {}  # 组 -> 剔除近零方差后的维度索引
        self.group_normal = {}     # 组 -> 正常留一距离 p5/p50/p95（解释用）
        self.raw_p5 = self.raw_p95 = None   # 逐维正常范围（解释用）
        self.ref_paths = []
        self.selection_report = None

    # ---------------- 基础 ----------------
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

    def _idx(self, dim):
        return freg.dim_index(self.groups, dim)

    def _group_dims(self, dim, keys=None):
        out = {}
        for k in (keys or self.groups):
            ids = self.group_valid_dims.get(k)
            if ids is None:
                ids = freg.dim_index([k], dim)
            ids = [i for i in ids if i < dim]
            if ids:
                out[k] = np.asarray(ids)
        return out

    def _group_dist(self, bank, v, keys=None):
        """-> {组: (N,) 组距离/组尺度}"""
        out = {}
        for k, ids in self._group_dims(bank.shape[1], keys).items():
            d = np.linalg.norm(bank[:, ids] - v[ids], axis=1)
            out[k] = d / self.group_scale.get(k, 1.0)
        return out

    def _dist(self, bank, v):
        """legacy：启用维 L2；grouped：组距离加权和。"""
        if self.scoring == "grouped" and self.group_weight:
            gd = self._group_dist(bank, v, [k for k, w in self.group_weight.items() if w > 0])
            if gd:
                return sum(self.group_weight[k] * d for k, d in gd.items())
        idx = self._idx(bank.shape[1])
        if len(idx) == bank.shape[1]:
            return np.linalg.norm(bank - v, axis=1)
        return np.linalg.norm(bank[:, idx] - v[idx], axis=1)

    # ---------------- 拟合 ----------------
    def fit(self, ctx):
        # 整图库（U27 主路径：demo4 t 分支式打分）
        imgs = ctx.get("train_imgs")
        if imgs:
            arr_w = np.stack([self._vec(im) for im in imgs])   # (N,200)
            self.norm_w = arr_w.mean(axis=0), arr_w.std(axis=0) + 1e-6
            self.bank_w = self._z(arr_w, self.norm_w)
            self.raw_p5 = np.percentile(arr_w, 5, axis=0).astype(np.float32)
            self.raw_p95 = np.percentile(arr_w, 95, axis=0).astype(np.float32)
            self.ref_paths = list(ctx.get("train_paths") or [])[:len(imgs)]
            self._fit_groups(arr_w, ctx.get("defect_imgs") or [])
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

    def _loo_group(self, k_ids):
        """正常库留一最近邻组距离（N,）——与 score_image 的跳自身口径一致。"""
        b = self.bank_w[:, k_ids].astype(np.float64)
        sq = (b ** 2).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * b @ b.T, 0.0)
        np.fill_diagonal(d2, np.inf)
        return np.sqrt(d2.min(axis=1)) if len(b) > 1 else np.zeros(len(b))

    def _fit_groups(self, arr_w, defect_imgs):
        dim = arr_w.shape[1]
        raw_std = arr_w.std(axis=0)
        raw_mean = np.abs(arr_w.mean(axis=0))
        near_const = raw_std <= 1e-6 * (1.0 + raw_mean)
        self.group_valid_dims, self.group_scale, self.group_normal = {}, {}, {}
        dropped = {}
        for k in freg.GROUP_KEYS:
            ids = [i for i in freg.dim_index([k], dim) if not near_const[i]]
            dropped[k] = [i for i in freg.dim_index([k], dim) if near_const[i]]
            if not ids:
                continue
            self.group_valid_dims[k] = ids
            loo = self._loo_group(ids)
            med = float(np.median(loo)) if len(loo) else 0.0
            self.group_scale[k] = med if med > 1e-9 else 1.0
            loo = loo / self.group_scale[k]
            self.group_normal[k] = {"p5": float(np.percentile(loo, 5)),
                                    "p50": float(np.percentile(loo, 50)),
                                    "p95": float(np.percentile(loo, 95))}
        report = {"spec": self.group_spec, "scoring": self.scoring,
                  "near_const_dims": {k: v for k, v in dropped.items() if v},
                  "constant_groups": [k for k in freg.GROUP_KEYS if k not in self.group_valid_dims],
                  "data": "fit:init_normal+init_defect（禁止 test）"}
        if self.scoring == "legacy" and self.group_spec != "auto":
            # legacy 必须与旧版数值逐位一致：不剔除常数组（其 z-clip 行为保持原样）
            candidates = list(self.groups)
        else:
            candidates = [k for k in self.groups if k in self.group_valid_dims]
        if self.group_spec == "auto":
            candidates = self._auto_select(candidates, defect_imgs, report)
        if not candidates:     # 全部常数：退回全组 legacy，避免空模型
            candidates = list(freg.GROUP_KEYS)
            report["fallback"] = "no_valid_group"
        self.groups = candidates
        self.group_weight = {k: (1.0 if k in candidates and k in self.group_valid_dims else 0.0)
                             for k in freg.GROUP_KEYS}
        report["enabled"] = list(self.groups)
        report["weights"] = dict(self.group_weight)
        self.selection_report = report

    def _normal_group_scores(self, ids):
        b = self.bank_w[:, ids].astype(np.float64)
        sq = (b ** 2).sum(1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * b @ b.T, 0.0))
        d.sort(axis=1)
        return d[:, 1:4].mean(axis=1)

    def _auto_select(self, candidates, defect_imgs, report):
        """开发集前向选组：单组 AUROC 方向/门槛预筛 + 组合增益前向（不只凭单组 AUC）。"""
        if len(defect_imgs) < self.auto_min_defects or len(self.bank_w) < 4:
            report["auto"] = {"status": "insufficient_dev", "n_defect": len(defect_imgs),
                              "reason": f"开发缺陷 < {self.auto_min_defects} 或正常 < 4，保留全部有效组"}
            return candidates
        dz = np.stack([self._z(self._vec(im), self.norm_w) for im in defect_imgs])
        neg, pos, single = {}, {}, {}
        for k in candidates:
            ids = self.group_valid_dims[k]
            sc = self.group_scale[k]
            neg[k] = self._normal_group_scores(ids) / sc
            pos[k] = np.array([np.sort(np.linalg.norm(self.bank_w[:, ids] - v[ids], axis=1))[1:4].mean()
                               for v in dz]) / sc
            single[k] = _auroc(neg[k], pos[k])
        pre = [k for k in candidates if single[k] is not None and single[k] >= self.auto_min_auroc]
        reversed_ = [k for k in candidates if single[k] is not None and single[k] < 0.5]
        order = sorted(pre, key=lambda k: -single[k])
        chosen, best, trail = [], 0.5, []
        for k in order:
            trial = chosen + [k]
            a = _auroc(sum(neg[t] for t in trial), sum(pos[t] for t in trial))
            trail.append({"add": k, "auroc": a})
            if not chosen or a >= best + self.auto_min_gain:
                chosen, best = trial, a
        report["auto"] = {"status": "selected" if chosen else "no_gain",
                          "n_normal": int(len(self.bank_w)), "n_defect": int(len(dz)),
                          "single_auroc": single, "reversed": reversed_,
                          "forward": trail, "dev_auroc": best if chosen else None}
        return chosen or candidates

    # ---------------- 打分 ----------------
    def score_image(self, img):
        """整图打分（U27）：kNN 最近 3 邻居 L2 距离均值（U99：跳过最近邻 [1:4] 消除
        train/good 的"自己距离 0"偏差——train 最近邻是自己，test 无自己可跳，口径
        不一致导致 test 分数系统性偏高 → CDF 外推 → fused 右移 → 决策阈值失效 U95）。
        保留 L2 判别力（AUROC 0.864 不变），只统一口径。"""
        if self.bank_w is None:
            return None
        v = self._z(self._vec(img), self.norm_w)
        d = self._dist(self.bank_w, v)
        s = float(np.sort(d)[1:4].mean())          # 跳过最近邻（train 为自己，test 为次近邻）
        return s, np.asarray([s]), [None]

    def score_tiles(self, tile_feats=None, tile_imgs=None):
        """fallback：逐 tile kNN 距离（L2），图像分数=tile top-3。返回 (T,) + None 热力图"""
        scores = []
        for t in tile_imgs:
            v = self._z(self._vec(t), self.norm_ref)
            d = self._dist(self.bank, v).min()
            scores.append(float(d))
        return np.asarray(scores), [None] * len(tile_imgs)

    # ---------------- 解释（AC-014）----------------
    def explain(self, img, top_dims=5):
        """测量值/正常范围/参考近邻/组距离分解。组距离是"偏离正常的程度"，
        不是性能增益，也不是融合贡献；分数不是缺陷概率。"""
        if self.bank_w is None:
            return {"status": "unavailable", "reason": "trad 整图库未建立"}
        raw = self._vec(img)
        v = self._z(raw, self.norm_w)
        d = self._dist(self.bank_w, v)
        order = np.argsort(d)
        nn = order[1:4] if len(order) > 3 else order
        gd = self._group_dist(self.bank_w, v, list(self.group_valid_dims))
        groups = []
        for k, dist in gd.items():
            g = freg.group(k)
            val = float(dist[nn].mean())
            rng = self.group_normal.get(k, {})
            groups.append({"key": k, "name": g["name"], "category": g["category"],
                           "enabled": k in self.groups, "weight": self.group_weight.get(k, 1.0),
                           "distance": val, "normal_p5": rng.get("p5"), "normal_p95": rng.get("p95"),
                           "out_of_range": bool(rng) and val > rng.get("p95", np.inf)})
        groups.sort(key=lambda x: -x["distance"])
        dims = []
        if self.raw_p5 is not None:
            n = min(len(raw), len(self.raw_p5))
            zabs = np.abs(v[:n])
            for i in np.argsort(-zabs)[:top_dims]:
                key = next((g["key"] for g in freg.TRAD_GROUPS
                            if g["slice"][0] <= i < g["slice"][1]), None)
                dims.append({"dim": int(i), "group": key, "value": float(raw[i]),
                             "normal_p5": float(self.raw_p5[i]), "normal_p95": float(self.raw_p95[i]),
                             "z": float(v[i])})
        refs = [{"index": int(i), "distance": float(d[i]),
                 "path": self.ref_paths[i] if i < len(self.ref_paths) else None} for i in nn]
        return {"status": "available", "score": float(d[nn].mean()), "scoring": self.scoring,
                "score_semantics": "异常程度（与正常库的距离），非缺陷概率",
                "enabled_groups": list(self.groups), "group_breakdown": groups,
                "top_dims": dims, "neighbors": refs}

    def signature(self):
        return {"registry_sha256": freg.registry_hash(), "groups": list(self.groups),
                "group_spec": self.group_spec, "scoring": self.scoring,
                "dimension": int(self.bank_w.shape[1]) if self.bank_w is not None else None}
