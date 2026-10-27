"""AC-006/007/008：分支误报预算、并集判定、预算细判调度、副作用分报。

图级 FPR 预算 α（未冻结时回落 1-tau_quantile，标注来源）在全局/局部分支间分配
（未冻结时等分，标注 equal_default_unfrozen）。基线判定 = 任一分支报警（并集）。
块级阈值：Šidák 多重比较 1-(1-α_l)^(1/n_tiles) 作为块分位，再用 holdout 正常图的
max-block 经验分位校验（小样本时以经验分位为准并标注）。
"""
import numpy as np


def sidak_block_alpha(alpha_image, n_tiles):
    n = max(1, int(n_tiles))
    return 1.0 - (1.0 - float(alpha_image)) ** (1.0 / n)


def resolve_budget(cfg):
    """→ (alpha, split{global,local}, sources)。"""
    dec = (cfg or {}).get("decision", {}) or {}
    bb = dec.get("branch_budget", {}) or {}
    src = {}
    if bb.get("image_fpr") is not None:
        alpha = float(bb["image_fpr"])
        src["alpha"] = "cfg.decision.branch_budget.image_fpr"
    else:
        alpha = 1.0 - float(dec.get("tau_quantile", 0.99))
        src["alpha"] = "unfrozen_default_from_tau_quantile"
    sp = bb.get("split")
    if sp:
        tot = float(sp.get("global", 0.5)) + float(sp.get("local", 0.5))
        split = {"global": float(sp.get("global", 0.5)) / tot,
                 "local": float(sp.get("local", 0.5)) / tot}
        src["split"] = "cfg.decision.branch_budget.split"
    else:
        split = {"global": 0.5, "local": 0.5}
        src["split"] = "equal_default_unfrozen"
    return alpha, split, src


def quantile_thresh(scores, alpha):
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return float("inf")
    return float(np.quantile(s, 1.0 - float(alpha)))


def fit_union_thresholds(global_normal, local_block_normal, local_image_normal,
                         n_tiles, cfg):
    """global_normal: 正常图 fused 分；local_block_normal: holdout 正常块校准分；
    local_image_normal: holdout 正常图 max-block 分；n_tiles: 每图局部块数。
    返回 dict（tau_global, tau_local, alpha_*, sources, report）。"""
    alpha, split, src = resolve_budget(cfg)
    a_g, a_l = alpha * split["global"], alpha * split["local"]
    tau_g = quantile_thresh(global_normal, a_g)
    a_blk = sidak_block_alpha(a_l, n_tiles)
    tau_blk = quantile_thresh(local_block_normal, a_blk)
    tau_img = quantile_thresh(local_image_normal, a_l)
    n_img = len(local_image_normal)
    # 经验校验：块独立假设下 Šidák 阈值应使 max-block FPR≈α_l；块相关时 Šidák 偏保守。
    # 取二者较大者（更保守，不超预算）；holdout 少于 1/α_l 张时经验分位不可靠，标注。
    tau_l = max(tau_blk, tau_img) if n_img else tau_blk
    src["tau_local"] = "max(sidak_block, empirical_max_block)"
    small = n_img < int(np.ceil(1.0 / max(a_l, 1e-9)))
    gn = np.asarray(global_normal, dtype=np.float64)
    ln = np.asarray(local_image_normal, dtype=np.float64)
    fpr_g = float((gn >= tau_g).mean()) if gn.size else None
    fpr_l = float((ln >= tau_l).mean()) if ln.size else None
    fpr_u = None
    if gn.size and ln.size and gn.size == ln.size:
        fpr_u = float(((gn >= tau_g) | (ln >= tau_l)).mean())
    return {"alpha": alpha, "alpha_global": a_g, "alpha_local": a_l,
            "alpha_block_sidak": a_blk, "n_tiles": int(n_tiles),
            "tau_global": tau_g, "tau_local": tau_l,
            "tau_local_sidak": tau_blk, "tau_local_empirical": tau_img,
            "sources": src,
            "report": {"fpr_global": fpr_g, "fpr_local": fpr_l, "fpr_union": fpr_u,
                       "n_holdout": n_img, "small_sample": bool(small),
                       "note": ("holdout 张数 < 1/α_l，经验 FPR 粒度不足，仅供参考"
                                if small else "")}}


def branch_ratio(s, tau):
    """分数/阈值比（=1 恰在阈值）。τ≤0 时（如负值原始分）用 1+(s-τ)/|τ| 保持单调与边界一致。"""
    s, tau = float(s), float(tau)
    if not np.isfinite(tau):
        return 0.0
    if tau > 0:
        return s / tau
    return 1.0 + (s - tau) / max(abs(tau), 1e-12)


def union_decide(s_global, s_local, tau_g, tau_l, gray_ratio):
    """基线：任一分支报警。返回 (decision, union_score, fired)。
    union_score = max(r_g, r_l)，r=branch_ratio（≥1 报警；≥gray_ratio 灰区）。"""
    rg = branch_ratio(s_global, tau_g)
    rl = branch_ratio(s_local, tau_l) if s_local is not None else 0.0
    u = max(rg, rl)
    fired = {"global": rg >= 1.0, "local": rl >= 1.0}
    if u >= 1.0:
        d = "anomaly"
    elif u >= gray_ratio:
        d = "gray"
    else:
        d = "normal"
    return d, u, fired


def side_effects(global_alarm, local_alarm, veto=False, local_veto=False):
    """AC-008 分报：added_by_local / confirmed / global_only / vetoed（否决新增漏检）。"""
    if global_alarm and local_alarm:
        k = "confirmed"
    elif local_alarm:
        k = "added_by_local"
    elif global_alarm:
        k = "vetoed" if (veto and local_veto) else "global_only"
    else:
        k = "none"
    return k


class BudgetScheduler:
    """AC-007：预算细判调度。

    ROI 内块必看；其余按全局热图在块内的最大值降序取前 max_tiles；剩余块用轮换游标
    每次抽检 rotate_n 块（保证长期全覆盖）；灰区/冲突时由调用方追加全量。
    """

    def __init__(self, max_tiles=4, rotate_n=1, roi=None):
        self.max_tiles = int(max_tiles)
        self.rotate_n = int(rotate_n)
        self.roi = roi
        self._cursor = 0

    @staticmethod
    def _in_roi(t, roi):
        if roi is None:
            return False
        y1, x1, y2, x2 = t
        return not (y2 <= roi[0] or x2 <= roi[1] or y1 >= roi[2] or x1 >= roi[3])

    def select(self, tiles, priority):
        """priority: 每块全局热度（越大越先看）。返回 (idx 列表, reasons dict)。"""
        n = len(tiles)
        reasons = {}
        must = [i for i in range(n) if self._in_roi(tiles[i], self.roi)]
        for i in must:
            reasons[i] = "roi"
        rest = [i for i in np.argsort(-np.asarray(priority, dtype=np.float64)) if i not in reasons]
        for i in rest[:max(0, self.max_tiles - len(must))]:
            reasons[int(i)] = "priority"
        left = [i for i in range(n) if i not in reasons]
        if left and self.rotate_n > 0:
            for k in range(min(self.rotate_n, len(left))):
                i = left[(self._cursor + k) % len(left)]
                reasons[i] = "rotate"
            self._cursor = (self._cursor + self.rotate_n) % max(1, len(left))
        return sorted(reasons), reasons


def tile_priority_from_heatmap(heat, tiles, src_size):
    """全局融合热图（任意分辨率）→ 每块在块内的最大值。"""
    if heat is None:
        return [0.0] * len(tiles)
    H, W = src_size
    h, w = heat.shape[:2]
    sy, sx = h / max(H, 1), w / max(W, 1)
    out = []
    for (y1, x1, y2, x2) in tiles:
        a, b = int(y1 * sy), max(int(y1 * sy) + 1, int(np.ceil(y2 * sy)))
        c, d = int(x1 * sx), max(int(x1 * sx) + 1, int(np.ceil(x2 * sx)))
        patch = heat[a:min(b, h), c:min(d, w)]
        out.append(float(patch.max()) if patch.size else 0.0)
    return out
