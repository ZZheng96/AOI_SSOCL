"""自动抽色工具（上位机调参控件）。

`extract_hsv_range` 从原 tuner 模块移出：
- 属于上位机调参控件（自动提取 HSV 颜色范围），不是算法模块；
- 供上位机/调参工具（tuner）单独调用，产出 feature_h/s/v_* 六个范围参数；
- 依赖 gamma 工具（algorithms.gamma.gamma）做可选亮度归一化预处理（惰性导入：
  use_autogamma=False 时无此依赖，部署环境缺 src/algorithms/gamma 也不报错）。
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple

_H_STEP = 3          # H 区间搜索步长
_H_MIN_RECALL = 0.6  # H 区间 recall 下限（至少保留 60% base 像素）
_H_J_GATE = 0.7      # H 约束门控：hist-J 达到该值才收窄 H，否则 H=[0,179]
_H_FLAT_TOL = 0.03   # H 区间扩展时允许的 J 下降容差


def _auto_gamma(img: np.ndarray, target: float) -> np.ndarray:
    """惰性导入 gamma：仅在需要时加载，缺失时给出明确错误。"""
    try:
        from algorithms.gamma.gamma import auto_gamma
    except ImportError as exc:
        raise RuntimeError(
            "use_autogamma=True 但 gamma 模块不可用（未部署 src/algorithms/gamma），"
            "请关闭 use_autogamma 或补齐依赖"
        ) from exc
    return auto_gamma(img, target)


def _search_precision_with_recall(
    ch_flat, pos_flat, neg_flat, n_bins,
    min_recall=0.3, fixed_lo=-1, step=3,
):
    """
    最大化 precision，约束 recall >= min_recall。

    precision = TP / (TP + FP)  -- 范围内正样本占比
    recall    = TP / total_pos   -- 正样本捕获率

    在保证 recall 的前提下，precision 越高 = 范围内噪声越少。
    """
    if len(ch_flat) == 0 or pos_flat.sum() == 0:
        lo = fixed_lo if fixed_lo >= 0 else 0
        return lo, n_bins - 1

    hp = np.bincount(ch_flat[pos_flat > 0], minlength=n_bins).astype(np.float64)
    hn = np.bincount(ch_flat[neg_flat > 0], minlength=n_bins).astype(np.float64)
    cp = np.cumsum(hp)
    cn = np.cumsum(hn)
    tp_total = cp[-1]

    def rs(c, lo, hi):
        return c[hi] - (c[lo - 1] if lo > 0 else 0)

    best_p = -1.0
    best_lo = fixed_lo if fixed_lo >= 0 else 0
    best_hi = n_bins - 1

    if fixed_lo >= 0:
        lo = fixed_lo
        for hi in range(lo, n_bins, step):
            tp = rs(cp, lo, hi)
            fp = rs(cn, lo, hi)
            r = tp / tp_total if tp_total > 0 else 0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            if r >= min_recall and p > best_p:
                best_p = p
                best_lo, best_hi = lo, hi
    else:
        for lo in range(0, n_bins, step):
            for hi in range(lo, n_bins, step):
                tp = rs(cp, lo, hi)
                fp = rs(cn, lo, hi)
                r = tp / tp_total if tp_total > 0 else 0
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                if r >= min_recall and p > best_p:
                    best_p = p
                    best_lo, best_hi = lo, hi

    # 如果没有满足 recall 约束的范围，取全范围
    if best_p < 0:
        best_lo = fixed_lo if fixed_lo >= 0 else 0
        best_hi = n_bins - 1

    return best_lo, best_hi


def _search_vs_2d(vf, sf, pf, nf, min_recall=0.3, step=3):
    """
    2D VS 联合搜索：V_lo=0, S_lo=0 固定，联合优化 V_hi + S_hi。

    使用 2D 累积直方图实现 O(1) 每次评估，numpy 向量化加速。
    S_lo 固定为 0（贴片底座 S 分布通常从 0 开始）。

    Returns:
        (v_hi, s_lo, s_hi)  -- s_lo 始终为 0
    """
    pos_mask = pf > 0
    neg_mask = nf > 0

    # 2D 直方图 H[v][s]
    H_pos = np.zeros((256, 256), dtype=np.float64)
    H_neg = np.zeros((256, 256), dtype=np.float64)
    np.add.at(H_pos, (vf[pos_mask], sf[pos_mask]), 1)
    np.add.at(H_neg, (vf[neg_mask], sf[neg_mask]), 1)

    # 2D 累积和：cum[v][s] = count(V<=v, S<=s)
    cum_pos = np.cumsum(np.cumsum(H_pos, axis=0), axis=1)
    cum_neg = np.cumsum(np.cumsum(H_neg, axis=0), axis=1)

    total_pos = cum_pos[-1, -1]
    total_neg = cum_neg[-1, -1]
    if total_pos == 0:
        return 255, 0, 255

    idx = np.arange(0, 256, step)
    n = len(idx)

    # S_lo=0 固定，搜索 (V_hi, S_hi)
    # TP = cum_pos[v_hi, s_hi]  (V in [0,v_hi], S in [0,s_hi])
    # FP = cum_neg[v_hi, s_hi]

    best_score = -1.0
    best_v_hi, best_s_hi = 255, 255

    for v_hi in idx:
        tp_row = cum_pos[v_hi, idx]   # (n,) cum[v_hi, s_hi]
        fp_row = cum_neg[v_hi, idx]

        recall = tp_row / total_pos
        denom = tp_row + fp_row
        precision = np.where(denom > 0, tp_row / np.maximum(denom, 1e-10), 0)

        # F1 平衡 precision 和 recall
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall + 1e-10), 0)

        valid = (recall >= min_recall) & (tp_row > 0)
        score = np.where(valid, 0.5 * precision + 0.5 * f1, -1)

        max_idx = int(np.argmax(score))
        max_val = score[max_idx]
        if max_val > best_score:
            best_score = max_val
            best_v_hi = int(v_hi)
            best_s_hi = int(idx[max_idx])

    if best_score < 0:
        return 255, 0, 255
    return best_v_hi, 0, best_s_hi


def _search_h_band(hf, pf, nf, vs_in, min_recall=_H_MIN_RECALL,
                   step=_H_STEP, flat_tol=_H_FLAT_TOL):
    """在 vs_in 掩码内统计 H 直方图，搜索最大化 Youden's J 的 H 区间。

    J = pos 覆盖率 - neg 覆盖率（均在 VS 区间内归一化）。
    约束 recall >= min_recall（保留足够 base 像素）；找到 J 最大的区间后
    只向高色相端扩展（只要 J 下降不超过 flat_tol），产出"右开口"
    H=[h_lo, 179] 这类参考区间；左边界固定在 J 最大处，避免引入噪声。
    门控由调用方用返回值 j_best 判断。

    Returns:
        (h_lo, h_hi, j_best)  -- j_best 为扩展后区间的 J
    """
    hp = np.bincount(hf[vs_in & (pf > 0)], minlength=180).astype(np.float64)
    hn = np.bincount(hf[vs_in & (nf > 0)], minlength=180).astype(np.float64)
    tp_total = hp.sum()
    if tp_total == 0:
        return 0, 179, 0.0
    tn_total = hn.sum()

    # 前缀和（pad 一个 0 便于闭区间求和）：cp[h+1] = count(H <= h)
    cp = np.concatenate([[0.0], np.cumsum(hp)])
    cn = np.concatenate([[0.0], np.cumsum(hn)])

    def seg(lo, hi):
        return cp[hi + 1] - cp[lo], cn[hi + 1] - cn[lo]

    # 步长网格上找 recall 满足约束时 J 最大的区间
    idx = np.arange(0, 180, step)
    lo_arr = idx[:, None]
    hi_arr = idx[None, :]
    tp_m = cp[hi_arr + 1] - cp[lo_arr]       # (n, n) 每格区间的 TP
    fp_m = cn[hi_arr + 1] - cn[lo_arr]
    valid = np.triu(np.ones((len(idx), len(idx)), dtype=bool))  # lo <= hi
    if tn_total > 0:
        j_m = np.where(valid, tp_m / tp_total - fp_m / tn_total, -1.0)
    else:
        j_m = np.where(valid, tp_m / tp_total, -1.0)
    j_m = np.where(valid & (tp_m >= tp_total * min_recall), j_m, -1.0)
    flat = int(np.argmax(j_m))
    j_max = j_m.ravel()[flat]
    if j_max < 0:
        return 0, 179, 0.0
    lo_i, hi_i = np.unravel_index(flat, j_m.shape)
    lo, hi = int(idx[lo_i]), int(idx[hi_i])

    # 只向高色相端扩展：J 不显著下降时放宽区间（稳健性，避免过拟合窄带）
    def _j_of(tp, fp):
        return tp / tp_total - (fp / tn_total if tn_total else 0.0)

    j_cur = j_max
    while hi < 179:
        tp, fp = seg(lo, hi + 1)
        j2 = _j_of(tp, fp)
        if j2 >= j_cur - flat_tol and tp >= tp_total * min_recall:
            hi += 1
            if j2 > j_cur:
                j_cur = j2
        else:
            break

    tp, fp = seg(lo, hi)
    j_final = _j_of(tp, fp)
    return lo, hi, j_final


def _search_hsv_3d(hf, sf, vf, pf, nf, min_recall=0.3, step=3):
    """
    HSV 三维联合搜索（V/S 二维联合 + H 一维联合 + 门控）。

    - 先做 VS 二维联合搜索（V_lo=0, S_lo=0 固定）得到 (v_hi, s_hi)；
    - 再在 VS 区间内对 H 做一维联合搜索（_search_h_band，最大化 Youden's J）；
    - 门控：H 区间对 base/背景的 J 达到 _H_J_GATE 才收窄 H；否则返回
      H=[0,179] 全范围（大多数贴片底座 H 分布散，不需要约束 H）。

    Returns:
        (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
    """
    v_hi, s_lo, s_hi = _search_vs_2d(vf, sf, pf, nf, min_recall=min_recall, step=step)
    vs_in = (vf <= v_hi) & (sf >= s_lo) & (sf <= s_hi)
    h_lo, h_hi, j_h = _search_h_band(hf, pf, nf, vs_in)
    if j_h >= _H_J_GATE:
        return (h_lo, h_hi, s_lo, s_hi, 0, v_hi)
    return (0, 179, s_lo, s_hi, 0, v_hi)


def extract_hsv_range(img, roi_box, use_autogamma=False, autogamma_target=0.35,
                      use_2d_search=False):
    """
    自动提取 HSV 颜色范围。

    Args:
        img: BGR 图像
        roi_box: (x, y, w, h) 底座框
        use_autogamma: 是否先做 AutoGamma 亮度归一化
        autogamma_target: AutoGamma 目标亮度
        use_2d_search: 是否启用 HSV 三维联合搜索（False=串行V->S, True=联合搜索）

    Returns:
        (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
    """
    if use_autogamma:
        img = _auto_gamma(img, autogamma_target)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    rx, ry, rw, rh = roi_box
    rx2, ry2 = min(w, rx + rw), min(h, ry + rh)
    rx, ry = max(0, rx), max(0, ry)
    roi_v = hsv[ry:ry2, rx:rx2, 2]
    if roi_v.size < 10:
        return (0, 179, 0, 255, 0, 255)
    _, roi_base = cv2.threshold(roi_v, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pos_mask = np.zeros((h, w), dtype=np.uint8)
    pos_mask[ry:ry2, rx:rx2] = roi_base
    neg_mask = np.ones((h, w), dtype=np.uint8)
    neg_mask[ry:ry2, rx:rx2] = 0

    hf = hsv[:, :, 0].ravel()
    sf = hsv[:, :, 1].ravel()
    vf = hsv[:, :, 2].ravel()
    pf = pos_mask.ravel()
    nf = neg_mask.ravel()

    if use_2d_search:
        # HSV 三维联合搜索（V_lo=0, S_lo=0 固定，H 联合搜索 + 门控）
        return _search_hsv_3d(hf, sf, vf, pf, nf)

    # 串行 V -> S 搜索
    v_lo, v_hi = _search_precision_with_recall(
        vf, pf, nf, 256, min_recall=0.3, fixed_lo=0)
    v_in = (vf >= v_lo) & (vf <= v_hi)
    s_lo, s_hi = _search_precision_with_recall(
        sf[v_in], pf[v_in], nf[v_in], 256, min_recall=0.5)

    # H: 在 V/S 区间内联合搜索（门控同上，J 足够强才收窄 H）
    vs_in = v_in & (sf >= s_lo) & (sf <= s_hi)
    h_lo, h_hi, j_h = _search_h_band(hf, pf, nf, vs_in)
    if j_h < _H_J_GATE:
        h_lo, h_hi = 0, 179

    return (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
