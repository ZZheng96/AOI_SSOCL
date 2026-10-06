"""不出脚(16)：标准图自适应 profile + 中心结构判定。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from .excess import (
    WrapMetrics,
    center_ellipse,
    is_excess_bulge,
    pin_core_area,
)
from .features import PadFeatures
from .mask_stats import SolderRelStats, compute_solder_rel_stats
from .segment import Masks, PadROI


@dataclass
class StandardProfile:
    solder_hsv_low: tuple[int, int, int]
    solder_hsv_high: tuple[int, int, int]
    center_gray: np.ndarray
    center_mask: np.ndarray
    center_v_peak: float
    center_grad_mean: float
    center_gray_std: float
    has_pin_structure: bool
    cyan_ratio: float
    pin_top_core_area: int = 0
    ring_v_mean: float = 0.0
    ring_contrast: float = 0.0
    solder_stats: SolderRelStats | None = None


def center_cyan_masks(hsv: np.ndarray, roi: PadROI):
    """中心椭圆区域内的亮青 mask (ROI 局部坐标)。返回 (占比, 中心椭圆, 亮青mask)。"""
    x, y, w, h = roi.rect
    sub = hsv[y:y + h, x:x + w]
    H = sub[:, :, 0]
    S = sub[:, :, 1]
    V = sub[:, :, 2]
    frac = C.TH["nolead_center_frac"]
    center = np.zeros((h, w), np.uint8)
    cv2.ellipse(center, (w // 2, h // 2), (int(w * frac), int(h * frac)),
                0, 0, 360, 255, -1)
    in_center = center > 0
    cyan = ((H >= C.TH["nolead_cyan_h_low"]) & (H <= C.TH["nolead_cyan_h_high"]) &
            (S >= C.TH["nolead_cyan_s_min"]) & (V >= C.TH["nolead_cyan_v_min"]))
    cyan_u8 = (cyan & in_center).astype(np.uint8) * 255
    ratio = float(cyan[in_center].mean()) if in_center.any() else 0.0
    return ratio, center, cyan_u8


def _center_cyan_ratio(hsv: np.ndarray, roi: PadROI) -> float:
    return center_cyan_masks(hsv, roi)[0]


def _annulus_mask(w: int, h: int,
                  inner_frac: float | None = None,
                  outer_frac: float | None = None) -> np.ndarray:
    inner_frac = inner_frac if inner_frac is not None else C.PROFILE.get("ring_inner_frac", 0.28)
    outer_frac = outer_frac if outer_frac is not None else C.PROFILE.get("ring_outer_frac", 0.72)
    cx, cy = w // 2, h // 2
    rx, ry = int(w * outer_frac), int(h * outer_frac)
    inner_rx, inner_ry = int(w * inner_frac), int(h * inner_frac)
    outer = np.zeros((h, w), np.uint8)
    inner = np.zeros((h, w), np.uint8)
    cv2.ellipse(outer, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    cv2.ellipse(inner, (cx, cy), (inner_rx, inner_ry), 0, 0, 360, 255, -1)
    return cv2.subtract(outer, inner)


def _adaptive_solder_hsv(hsv: np.ndarray, solder: np.ndarray, roi: PadROI | None = None):
    g_lo = np.array(C.SOLDER_HSV_LOW, np.int32)
    g_hi = np.array(C.SOLDER_HSV_HIGH, np.int32)

    sel = None
    if roi is not None:
        x, y, w, h = roi.rect
        ring = _annulus_mask(w, h) > 0
        sel = ring
        px = hsv[y:y + h, x:x + w][sel]
    if sel is None or px.shape[0] < 80:
        sel = solder > 0
        if sel.sum() < 80:
            return tuple(g_lo), tuple(g_hi)
        px = hsv[sel]

    margin_h = C.PROFILE.get("hsv_margin_h", 10)
    margin_sv = C.PROFILE.get("hsv_margin_sv", 35)
    p_h = np.percentile(px[:, 0], [3, 97])
    p_s = np.percentile(px[:, 1], [5, 95])
    p_v = np.percentile(px[:, 2], [5, 95])
    lo = np.array([
        max(g_lo[0], int(p_h[0]) - margin_h),
        max(g_lo[1], int(p_s[0]) - margin_sv),
        max(g_lo[2], int(p_v[0]) - margin_sv),
    ], np.int32)
    hi = np.array([
        min(g_hi[0], int(p_h[1]) + margin_h),
        min(g_hi[1], int(p_s[1]) + margin_sv),
        min(g_hi[2], int(p_v[1]) + margin_sv),
    ], np.int32)
    lo = np.minimum(lo, hi)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def _pin_transition_ring_metrics(bgr: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """引脚高亮与外围之间的过渡环: (环带 V 均值, 中心峰值-环带 V 对比度)。"""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    inner = float(C.TH["nolead_transition_ring_inner"])
    outer = float(C.TH["nolead_transition_ring_outer"])
    cx, cy = w // 2, h // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
    ring = (rr >= inner) & (rr <= outer)
    center = rr <= inner
    if not ring.any() or not center.any():
        return 0.0, 0.0
    ring_v = float(v[ring].mean())
    contrast = float(v[center].max() - ring_v)
    return ring_v, contrast


def build_profile(std_bgr: np.ndarray, roi: PadROI, masks: Masks) -> StandardProfile:
    x, y, w, h = roi.rect
    sub = std_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    hsv_local = masks.hsv[y:y + h, x:x + w]
    center = center_ellipse(w, h)
    in_c = center > 0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    v = hsv_local[:, :, 2]
    if in_c.any():
        center_v_peak = float(v[in_c].max())
        center_grad_mean = float(grad[in_c].mean())
        center_gray_std = float(max(gray[in_c].std(), 8.0))
    else:
        center_v_peak = float(v.max())
        center_grad_mean = float(grad.mean())
        center_gray_std = float(max(gray.std(), 8.0))

    cyan_ratio = _center_cyan_ratio(masks.hsv, roi)

    has_pin = (center_v_peak >= C.TH["nolead_std_v_peak_min"] or
               center_grad_mean >= C.TH["nolead_std_grad_min"])
    ring_v, ring_contrast = _pin_transition_ring_metrics(sub, w, h)

    h_lo, h_hi = _adaptive_solder_hsv(masks.hsv, masks.solder, roi)
    pin_top_core = pin_core_area(std_bgr, roi)
    solder_roi = masks.solder[y:y + h, x:x + w]
    hsv_roi = masks.hsv[y:y + h, x:x + w]
    solder_stats = compute_solder_rel_stats(hsv_roi, solder_roi)
    return StandardProfile(
        solder_hsv_low=h_lo,
        solder_hsv_high=h_hi,
        center_gray=gray,
        center_mask=center,
        center_v_peak=center_v_peak,
        center_grad_mean=center_grad_mean,
        center_gray_std=center_gray_std,
        has_pin_structure=has_pin,
        cyan_ratio=cyan_ratio,
        pin_top_core_area=pin_top_core,
        ring_v_mean=ring_v,
        ring_contrast=ring_contrast,
        solder_stats=solder_stats,
    )


def center_structure_metrics(bgr: np.ndarray, roi: PadROI, profile: StandardProfile):
    x, y, w, h = roi.rect
    sub = bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv_local = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    center = profile.center_mask
    in_c = center > 0
    if not in_c.any():
        return {"patch_ncc": 1.0, "diff_norm": 0.0, "peak_ratio": 1.0}

    std_g = profile.center_gray.astype(np.float32)
    diff = np.abs(gray - std_g)
    diff_norm = float(diff[in_c].mean() / profile.center_gray_std)

    sg = std_g[in_c]
    tg = gray[in_c]
    sg = sg - sg.mean()
    tg = tg - tg.mean()
    denom = (np.linalg.norm(sg) * np.linalg.norm(tg) + 1e-6)
    patch_ncc = float(np.dot(sg, tg) / denom)

    v = hsv_local[:, :, 2]
    test_peak = float(v[in_c].max())
    peak_ratio = test_peak / max(profile.center_v_peak, 1.0)

    return {
        "patch_ncc": patch_ncc,
        "diff_norm": diff_norm,
        "peak_ratio": peak_ratio,
    }


def _nolead_evidence(profile: StandardProfile, sm: dict, cb_t: float, cb_g: float,
                     ring_missing: bool) -> dict[str, bool]:
    """不出脚各独立证据项。"""
    soft = C.TH["nolead_center_diff_soft"]
    return {
        "ring": ring_missing,
        "diff": sm["diff_norm"] > C.TH["nolead_center_diff_min"],
        "ncc": (sm["patch_ncc"] < C.TH["nolead_patch_ncc_max"] and
                sm["diff_norm"] > soft),
        "peak": sm["peak_ratio"] < C.TH["nolead_peak_ratio_max"],
        "cyan": (cb_t - cb_g > C.TH["nolead_center_cyan_delta"] and
                 cb_t >= C.TH["nolead_center_cyan_abs"]),
    }


def _nolead_fusion_score(ev: dict[str, bool]) -> int:
    """多特征投票得分；过渡环缺失计双倍权重。"""
    ring_w = int(C.TH.get("nolead_ring_vote_weight", 2))
    score = ring_w if ev["ring"] else 0
    for k in ("diff", "ncc", "peak", "cyan"):
        if ev[k]:
            score += 1
    return score


def _nolead_primary_reason(ev: dict[str, bool]) -> str:
    """按优先级返回主因标签。"""
    order = (
        ("ring", "过渡环缺失"),
        ("diff", "中心差异"),
        ("ncc", "结构NCC"),
        ("peak", "亮峰缺失"),
        ("cyan", "亮青差异"),
    )
    for key, label in order:
        if ev[key]:
            return label
    return "多特征融合"


def evaluate_nolead(bgr, roi: PadROI, masks: Masks, profile: StandardProfile,
                    wrap: WrapMetrics | None = None,
                    feat_t: PadFeatures | None = None,
                    feat_g: PadFeatures | None = None):
    """不出脚综合判定。
    """
    cb_t = _center_cyan_ratio(masks.hsv, roi)
    cb_g = profile.cyan_ratio

    x, y, w, h = roi.rect
    sm = center_structure_metrics(bgr, roi, profile)

    ring_v_t, ring_contrast_t = _pin_transition_ring_metrics(
        bgr[y:y + h, x:x + w], w, h)
    ring_v_delta = ring_v_t - profile.ring_v_mean
    ring_contrast_drop = profile.ring_contrast - ring_contrast_t

    diff_ok = sm["diff_norm"] <= C.TH["nolead_diff_ok_max"]
    cyan_ok = abs(cb_t - cb_g) <= C.TH["nolead_cyan_match_max"]
    similar = diff_ok and cyan_ok and sm["patch_ncc"] >= 0.0
    pin_likely_min = float(C.TH.get("nolead_pin_likely_peak_min", 0.98))
    pin_visible_diff_max = float(C.TH["nolead_pin_visible_diff_max"])
    pin_high_diff_min = float(C.TH.get("nolead_pin_high_diff_min", 1.65))
    pin_likely = sm["peak_ratio"] >= pin_likely_min
    # 中心亮于标准且差异未过大 => 引脚仍顶出, 抑制不出脚
    pin_peak = pin_likely and sm["diff_norm"] <= pin_visible_diff_max

    # 中心亮青远超标准(锡面填满中心) => 亮峰判据失效, 交回证据链正常判定
    cyan_override = (
        (cb_t - cb_g) > C.TH.get("nolead_cyan_override_delta", 0.25) and
        cb_t >= C.TH["nolead_center_cyan_abs"]
    )
    if cyan_override:
        pin_likely = False
        pin_peak = False

    ring_missing = (
        ring_v_delta >= C.TH["nolead_ring_v_delta_min"] and
        ring_contrast_drop >= C.TH["nolead_ring_contrast_drop_min"]
    )
    # 引脚顶出时过渡环形态受光照/校准影响大, 不再单独作为不出脚证据
    if pin_likely:
        ring_missing = False

    hit = False
    reason = ""
    min_votes = int(C.TH.get("nolead_min_votes", 1))
    if similar:
        reason = "与标准一致"
    elif pin_peak:
        reason = "引脚可见"
    elif profile.has_pin_structure:
        ev = _nolead_evidence(profile, sm, cb_t, cb_g, ring_missing)
        if pin_likely:
            # 引脚可见时仅当中心差异极大才计 diff/NCC 证据
            if not (ev["diff"] and sm["diff_norm"] >= pin_high_diff_min):
                ev["diff"] = False
            if ev["ncc"] and sm["diff_norm"] < pin_high_diff_min:
                ev["ncc"] = False
        if _nolead_fusion_score(ev) >= min_votes:
            hit, reason = True, _nolead_primary_reason(ev)
        else:
            reason = "结构接近"
    else:
        ev = _nolead_evidence(profile, sm, cb_t, cb_g, False)
        if ev["cyan"]:
            hit, reason = True, "亮青差异"
        else:
            reason = "亮青正常"

    if not hit:
        score = 0.0
    else:
        score_parts = [
            abs(cb_t - cb_g) / 0.20,
            max(0.0, 1.0 - sm["patch_ncc"]),
            sm["diff_norm"] / 2.0,
            max(0.0, 1.0 - sm["peak_ratio"]),
            ring_contrast_drop / 50.0,
        ]
        score = float(min(1.0, max(score_parts)))

    # 引脚顶仍在 + 包锡型多锡 => 中心差异来自饱满锡反光, 非不出脚
    if (hit and wrap is not None and feat_t is not None and feat_g is not None and
            is_excess_bulge(wrap, feat_t, feat_g, sm)):
        hit = False
        reason = "引脚保留"
        score = 0.0

    return {
        **sm,
        "cb_t": cb_t,
        "cb_g": cb_g,
        "ring_v_delta": ring_v_delta,
        "ring_contrast_drop": ring_contrast_drop,
        "hit": hit,
        "reason": reason,
        "score": score,
    }
