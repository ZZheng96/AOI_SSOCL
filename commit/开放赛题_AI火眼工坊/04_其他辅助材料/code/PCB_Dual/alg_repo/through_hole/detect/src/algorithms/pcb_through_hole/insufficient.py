"""少锡(13) 缺陷判定。"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from .features import PadFeatures
from .mask_stats import (
    compute_solder_rel_stats,
    exposed_pixel_ok,
    solder_core_mask,
)
from .segment import Masks, PadROI, edge_ring_mask

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _valid_detection_mask(sg: np.ndarray, w: int, h: int) -> np.ndarray:
    """少锡有效检测 Mask = 标准锡面核心 (扣除边缘公差圈)。
    """
    return solder_core_mask(sg, w, h)


def _texture_drop(feat_t: PadFeatures, feat_g: PadFeatures) -> float:
    if feat_g.lap_var <= 1e-6:
        return 0.0
    return max(0.0, (feat_g.lap_var - feat_t.lap_var) / feat_g.lap_var)


def _exposed_evidence(ss: np.ndarray, ts: np.ndarray, red: np.ndarray,
                      hsv: np.ndarray, valid_mask: np.ndarray, valid_area: int):
    V, S = hsv[:, :, 2], hsv[:, :, 1]

    stats_g = compute_solder_rel_stats(hsv, ss)
    exposed_all = ((ts == 0) & (red == 0) & exposed_pixel_ok(V, S, stats_g))
    exposed_u8 = exposed_all.astype(np.uint8) * 255
    exposed_u8 = cv2.morphologyEx(exposed_u8, cv2.MORPH_OPEN, _K3)
    exposed_u8 = cv2.bitwise_and(exposed_u8, valid_mask)
    exposed_ratio = cv2.countNonZero(exposed_u8) / valid_area

    red_in_valid = cv2.bitwise_and(red, valid_mask)
    red_ratio = cv2.countNonZero(red_in_valid) / valid_area
    return exposed_ratio, red_ratio, exposed_u8


def _thin_support_ok(ar: float, exposed_ratio: float, ring_red: float) -> bool:
    """覆盖率仍较高时，薄锡需有边缘红或少量裸露佐证，排除整圈配准偏差。"""
    ar_lo = float(C.TH.get("insuf_thin_support_ar_max", 0.72))
    if ar <= ar_lo:
        return True
    exp_min = float(C.TH.get("insuf_thin_support_exposed_min", 0.03))
    ring_min = float(C.TH.get("insuf_thin_support_ring_min", 0.02))
    return exposed_ratio >= exp_min or ring_red >= ring_min


def _insuf_metrics(ar: float = 1.0, exposed: float = 0.0, red: float = 0.0,
                   ring: float = 0.0, tex: float = 0.0) -> dict[str, float]:
    """`analyze_insufficient` 判定依据的原始数值特征，供 suppress_* 交叉判定直接读取。"""
    return {
        "ar": round(float(ar), 2),
        "exposed": round(float(exposed), 2),
        "red": round(float(red), 2),
        "ring": round(float(ring), 2),
        "tex": round(float(tex), 2),
    }


def _classify_insuf_subtype(exposed: float, ring_red: float) -> str:
    """少锡分型：边缘渐变=薄锡，硬边缺口=露铜。

    - 薄锡：异常主要在边缘环带（红/粉过渡明显）
    - 露铜：焊盘内出现明显裸露缺口，边缘红很少（边界清楚的缺块）
    """
    thin_ring = float(C.TH.get("insuf_thin_ring_prefer", 0.03))
    copper_exp = float(C.TH.get("insuf_copper_exposed_prefer", 0.10))
    # 边缘红达到偏好阈值 → 薄锡
    if ring_red >= thin_ring:
        return "薄锡"
    # 核心裸露高且边缘几乎无红 → 露铜硬缺口
    if exposed >= copper_exp and ring_red < thin_ring:
        return "露铜"
    # 边缘红相对裸露更突出 → 薄锡
    if ring_red > 1e-6 and ring_red >= exposed * 0.4:
        return "薄锡"
    if exposed >= float(C.TH.get("insuf_exposed_hard_min", 0.06)):
        return "露铜"
    return "薄锡"


def suppress_insuf_for_nolead(insuf: tuple, nolead: dict) -> bool:
    """不出脚致中心平顶/裸露时，不算少锡。"""
    if not (insuf[1] and nolead.get("hit")):
        return False
    if insuf[2] != "露铜":
        return False
    m = insuf[5]
    ar = m.get("ar", 1.0)
    red = m.get("red", 0.0)
    ring = m.get("ring", 0.0)
    ar_min = float(C.TH.get("insuf_nolead_center_ar_min", 0.82))
    red_max = float(C.TH.get("insuf_nolead_center_red_max", 0.04))
    ring_max = float(C.TH.get("insuf_nolead_center_ring_max", 0.06))
    if ar < ar_min or red >= red_max or ring >= ring_max:
        return False
    ring_drop = float(nolead.get("ring_contrast_drop", 0.0))
    diff = float(nolead.get("diff_norm", 0.0))
    ring_min = float(C.TH.get("nolead_ring_contrast_drop_min", 28.0))
    diff_min = float(C.TH.get("insuf_nolead_center_diff_min", 2.0))
    return ring_drop >= ring_min or diff >= diff_min


def suppress_nolead_for_insuf(insuf: tuple, nolead: dict) -> bool:
    """严重少锡致锡偏位时，中心差异/过渡环变化不算不出脚。"""
    if not (insuf[1] and nolead.get("hit")):
        return False
    peak_min = float(C.TH.get("nolead_pin_likely_peak_min", 0.98))
    if nolead.get("peak_ratio", 0.0) < peak_min:
        return False
    if insuf[2] == "薄锡":
        return True
    if insuf[2] != "露铜":
        return False
    m = insuf[5]
    ar = m.get("ar", 1.0)
    exposed = m.get("exposed", 0.0)
    ring = m.get("ring", 0.0)
    ar_max = float(C.TH.get("insuf_shift_nolead_ar_max", 0.65))
    exp_min = float(C.TH.get("insuf_shift_nolead_exposed_min", 0.12))
    ring_min = float(C.TH.get("insuf_shift_nolead_ring_min", 0.12))
    return ar <= ar_max or exposed >= exp_min or ring >= ring_min


def analyze_insufficient(roi: PadROI, masks: Masks, masks_g: Masks,
                         feat_t: PadFeatures, feat_g: PadFeatures):
    """少锡分析：覆盖率 + 异常证据命中；分型按工艺外观。
    """
    x, y, w, h = roi.rect
    ts = masks.solder[y:y + h, x:x + w]
    ss = masks_g.solder[y:y + h, x:x + w]
    red = masks.red[y:y + h, x:x + w]
    hsv = masks.hsv[y:y + h, x:x + w]

    valid_mask = _valid_detection_mask(ss, w, h)
    valid_area = max(1, cv2.countNonZero(valid_mask))

    covered = cv2.bitwise_and(ts, valid_mask)
    ar = cv2.countNonZero(covered) / valid_area
    uncovered = cv2.bitwise_and(cv2.bitwise_not(ts), valid_mask)

    exposed_core, red_core, exposed_mask = _exposed_evidence(
        ss, ts, red, hsv, valid_mask, valid_area)

    if ar < C.TH["insuf_area_ratio_min"]:
        return uncovered, False, "", 0.0, f"面积比过低({ar:.2f})", _insuf_metrics()

    tex_drop = _texture_drop(feat_t, feat_g)
    ring_red = float(feat_t.ring_red_ratio)
    red_ratio = float(feat_t.red_ratio)

    area_max = C.insuf_th("insuf_area_ratio_max")
    red_min = C.insuf_th("insuf_red_ratio_min")
    ring_min = C.insuf_th("insuf_ring_red_min")
    thin_max = C.insuf_th("insuf_thin_area_max")
    exposed_min = C.insuf_th("insuf_exposed_core_min")

    hard_exposed = float(C.TH.get("insuf_exposed_hard_min", 0.06))
    hard_red = float(C.TH.get("insuf_exposed_hard_red_min", 0.015))

    # --- 命中：硬缺口 / 覆盖不足+颜色 / 边缘渐变 ---
    hard_gap_hit = (
        exposed_core >= hard_exposed or red_core >= hard_red
    )
    coverage_color_hit = (
        ar <= area_max and (
            red_ratio >= red_min or exposed_core >= exposed_min
        )
    )
    edge_gradient_hit = (
        ar <= thin_max
        and ring_red >= ring_min
        and _thin_support_ok(ar, exposed_core, ring_red)
    )

    if not (hard_gap_hit or coverage_color_hit or edge_gradient_hit):
        metrics = _insuf_metrics(
            ar=ar, exposed=exposed_core, red=red_ratio, ring=ring_red, tex=tex_drop)
        return uncovered, False, "", 0.0, (
            f"未命中 ar={ar:.2f} red={red_ratio:.2f} "
            f"裸露={exposed_core:.2f} ring={ring_red:.2f}"), metrics

    ring = cv2.bitwise_and(edge_ring_mask(w, h), red)
    basis = cv2.bitwise_or(exposed_mask, ring)

    subtype = _classify_insuf_subtype(exposed_core, ring_red)
    if subtype == "露铜":
        score = float(min(1.0, exposed_core * 2.5 + (1 - ar) * 0.5 +
                          min(red_ratio, 0.35) * 1.2))
    else:
        score = float(min(1.0, (1 - ar) * 0.35 +
                          max(ring_red, 0.05) * 1.2 + tex_drop * 0.25))

    reason = (f"{subtype} ar={ar:.2f} 裸露={exposed_core:.2f} "
              f"red={red_ratio:.2f} ring={ring_red:.2f} tex={tex_drop:.2f}")
    metrics = _insuf_metrics(
        ar=ar, exposed=exposed_core, red=red_ratio, ring=ring_red, tex=tex_drop)
    return basis, True, subtype, score, reason, metrics
