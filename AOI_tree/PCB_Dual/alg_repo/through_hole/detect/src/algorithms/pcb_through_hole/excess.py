"""多锡(14) 缺陷判定（外扩型 + 包锡型）。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from .features import PadFeatures
from .mask_stats import pin_core_min_pixels
from .segment import Masks, PadROI


def _area_ratio(feat_t: PadFeatures, feat_g: PadFeatures) -> float:
    return feat_t.solder_area / max(1.0, feat_g.solder_area)


@dataclass
class WrapMetrics:
    """相对标准图的引脚顶与中心饱和青差分。"""
    pin_core_std: int
    pin_core_test: int
    pin_core_ret: float
    extra_cyan: float
    delta_cyan: float
    extra_bright: float
    basis_mask: np.ndarray | None = None


def _pin_core_ellipse(w: int, h: int) -> np.ndarray:
    frac = C.TH["pin_top_frac"]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * frac), int(h * frac)),
                0, 0, 360, 255, -1)
    return mask


def center_ellipse(w: int, h: int) -> np.ndarray:
    """中心椭圆区域 (半径比例见 `nolead_center_frac`)；`nolead.build_profile`等亦复用此函数，避免重复实现。"""
    frac = C.TH["nolead_center_frac"]
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * frac), int(h * frac)),
                0, 0, 360, 255, -1)
    return mask


def pin_core_area(bgr: np.ndarray, roi: PadROI) -> int:
    """中心引脚顶暗核面积 (灰度相对中心中位数偏低区域)。"""
    x, y, w, h = roi.rect
    gray = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pe = _pin_core_ellipse(w, h) > 0
    if not pe.any():
        return 0
    med = float(np.median(gray[pe]))
    tol = C.TH["pin_top_dark_tol"]
    return int(((gray < med + tol) & pe).sum())


def _cyan_mask(hsv: np.ndarray) -> np.ndarray:
    h_lo = C.TH["nolead_cyan_h_low"]
    h_hi = C.TH["nolead_cyan_h_high"]
    s_min = C.TH["nolead_cyan_s_min"]
    v_min = C.TH["nolead_cyan_v_min"]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= h_lo) & (H <= h_hi) & (S >= s_min) & (V >= v_min))


def compute_wrap_metrics(bgr: np.ndarray, masks: Masks, masks_g: Masks,
                         roi: PadROI, pin_top_core_std: int) -> WrapMetrics:
    """引脚顶保留 + 中心饱和青/高亮相对标准的增量。"""
    x, y, w, h = roi.rect
    pin_std = pin_top_core_std
    pin_test = pin_core_area(bgr, roi)

    ht = masks.hsv[y:y + h, x:x + w]
    hg = masks_g.hsv[y:y + h, x:x + w]
    center = center_ellipse(w, h) > 0
    cyan_t = _cyan_mask(ht)
    cyan_g = _cyan_mask(hg)
    bright_t = masks.bright[y:y + h, x:x + w] > 0
    bright_g = masks_g.bright[y:y + h, x:x + w] > 0

    ic = center
    delta_cyan = float(cyan_t[ic].mean() - cyan_g[ic].mean()) if ic.any() else 0.0
    extra_cyan = float((cyan_t & ~cyan_g & ic).mean()) if ic.any() else 0.0
    extra_bright = float((bright_t & ~bright_g & ic).mean()) if ic.any() else 0.0

    extra = (cyan_t & ~cyan_g) | (bright_t & ~bright_g)
    extra_u8 = (extra & ic).astype(np.uint8) * 255

    ret = pin_test / max(1, pin_std)
    return WrapMetrics(
        pin_core_std=pin_std,
        pin_core_test=pin_test,
        pin_core_ret=ret,
        extra_cyan=extra_cyan,
        delta_cyan=delta_cyan,
        extra_bright=extra_bright,
        basis_mask=extra_u8,
    )


def is_excess_bulge(wrap: WrapMetrics, feat_t, feat_g, sm: dict) -> bool:
    """包锡型多锡: 引脚顶仍在, 中心饱和青/高光相对标准增多, 结构差异显著。"""
    roi_side = int(max(1, round(feat_g.roi_area ** 0.5)))
    pin_min = pin_core_min_pixels(roi_side, roi_side, wrap.pin_core_std)
    if wrap.pin_core_std < pin_min:
        return False

    ar = feat_t.solder_area / max(1.0, feat_g.solder_area)
    diff = sm.get("diff_norm", 0.0)
    ncc = sm.get("patch_ncc", 1.0)

    return (
        C.TH["excess_bulge_pin_ret_min"] <= wrap.pin_core_ret <= C.TH["excess_bulge_pin_ret_max"]
        and wrap.extra_cyan >= C.TH["excess_bulge_extra_cyan_min"]
        and C.TH["excess_bulge_diff_min"] <= diff <= C.TH["excess_bulge_diff_max"]
        and ncc <= C.TH["excess_bulge_ncc_max"]
        and C.TH["excess_bulge_area_min"] <= ar <= C.TH["excess_bulge_area_max"]
    )


def analyze_excess(roi: PadROI, masks: Masks, masks_g: Masks,
                   feat_t: PadFeatures, feat_g: PadFeatures,
                   wrap: WrapMetrics | None = None,
                   nolead_sm: dict | None = None):
    """多锡分析: 外扩型 (面积增大) + 包锡型 (引脚顶保留 + cyan/高光差分)。"""
    ar = _area_ratio(feat_t, feat_g)

    # --- 包锡型: 引脚仍在, 中心饱和青/高光相对标准增多 ---
    if wrap is not None and nolead_sm is not None and is_excess_bulge(wrap, feat_t, feat_g, nolead_sm):
        basis = wrap.basis_mask
        score = float(min(1.0, wrap.extra_cyan * 8.0 + max(0.0, nolead_sm["diff_norm"] - 1.0) * 0.35))
        reason = (f"包锡 pin={wrap.pin_core_ret:.2f} ec={wrap.extra_cyan:.3f} "
                  f"diff={nolead_sm['diff_norm']:.2f} ar={ar:.2f}")
        return basis, True, score, reason

    # --- 外扩型: 面积增大 + 高亮/形态 + 掩膜差证据 ---
    if ar < C.TH["excess_area_ratio_min"]:
        return None, False, 0.0, f"面积比不足({ar:.2f})"

    x, y, w, h = roi.rect
    extra = cv2.subtract(masks.solder[y:y + h, x:x + w],
                         masks_g.solder[y:y + h, x:x + w])
    solder_g_area = max(1.0, float(cv2.countNonZero(masks_g.solder[y:y + h, x:x + w])))
    extra_frac = cv2.countNonZero(extra) / solder_g_area
    extra_ok = extra_frac >= C.TH.get("excess_extra_solder_frac_min", 0.025)

    morph_ok = (feat_t.circularity >= C.TH["excess_circularity_min"] or
                feat_t.solidity >= C.TH["excess_solidity_min"])
    highlight_ok = (feat_t.highlight_ratio >= C.TH["excess_highlight_ratio_min"] and
                    feat_t.bright_closure >= C.TH["excess_bright_closure_min"])
    morph_pair = ar >= C.TH["excess_area_ratio_min"] + 0.08 and morph_ok

    if not extra_ok:
        return None, False, 0.0, f"外扩掩膜差不足 extra={extra_frac:.3f}"
    if not highlight_ok and not morph_pair:
        return None, False, 0.0, (
            f"高亮/形态不足 hl={feat_t.highlight_ratio:.2f} "
            f"closure={feat_t.bright_closure:.2f} circ={feat_t.circularity:.2f}")

    extra = cv2.bitwise_or(extra, masks.bright[y:y + h, x:x + w])

    score = float(min(1.0, (ar - 1.0) * 0.8 + feat_t.highlight_ratio * 2.5 +
                       feat_t.circularity * 0.15))
    reason = (f"面积比={ar:.2f} 高亮={feat_t.highlight_ratio:.2f} "
              f"圆度={feat_t.circularity:.2f} 闭合={feat_t.bright_closure:.2f}")
    return extra, True, score, reason
