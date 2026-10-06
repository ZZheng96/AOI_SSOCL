"""规则 12 孔洞：标准侧 `VoidStdCache` 缓存一次，测试图做 HSV 分类 + 连通域判定。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from . import config as C
from .mask_stats import (
    SolderRelStats,
    build_edge_artifact_context,
    compute_solder_rel_stats,
    is_shallow_edge_artifact,
    void_desat_threshold,
    void_s_threshold,
    void_v_threshold,
)
from .detection import Detection
from .segment import Masks, PadROI, ellipse_kernel

_K_VALID = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K_BD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K_VOID = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (C.VOID_MORPH_KERNEL, C.VOID_MORPH_KERNEL))


def erode_valid_roi(valid_roi: np.ndarray | None, h: int, w: int) -> np.ndarray:
    """校准有效区掩膜；轻微腐蚀以避开黑边过渡带。"""
    if valid_roi is None:
        return np.full((h, w), 255, np.uint8)
    return cv2.erode(valid_roi, _K_VALID)


def _void_component_metrics(comp: np.ndarray, solder_t: np.ndarray, void_evidence: np.ndarray,
                            core: np.ndarray, v: np.ndarray, v_max: int,
                            s: np.ndarray | None = None, s_desat_max: int = 70):
    """单个孔洞候选连通域的判定指标：enc(边界落在测试锡面比例)/miss(与颜色
    证据重叠比例)/dark(暗区占比)/in_core(落在核心比例)/desat(低饱和占比)。
    """
    area = cv2.countNonZero(comp)
    if area <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    bd = cv2.dilate(comp, _K_BD) - cv2.erode(comp, _K_BD)
    bd_cnt = max(1, cv2.countNonZero(bd))
    enc = cv2.countNonZero(cv2.bitwise_and(bd, solder_t)) / bd_cnt
    miss = cv2.countNonZero(cv2.bitwise_and(comp, void_evidence)) / area
    dark = cv2.countNonZero(cv2.bitwise_and(comp, (v < v_max).astype(np.uint8) * 255)) / area
    in_core = cv2.countNonZero(cv2.bitwise_and(comp, core)) / area
    if s is None:
        desat = 1.0
    else:
        desat = cv2.countNonZero(
            cv2.bitwise_and(comp, (s < s_desat_max).astype(np.uint8) * 255)) / area
    return enc, miss, dark, in_core, desat


def _bbox_aspect_ratio(bw: int, bh: int) -> float:
    """外接矩形长短边比；孔洞近圆/近椭圆，线状噪声比值很大。"""
    return max(bw, bh) / max(1, min(bw, bh))


def _void_shape_ok(area: int, bw: int, bh: int) -> bool:
    """面积填充率 + 长宽比，剔除细长条误检。"""
    fill = area / max(1, bw * bh)
    if fill < C.TH["void_min_circularity"]:
        return False
    max_ar = float(C.TH.get("void_max_aspect_ratio", 4.0))
    if _bbox_aspect_ratio(bw, bh) > max_ar:
        return False
    return True


def _invalid_side_margins(valid_roi: np.ndarray | None,
                          roi_w: int, roi_h: int) -> tuple[int, int, int, int]:
    """按 valid 掩膜推断四边黑边禁区 (left, top, right, bottom)，无黑边侧为 0。"""
    fixed = int(C.TH.get("void_border_margin_px", 3))
    if valid_roi is None:
        return fixed, fixed, fixed, fixed
    h, w = valid_roi.shape[:2]
    if h != roi_h or w != roi_w:
        valid_roi = cv2.resize(valid_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    rows = np.any(valid_roi > 0, axis=1)
    cols = np.any(valid_roi > 0, axis=0)
    if not rows.any() or not cols.any():
        return fixed, fixed, fixed, fixed
    top = int(np.argmax(rows))
    bottom = roi_h - 1 - int(np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = roi_w - 1 - int(np.argmax(cols[::-1]))
    return (
        fixed if left > 0 else 0,
        fixed if top > 0 else 0,
        fixed if right < roi_w - 1 else 0,
        fixed if bottom < roi_h - 1 else 0,
    )


def _void_touches_invalid_border(bx: int, by: int, bw: int, bh: int,
                                 roi_w: int, roi_h: int,
                                 margins: tuple[int, int, int, int]) -> bool:
    """仅当候选外接框贴近有黑边的 ROI 侧时才剔除。"""
    left, top, right, bottom = margins
    if left and bx < left:
        return True
    if top and by < top:
        return True
    if right and bx + bw > roi_w - right:
        return True
    if bottom and by + bh > roi_h - bottom:
        return True
    return False


def _void_is_edge_shadow(bx: int, by: int, bw: int, bh: int,
                        enc: float, dark: float,
                        roi_w: int, roi_h: int,
                        margins: tuple[int, int, int, int]) -> bool:
    """ROI 边缘阴影/校准黑边误检：低包络 + 暗度不足。"""
    enc_max = float(C.TH.get("void_edge_shadow_max_enc", 0.32))
    dark_min = float(C.TH.get("void_edge_shadow_min_dark", 0.92))
    if enc >= enc_max or dark >= dark_min:
        return False
    border_px = int(C.TH.get("void_edge_shadow_border_px", 8))
    near_border = (
        bx < border_px or by < border_px or
        bx + bw > roi_w - border_px or by + bh > roi_h - border_px
    )
    if near_border:
        return True
    left, top, right, bottom = margins
    if left and bx < left:
        return True
    if top and by < top:
        return True
    if right and bx + bw > roi_w - right:
        return True
    if bottom and by + bh > roi_h - bottom:
        return True
    return False


def _void_passes(enc: float, miss: float, dark: float, core: float, area: float,
                 solder_area: float, desat: float = 1.0) -> bool:
    """孔洞简化判定：面积占比 + 暗度（颜色证据）为主。
    """
    _ = core  # 兼容旧调用签名，简化后不再分路径使用
    if desat < float(C.TH.get("void_min_desat_frac", 0.75)):
        return False

    # 以相对锡面面积占比为主；仅保留极小像素底噪，避免 0 面积误过。
    frac = float(C.TH.get("void_decide_area_frac", 0.0035))
    min_area = max(1.0, frac * max(1.0, solder_area))
    dark_min = float(C.TH.get("void_decide_min_dark", 0.90))
    miss_min = float(C.TH.get("void_decide_min_miss", 0.90))
    enc_min = float(C.TH.get("void_enc_min", 0.28))
    enc_max = float(C.TH.get("void_enc_max", 0.52))

    if area < min_area:
        return False
    if dark < dark_min or miss < miss_min:
        return False
    if not (enc_min <= enc <= enc_max):
        return False
    return True


def _void_v_max(stats: SolderRelStats | None = None) -> int:
    """判定 dark 指标 / _void_passes 用（随 void_preset 挡位变化）。"""
    cap = int(C.VOID_V_MAX)
    if stats is None:
        return cap
    return void_v_threshold(stats, abs_cap=cap)


def _void_dark_s_max(stats: SolderRelStats | None = None) -> int:
    cap = int(C.TH.get("void_dark_s_max", 130))
    if stats is None:
        return cap
    return void_s_threshold(stats, abs_cap=cap)


def _void_v_for_mask(stats: SolderRelStats | None = None) -> int:
    """构造候选掩膜用（固定 MED 基准，避免 void_preset 改变连通域形态）。"""
    cap = int(getattr(C, "SOLDER_VOID_STRIP_V_MAX", C.VOID_V_MAX))
    if stats is None:
        return cap
    return void_v_threshold(stats, abs_cap=cap)


def _void_s_for_mask(stats: SolderRelStats | None = None) -> int:
    cap = int(getattr(C, "SOLDER_VOID_STRIP_S_MAX", _void_dark_s_max()))
    if stats is None:
        return cap
    return void_s_threshold(stats, abs_cap=cap)


def _min_area(roi_w: int, roi_h: int, solder_area: float) -> float:
    """候选连通域最小面积：按相对 ROI/锡面面积占比，不再用绝对像素作主门槛。"""
    ref = max(float(roi_w * roi_h), solder_area)
    frac = float(C.TH.get("void_min_area_ratio", 0.005))
    return max(1.0, frac * max(1.0, ref))


@dataclass
class VoidStdCache:
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    pad_id: int
    solder_g: np.ndarray
    red_g: np.ndarray
    region: np.ndarray
    core: np.ndarray
    solder_stats: SolderRelStats
    solder_area: float


def build_void_std_cache(roi: PadROI, masks_g: Masks) -> VoidStdCache:
    x, y, w, h = roi.rect
    solder_g = masks_g.solder[y:y + h, x:x + w]
    red_g = masks_g.red[y:y + h, x:x + w]
    hsv_g = masks_g.hsv[y:y + h, x:x + w]
    region = cv2.dilate(solder_g, _K_VOID)
    core = cv2.erode(solder_g, ellipse_kernel(9))
    stats = compute_solder_rel_stats(hsv_g, solder_g)
    return VoidStdCache(
        x, y, w, h, roi.id, solder_g, red_g, region, core,
        stats, float(max(1, cv2.countNonZero(solder_g))),
    )


def analyze_voids(std: VoidStdCache, hsv_t: np.ndarray,
                  solder_t: np.ndarray, red_t: np.ndarray,
                  valid_roi: np.ndarray | None = None):
    """孔洞分析: 仅在 ROI 数组上判定（阈值随 config.yaml 中 void_preset 生效）。
    """
    stats = std.solder_stats
    v_max = _void_v_max(stats)
    v_mask = _void_v_for_mask(stats)
    s_mask = _void_s_for_mask(stats)
    min_area = _min_area(std.roi_w, std.roi_h, std.solder_area)
    compact_floor = float(C.TH.get("void_core_compact_min_area", 250))
    side_margins = _invalid_side_margins(valid_roi, std.roi_w, std.roi_h)

    v = hsv_t[:, :, 2]
    s = hsv_t[:, :, 1]
    sg, region, core = std.solder_g, std.region, std.core
    valid_e = erode_valid_roi(valid_roi, std.roi_h, std.roi_w)
    solder_t = cv2.bitwise_and(solder_t, valid_e)

    ve = valid_e > 0
    dark = ((v < v_mask) & (region > 0) & (solder_t == 0) & (red_t == 0) & ve)
    dark_in_std = ((v < v_mask) & (s < s_mask) & (sg > 0) & (red_t == 0) & ve)

    combined_raw = cv2.bitwise_or(
        dark.astype(np.uint8) * 255, dark_in_std.astype(np.uint8) * 255)
    combined = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, _K_VOID)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _K_VOID)
    combined = cv2.bitwise_and(combined, valid_e)

    void_evidence = dark_in_std.astype(np.uint8) * 255
    s_desat_max = void_desat_threshold(stats)

    rx, ry = std.roi_x, std.roi_y
    cand: list[Detection] = []
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    edge_ctx = build_edge_artifact_context(sg, std.roi_w, std.roi_h)
    for i in range(1, n):
        area = cc_stats[i, cv2.CC_STAT_AREA]
        if area < min_area and area < compact_floor:
            continue
        bx = cc_stats[i, cv2.CC_STAT_LEFT]
        by = cc_stats[i, cv2.CC_STAT_TOP]
        bw = cc_stats[i, cv2.CC_STAT_WIDTH]
        bh = cc_stats[i, cv2.CC_STAT_HEIGHT]
        if not _void_shape_ok(area, bw, bh):
            continue
        if _void_touches_invalid_border(
                bx, by, bw, bh, std.roi_w, std.roi_h, side_margins):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        enc, miss, dark_f, in_core, desat = _void_component_metrics(
            comp, solder_t, void_evidence, core, v, v_max,
            s=s, s_desat_max=s_desat_max)
        if _void_is_edge_shadow(
                bx, by, bw, bh, enc, dark_f, std.roi_w, std.roi_h, side_margins):
            continue
        if is_shallow_edge_artifact(comp, sg, std.roi_w, std.roi_h, ctx=edge_ctx):
            continue
        if not _void_passes(
                enc, miss, dark_f, in_core, float(area), std.solder_area, desat):
            continue
        score = min(1.0, area / (3 * min_area))
        cand.append(Detection(
            C.DEFECT_ID_VOID, score, (rx + bx, ry + by, bw, bh),
            f"孔洞 area={area} enc={enc:.2f} miss={miss:.2f} dark={dark_f:.2f} "
            f"desat={desat:.2f}",
            std.pad_id))

    max_n = int(C.TH["void_max_count"])
    if len(cand) > max_n:
        cand.sort(key=lambda d: d.score, reverse=True)
        cand = cand[:max_n]

    return combined, cand
