"""HSV 分割与 ROI 提取"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import cv2

from . import config as C
from .mask_stats import (
    SolderRelStats,
    compute_solder_rel_stats,
    highlight_s_threshold,
    highlight_v_threshold,
    void_s_threshold,
    void_v_threshold,
)


@dataclass
class Masks:
    hsv: np.ndarray
    solder: np.ndarray
    red: np.ndarray
    bright: np.ndarray


@dataclass
class PadROI:
    id: int
    x: int
    y: int
    w: int
    h: int
    cx: float = 0.0
    cy: float = 0.0

    @property
    def rect(self):
        return (self.x, self.y, self.w, self.h)


@lru_cache(maxsize=64)
def ellipse_kernel(ksize: int) -> np.ndarray:
    """按 ksize 缓存的圆结构元，供 inspect() 单次调用内反复使用同一尺寸的核。
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


@lru_cache(maxsize=32)
def _edge_ring_mask_cached(w: int, h: int, inner: float, outer: float) -> np.ndarray:
    cx, cy = w // 2, h // 2
    out = np.zeros((h, w), np.uint8)
    inn = np.zeros((h, w), np.uint8)
    cv2.ellipse(out, (cx, cy), (int(w * outer), int(h * outer)), 0, 0, 360, 255, -1)
    cv2.ellipse(inn, (cx, cy), (int(w * inner), int(h * inner)), 0, 0, 360, 255, -1)
    return cv2.subtract(out, inn)


def edge_ring_mask(w: int, h: int) -> np.ndarray:
    """ROI 内的圆环带 mask (外圆减内圆)，按 (w,h,inner,outer) 缓存复用。"""
    return _edge_ring_mask_cached(w, h, C.EDGE_RING_INNER, C.EDGE_RING_OUTER)


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    red1 = cv2.inRange(hsv, C.RED1_HSV_LOW, C.RED1_HSV_HIGH)
    red2 = cv2.inRange(hsv, C.RED2_HSV_LOW, C.RED2_HSV_HIGH)
    return cv2.bitwise_or(red1, red2)


def _split_solder_and_copper_red(hsv: np.ndarray, solder: np.ndarray):
    """按锡面膨胀边界拆分红色像素: 落在锡面附近的合并回锡面 (锡面反光)，
    落在锡面外的作为裸铜红 (copper) 单独返回。"""
    red_raw = _red_mask(hsv)
    band = int(getattr(C, "SOLDER_RED_BAND", 13))
    k = ellipse_kernel(band)
    inside = cv2.dilate(solder, k)
    solder_red = cv2.bitwise_and(red_raw, inside)
    solder = cv2.bitwise_or(solder, solder_red)
    copper = cv2.bitwise_and(red_raw, cv2.bitwise_not(inside))
    return solder, copper


def _punch_center_hole(solder: np.ndarray, roi_rect: tuple | None) -> np.ndarray:
    if roi_rect is None or not getattr(C, "SOLDER_PUNCH_CENTER", False):
        return solder
    x, y, w, h = roi_rect
    sub = solder[y:y + h, x:x + w]
    frac = 0.26
    cx, cy = w // 2, h // 2
    hole = np.zeros((h, w), np.uint8)
    cv2.ellipse(hole, (cx, cy), (int(w * frac), int(h * frac)), 0, 0, 360, 255, -1)
    sub = cv2.bitwise_and(sub, cv2.bitwise_not(hole))
    out = solder.copy()
    out[y:y + h, x:x + w] = sub
    return out


def _expand_solder_in_roi(solder: np.ndarray, hsv: np.ndarray,
                          roi_rect: tuple, lo: tuple, hi: tuple) -> np.ndarray:
    x, y, w, h = roi_rect
    sub_hsv = hsv[y:y + h, x:x + w]
    narrow = cv2.inRange(sub_hsv, lo, hi)
    wide = cv2.inRange(sub_hsv, C.SOLDER_WIDE_HSV_LOW, C.SOLDER_WIDE_HSV_HIGH)
    k = ellipse_kernel(7)
    region = cv2.dilate(narrow, k)
    merged = cv2.bitwise_or(narrow, cv2.bitwise_and(wide, region))
    out = solder.copy()
    out[y:y + h, x:x + w] = merged
    return out


def _merge_with_prior(hsv: np.ndarray, prior: np.ndarray,
                      lo: tuple, hi: tuple) -> np.ndarray:
    dil = int(C.PROFILE.get("prior_dilate", 17))
    k = ellipse_kernel(dil)
    region = cv2.dilate(prior, k)
    wide = cv2.inRange(hsv, C.SOLDER_WIDE_HSV_LOW, C.SOLDER_WIDE_HSV_HIGH)
    narrow = cv2.inRange(hsv, lo, hi)
    merged = cv2.bitwise_or(narrow, cv2.bitwise_and(wide, region))
    return cv2.bitwise_and(merged, region)


def _morph_solder(mask: np.ndarray, *, light: bool = True) -> np.ndarray:
    k_close = int(getattr(C, "SOLDER_MORPH_CLOSE", 5))
    k_open = int(getattr(C, "SOLDER_MORPH_OPEN", 0))
    med = int(getattr(C, "SOLDER_MEDIAN", 0))
    m = mask
    if med >= 3:
        m = cv2.medianBlur(m, med | 1)
    if k_close >= 3:
        el = ellipse_kernel(k_close)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, el)
    if k_open >= 3:
        el = ellipse_kernel(k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, el)
    return m


def _keep_largest_component(mask: np.ndarray, min_frac: float) -> np.ndarray:
    h, w = mask.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = int(np.argmax(areas)) + 1
    best_a = int(areas[best_i - 1])
    if best_a < min_frac * h * w:
        return mask
    out = np.zeros_like(mask)
    out[labels == best_i] = 255
    return out


def _void_like_mask(hsv: np.ndarray, solder: np.ndarray,
                    ref_stats: SolderRelStats | None = None) -> np.ndarray:
    """孔洞类暗区的相对 V/S 判据，用于把这类区域从锡面 mask 里排除
    (避免孔洞被误纳入锡面分割结果)。"""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    stats = ref_stats or compute_solder_rel_stats(hsv, solder)
    strip_v = void_v_threshold(
        stats, abs_cap=int(getattr(C, "SOLDER_VOID_STRIP_V_MAX", C.VOID_V_MAX)))
    strip_s = void_s_threshold(
        stats, abs_cap=int(getattr(C, "SOLDER_VOID_STRIP_S_MAX",
                                   C.TH.get("void_dark_s_max", 130))))
    return (v < strip_v) & (s < strip_s)


def _bright_mask(hsv: np.ndarray, solder: np.ndarray,
                 ref_stats: SolderRelStats | None = None) -> np.ndarray:
    """高亮/反光区域的相对 V/S 判据。"""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    stats = ref_stats or compute_solder_rel_stats(hsv, solder)
    v_min = highlight_v_threshold(stats)
    s_max = highlight_s_threshold(stats)
    return ((v >= v_min) & (s <= s_max)).astype(np.uint8) * 255


def compute_masks(bgr: np.ndarray,
                  solder_hsv_low: tuple | None = None,
                  solder_hsv_high: tuple | None = None,
                  prior_mask: np.ndarray | None = None,
                  roi_rect: tuple | None = None,
                  ref_stats: SolderRelStats | None = None) -> Masks:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = solder_hsv_low or C.SOLDER_HSV_LOW
    hi = solder_hsv_high or C.SOLDER_HSV_HIGH

    if prior_mask is not None and getattr(C, "SOLDER_USE_STANDARD_PRIOR", False):
        solder = _merge_with_prior(hsv, prior_mask, lo, hi)
    elif roi_rect is not None:
        solder = cv2.inRange(hsv, lo, hi)
        solder = _expand_solder_in_roi(solder, hsv, roi_rect, lo, hi)
    else:
        solder = cv2.inRange(hsv, lo, hi)

    solder = _morph_solder(solder)
    if getattr(C, "SOLDER_KEEP_LARGEST", False):
        solder = _keep_largest_component(
            solder, getattr(C, "SOLDER_MIN_AREA_FRAC", 0.02))

    if getattr(C, "SOLDER_EXCLUDE_RED", False):
        solder, copper_red = _split_solder_and_copper_red(hsv, solder)
    else:
        copper_red = _red_mask(hsv)
    red = _morph_solder(copper_red) if C.MORPH_KERNEL else copper_red

    void_like = _void_like_mask(hsv, solder, ref_stats)
    solder = cv2.bitwise_and(solder, cv2.bitwise_not(void_like.astype(np.uint8) * 255))
    solder = _punch_center_hole(solder, roi_rect)

    bright = _bright_mask(hsv, solder, ref_stats)
    return Masks(hsv=hsv, solder=solder, red=red, bright=bright)


def compute_masks_from_solder(bgr: np.ndarray, solder_mask: np.ndarray,
                              ref_stats: SolderRelStats | None = None) -> Masks:
    """已有锡面 Mask (标注/复用) 时，只补算其余派生 Mask，不重新分割锡面本身。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if getattr(C, "SOLDER_EXCLUDE_RED", False):
        _, copper_red = _split_solder_and_copper_red(hsv, solder_mask)
    else:
        copper_red = _red_mask(hsv)
    red = _morph_solder(copper_red) if C.MORPH_KERNEL else copper_red
    bright = _bright_mask(hsv, solder_mask, ref_stats)
    return Masks(hsv=hsv, solder=solder_mask, red=red, bright=bright)


def _bbox_roi_from_solder(mask: np.ndarray, w_img: int, h_img: int,
                          margin: int = 8, inset: float = 0.06,
                          min_area_frac: float = 0.05,
                          *, union_bbox: bool = False) -> PadROI:
    """由锡面 mask 得 ROI；默认最大连通域，union_bbox 时用前景并集外接框。"""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if union_bbox:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x0 = max(0, int(xs.min()) - margin)
            y0 = max(0, int(ys.min()) - margin)
            x1 = min(w_img, int(xs.max()) + margin + 1)
            y1 = min(h_img, int(ys.max()) + margin + 1)
            return PadROI(0, x0, y0, x1 - x0, y1 - y0,
                          float(xs.mean()), float(ys.mean()))
    else:
        best = None
        best_area = 0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best = i
        if best is not None and best_area >= min_area_frac * w_img * h_img:
            x = stats[best, cv2.CC_STAT_LEFT]
            y = stats[best, cv2.CC_STAT_TOP]
            w = stats[best, cv2.CC_STAT_WIDTH]
            h = stats[best, cv2.CC_STAT_HEIGHT]
            x0 = max(0, int(x - margin))
            y0 = max(0, int(y - margin))
            x1 = min(w_img, int(x + w + margin))
            y1 = min(h_img, int(y + h + margin))
            return PadROI(0, x0, y0, x1 - x0, y1 - y0,
                          float(centroids[best][0]), float(centroids[best][1]))
    dx = int(w_img * inset)
    dy = int(h_img * inset)
    return PadROI(0, dx, dy, w_img - 2 * dx, h_img - 2 * dy,
                  w_img / 2.0, h_img / 2.0)


def detect_main_roi(std_bgr: np.ndarray, inset: float = 0.06) -> PadROI:
    """自动分割定位主焊点 ROI (无标注/无先验时的分割 mask 兆底路径)。"""
    h_img, w_img = std_bgr.shape[:2]
    masks = compute_masks(std_bgr)
    return _bbox_roi_from_solder(masks.solder, w_img, h_img, inset=inset)


def roi_from_mask(mask: np.ndarray, inset: float = 0.06,
                  *, union_bbox: bool = False) -> PadROI:
    """由已标注锡面 mask 得到 ROI（外接框+margin）。"""
    h_img, w_img = mask.shape[:2]
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return _bbox_roi_from_solder(
        mask_bin, w_img, h_img, inset=inset, union_bbox=union_bbox)
