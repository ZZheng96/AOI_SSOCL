"""连锡颜色：从当前焊点区自适应学锡色（Lab），候选须「像锡」且相对模板有变化。"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def erode_mask(mask: np.ndarray, frac: float = 0.18, min_px: int = 4, max_px: Optional[int] = None) -> np.ndarray:
    """向内收缩标注区域，避免采样到边缘的绿色阻焊圈/抗锯齿混合像素。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.copy()
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    size = max(min_px, int(round(frac * min(w, h))))
    if max_px:
        size = min(size, max_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    eroded = cv2.erode(mask, k)
    if not np.any(eroded):
        # 区域太小被完全腐蚀掉时，退化为较小的核，保底不返回空
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(mask, k2)
    return eroded


def robust_color_stats(
    lab_img: np.ndarray, mask: np.ndarray, lo: float = 10, hi: float = 90
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """在 mask 区域内统计 Lab 颜色的稳健中心/离散度。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    samples = lab_img[ys, xs]
    medians = np.zeros(3, dtype=np.float32)
    robust_std = np.zeros(3, dtype=np.float32)
    for c in range(3):
        v = samples[:, c]
        p_lo, p_hi = np.percentile(v, [lo, hi])
        v_clip = v[(v >= p_lo) & (v <= p_hi)]
        if len(v_clip) == 0:
            v_clip = v
        med = float(np.median(v_clip))
        mad = float(np.median(np.abs(v_clip - med)))
        medians[c] = med
        robust_std[c] = mad * 1.4826  # 使 MAD 在正态分布假设下与标准差可比
    return medians, robust_std, len(xs)


def classify_solder(
    lab_img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 1.0, 1.0),
    std_floor: Tuple[float, float, float] = (4.0, 3.0, 3.0),
    std_cap: Tuple[float, float, float] = (16.0, 12.0, 12.0),
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """按加权归一化距离对每个像素分类：是否与参考"锡色"接近。
    """
    diff = lab_img - mean.reshape(1, 1, 3)
    std_eff = np.clip(std, std_floor, std_cap).reshape(1, 1, 3).astype(np.float32)
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    d2 = np.sum(w * (diff / std_eff) ** 2, axis=2)
    dist = np.sqrt(d2)
    return dist <= k, dist


def classify_change(
    test_lab: np.ndarray,
    template_lab: np.ndarray,
    test_ref_mean: np.ndarray,
    test_ref_std: np.ndarray,
    templ_ref_mean: np.ndarray,
    templ_ref_std: np.ndarray,
    weights: Tuple[float, float, float] = (0.5, 1.0, 1.0),
    tol: Tuple[float, float, float] = (10.0, 6.0, 6.0),
    scale_clip: Tuple[float, float] = (0.5, 2.0),
    k: float = 1.0,
    blur_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """判断像素相对标准图（模板）同一位置，是否发生了"实质性变化"。
    """
    scale = np.clip(
        templ_ref_std / np.maximum(test_ref_std, 1e-3), scale_clip[0], scale_clip[1]
    ).reshape(1, 1, 3)
    corrected = (test_lab - test_ref_mean.reshape(1, 1, 3)) * scale + templ_ref_mean.reshape(1, 1, 3)
    diff = corrected - template_lab
    if blur_ksize and blur_ksize > 1:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    tol_arr = np.array(tol, dtype=np.float32).reshape(1, 1, 3)
    d2 = np.sum(w * (diff / tol_arr) ** 2, axis=2)
    dist = np.sqrt(d2)
    return dist > k, dist


def gap_clearance_widths(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    lab_img: np.ndarray,
    bg_percentile: float = 70.0,
    bg_k: float = 3.0,
) -> Tuple[List[int], str, List[Tuple[int, int, int]]]:
    """沿两焊点连线方向，逐行/列测量它们之间"背景色"缝隙的宽度。
    """
    h, w = mask_a.shape
    ys_a, xs_a = np.where(mask_a > 0)
    ys_b, xs_b = np.where(mask_b > 0)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return [], "x", []
    ca = (float(np.mean(ys_a)), float(np.mean(xs_a)))
    cb = (float(np.mean(ys_b)), float(np.mean(xs_b)))
    axis = "x" if abs(cb[1] - ca[1]) >= abs(cb[0] - ca[0]) else "y"

    raw_segments = []  # list of (idx, lo, hi) 几何缝隙（未做颜色筛选）
    if axis == "x":
        left_is_a = ca[1] < cb[1]
        for y in range(h):
            ra = np.where(mask_a[y] > 0)[0]
            rb = np.where(mask_b[y] > 0)[0]
            if len(ra) == 0 or len(rb) == 0:
                continue
            lo, hi = (ra.max() + 1, rb.min()) if left_is_a else (rb.max() + 1, ra.min())
            if hi > lo:
                raw_segments.append((y, lo, hi))
    else:
        top_is_a = ca[0] < cb[0]
        for x in range(w):
            ca_col = np.where(mask_a[:, x] > 0)[0]
            cb_col = np.where(mask_b[:, x] > 0)[0]
            if len(ca_col) == 0 or len(cb_col) == 0:
                continue
            lo, hi = (ca_col.max() + 1, cb_col.min()) if top_is_a else (cb_col.max() + 1, ca_col.min())
            if hi > lo:
                raw_segments.append((x, lo, hi))

    if not raw_segments:
        return [], axis, []

    geo_widths = [hi - lo for _, lo, hi in raw_segments]
    wide_thresh = np.percentile(geo_widths, bg_percentile)
    bg_pixels = []
    for idx, lo, hi in raw_segments:
        if (hi - lo) >= wide_thresh:
            seg = lab_img[idx, lo:hi] if axis == "x" else lab_img[lo:hi, idx]
            bg_pixels.append(seg)
    bg_pixels = np.concatenate(bg_pixels, axis=0)
    bg_med = np.median(bg_pixels, axis=0)
    bg_mad = np.maximum(np.median(np.abs(bg_pixels - bg_med), axis=0) * 1.4826, 2.0)

    widths = []
    for idx, lo, hi in raw_segments:
        seg = lab_img[idx, lo:hi] if axis == "x" else lab_img[lo:hi, idx]
        d = np.sqrt(np.sum(((seg - bg_med) / bg_mad) ** 2, axis=1))
        widths.append(int(np.count_nonzero(d <= bg_k)))
    return widths, axis, raw_segments
