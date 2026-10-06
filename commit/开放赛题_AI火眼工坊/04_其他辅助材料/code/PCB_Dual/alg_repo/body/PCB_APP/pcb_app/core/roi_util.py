"""ROI 裁剪工具：与 pcb_defect_detector.detector 行为一致。

当 Ubuntu/旧版算法包 ``detect()`` 尚不支持 ``roi=`` 参数时，由 PCB_APP 在调用前
自行裁剪模板/待检图，并把缺陷坐标偏移回整图坐标系。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def normalize_roi(roi: Optional[Mapping[str, Any]], img_w: int, img_h: int):
    """把 roi 描述归一化为 (x, y, w, h, mask_or_None)。"""
    if not roi:
        return None
    shape = str(roi.get("shape", "rect")).lower()
    if shape == "circle":
        cx = float(roi.get("cx", 0.0))
        cy = float(roi.get("cy", 0.0))
        r = float(roi.get("r", 0.0))
        if r <= 0:
            return None
        x0 = cx - r
        y0 = cy - r
        w = 2.0 * r
        h = 2.0 * r
    else:
        x0 = float(roi.get("x", 0.0))
        y0 = float(roi.get("y", 0.0))
        w = float(roi.get("w", 0.0))
        h = float(roi.get("h", 0.0))
        if w <= 0 or h <= 0:
            return None

    xi = max(0, int(round(x0)))
    yi = max(0, int(round(y0)))
    x2 = min(img_w, int(round(x0 + w)))
    y2 = min(img_h, int(round(y0 + h)))
    if x2 - xi < 2 or y2 - yi < 2:
        return None
    cw = x2 - xi
    ch = y2 - yi

    mask = None
    if shape == "circle":
        yy, xx = np.ogrid[:ch, :cw]
        ccx = float(roi.get("cx", 0.0)) - xi
        ccy = float(roi.get("cy", 0.0)) - yi
        r = float(roi.get("r", 0.0))
        mask = ((xx - ccx) ** 2 + (yy - ccy) ** 2 <= r * r).astype(np.uint8) * 255
    return (xi, yi, cw, ch, mask)


def crop_with_roi(img: np.ndarray, norm_roi) -> np.ndarray:
    xi, yi, cw, ch, mask = norm_roi
    crop = img[yi:yi + ch, xi:xi + cw].copy()
    if mask is not None:
        crop[mask == 0] = 0
    return crop


def apply_roi(
    template: np.ndarray,
    test: np.ndarray,
    roi: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """按 roi 裁剪模板/待检图，返回 (tpl_crop, tst_crop, off_x, off_y)。"""
    if not roi:
        return template, test, 0, 0
    norm = normalize_roi(roi, test.shape[1], test.shape[0])
    if norm is None:
        return template, test, 0, 0
    off_x, off_y = norm[0], norm[1]
    tst = crop_with_roi(test, norm)
    tnorm = normalize_roi(roi, template.shape[1], template.shape[0])
    tpl = crop_with_roi(template, tnorm if tnorm is not None else norm)
    return tpl, tst, off_x, off_y
