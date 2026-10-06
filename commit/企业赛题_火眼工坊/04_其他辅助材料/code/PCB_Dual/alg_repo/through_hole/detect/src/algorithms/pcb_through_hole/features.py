"""每个焊点 ROI 的特征提取"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import cv2

from . import config as C
from .segment import Masks, PadROI, compute_masks, edge_ring_mask


@dataclass
class PadFeatures:
    pad_id: int
    roi_area: int
    solder_area: float
    red_area: float
    bright_area: float
    circularity: float          # 锡面最大连通域圆度 (多锡)
    solidity: float             # area / convex_hull_area
    red_ratio: float            # 露铜红占 ROI 比
    ring_red_ratio: float       # 边缘环带红占比 (薄锡)
    highlight_ratio: float      # 高亮 / 锡面
    bright_closure: float       # 锡面内高亮 / 全部高亮
    lap_var: float              # 锡面内 Laplacian 方差 (纹理)
    contour: np.ndarray | None = None


def _largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 0 or area <= 0:
        return 0.0
    return 4 * math.pi * area / (peri * peri)


def extract(bgr: np.ndarray, roi: PadROI,
            masks: Masks | None = None) -> PadFeatures:
    x, y, w, h = roi.rect
    sub_bgr = bgr[y:y + h, x:x + w]
    m = masks if masks is not None else compute_masks(bgr)
    solder = m.solder[y:y + h, x:x + w]
    red = m.red[y:y + h, x:x + w]
    bright = m.bright[y:y + h, x:x + w]
    hsv = m.hsv[y:y + h, x:x + w]

    roi_area = w * h
    solder_area = float(cv2.countNonZero(solder))
    red_area = float(cv2.countNonZero(red))
    bright_area = float(cv2.countNonZero(bright))

    cnt = _largest_contour(solder)
    circ = solidity = 0.0
    if cnt is not None and cv2.contourArea(cnt) > 0:
        area = cv2.contourArea(cnt)
        circ = _circularity(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0.0

    red_ratio = red_area / max(1, roi_area)
    highlight_ratio = bright_area / max(1.0, solder_area)

    ring = edge_ring_mask(w, h)
    ring_cnt = max(1, cv2.countNonZero(ring))
    ring_red_ratio = cv2.countNonZero(cv2.bitwise_and(red, ring)) / ring_cnt

    bright_in_solder = cv2.countNonZero(cv2.bitwise_and(bright, solder))
    bright_closure = bright_in_solder / max(1.0, bright_area)

    gray = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if solder_area > 30:
        # cv2.meanStdDev(mask=...) 等价于 lap[solder > 0].var()（同为总体方差,
        # ddof=0），但走 OpenCV 内建实现，避免布尔索引产生的临时数组拷贝。
        _, stddev = cv2.meanStdDev(lap, mask=solder)
        lap_var = float(stddev[0, 0] ** 2)
    else:
        lap_var = float(lap.var())

    return PadFeatures(
        pad_id=roi.id,
        roi_area=roi_area,
        solder_area=solder_area,
        red_area=red_area,
        bright_area=bright_area,
        circularity=circ,
        solidity=solidity,
        red_ratio=red_ratio,
        ring_red_ratio=ring_red_ratio,
        highlight_ratio=highlight_ratio,
        bright_closure=bright_closure,
        lap_var=lap_var,
        contour=cnt,
    )
