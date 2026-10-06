"""检测结果公共类型。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segment import PadROI


@dataclass
class Detection:
    defect_id: int
    score: float
    bbox: tuple
    reason: str
    pad_id: int = -1

    def __post_init__(self):
        self.bbox = tuple(int(v) for v in self.bbox)
        self.pad_id = int(self.pad_id)


def bbox_from_mask(mask: np.ndarray, roi: PadROI) -> tuple:
    """Mask 外接框映射回整图坐标；空 Mask 时回退到 ROI 矩形。"""
    x, y, w, h = roi.rect
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return roi.rect
    bx, by = xs.min(), ys.min()
    return (x + int(bx), y + int(by), int(xs.max() - bx + 1), int(ys.max() - by + 1))
