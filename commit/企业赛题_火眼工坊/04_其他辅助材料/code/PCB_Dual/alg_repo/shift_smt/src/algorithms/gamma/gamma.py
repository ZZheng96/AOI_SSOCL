"""Gamma 归一化工具（上位机调参控件）。

`auto_gamma` 从原算法 `_shift_core.py` 中移出：
- 属于上位机调参控件（亮度归一化预处理），不是算法模块；
- 供上位机/调参工具（tuner、color_extractor）单独调用；
- 算法运行时 `use_autogamma=True` 时也从本模块引用，保证行为一致。
"""

from __future__ import annotations

import cv2
import numpy as np


def auto_gamma(image: np.ndarray, target: float = 0.35) -> np.ndarray:
    """
    自动 gamma 校正，归一化图像平均亮度到目标值。

    通过调整 gamma 使灰度均值接近 target（0~1），用于消除不同光照条件差异。
    gamma > 1 变暗，gamma < 1 变亮。

    Args:
        image: BGR 图像
        target: 目标平均亮度（0~1），默认 0.35（偏暗，适合底座检测）

    Returns:
        gamma 校正后的 BGR 图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean() / 255.0
    if mean_val < 0.01 or mean_val > 0.99:
        return image
    gamma = np.clip(np.log(target) / np.log(mean_val), 0.3, 3.0)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)
