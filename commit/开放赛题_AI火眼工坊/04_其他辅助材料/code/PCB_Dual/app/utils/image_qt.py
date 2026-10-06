from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def make_diff_heatmap(std_bgr: np.ndarray, test_bgr: np.ndarray) -> np.ndarray:
    """标准图/测试图差异热力图（P2 差异可视化）：尺寸不一致时先把标准图
    缩放到测试图大小，再取灰度差分并做 JET 伪彩映射。"""
    h, w = test_bgr.shape[:2]
    std_r = std_bgr
    if std_bgr.shape[:2] != (h, w):
        std_r = cv2.resize(std_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    gray_std = cv2.cvtColor(std_r, cv2.COLOR_BGR2GRAY) if std_r.ndim == 3 else std_r
    gray_test = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY) if test_bgr.ndim == 3 else test_bgr
    diff = cv2.absdiff(gray_std, gray_test)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image is None:
        return QPixmap()
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
