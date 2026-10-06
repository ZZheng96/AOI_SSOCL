"""Unicode 安全的图像读写（Windows 中文路径 cv2.imread 会静默失败）。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """中文路径安全的图像读取，失败返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """中文路径安全的图像写入。"""
    try:
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False
