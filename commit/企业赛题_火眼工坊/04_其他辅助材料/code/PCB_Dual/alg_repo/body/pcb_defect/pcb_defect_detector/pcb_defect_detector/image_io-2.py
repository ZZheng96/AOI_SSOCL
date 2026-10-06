from __future__ import annotations

import os

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> "np.ndarray | None":
    """读取图片，兼容中文路径。失败返回 None。

    路径含非 ASCII 字符时，cv2.imread 在 Windows 下会乱码并打印警告，
    因此直接走 np.fromfile + cv2.imdecode；纯 ASCII 路径才用更快的 cv2.imread。
    """
    try:
        if all(ord(c) < 128 for c in str(path)):
            img = cv2.imread(path, flags)
            if img is not None:
                return img
    except Exception:
        pass
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str, image: np.ndarray, params=None) -> bool:
    """写入图片，兼容中文路径。"""
    params = list(params) if params else []
    try:
        ok = cv2.imwrite(path, image, params)
        if ok:
            return True
    except Exception:
        pass
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image, params)
    if ok:
        buf.tofile(path)
        return True
    return False
