from __future__ import annotations

import os

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> "np.ndarray | None":
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
    img = cv2.imdecode(data, flags)
    if img is not None:
        return img
    # 2026-08-31 评审修复：OpenCV 读不出的 RGBA/特殊编码用 PIL 兜底，
    # 避免产线相机 RGBA 出图时链路静默 None。
    try:
        import io

        from PIL import Image
        with Image.open(io.BytesIO(bytes(data))) as im:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.array(im.convert("L"))
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imwrite_unicode(path: str, image: np.ndarray, params=None) -> bool:
    params = list(params) if params else []
    path = str(path)
    if all(ord(c) < 128 for c in path):
        try:
            if cv2.imwrite(path, image, params):
                return True
        except Exception:
            pass
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True
