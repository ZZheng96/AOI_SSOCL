import os
import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread 不支持非ASCII路径 → np.fromfile + cv2.imdecode"""
    try:
        return cv2.imread(path, flags)
    except Exception:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)


def imwrite_unicode(path, image, params=None):
    """cv2.imwrite 不支持非ASCII路径 → cv2.imencode + np.tofile"""
    try:
        return cv2.imwrite(path, image, params or [])
    except Exception:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = '.png'
        params = params or []
        ok, buf = cv2.imencode(ext, image, params)
        if ok:
            buf.tofile(path)
        return ok
