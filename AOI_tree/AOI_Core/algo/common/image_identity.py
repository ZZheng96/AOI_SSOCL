"""轻量近重复检查：缩略图差异与感知哈希同时通过，避免仅按路径去重。"""
import cv2
import numpy as np


def image_signature(image):
    rgb = np.asarray(image)
    thumb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY) if thumb.ndim == 3 else thumb
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return {"pixels": thumb.astype(np.uint8).reshape(-1).tolist(),
            "dhash": np.packbits(bits).tolist()}


def near_duplicate(a, b):
    if not a or not b or len(a["pixels"]) != len(b["pixels"]):
        return False
    distance = sum((int(x) ^ int(y)).bit_count()
                   for x, y in zip(a["dhash"], b["dhash"]))
    error = np.abs(np.asarray(a["pixels"], dtype=np.float32)
                   - np.asarray(b["pixels"], dtype=np.float32)).mean()
    return distance <= 6 and error <= 2.0
