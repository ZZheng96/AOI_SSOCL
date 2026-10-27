"""图像 IO：兼容中文路径（cv2.imread 在非 ASCII 路径下会静默失败）"""
import numpy as np
import cv2


def load_image(path, max_side=None):
    """-> HxWx3 uint8 RGB；max_side 可选限长边（M0 全分辨率，不用）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)
    return img
