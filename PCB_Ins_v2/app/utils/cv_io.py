from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read image supporting non-ASCII paths (e.g. Chinese on Windows).

    2026-08-31 评审修复：OpenCV 读不出的 RGBA/特殊编码用 PIL 兜底转换，
    避免产线相机 RGBA 出图时链路返回 None 静默失败。"""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    if img is not None:
        return img
    try:
        import io

        from PIL import Image
        with Image.open(io.BytesIO(bytes(data))) as im:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.array(im.convert("L"))
            arr = np.array(im.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001 兜底失败按原语义返回 None
        return None


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Write image supporting non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except OSError:
        return False
