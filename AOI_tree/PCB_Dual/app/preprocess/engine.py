from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

COLOR_OPTIONS = [
    "NONE",
    "GRAYSCALE",
    "BGR_RED",
    "BGR_GREEN",
    "BGR_BLUE",
    "HSV_HUE",
    "HSV_SATURATION",
    "HSV_VALUE",
    "HLS_HUE",
    "HLS_LIGHTNESS",
    "HLS_SATURATION",
    "BGR_RB",
    "BGR_RG",
    "BGR_GB",
    "MAX_MIN",
]

ENHANCE_OPTIONS = ["NONE", "LINEAR_AUTO"]
FILTER_OPTIONS = ["NONE", "GAUSSIAN", "MEDIAN", "BILATERAL"]
MORPH_OPTIONS = ["NONE", "ERODE", "DILATE", "OPEN", "CLOSE"]


@dataclass
class PreprocessParams:
    color: str = "NONE"
    enhance: str = "NONE"
    filter: str = "NONE"
    morphology: str = "NONE"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PreprocessParams":
        data = data or {}
        return cls(
            color=data.get("color", "NONE"),
            enhance=data.get("enhance", "NONE"),
            filter=data.get("filter", "NONE"),
            morphology=data.get("morphology", "NONE"),
        )


def _to_u8(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    cmin, cmax = float(channel.min()), float(channel.max())
    if cmax <= cmin:
        return np.zeros_like(channel, dtype=np.uint8)
    scaled = (channel - cmin) * (255.0 / (cmax - cmin))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _apply_color(bgr: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "NONE").upper()
    if mode == "NONE":
        return bgr.copy()

    b, g, r = cv2.split(bgr)

    if mode == "GRAYSCALE":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if mode == "BGR_RED":
        return r
    if mode == "BGR_GREEN":
        return g
    if mode == "BGR_BLUE":
        return b
    if mode == "BGR_RB":
        return cv2.absdiff(r, b)
    if mode == "BGR_RG":
        return cv2.absdiff(r, g)
    if mode == "BGR_GB":
        return cv2.absdiff(g, b)
    if mode == "MAX_MIN":
        stacked = np.stack([b, g, r], axis=-1)
        return (stacked.max(axis=-1) - stacked.min(axis=-1)).astype(np.uint8)

    if mode.startswith("HSV_"):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mapping = {
            "HSV_HUE": _to_u8(h.astype(np.float32) * (255.0 / 179.0)),
            "HSV_SATURATION": s,
            "HSV_VALUE": v,
        }
        if mode in mapping:
            return mapping[mode]

    if mode.startswith("HLS_"):
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)
        mapping = {
            "HLS_HUE": _to_u8(h.astype(np.float32) * (255.0 / 179.0)),
            "HLS_LIGHTNESS": l,
            "HLS_SATURATION": s,
        }
        if mode in mapping:
            return mapping[mode]

    return bgr.copy()


def _auto_contrast(image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
    if image.ndim == 2:
        return _auto_contrast_single(image, clip_percent)
    channels = cv2.split(image)
    return cv2.merge([_auto_contrast_single(ch, clip_percent) for ch in channels])


def _auto_contrast_single(channel: np.ndarray, clip_percent: float) -> np.ndarray:
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    total = channel.size
    clip = total * (clip_percent / 100.0)

    low = 0
    acc = 0.0
    for i, v in enumerate(hist):
        acc += v
        if acc >= clip:
            low = i
            break

    high = 255
    acc = 0.0
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= clip:
            high = i
            break

    if high <= low:
        return channel.copy()
    # 查表实现，与逐像素 float 计算结果一致，避免整图 float32 中间量
    scale = 255.0 / (high - low)
    lut = np.clip((np.arange(256, dtype=np.float32) - low) * scale, 0, 255).astype(np.uint8)
    return cv2.LUT(channel, lut)


def _apply_filter(image: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "NONE").upper()
    if mode == "NONE":
        return image
    if mode == "GAUSSIAN":
        return cv2.GaussianBlur(image, (3, 3), 0)
    if mode == "MEDIAN":
        return cv2.medianBlur(image, 3)
    if mode == "BILATERAL":
        if image.ndim == 2:
            return cv2.bilateralFilter(image, 5, 50, 50)
        return cv2.bilateralFilter(image, 5, 50, 50)
    return image


def _apply_morphology(image: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "NONE").upper()
    if mode == "NONE":
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    op = {
        "ERODE": cv2.MORPH_ERODE,
        "DILATE": cv2.MORPH_DILATE,
        "OPEN": cv2.MORPH_OPEN,
        "CLOSE": cv2.MORPH_CLOSE,
    }.get(mode)
    if op is None:
        return image
    return cv2.morphologyEx(image, op, kernel, iterations=1)


def run_preprocess(image_bgr: np.ndarray, params: PreprocessParams | dict | None) -> np.ndarray:
    """Run preprocess pipeline. Input/output are OpenCV BGR or gray uint8."""
    if image_bgr is None:
        raise ValueError("image is None")
    p = params if isinstance(params, PreprocessParams) else PreprocessParams.from_dict(params)

    out = _apply_color(image_bgr, p.color)
    if (p.enhance or "NONE").upper() == "LINEAR_AUTO":
        out = _auto_contrast(out)
    out = _apply_filter(out, p.filter)
    out = _apply_morphology(out, p.morphology)
    return out
