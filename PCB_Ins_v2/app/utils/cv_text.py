"""OpenCV 图像上的中文/Unicode 文字绘制（Hershey 字体无法渲染汉字）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
)


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            try:
                return ImageFont.truetype(str(path), size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def put_text_bgr(
    image_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    font_size: int = 16,
    thickness_box: bool = False,
) -> tuple[int, int]:
    """在 BGR 图上绘制文字，返回文字宽高。``org`` 为文字左下角基线附近（与 cv2.putText 接近）。"""
    if not text:
        return 0, 0
    font = _load_font(max(10, int(font_size)))
    # PIL 用 RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = int(org[0]), int(org[1])
    # org 近似为基线；改为左上角绘制以便与框对齐
    top = max(0, y - th)
    left = max(0, x)
    if thickness_box:
        pad = 2
        draw.rectangle(
            (left - pad, top - pad, left + tw + pad, top + th + pad),
            fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
        )
        fill = (255, 255, 255)
    else:
        fill = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((left, top), text, font=font, fill=fill)
    image_bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return tw, th
