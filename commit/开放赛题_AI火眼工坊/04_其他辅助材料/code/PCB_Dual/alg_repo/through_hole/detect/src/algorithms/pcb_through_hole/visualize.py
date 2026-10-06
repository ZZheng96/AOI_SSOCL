"""结果可视化：OK/NG/待复检标注图（含 PIL 中文绘制）。"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C
from .pipeline import InspectResult

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = ("", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = font
                return font
            except OSError:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _text_size(text: str, font) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_by_char(text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        if _text_size(trial, font)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines or [""]


def _wrap_line(text: str, font, max_width: int) -> list[str]:
    if _text_size(text, font)[0] <= max_width:
        return [text]
    return _wrap_by_char(text, font, max_width)


def _text_block_size(lines: list[str], font_size: int, line_gap: int = 2) -> tuple[int, int]:
    font = _load_font(font_size)
    w = h = 0
    for i, line in enumerate(lines):
        tw, th = _text_size(line, font)
        w = max(w, tw)
        h += th + (line_gap if i else 0)
    return w, h


def _draw_text(
    bgr: np.ndarray, text: str, pos: tuple[int, int], *,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 16, bg_bgr: tuple[int, int, int] | None = None, pad: int = 2,
) -> np.ndarray:
    font = _load_font(font_size)
    tw, th = _text_size(text, font)
    x, y = pos
    if bg_bgr is not None:
        cv2.rectangle(bgr, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), bg_bgr, -1)
        x += pad
        y += pad
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ImageDraw.Draw(pil).text((x, y), text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_text_lines(
    bgr: np.ndarray, lines: list[str], pos: tuple[int, int], *,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 16, line_gap: int = 2,
) -> np.ndarray:
    if not lines:
        return bgr
    font = _load_font(font_size)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    fill = (color_bgr[2], color_bgr[1], color_bgr[0])
    x0, y = pos
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill)
        _, th = _text_size(line, font)
        y += th + line_gap
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _layout_status_banner(
    text: str, img_w: int, *, img_h: int | None = None, font_size: int = 22, margin: int = 2,
) -> tuple[list[str], int, int]:
    max_w = max(1, img_w - 2 * margin)
    max_h = max(12, int(img_h * 0.38)) if img_h is not None else None
    if text.startswith("NG-") and "、" in text[3:]:
        parts = text[3:].split("、")
        raw_lines = [f"NG-{parts[0]}"] + parts[1:]
    else:
        raw_lines = [text]
    for fs in range(font_size, 8, -1):
        font = _load_font(fs)
        wrapped: list[str] = []
        for line in raw_lines:
            wrapped.extend(_wrap_line(line, font, max_w))
        _, block_h = _text_block_size(wrapped, fs, line_gap=2)
        if max_h is None or block_h <= max_h:
            return wrapped, fs, block_h + margin
    return [text[: max(1, img_w // 8)]], 8, 12


def _layout_box_label(
    lines: list[str], *, box_x: int, box_y: int, box_w: int, box_h: int,
    img_w: int, img_h: int, font_size: int = 14, line_gap: int = 2,
    reserved_top: int = 0, margin: int = 2,
) -> tuple[list[str], int, int, int]:
    max_w = max(20, min(img_w - 2 * margin, int(img_w * 0.92)))
    for fs in range(font_size, 8, -1):
        font = _load_font(fs)
        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(_wrap_line(line, font, max_w))
        block_w, block_h = _text_block_size(wrapped, fs, line_gap)
        if block_w <= max_w:
            break
    else:
        fs = 8
        wrapped = [lines[0][:12] + "…"] if lines else [""]
        block_w, block_h = _text_block_size(wrapped, fs, line_gap)
    candidates = [
        (box_x, box_y - block_h - margin),
        (box_x, box_y + box_h + margin),
        (box_x + box_w + margin, box_y),
        (box_x - block_w - margin, box_y),
    ]
    lx, ly = candidates[0]
    for cx, cy in candidates:
        if (margin <= cx <= img_w - block_w - margin and
                reserved_top + margin <= cy <= img_h - block_h - margin):
            lx, ly = cx, cy
            break
    lx = max(margin, min(lx, img_w - block_w - margin))
    ly = max(reserved_top + margin, min(ly, img_h - block_h - margin))
    return wrapped, lx, ly, fs


def _resolve_label_overlap(
    lx: int, ly: int, block_w: int, block_h: int,
    placed: list[tuple[int, int, int, int]], img_h: int, margin: int = 2,
) -> int:
    y = ly
    for _ in range(20):
        overlap = False
        for px, py, pw, ph in placed:
            if not (lx + block_w < px or lx > px + pw or y + block_h < py or y > py + ph):
                overlap = True
                y = py + ph + margin
                break
        if not overlap:
            break
    return min(y, max(margin, img_h - block_h - margin))


def _clamp_bbox(x, y, w, h, img_w, img_h):
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def draw_annotation(result: InspectResult):
    """在原图上绘制 OK/NG/待复检标注与缺陷框。"""
    base = result.original.copy()
    img_h, img_w = base.shape[:2]
    status = result.status or ("NG" if result.is_defect else "OK")

    if status == "待复检":
        reason = result.review_reason or "数据可靠性不足"
        cov_txt = f" cov={result.coverage:.2f}" if result.coverage is not None else ""
        status_lines, status_fs, _ = _layout_status_banner(
            f"待复检-{reason}{cov_txt}", img_w, img_h=img_h)
        return _draw_text_lines(
            base, status_lines, (2, 2), color_bgr=(160, 160, 160), font_size=status_fs)

    is_ng = status == "NG"
    if is_ng:
        names = "、".join(C.DEFECT_NAMES[i] for i in result.defect_ids)
        status_lines, status_fs, reserved_top = _layout_status_banner(
            f"NG-{names}", img_w, img_h=img_h)
        base = _draw_text_lines(
            base, status_lines, (2, 2), color_bgr=(0, 0, 255), font_size=status_fs)
    else:
        reserved_top = 0
        status_lines, status_fs = [], 14
        base = _draw_text(base, "OK", (8, 8), color_bgr=(0, 200, 0), font_size=22)

    placed: list[tuple[int, int, int, int]] = []
    if is_ng:
        sbw, sbh = _text_block_size(status_lines, status_fs, 2)
        placed.append((2, 2, sbw, sbh))

    meta = result.align_meta
    for d in result.detections:
        color = C.DEFECT_COLORS.get(d.defect_id, (0, 255, 0))
        x, y, w, h = meta.map_bbox_to_original(d.bbox)
        x, y, w, h = _clamp_bbox(x, y, w, h, img_w, img_h)
        cv2.rectangle(base, (x, y), (x + w, y + h), color, 2)
        lines = [f"{d.defect_id}-{C.DEFECT_NAMES[d.defect_id]}"]
        if d.reason:
            lines.append(d.reason)
        lines.append(f"score={d.score:.2f}")
        wrapped, lx, ly, fs = _layout_box_label(
            lines, box_x=x, box_y=y, box_w=w, box_h=h,
            img_w=img_w, img_h=img_h, reserved_top=reserved_top)
        block_w, block_h = _text_block_size(wrapped, fs, 2)
        ly = _resolve_label_overlap(lx, ly, block_w, block_h, placed, img_h)
        base = _draw_text_lines(base, wrapped, (lx, ly), color_bgr=color, font_size=fs)
        placed.append((lx, ly, block_w, block_h))
    return base
