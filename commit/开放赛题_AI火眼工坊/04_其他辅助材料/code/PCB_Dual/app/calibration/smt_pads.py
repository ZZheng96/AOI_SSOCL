"""贴片锡焊检测的焊盘框(pad_frames)标定：在标准图上框选若干矩形，
每个矩形对应一个焊盘。该算法没有焊盘框无法运行，未标定时上层适配器会
判定为 SKIP。"""
from __future__ import annotations

from app.calibration.store import load_calib, update_calib


def get_smt_pads(template_name: str | None) -> list[tuple[int, int, int, int]]:
    data = load_calib(template_name)
    pads = data.get("smt_pads") or []
    out = []
    for p in pads:
        try:
            x, y, w, h = (int(round(v)) for v in p)
            if w > 0 and h > 0:
                out.append((x, y, w, h))
        except (TypeError, ValueError):
            continue
    return out


def set_smt_pads(template_name: str | None, pads: list[tuple[float, float, float, float]]) -> None:
    clean = [[int(round(x)), int(round(y)), int(round(w)), int(round(h))] for (x, y, w, h) in pads]
    update_calib(template_name, smt_pads=clean)


def _get_rects(template_name: str | None, key: str) -> list[tuple[int, int, int, int]]:
    data = load_calib(template_name)
    rects = data.get(key) or []
    out = []
    for r in rects:
        try:
            x, y, w, h = (int(round(v)) for v in r)
            if w > 0 and h > 0:
                out.append((x, y, w, h))
        except (TypeError, ValueError):
            continue
    return out


def get_smt_toe(template_name: str | None) -> list[tuple[int, int, int, int]]:
    """gull-wing 引脚"虚焊"判定用的 toe（脚尖）标注框，可选。"""
    return _get_rects(template_name, "smt_toe")


def set_smt_toe(template_name: str | None, rects: list[tuple[float, float, float, float]]) -> None:
    clean = [[int(round(x)), int(round(y)), int(round(w)), int(round(h))] for (x, y, w, h) in rects]
    update_calib(template_name, smt_toe=clean)


def get_smt_rim(template_name: str | None) -> list[tuple[int, int, int, int]]:
    """gull-wing 引脚"虚焊"判定用的 rim（焊盘边缘）标注框，可选。"""
    return _get_rects(template_name, "smt_rim")


def set_smt_rim(template_name: str | None, rects: list[tuple[float, float, float, float]]) -> None:
    clean = [[int(round(x)), int(round(y)), int(round(w)), int(round(h))] for (x, y, w, h) in rects]
    update_calib(template_name, smt_rim=clean)
