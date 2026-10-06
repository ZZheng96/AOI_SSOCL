"""贴片元件移位检测的底座框（base 框）标定。

ShiftSmtAllAlg（shift2）需要底座框 ``roi_bbox``（原图坐标 x,y,w,h）作为
滑窗模板尺寸与移位判定参考框。每个底座框对应一个待检元件，支持多个。
检测时按框逐个送检；未标定时适配器判定为 SKIP。
"""
from __future__ import annotations

from app.calibration.store import load_calib, update_calib


def get_shift_bases(template_name: str | None) -> list[tuple[int, int, int, int]]:
    data = load_calib(template_name)
    raw = data.get("shift_base") or []
    out: list[tuple[int, int, int, int]] = []
    for p in raw:
        try:
            if isinstance(p, dict):
                x, y, w, h = float(p["x"]), float(p["y"]), float(p["w"]), float(p["h"])
            else:
                x, y, w, h = (float(v) for v in p)
        except (KeyError, TypeError, ValueError):
            continue
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w > 0 and h > 0:
            out.append((x, y, w, h))
    return out


def set_shift_bases(template_name: str | None, bases: list[tuple[float, float, float, float]]) -> None:
    clean = [[int(round(x)), int(round(y)), int(round(w)), int(round(h))]
             for (x, y, w, h) in bases if float(w) > 0 and float(h) > 0]
    update_calib(template_name, shift_base=clean)
