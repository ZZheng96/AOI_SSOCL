"""元件本体检测的重点区域框选。

在标准图上框选若干矩形/圆形 ROI；检测时按 ROI 裁剪标准图与测试图后送检。
未框选时退回整图检测（与参考验证台一致）。
"""
from __future__ import annotations

from typing import Any

from app.calibration.store import load_calib, update_calib


def _normalize_roi(roi: Any) -> dict | None:
    if not isinstance(roi, dict) or not roi:
        return None
    shape = str(roi.get("shape", "rect") or "rect").lower()
    if shape == "circle":
        try:
            r = float(roi["r"])
            if r <= 0:
                return None
            return {
                "shape": "circle",
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "r": r,
            }
        except (KeyError, TypeError, ValueError):
            return None
    try:
        w = float(roi["w"])
        h = float(roi["h"])
        if w <= 0 or h <= 0:
            return None
        return {
            "shape": "rect",
            "x": float(roi["x"]),
            "y": float(roi["y"]),
            "w": w,
            "h": h,
        }
    except (KeyError, TypeError, ValueError):
        return None


def get_body_rois(template_name: str | None) -> list[dict]:
    data = load_calib(template_name)
    raw = data.get("body_rois") or []
    out: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            n = _normalize_roi(item)
            if n is not None:
                out.append(n)
    return out


def set_body_rois(template_name: str | None, rois: list[dict] | None) -> None:
    clean = []
    for r in rois or []:
        n = _normalize_roi(r)
        if n is not None:
            clean.append(n)
    update_calib(template_name, body_rois=clean)
