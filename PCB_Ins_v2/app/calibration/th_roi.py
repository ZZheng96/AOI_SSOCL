"""插件焊点检测的锡面区域标定。

- 未画框：按 ``shape_choice`` 自动（ellipse / contour）。
- 画 1 个框：人工单焊点 ``solder_roi``，跑盘内四类缺陷。
- 画 ≥2 个框：多焊点 ``bridge_joints``，算法只跑连锡。
"""
from __future__ import annotations

from typing import Any

from app.calibration.store import load_calib, update_calib

_DEFAULT_SHAPE_CHOICE = "ellipse"  # "ellipse" | "contour"


def _normalize_roi(roi: Any) -> dict | None:
    if not isinstance(roi, dict) or not roi:
        return None
    shape = str(roi.get("shape", "rect") or "rect").lower()
    if shape == "circle":
        try:
            return {
                "shape": "circle",
                "cx": float(roi["cx"]),
                "cy": float(roi["cy"]),
                "r": float(roi["r"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return {
            "shape": "rect",
            "x": float(roi["x"]),
            "y": float(roi["y"]),
            "w": float(roi["w"]),
            "h": float(roi["h"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def get_th_roi(template_name: str | None) -> dict[str, Any]:
    data = load_calib(template_name)
    th = data.get("th_roi") or {}
    pads_raw = th.get("pads")
    pads: list[dict] = []
    if isinstance(pads_raw, list):
        for item in pads_raw:
            n = _normalize_roi(item)
            if n is not None:
                pads.append(n)
    # 兼容旧数据：只有单个 roi
    if not pads:
        legacy = _normalize_roi(th.get("roi"))
        if legacy is not None:
            pads = [legacy]
    return {
        "shape_choice": th.get("shape_choice") or _DEFAULT_SHAPE_CHOICE,
        "roi": pads[0] if len(pads) == 1 else None,
        "pads": pads,
    }


def get_th_pad_count(template_name: str | None) -> int:
    return len(get_th_roi(template_name).get("pads") or [])


def set_th_shape_choice(template_name: str | None, shape_choice: str) -> None:
    th = get_th_roi(template_name)
    th["shape_choice"] = shape_choice if shape_choice in ("ellipse", "contour") else _DEFAULT_SHAPE_CHOICE
    update_calib(
        template_name,
        th_roi={
            "shape_choice": th["shape_choice"],
            "pads": th.get("pads") or [],
            "roi": th.get("roi"),
        },
    )


def set_th_manual_roi(template_name: str | None, roi: dict | None) -> None:
    """兼容旧接口：写入 0/1 个框。"""
    pads = [_normalize_roi(roi)] if roi else []
    pads = [p for p in pads if p is not None]
    set_th_pads(template_name, pads)


def set_th_pads(template_name: str | None, pads: list[dict] | None) -> None:
    clean = []
    for p in pads or []:
        n = _normalize_roi(p)
        if n is not None:
            clean.append(n)
    th = get_th_roi(template_name)
    update_calib(
        template_name,
        th_roi={
            "shape_choice": th.get("shape_choice") or _DEFAULT_SHAPE_CHOICE,
            "pads": clean,
            "roi": clean[0] if len(clean) == 1 else None,
        },
    )


def effective_solder_mode(template_name: str | None) -> str:
    th = get_th_roi(template_name)
    if th.get("pads"):
        return "roi"
    return th.get("shape_choice", _DEFAULT_SHAPE_CHOICE)


def th_run_config_from_pads(pads: list[dict] | None, *, shape_choice: str | None = None) -> dict[str, Any]:
    """由焊点框列表生成算法输入，不读全局标定文件。"""
    pads = list(pads or [])
    if len(pads) >= 2:
        joints = []
        for i, p in enumerate(pads):
            shape = p.get("shape", "rect")
            if shape == "circle":
                params = {"cx": p["cx"], "cy": p["cy"], "r": p["r"]}
                jtype = "circle"
            else:
                params = {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]}
                jtype = "rect"
            joints.append({
                "id": i + 1,
                "type": jtype,
                "params": params,
                "role": "pad",
                "label": f"pad{i + 1}",
            })
        return {
            "solder_mode": "roi",
            "solder_roi": pads[0],
            "bridge_joints": joints,
        }
    if len(pads) == 1:
        return {
            "solder_mode": "roi",
            "solder_roi": pads[0],
            "bridge_joints": None,
        }
    return {
        "solder_mode": shape_choice or _DEFAULT_SHAPE_CHOICE,
        "solder_roi": None,
        "bridge_joints": None,
    }


def th_run_config(template_name: str | None) -> dict[str, Any]:
    """兼容：从 legacy 标定文件生成 solder_mode / solder_roi / bridge_joints。"""
    th = get_th_roi(template_name)
    return th_run_config_from_pads(th.get("pads") or [], shape_choice=th.get("shape_choice"))
