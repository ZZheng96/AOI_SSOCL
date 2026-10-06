"""模板检测区域：region_sets ↔ legacy calibration 互转。

关联链：标准图 → 模板 → 检测项(algorithm) → 区域种类(region_kind) → region_sets。
适配器仍读 template_calib，因此保存时把 region_sets 同步为 calibration。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def region_kind_for(algorithm_id: str) -> str | None:
    try:
        from app.detect.catalog import get_catalog

        kind = get_catalog().region_kind(algorithm_id)
        if kind and kind != "none":
            return kind
        if get_catalog().get(algorithm_id) is not None:
            return None
    except Exception:
        pass
    if algorithm_id.startswith("body_"):
        return "body"
    if algorithm_id.startswith("tht_"):
        return "th"
    if algorithm_id.startswith("solder_"):
        return "smt"
    if algorithm_id.startswith("board_"):
        return "gold"
    return None


def region_targets_for_kind(kind: str | None) -> list[str]:
    """UI / 存储用的具体图层键。smt 拆成 pads/toe/rim。"""
    if kind == "body":
        return ["body"]
    if kind == "th":
        return ["th"]
    if kind == "smt":
        return ["smt_pads", "smt_toe", "smt_rim"]
    if kind == "gold":
        return ["gold"]
    if kind == "shift":
        return ["shift"]
    return []


def region_sets_from_calibration(calibration: dict[str, Any] | None) -> dict[str, list[dict]]:
    calib = dict(calibration or {})
    out: dict[str, list[dict]] = {}

    body = calib.get("body_rois") or []
    if isinstance(body, list) and body:
        out["body"] = [dict(x) for x in body if isinstance(x, dict)]

    th = calib.get("th_roi") or {}
    if isinstance(th, dict):
        pads = th.get("pads") or []
        if isinstance(pads, list) and pads:
            out["th"] = [dict(x) for x in pads if isinstance(x, dict)]

    for key in ("smt_pads", "smt_toe", "smt_rim"):
        raw = calib.get(key) or []
        if isinstance(raw, list) and raw:
            shapes = []
            for item in raw:
                if isinstance(item, dict) and item.get("shape"):
                    shapes.append(dict(item))
                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    x, y, w, h = item[:4]
                    shapes.append({"shape": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
            if shapes:
                out[key] = shapes

    gold = calib.get("gold_seed") or {}
    if isinstance(gold, dict):
        seeds = gold.get("seeds") or []
        if isinstance(seeds, list) and seeds:
            pts = []
            for s in seeds:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    pts.append({"shape": "point", "x": float(s[0]), "y": float(s[1])})
                elif isinstance(s, dict) and "x" in s and "y" in s:
                    pts.append({"shape": "point", "x": float(s["x"]), "y": float(s["y"])})
            if pts:
                out["gold"] = pts

    shift = calib.get("shift_base") or []
    if isinstance(shift, list) and shift:
        shapes = []
        for item in shift:
            if isinstance(item, dict) and item.get("shape"):
                shapes.append(dict(item))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                x, y, w, h = item[:4]
                shapes.append({"shape": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
        if shapes:
            out["shift"] = shapes
    return out


def calibration_from_region_sets(
    region_sets: dict[str, list[dict]] | None,
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用 region_sets 覆盖生成 legacy calibration，保留未建模字段（如 gold tol）。"""
    calib = deepcopy(base) if base else {}
    sets = region_sets or {}

    if "body" in sets:
        calib["body_rois"] = [dict(s) for s in sets.get("body") or []]

    if "th" in sets:
        pads = [dict(s) for s in sets.get("th") or []]
        prev = dict(calib.get("th_roi") or {}) if isinstance(calib.get("th_roi"), dict) else {}
        prev["pads"] = pads
        if pads:
            prev["roi"] = dict(pads[0])
            shape0 = str(pads[0].get("shape") or "rect")
            prev["shape_choice"] = "ellipse" if shape0 == "circle" else "rect"
        calib["th_roi"] = prev

    for key in ("smt_pads", "smt_toe", "smt_rim"):
        if key in sets:
            rects = []
            for s in sets.get(key) or []:
                if s.get("shape") == "rect":
                    rects.append((float(s["x"]), float(s["y"]), float(s["w"]), float(s["h"])))
            calib[key] = rects

    if "gold" in sets:
        seeds = []
        for s in sets.get("gold") or []:
            if s.get("shape") == "point":
                seeds.append([float(s["x"]), float(s["y"])])
        prev = dict(calib.get("gold_seed") or {}) if isinstance(calib.get("gold_seed"), dict) else {}
        prev["seeds"] = seeds
        prev.setdefault("tol", 1.0)
        prev.setdefault("ranges", [])
        calib["gold_seed"] = prev

    if "shift" in sets:
        rects = []
        for s in sets.get("shift") or []:
            if s.get("shape") == "rect":
                rects.append((float(s["x"]), float(s["y"]), float(s["w"]), float(s["h"])))
        calib["shift_base"] = rects

    return calib


def count_regions(region_sets: dict[str, list[dict]] | None) -> int:
    return sum(len(v) for v in (region_sets or {}).values())
