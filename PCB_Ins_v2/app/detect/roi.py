"""从模板 region_sets / Context 提取标准 ROI，供模块 is_ready / run 使用。"""
from __future__ import annotations

from typing import Any, Iterable

from app.detect.contract import Roi
from app.template.regions import region_targets_for_kind


def rois_from_region_sets(region_sets: dict[str, list[dict]] | None, region_kind: str | None) -> list[Roi]:
    if not region_kind or region_kind == "none":
        return []
    sets = region_sets or {}
    layers = region_targets_for_kind(region_kind)
    if not layers:
        layers = [region_kind]
    out: list[Roi] = []
    for layer in layers:
        for i, shape in enumerate(sets.get(layer) or []):
            if not isinstance(shape, dict):
                continue
            out.append(
                Roi(
                    id=str(shape.get("id") or f"{layer}:{i}"),
                    kind=region_kind,
                    layer=layer,
                    shape=dict(shape),
                )
            )
    return out


def filter_rois(rois: Iterable[Roi], *, kind: str | None = None, layer: str | None = None) -> list[Roi]:
    out = []
    for roi in rois:
        if kind and roi.kind != kind:
            continue
        if layer and roi.layer != layer:
            continue
        out.append(roi)
    return out


def shapes_of(rois: Iterable[Roi], *, kind: str | None = None, layer: str | None = None) -> list[dict[str, Any]]:
    return [dict(r.shape) for r in filter_rois(rois, kind=kind, layer=layer)]


def rects_of(rois: Iterable[Roi], *, kind: str | None = None, layer: str | None = None) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for shape in shapes_of(rois, kind=kind, layer=layer):
        if str(shape.get("shape") or "rect") != "rect":
            continue
        try:
            x, y, w, h = (int(round(float(shape[k]))) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            rects.append((x, y, w, h))
    return rects
