"""CAD 导入 / 拼板复制 / FOV 划分 API（P2：借鉴 Java AOI 编程设计器/CAD流程）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.cad import (compute_fov_grid, duplicate_panels, parse_cad_file)
from app.template.store import TemplateStore

router = APIRouter()


def _load(template_id: str):
    tpl = TemplateStore().load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    return tpl


@router.post("/templates/{template_id}/cad-import")
def cad_import(template_id: str, payload: dict) -> dict:
    """导入 CAD 器件表到模板。

    body: {cad_path, field_mapping?}
    """
    tpl = _load(template_id)
    cad_path = payload.get("cad_path")
    if not cad_path:
        raise HTTPException(status_code=400, detail="需要 cad_path")
    try:
        items = parse_cad_file(cad_path, payload.get("field_mapping"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"CAD 解析失败: {exc}")
    if not items:
        raise HTTPException(status_code=400, detail="CAD 文件无有效器件（需要 位号+x_mm+y_mm 列）")
    tpl.cad_items = items
    TemplateStore().save(tpl)
    return {"template_id": template_id, "n_items": len(items),
            "sample": items[:3]}


@router.get("/templates/{template_id}/cad")
def get_cad(template_id: str) -> dict:
    tpl = _load(template_id)
    return {"template_id": template_id, "n_items": len(tpl.cad_items),
            "items": tpl.cad_items, "panels": tpl.panels,
            "fov_grid": tpl.fov_grid}


@router.post("/templates/{template_id}/panels")
def build_panels(template_id: str, payload: dict) -> dict:
    """拼板复制。

    body: {panel_dx_mm, panel_dy_mm, cols, rows, rotation}
    """
    tpl = _load(template_id)
    if not tpl.cad_items:
        raise HTTPException(status_code=400, detail="模板无 CAD 器件，先 cad-import")
    try:
        duplicated = duplicate_panels(
            tpl.cad_items,
            panel_dx_mm=float(payload["panel_dx_mm"]),
            panel_dy_mm=float(payload["panel_dy_mm"]),
            cols=int(payload.get("cols", 1)),
            rows=int(payload.get("rows", 1)),
            rotation=float(payload.get("rotation", 0.0)),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"缺少参数: {exc}")
    tpl.panels = {"panel_dx_mm": float(payload["panel_dx_mm"]),
                  "panel_dy_mm": float(payload["panel_dy_mm"]),
                  "cols": int(payload.get("cols", 1)),
                  "rows": int(payload.get("rows", 1)),
                  "rotation": float(payload.get("rotation", 0.0))}
    tpl.cad_items = duplicated
    TemplateStore().save(tpl)
    return {"template_id": template_id, "n_items": len(duplicated),
            "panels": tpl.panels}


@router.post("/templates/{template_id}/fov")
def build_fov(template_id: str, payload: dict) -> dict:
    """FOV 网格划分。

    body: {board_w_mm, board_h_mm, fov_w_mm, fov_h_mm, overlap?}
    """
    tpl = _load(template_id)
    try:
        grid = compute_fov_grid(
            float(payload["board_w_mm"]), float(payload["board_h_mm"]),
            float(payload["fov_w_mm"]), float(payload["fov_h_mm"]),
            float(payload.get("overlap", 0.1)),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"缺少参数: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    tpl.fov_grid = grid
    TemplateStore().save(tpl)
    return {"template_id": template_id, "n_fov": len(grid),
            "grid": grid[:5], "grid_full": grid}
