"""模板管理 API：CRUD / 发布 / 绑定品类模型。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.template.model import InspectionTemplate
from app.template.store import TemplateStore

router = APIRouter()
_store = TemplateStore()


@router.get("/templates")
def list_templates(status: str | None = None, category: str | None = None) -> dict:
    items = _store.list_templates(status=status, category=category)
    return {"total": len(items), "items": [t.to_dict() for t in items]}


@router.get("/templates/{template_id}")
def get_template(template_id: str) -> dict:
    tpl = _store.load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    return tpl.to_dict()


@router.post("/templates")
def create_template(payload: dict) -> dict:
    """从标准图创建模板（草稿）。"""
    path = payload.get("standard_image_path") or payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="缺少 standard_image_path")
    try:
        tpl = _store.create_from_standard_image(
            path,
            display_name=payload.get("display_name") or "",
            category=payload.get("category") or "",
            notes=payload.get("notes") or "",
            algorithm_ids=payload.get("algorithm_ids"),
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return tpl.to_dict()


@router.put("/templates/{template_id}")
def update_template(template_id: str, payload: dict) -> dict:
    """更新模板配置（engine_mode / model_category / 检测项参数等）。"""
    tpl = _store.load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    if "engine_mode" in payload:
        mode = str(payload["engine_mode"])
        if mode not in ("traditional", "feature", "dual"):
            raise HTTPException(status_code=400, detail="engine_mode 须为 traditional/feature/dual")
        tpl.engine_mode = mode
    if "model_category" in payload:
        tpl.model_category = str(payload["model_category"] or "")
    if "category" in payload:
        tpl.category = str(payload["category"] or "")
    if "display_name" in payload:
        tpl.display_name = str(payload["display_name"] or tpl.id)
    if "algorithm_ids" in payload:
        tpl.set_algorithm_ids(list(payload["algorithm_ids"]))
    if "params" in payload and isinstance(payload["params"], dict):
        for alg_id, p in payload["params"].items():
            if isinstance(p, dict):
                tpl.upsert_params(alg_id, p)
    _store.save(tpl)
    return tpl.to_dict()


@router.post("/templates/{template_id}/publish")
def publish_template(template_id: str) -> dict:
    tpl = _store.load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    try:
        _store.publish(tpl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return tpl.to_dict()


@router.get("/templates/{template_id}/checklist")
def template_checklist(template_id: str) -> dict:
    tpl = _store.load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    issues = _store.publish_checklist(tpl)
    blockers = [i for i in issues if i.startswith("[阻断]")]
    return {"issues": issues, "publishable": not blockers}
