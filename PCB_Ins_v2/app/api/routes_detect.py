"""检测 API：双检（传统+特征）/ 单传统 / 单特征。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.inspect.service import InspectService
from app.template.store import TemplateStore

router = APIRouter()
_service = InspectService()
_store = TemplateStore()


def _load_template(template_id: str):
    tpl = _store.load(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    return tpl


def _resolve_image(image_path: str) -> str:
    p = Path(image_path)
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"测试图不存在: {image_path}")
    return str(p)


@router.post("/detect/dual")
def detect_dual(payload: dict) -> dict:
    """双检：传统引擎（模板差分）+ 特征引擎（品类模型）→ 融合判定，落库。

    body: {template_id, image_path, allow_stub?, persist?, workorder_id?}
    """
    from app.inspect.detect_service import get_detection_service
    template_id = payload.get("template_id") or payload.get("template")
    image_path = payload.get("image_path") or payload.get("path")
    if not template_id or not image_path:
        raise HTTPException(status_code=400, detail="需要 template_id 与 image_path")
    tpl = _load_template(template_id)
    out = get_detection_service().detect_dual(
        tpl, _resolve_image(image_path),
        workorder_id=payload.get("workorder_id"),
        target_type=payload.get("target_type", "image"),
        persist=bool(payload.get("persist", True)),
        allow_stub=bool(payload.get("allow_stub", False)),
    )
    return out


@router.post("/detect/traditional")
def detect_traditional(payload: dict) -> dict:
    """单传统引擎（模板差分）。"""
    template_id = payload.get("template_id") or payload.get("template")
    image_path = payload.get("image_path") or payload.get("path")
    if not template_id or not image_path:
        raise HTTPException(status_code=400, detail="需要 template_id 与 image_path")
    tpl = _load_template(template_id)
    run = _service.run_with_template(tpl, _resolve_image(image_path),
                                     test_path=image_path,
                                     allow_stub=bool(payload.get("allow_stub", False)),
                                     archive=False)
    s = run.summary
    return {
        "overall": s.overall,
        "overall_ok": s.overall_ok,
        "ng_count": s.ng_count,
        "elapsed_ms": s.elapsed_ms,
        "gate_blocked": s.gate_blocked,
        "gate_message": s.gate_message,
        "results": [
            {"status": r.status, "display_name": r.display_name, "message": r.message,
             "defect_count": r.defect_count, "elapsed_ms": r.elapsed_ms,
             "boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h, "label": b.label}
                       for b in r.boxes]}
            for r in s.results
        ],
    }


@router.post("/detect/feature")
def detect_feature(payload: dict) -> dict:
    """单特征引擎（品类模型）。body: {category, image_path}"""
    category = payload.get("category")
    image_path = payload.get("image_path") or payload.get("path")
    if not category or not image_path:
        raise HTTPException(status_code=400, detail="需要 category 与 image_path")
    try:
        from app.engines.feature import get_engine
        from app.inspect.fusion import feature_to_dict
        engine = get_engine()
        r = engine.predict_image_path(category, _resolve_image(image_path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"特征引擎失败: {exc}")
    return feature_to_dict(r)


# ── M10b：PLC 触发拍照检测（AOI 上产线标准形态）──────────────
@router.post("/detect/trigger")
def detect_trigger(payload: dict) -> dict:
    """PLC/光电触发：从 plc.trigger_dir 取最早一张待检图 → 双检 →
    移入 processed/ → 返回判定结果（OK/NG）。"""
    from app.config import get_settings
    settings = get_settings()
    tdir = Path(settings.get("plc", "trigger_dir") or "")
    if not tdir or not tdir.is_dir():
        raise HTTPException(status_code=400,
                            detail="plc.trigger_dir 未配置（configs/default.yaml plc 段）")
    exts = {e.lower() for e in (settings.get("plc", "exts") or [".png", ".jpg", ".jpeg", ".bmp"])}
    pending = sorted((p for p in tdir.iterdir()
                      if p.is_file() and p.suffix.lower() in exts),
                     key=lambda p: p.stat().st_mtime)
    if not pending:
        raise HTTPException(status_code=404, detail="触发目录无待检图")
    img = pending[0]
    template_ref = payload.get("template_ref") or settings.get("plc", "template_ref") or ""
    try:
        from app.inspect.detect_service import get_detection_service
        if template_ref:
            tpl = _store.load(template_ref)
            if tpl is None:
                raise HTTPException(status_code=400,
                                    detail=f"PLC 绑定的模板不存在: {template_ref}")
            out = get_detection_service().detect_dual(tpl, img,
                                                      target_type="plc",
                                                      persist=True)
        else:
            category = payload.get("category") or settings.get("plc", "category") or "default"
            from app.engines.feature import get_engine
            from app.inspect.fusion import feature_to_dict
            r = get_engine().predict_image_path(category, str(img))
            out = feature_to_dict(r)
        done_dir = tdir / "processed"
        done_dir.mkdir(exist_ok=True)
        try:
            img.rename(done_dir / img.name)
        except OSError:
            pass
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PLC 触发检测失败: {e}")


# ── 检测记录查询 ─────────────────────────────────────────────
@router.get("/detections")
def list_detections(category: str | None = None,
                    is_anomaly: bool | None = None,
                    target_type: str | None = None,
                    workorder_id: int | None = None,
                    page: int = 1, page_size: int = 50) -> dict:
    from app.db.database import session_scope
    from app.db.models import Detection
    with session_scope() as s:
        q = s.query(Detection)
        if category:
            q = q.filter(Detection.category == category)
        if is_anomaly is not None:
            q = q.filter(Detection.is_anomaly == is_anomaly)
        if target_type:
            q = q.filter(Detection.target_type == target_type)
        if workorder_id is not None:
            q = q.filter(Detection.workorder_id == workorder_id)
        total = q.count()
        rows = (q.order_by(Detection.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
        items = []
        for r in rows:
            items.append(_det_dict(r))
        return {"total": total, "items": items}


@router.get("/detections/{detection_id}")
def get_detection(detection_id: int) -> dict:
    from app.db.database import session_scope
    from app.db.models import Detection
    with session_scope() as s:
        row = s.get(Detection, detection_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
        return _det_dict(row)


def _det_dict(r) -> dict:
    return {
        "id": r.id, "target_type": r.target_type, "workorder_id": r.workorder_id,
        "image_path": r.image_path, "category": r.category,
        "engine_mode": r.engine_mode, "template_id": r.template_id,
        "template_version": r.template_version,
        "traditional_overall": r.traditional_overall,
        "feature_decision": r.feature_decision,
        "final_score": r.final_score, "is_anomaly": r.is_anomaly,
        "latency_ms": r.latency_ms,
        "n_tiles": r.n_tiles or {},
        "dual_json": r.dual_json,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }
