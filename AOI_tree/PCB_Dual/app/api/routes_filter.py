"""不良过滤 + 告警 API（P2：借鉴 Java AOI ng_filter / alarm_config+record）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.database import session_scope
from app.db.models import AlarmRecord, NgFilter

router = APIRouter()


# ── 不良过滤规则 ─────────────────────────────────────────────
@router.get("/ng-filters")
def list_ng_filters(defect_type: str | None = None, enabled: bool | None = None) -> dict:
    with session_scope() as s:
        q = s.query(NgFilter)
        if defect_type:
            q = q.filter(NgFilter.defect_type == defect_type)
        if enabled is not None:
            q = q.filter(NgFilter.enabled == enabled)
        rows = q.all()
        return {"items": [_nf_dict(r) for r in rows]}


@router.post("/ng-filters")
def create_ng_filter(payload: dict) -> dict:
    defect_type = payload.get("defect_type")
    if not defect_type:
        raise HTTPException(status_code=400, detail="需要 defect_type")
    with session_scope() as s:
        r = NgFilter(
            defect_type=defect_type,
            name=payload.get("name") or defect_type,
            filter_content=payload.get("filter_content", "area"),
            filter_condition=payload.get("filter_condition", "LT"),
            value_min=payload.get("value_min"),
            value_max=payload.get("value_max"),
            unit=payload.get("unit", "px"),
            rule_json=payload.get("rule_json") or {},
            enabled=bool(payload.get("enabled", True)),
            template_ref=payload.get("template_ref"),
        )
        s.add(r)
        s.flush()
        return _nf_dict(r)


@router.put("/ng-filters/{rule_id}")
def update_ng_filter(rule_id: int, payload: dict) -> dict:
    with session_scope() as s:
        r = s.get(NgFilter, rule_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"规则不存在: {rule_id}")
        for k in ("name", "filter_content", "filter_condition", "value_min",
                  "value_max", "unit", "rule_json", "template_ref"):
            if k in payload:
                setattr(r, k, payload[k])
        if "enabled" in payload:
            r.enabled = bool(payload["enabled"])
        return _nf_dict(r)


@router.delete("/ng-filters/{rule_id}")
def delete_ng_filter(rule_id: int) -> dict:
    with session_scope() as s:
        r = s.get(NgFilter, rule_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"规则不存在: {rule_id}")
        s.delete(r)
    return {"ok": True}


def _nf_dict(r: NgFilter) -> dict:
    return {"id": r.id, "defect_type": r.defect_type, "name": r.name,
            "filter_content": r.filter_content, "filter_condition": r.filter_condition,
            "value_min": r.value_min, "value_max": r.value_max, "unit": r.unit,
            "rule_json": r.rule_json, "enabled": r.enabled,
            "template_ref": r.template_ref}


# ── 告警 ─────────────────────────────────────────────────────
@router.post("/alarms")
def create_alarm(payload: dict) -> dict:
    """告警上报（检测 NG/任务失败/系统异常）。"""
    with session_scope() as s:
        a = AlarmRecord(
            alarm_type=payload.get("alarm_type", "system"),
            level=payload.get("level", "WARN"),
            source=payload.get("source", ""),
            content=payload.get("content", ""),
            ref_id=payload.get("ref_id"),
            ip_address=payload.get("ip_address"),
            process_status=payload.get("process_status", "pending"),
        )
        s.add(a)
        s.flush()
        return {"id": a.id, "alarm_type": a.alarm_type, "level": a.level,
                "content": a.content}


@router.get("/alarms")
def list_alarms(alarm_type: str | None = None, level: str | None = None,
                process_status: str | None = None,
                page: int = 1, page_size: int = 50) -> dict:
    with session_scope() as s:
        q = s.query(AlarmRecord)
        if alarm_type:
            q = q.filter(AlarmRecord.alarm_type == alarm_type)
        if level:
            q = q.filter(AlarmRecord.level == level)
        if process_status:
            q = q.filter(AlarmRecord.process_status == process_status)
        total = q.count()
        rows = (q.order_by(AlarmRecord.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [
            {"id": a.id, "alarm_type": a.alarm_type, "level": a.level,
             "source": a.source, "content": a.content, "ref_id": a.ref_id,
             "ip_address": a.ip_address, "process_status": a.process_status,
             "created_at": a.created_at.isoformat() if a.created_at else None}
            for a in rows]}


@router.post("/alarms/{alarm_id}/process")
def process_alarm(alarm_id: int, payload: dict) -> dict:
    """处理告警（pending → processed/ignored）。"""
    with session_scope() as s:
        a = s.get(AlarmRecord, alarm_id)
        if a is None:
            raise HTTPException(status_code=404, detail=f"告警不存在: {alarm_id}")
        a.process_status = payload.get("process_status", "processed")
        a.remark = payload.get("remark", a.remark)
        return {"id": a.id, "process_status": a.process_status}
