"""E_open 样本处置：只读历史证据，人工确认不自动触发生产学习。"""
from datetime import datetime
from typing import Literal

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.security import require_role
from ..db.database import session_scope
from ..db.models import Detection, Feedback
from ._wo_shared import archived_workorder_ids


class OpenDisposition(BaseModel):
    status: Literal["confirmed_defect", "confirmed_normal", "template_candidate", "deferred", "ignored"]
    operator: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    event_id: str = Field(min_length=1, max_length=100)
    expected_revision: int = Field(ge=0)
    # AC-034 高 open 原因归类：漂移/噪声/配准/新缺陷/未知（可选，旧客户端不传）
    cause: Literal["drift", "noise", "registration", "new_defect", "unknown"] | None = None


# 非缺陷原因不得与 confirmed_defect 同时出现；new_defect 必须是 confirmed_defect
_NON_DEFECT_CAUSES = {"drift", "noise", "registration"}
# new_defect 只登记候选准入事件；上线须走 hard_admission 全部门禁 + 快照版本绑定
ADMISSION_CHECKS = ["gain", "redundancy", "regression", "g1", "g2", "version"]


def admission_candidate(cause):
    if cause != "new_defect":
        return None
    return {"kind": "feature_admission", "status": "candidate", "auto_online": False,
            "required_checks": ADMISSION_CHECKS,
            "entry": "hard_admission.evaluate_feature_admission（独立难例池试算，不注册/不激活）"}


def new_item(score=None, threshold=None):
    return {"status": "queued", "revision": 0, "e_open_score": score,
            "e_open_threshold": threshold, "operator": None, "disposed_at": None,
            "reason": None, "feedback_id": None, "events": []}


def item_dict(det):
    nt = det.n_tiles or {}
    item = nt.get("open_alert_item") or new_item(nt.get("e_open_score"), nt.get("e_open_threshold"))
    return {**item, "detection_id": det.id, "image_id": det.image_id,
            "image_path": det.image_path, "overlay_path": det.overlay_path,
            "category": det.category, "workorder_id": det.workorder_id,
            "target_type": det.target_type, "model_version": nt.get("model_version"),
            "evidence_status": "recorded" if item.get("e_open_score") is not None and item.get("e_open_threshold") is not None else "legacy_unknown",
            "created_at": det.created_at.isoformat() if det.created_at else None,
            "is_defect": item["status"] == "confirmed_defect",
            "note": "未知信号不等于缺陷；处置仅留存反馈，未训练或发布模型"}


def register(router):
    @router.get("/review/open_alerts")
    def list_items(category: str | None = None, workorder_id: int | None = None,
                   status: str | None = None, page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, le=200)):
        with session_scope() as s:
            q = s.query(Detection).filter(Detection.n_tiles["open_alert"].as_boolean().is_(True))
            arch = archived_workorder_ids(s)
            if arch:
                q = q.filter(Detection.workorder_id.is_(None) | ~Detection.workorder_id.in_(arch))
            if category:
                q = q.filter(Detection.category == category)
            if workorder_id is not None:
                q = q.filter(Detection.workorder_id == workorder_id)
            if status:
                if status not in {"queued", "confirmed_defect", "confirmed_normal", "template_candidate", "deferred", "ignored"}:
                    raise HTTPException(422, "未知处置状态")
                st = Detection.n_tiles["open_alert_item"]["status"].as_string()
                q = q.filter((st == status) | st.is_(None)) if status == "queued" else q.filter(st == status)
            total = q.count()
            rows = q.order_by(Detection.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
            return {"total": total, "page": page, "page_size": page_size, "items": [item_dict(d) for d in rows]}

    @router.get("/review/open_alerts/{detection_id}")
    def detail(detection_id: int):
        with session_scope() as s:
            det = s.get(Detection, detection_id)
            if det is None or not (det.n_tiles or {}).get("open_alert"):
                raise HTTPException(404, "开放告警样本不存在")
            return item_dict(det)

    @router.post("/review/open_alerts/{detection_id}/disposition",
                 dependencies=[Depends(require_role("engineer"))])
    def dispose(detection_id: int, req: OpenDisposition):
        if not req.operator.strip() or not req.reason.strip() or not req.event_id.strip():
            raise HTTPException(422, "处置人、原因和事件ID不能为空")
        if req.cause == "new_defect" and req.status != "confirmed_defect":
            raise HTTPException(422, "原因为新缺陷时处置状态须为 confirmed_defect")
        if req.cause in _NON_DEFECT_CAUSES and req.status == "confirmed_defect":
            raise HTTPException(422, "漂移/噪声/配准原因不能确认为缺陷")
        with session_scope() as s:
            det = s.get(Detection, detection_id)
            if det is None or not (det.n_tiles or {}).get("open_alert"):
                raise HTTPException(404, "开放告警样本不存在")
            old = det.n_tiles
            item = item_dict(det)
            payload = {"status": req.status, "operator": req.operator, "reason": req.reason,
                       "expected_revision": req.expected_revision}
            if req.cause is not None:   # 不传时保持旧事件 payload 一致（幂等比较）
                payload["cause"] = req.cause
            for event in item["events"]:
                if event["event_id"] == req.event_id:
                    if event["payload"] != payload:
                        raise HTTPException(409, "事件ID已对应其他处置")
                    return {"item": item, "event": event, "idempotent": True}
            if item["revision"] != req.expected_revision:
                raise HTTPException(409, "样本已被处置，请刷新后重试")
            now = datetime.now().isoformat()
            event = {"event_id": req.event_id, "payload": payload, "at": now, "feedback_id": None}
            state = {k: item[k] for k in new_item()}
            state.update(status=req.status, revision=item["revision"] + 1,
                         operator=req.operator, reason=req.reason, disposed_at=now,
                         events=[*item["events"], event], feedback_id=None,
                         cause=req.cause, admission=admission_candidate(req.cause))
            updated = {**old, "open_alert_item": state}
            changed = (s.query(Detection).filter(Detection.id == detection_id,
                       Detection.n_tiles == old)
                       .update({Detection.n_tiles: updated}, synchronize_session=False))
            if changed != 1:
                raise HTTPException(409, "并发处置冲突，请刷新后重试")
            label = {"confirmed_defect": 1, "confirmed_normal": 0}.get(req.status, -1)
            fb = Feedback(detection_id=detection_id, feedback_type="open_alert", operator_label=label,
                          operator=req.operator, comment=req.reason,
                          source_event_id=f"open:{detection_id}:{req.event_id}", consumed=False,
                          model_version_before=str(item["model_version"]) if item["model_version"] is not None else None,
                          learning_trace={"candidate_only": True, "learning_status": "not_requested",
                                          "open_alert_status": req.status, "event_id": req.event_id,
                                          "open_cause": req.cause,
                                          "admission": admission_candidate(req.cause)})
            s.add(fb)
            s.flush()
            state["feedback_id"] = event["feedback_id"] = fb.id
            det.n_tiles = {**old, "open_alert_item": state}
            s.flush()
            return {"item": item_dict(det), "event": event, "idempotent": False}
