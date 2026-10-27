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
    # new_defect 时的缺陷类型名（自动候选按 品类/类型 归并；不传记为 unnamed）
    defect_type: str | None = Field(default=None, max_length=64)


# 非缺陷原因不得与 confirmed_defect 同时出现；new_defect 必须是 confirmed_defect
_NON_DEFECT_CAUSES = {"drift", "noise", "registration"}
# new_defect 只登记候选准入事件；上线须走 hard_admission 全部门禁 + 快照版本绑定
ADMISSION_CHECKS = ["gain", "redundancy", "regression", "g1", "g2", "version"]


def admission_candidate(cause, autoreg=None):
    if cause != "new_defect":
        return None
    out = {"kind": "feature_admission", "status": "candidate", "auto_online": False,
           "required_checks": ADMISSION_CHECKS,
           "entry": "hard_admission.evaluate_feature_admission（独立难例池试算，不注册/不激活）"}
    if autoreg is not None:
        out.update(autoregister=autoreg, revision_id=autoreg.get("revision_id"))
    return out


def _revision_store():
    """与 routes_models._feature_revision_store 同一注册表（测试可 patch）。"""
    from algo.feature_revision import FeatureSetRevision
    from ..core.config import get_settings
    return FeatureSetRevision(get_settings().storage("feature_revisions") / "registry.json")


def candidate_manifest(category, defect_type, model_version):
    """新命名组候选 manifest：沿用当前注册表组作基线 + proposed_group 元数据（未实现提取器）。
    只确定于 (品类, 缺陷类型, 快照版本)，便于归并；admission_required=True。"""
    from algo.feature_revision import production_hashes
    from algo.features import registry as freg
    groups = [{"key": g["key"], "dim": int(g["dim"]), "slice": list(g["slice"]),
               "params": dict(g["params"])} for g in freg.TRAD_GROUPS]
    return {"schema_version": 1, "groups": groups, "dimension": int(freg.TRAD_DIM),
            "enabled_groups": [g["key"] for g in groups], "scoring": "legacy",
            "category": category, "implementation_hashes": production_hashes(),
            "preprocess": dict(freg.PREPROCESS), "seed": 0,
            "dataset_split": {"category": category, "source": "open_alert_new_defect",
                              "defect_type": defect_type},
            "model_snapshot": {"category": category, "version": int(model_version),
                               "artifact_hashes": {"meta": "unknown"}},
            "origin": "open_autoregister", "admission_required": True,
            "proposed_group": {"name": defect_type, "status": "candidate",
                               "extractor": "not_implemented"},
            "note": f"E_open new_defect 自动候选：{category}/{defect_type}"[:200]}


def autoregister(det, defect_type, event_id):
    """new_defect → 候选 draft（同 品类/类型 归并计数）。失败不阻断人工处置。"""
    defect_type = (defect_type or "").strip()[:64] or "unnamed"
    category = det.category or "unknown"
    model_version = (det.n_tiles or {}).get("model_version")
    base = {"key": f"{category}/{defect_type}", "auto_online": False, "revision_id": None}
    if model_version is None:
        return {**base, "action": "skipped", "reason": "检测记录无模型版本，无法绑定快照血缘"}
    sample = {"detection_id": det.id, "image_id": det.image_id, "image_path": det.image_path,
              "event_id": event_id, "model_version": model_version}
    try:
        return _revision_store().register_open_candidate(
            base["key"], candidate_manifest(category, defect_type, model_version), sample)
    except (ValueError, KeyError, TypeError, OSError) as e:
        return {**base, "action": "failed", "reason": str(e)[:200]}


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

    @router.get("/review/open_alerts/summary")
    def summary(category: str | None = None):
        """AC-052：处置→反馈→学习 链路汇总（按原因/状态/学习状态计数；只读，不训练）。"""
        with session_scope() as s:
            q = s.query(Detection).filter(Detection.n_tiles["open_alert"].as_boolean().is_(True))
            if category:
                q = q.filter(Detection.category == category)
            items = [item_dict(d) for d in q.all()]
            fb_ids = [i["feedback_id"] for i in items if i.get("feedback_id")]
            fbs = {f.id: f for f in s.query(Feedback).filter(Feedback.id.in_(fb_ids)).all()} if fb_ids else {}
            by_status, by_cause, by_learning = {}, {}, {}
            for i in items:
                by_status[i["status"]] = by_status.get(i["status"], 0) + 1
                c = i.get("cause") or "unclassified"
                by_cause[c] = by_cause.get(c, 0) + 1
                fb = fbs.get(i.get("feedback_id"))
                ls = ((fb.learning_trace or {}).get("learning_status") if fb else None) or "no_feedback"
                by_learning[ls] = by_learning.get(ls, 0) + 1
            return {"total": len(items), "by_status": by_status, "by_cause": by_cause,
                    "by_learning_status": by_learning,
                    "admission_candidates": sum(1 for i in items if i.get("admission")),
                    "note": "open≠缺陷；告警不自动训练，学习状态仅反映反馈事件是否被显式候选消费"}

    @router.get("/review/open_alerts/autoregistered")
    def autoregistered(category: str | None = None):
        """new_defect 自动建的候选 draft（只读）；上线须人工发起 admission + 门禁。"""
        store = _revision_store()
        rows, ledger = store.list(), store.open_candidates()
        items = []
        for key, entry in sorted(ledger.items()):
            if category and not key.startswith(f"{category}/"):
                continue
            row = rows.get(entry["revision_id"]) or {}
            items.append({**entry, "state": row.get("state", "missing"), "auto_online": False,
                          "admission_required": True,
                          "next_step": "人工 POST /models/feature-revisions/{rid}/admission 试算后走门禁"})
        return {"total": len(items), "items": items,
                "note": "自动注册仅建候选草案并归并计数，不自动评估/激活/上线"}

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
            if req.defect_type is not None:
                payload["defect_type"] = req.defect_type
            for event in item["events"]:
                if event["event_id"] == req.event_id:
                    if event["payload"] != payload:
                        raise HTTPException(409, "事件ID已对应其他处置")
                    return {"item": item, "event": event, "idempotent": True,
                            "autoregister": event.get("autoregister")}
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
            # CAS 成功后才自动建候选 draft（只登记，不评估/不激活/不绑定生产）
            autoreg = (autoregister(det, req.defect_type, req.event_id)
                       if req.cause == "new_defect" else None)
            state["admission"] = admission_candidate(req.cause, autoreg)
            event["autoregister"] = autoreg
            label = {"confirmed_defect": 1, "confirmed_normal": 0}.get(req.status, -1)
            fb = Feedback(detection_id=detection_id, feedback_type="open_alert", operator_label=label,
                          operator=req.operator, comment=req.reason,
                          source_event_id=f"open:{detection_id}:{req.event_id}", consumed=False,
                          model_version_before=str(item["model_version"]) if item["model_version"] is not None else None,
                          learning_trace={"candidate_only": True, "learning_status": "not_requested",
                                          "open_alert_status": req.status, "event_id": req.event_id,
                                          "open_cause": req.cause,
                                          "admission": state["admission"]})
            s.add(fb)
            s.flush()
            state["feedback_id"] = event["feedback_id"] = fb.id
            det.n_tiles = {**old, "open_alert_item": state}
            s.flush()
            return {"item": item_dict(det), "event": event, "idempotent": False,
                    "autoregister": autoreg}
