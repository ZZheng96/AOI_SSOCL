"""SPC 统计 API（借鉴 reference5/UI算法 Java AOI 的 SPC 设计 + 复判.md）。

维度：
- 工单级：检测次数 / 不良总数 / 不良率 / 良率 / 误报数 / 误报率
- 缺陷类型分布（来自 dual_json 的缺陷框 label）
- 误报按缺陷类型 / 料号（category）统计（复判反馈 wrong 判定）
"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException

from app.db.database import session_scope
from app.db.models import DefectRecord, Detection, Feedback, Inspect, WorkOrder

router = APIRouter()


@router.get("/workorders/{wo_id}/spc-summary")
def spc_summary(wo_id: int) -> dict:
    """工单 SPC 汇总：检测/不良/良率 + 缺陷类型分布 + 误报统计（缺陷明细优先，旧记录回退 dual_json）。"""
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        dets = (s.query(Detection)
                .filter(Detection.workorder_id == wo_id).all())
        fbs = (s.query(Feedback, Detection)
               .join(Detection, Detection.id == Feedback.detection_id)
               .filter(Detection.workorder_id == wo_id).all())

        total = len(dets)
        ng_count = sum(1 for d in dets if d.is_anomaly)
        ok_count = total - ng_count
        ng_rate = (ng_count / total) if total else 0.0
        pass_rate = (ok_count / total) if total else 0.0

        # 缺陷类型分布：优先 defect_records 明细（P0 落表），旧记录回退 dual_json
        defect_types: Counter[str] = Counter()
        n_defects = 0
        det_ids = [d.id for d in dets]
        if det_ids:
            rows = (s.query(DefectRecord)
                    .filter(DefectRecord.detection_id.in_(det_ids)).all())
            if rows:
                for r in rows:
                    defect_types[r.defect_type] += 1
                    n_defects += 1
            else:
                for d in dets:
                    dual = d.dual_json or {}
                    for b in dual.get("boxes") or []:
                        label = str(b.get("label") or "未知")
                        defect_types[label] += 1
                        n_defects += 1

        # 误报统计：复判 wrong（false_positive / false_negative）且检测为 NG/OK 的判定
        fp_count = sum(1 for fb, _d in fbs if fb.feedback_type == "false_positive")
        fn_count = sum(1 for fb, _d in fbs if fb.feedback_type == "false_negative")
        confirmed_ng = sum(1 for fb, _d in fbs if fb.feedback_type == "confirmed_ng")
        false_alarm_rate = (fp_count / ng_count) if ng_count else 0.0

        # 误报按缺陷类型分布（反馈对应的原检测框 label）
        fp_by_type: Counter[str] = Counter()
        for fb, d in fbs:
            if fb.feedback_type != "false_positive":
                continue
            dual = d.dual_json or {}
            for b in dual.get("boxes") or []:
                fp_by_type[str(b.get("label") or "未知")] += 1

        return {
            "workorder_id": wo_id,
            "workorder_name": wo.name,
            "total": total,
            "ok": ok_count,
            "ng": ng_count,
            "ng_rate": round(ng_rate, 4),
            "pass_rate": round(pass_rate, 4),
            "n_defects": n_defects,
            "defect_types": dict(defect_types.most_common()),
            "feedback": {
                "false_positive": fp_count,
                "false_negative": fn_count,
                "confirmed_ng": confirmed_ng,
                "false_alarm_rate": round(false_alarm_rate, 4),
                "fp_by_type": dict(fp_by_type.most_common()),
            },
        }


@router.get("/workorders/{wo_id}/defect-types")
def defect_type_stat(wo_id: int) -> dict:
    """按缺陷类型统计：类型 × 数量 × 对应检测记录数（SPC 缺陷维度）。"""
    with session_scope() as s:
        dets = (s.query(Detection)
                .filter(Detection.workorder_id == wo_id,
                        Detection.is_anomaly.is_(True)).all())
        rows: dict[str, dict] = {}
        det_ids = [d.id for d in dets]
        if det_ids:
            records = (s.query(DefectRecord)
                       .filter(DefectRecord.detection_id.in_(det_ids)).all())
            if records:
                for r in records:
                    e = rows.setdefault(r.defect_type, {"count": 0, "detection_ids": []})
                    e["count"] += 1
                    e["detection_ids"].append(r.detection_id)
        # 旧记录无明细 → 回退 dual_json
        if not rows:
            for d in dets:
                dual = d.dual_json or {}
                for b in dual.get("boxes") or []:
                    label = str(b.get("label") or "未知")
                    e = rows.setdefault(label, {"count": 0, "detection_ids": []})
                    e["count"] += 1
                    e["detection_ids"].append(d.id)
        out = []
        for label, e in sorted(rows.items(), key=lambda kv: -kv[1]["count"]):
            out.append({"defect_type": label, "count": e["count"],
                        "n_detections": len(set(e["detection_ids"]))})
        return {"workorder_id": wo_id, "items": out}


# ── 板级检测（P0：inspects 表）──────────────────────────────
@router.get("/inspects")
def list_inspects(workorder_id: int | None = None,
                  pass_: bool | None = None,
                  repair_state: str | None = None,
                  page: int = 1, page_size: int = 50) -> dict:
    """板级检测列表（对应 Java AOI inspect 主表）。"""
    with session_scope() as s:
        q = s.query(Inspect)
        if workorder_id is not None:
            q = q.filter(Inspect.workorder_id == workorder_id)
        if pass_ is not None:
            q = q.filter(Inspect.pass_ == pass_)
        if repair_state:
            q = q.filter(Inspect.repair_state == repair_state)
        total = q.count()
        rows = (q.order_by(Inspect.id.desc())
                .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [_insp_dict(r) for r in rows]}


@router.get("/inspects/{inspect_id}")
def get_inspect(inspect_id: int) -> dict:
    """板级检测详情：板信息 + 全部缺陷明细（对应 Java AOI getDefectsByInspectId）。"""
    from app.db.models import Detection as _Det
    with session_scope() as s:
        insp = s.get(Inspect, inspect_id)
        if insp is None:
            raise HTTPException(status_code=404, detail=f"板级检测不存在: {inspect_id}")
        defects = (s.query(DefectRecord)
                   .filter(DefectRecord.inspect_id == inspect_id).all())
        dets = (s.query(_Det).filter(_Det.inspect_id == inspect_id).all())
        d = _insp_dict(insp)
        d["defects"] = [
            {"id": r.id, "defect_type": r.defect_type, "engine": r.engine,
             "x": r.x, "y": r.y, "w": r.width, "h": r.height,
             "board_x": r.board_x, "board_y": r.board_y,
             "position_on": r.position_on, "score": r.score,
             "rejudged": r.rejudged, "final_error_type": r.final_error_type}
            for r in defects
        ]
        d["detections"] = [
            {"id": dd.id, "image_path": dd.image_path, "is_anomaly": dd.is_anomaly,
             "engine_mode": dd.engine_mode, "feature_decision": dd.feature_decision}
            for dd in dets
        ]
        return d


def _insp_dict(insp: Inspect) -> dict:
    return {
        "id": insp.id, "workorder_id": insp.workorder_id,
        "template_id": insp.template_id, "template_version": insp.template_version,
        "number": insp.number, "board_barcode": insp.board_barcode,
        "track_index": insp.track_index,
        "total_count": insp.total_count, "ng_count": insp.ng_count,
        "pass": insp.pass_, "repair_state": insp.repair_state,
        "repair_user": insp.repair_user, "synced": insp.synced,
        "inspect_time": insp.inspect_time.isoformat() if insp.inspect_time else None,
        "finish_time": insp.finish_time.isoformat() if insp.finish_time else None,
    }


@router.get("/detections/{detection_id}/records")
def detect_records(detection_id: int) -> dict:
    """检测框级明细（对应 Java AOI detect_record）：从 dual_json 提取逐缺陷记录。"""
    with session_scope() as s:
        d = s.get(Detection, detection_id)
        if d is None:
            raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
        dual = d.dual_json or {}
        records = []
        for b in dual.get("boxes") or []:
            records.append({
                "engine": b.get("engine"),
                "label": b.get("label"),
                "x": b.get("x"), "y": b.get("y"), "w": b.get("w"), "h": b.get("h"),
                "result": "NG",
            })
        for r in (dual.get("traditional") or {}).get("results") or []:
            for b in r.get("boxes") or []:
                records.append({
                    "engine": "traditional",
                    "label": b.get("label"),
                    "x": b.get("x"), "y": b.get("y"), "w": b.get("w"), "h": b.get("h"),
                    "result": "NG",
                    "item": r.get("display_name"),
                })
        feat = dual.get("feature") or {}
        if feat.get("decision") == "anomaly":
            records.append({
                "engine": "feature", "label": f"AOI·{feat.get('triggered_slot')}",
                "result": "NG", "score": feat.get("score"),
            })
        return {"detection_id": detection_id, "engine_mode": d.engine_mode,
                "overall": dual.get("overall"), "records": records}
