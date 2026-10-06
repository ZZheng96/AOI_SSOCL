"""SPC 统计 API（借鉴 reference5/UI算法 Java AOI 的 SPC 设计 + 复判.md）。

维度：
- 工单级：检测次数 / 不良总数 / 不良率 / 良率 / 误报数 / 误报率
- 缺陷类型分布（来自 dual_json 的缺陷框 label）
- 误报按缺陷类型 / 料号（category）统计（复判反馈 wrong 判定）
"""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException
from sqlalchemy import case, func, select

from app.db.database import session_scope
from app.db.models import DefectRecord, Detection, Feedback, Inspect, WorkOrder
from app.inspect.fusion import result_interpretation

router = APIRouter()


def _overall_status():
    """有融合摘要时以真实判定为准；历史无摘要记录回退布尔判定。"""
    return case(
        (Detection.dual_json["overall"].as_string().in_(["OK", "NG", "GRAY", "ERROR"]),
         Detection.dual_json["overall"].as_string()),
        (Detection.is_anomaly.is_(True), "NG"),
        else_="OK",
    )


def _legacy_boxes(s, wo_id: int, *, ng_only: bool = False):
    """仅提取没有缺陷明细的历史记录，避免加载完整检测对象。"""
    has_records = select(DefectRecord.id).where(DefectRecord.detection_id == Detection.id).exists()
    q = (s.query(Detection.id, Detection.dual_json)
         .filter(Detection.workorder_id == wo_id, ~has_records))
    if ng_only:
        q = q.filter(_overall_status() == "NG")
    return q


@router.get("/workorders/{wo_id}/spc-summary")
def spc_summary(wo_id: int) -> dict:
    """工单 SPC 汇总：检测/不良/良率 + 缺陷类型分布 + 误报统计（缺陷明细优先，旧记录回退 dual_json）。"""
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")

        status = _overall_status()
        status_counts = dict(s.query(status, func.count(Detection.id))
                             .filter(Detection.workorder_id == wo_id)
                             .group_by(status).all())
        total = sum(status_counts.values())
        ok_count = status_counts.get("OK", 0)
        ng_count = status_counts.get("NG", 0)
        gray_count = status_counts.get("GRAY", 0)
        error_count = status_counts.get("ERROR", 0)
        # 直通率口径：pass_rate = OK/(OK+NG)，ng_rate = NG/(OK+NG)；GRAY/ERROR 单列不计入分母
        judged = ok_count + ng_count
        ng_rate = (ng_count / judged) if judged else 0.0
        pass_rate = (ok_count / judged) if judged else 0.0

        defect_types: Counter[str] = Counter()
        for label, count in (s.query(DefectRecord.defect_type, func.count(DefectRecord.id))
                             .join(Detection, Detection.id == DefectRecord.detection_id)
                             .filter(Detection.workorder_id == wo_id, _overall_status() == "NG")
                             .group_by(DefectRecord.defect_type)):
            defect_types[label] += count
        for _id, dual in _legacy_boxes(s, wo_id, ng_only=True):
            for b in (dual or {}).get("boxes") or []:
                defect_types[str(b.get("label") or "未知")] += 1
        n_defects = sum(defect_types.values())

        fb_counts = dict(s.query(Feedback.feedback_type, func.count(Feedback.id))
                         .join(Detection, Detection.id == Feedback.detection_id)
                         .filter(Detection.workorder_id == wo_id, Feedback.invalidated.is_(False))
                         .group_by(Feedback.feedback_type).all())
        fp_count = fb_counts.get("false_positive", 0)
        fn_count = fb_counts.get("false_negative", 0)
        confirmed_ng = fb_counts.get("confirmed_ng", 0)
        false_alarm_rate = (fp_count / ng_count) if ng_count else 0.0

        fp_by_type: Counter[str] = Counter()
        fp_rows = (s.query(Detection.dual_json)
                   .join(Feedback, Feedback.detection_id == Detection.id)
                   .filter(Detection.workorder_id == wo_id,
                           Feedback.feedback_type == "false_positive",
                           Feedback.invalidated.is_(False)))
        for (dual,) in fp_rows:
            for b in (dual or {}).get("boxes") or []:
                fp_by_type[str(b.get("label") or "未知")] += 1

        return {
            "workorder_id": wo_id,
            "workorder_name": wo.name,
            "total": total,
            "ok": ok_count,
            "ng": ng_count,
            "gray": gray_count,
            "error": error_count,
            "final_status_counts": {
                "PASS": ok_count, "ANOMALY": ng_count,
                "REVIEW": gray_count, "ERROR": error_count,
            },
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
        rows: dict[str, dict] = {}
        for label, count, n_detections in (
            s.query(DefectRecord.defect_type, func.count(DefectRecord.id),
                    func.count(func.distinct(DefectRecord.detection_id)))
             .join(Detection, Detection.id == DefectRecord.detection_id)
             .filter(Detection.workorder_id == wo_id, _overall_status() == "NG")
             .group_by(DefectRecord.defect_type)
        ):
            rows[label] = {"count": count, "n_detections": n_detections}
        legacy_ids: dict[str, set[int]] = {}
        for det_id, dual in _legacy_boxes(s, wo_id, ng_only=True):
            for b in (dual or {}).get("boxes") or []:
                label = str(b.get("label") or "未知")
                entry = rows.setdefault(label, {"count": 0, "n_detections": 0})
                entry["count"] += 1
                legacy_ids.setdefault(label, set()).add(det_id)
        for label, ids in legacy_ids.items():
            rows[label]["n_detections"] += len(ids)
        out = [{"defect_type": label, **entry}
               for label, entry in sorted(rows.items(), key=lambda kv: -kv[1]["count"])]
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
            {"id": dd.id, "image_path": dd.image_path,
             **result_interpretation((dd.dual_json or {}).get("overall")),
             "is_anomaly": dd.is_anomaly,
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
                "overall": dual.get("overall"),
                **result_interpretation(dual.get("overall")), "records": records}
