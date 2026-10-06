"""工单制接口 - 统计 / 产线控制 / 队列 / 学习 / 报告（自 routes_workorder 拆分）。"""
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func

from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import Detection, Feedback, Image as ImageRow, WorkOrder
from ._wo_shared import (_batch_stats_by_ids, _source_capability,
                         _workorder_detect_batches, _workorder_item,
                         _workorder_sources)
from .schemas import QueueDatasetRequest

router = APIRouter()


# ══════════════════ 工单口径统计 ══════════════════
@router.get("/workorders/{workorder_id}/stats")
def api_workorder_stats(workorder_id: int, range: str = "all"):
    """单工单统计（工单为统计基础）：range=all/today/7d。"""
    days = {"today": 0, "7d": 7, "all": None}.get(range, None)
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        item = _workorder_item(s, wo, range_days=days)
    # 时序（按日聚合，仅工单品类）
    cats = item["categories"]
    with session_scope() as s:
        series = _wo_timeseries(s, cats, range)
    item["timeseries"] = series
    return item


def _wo_timeseries(s, cats: list, range: str) -> list:
    """工单品类的按日检测聚合（近 14 天 / today 当天）。"""
    day_col = func.date(Detection.created_at)
    q = (s.query(day_col.label("d"),
                 func.count(Detection.id).label("n"),
                 func.sum(Detection.is_anomaly).label("a")))
    if cats:
        q = q.filter(Detection.category.in_(cats))
    else:
        q = q.filter(Detection.id < 0)
    if range == "today":
        q = q.filter(day_col == date.today())
    else:
        since = date.today() - timedelta(days=13)
        q = q.filter(day_col >= since)
    rows = q.group_by(day_col).order_by(day_col.asc()).all()
    return [{"date": str(d), "n_inspected": int(n or 0),
             "n_anomaly": int(a or 0)} for d, n, a in rows]


# ══════════════════ 产线控制（前端反馈 v5）══════════════════
@router.post("/workorders/{workorder_id}/pipeline/{action}",
             dependencies=[Depends(require_role("engineer"))])
def api_pipeline_control(workorder_id: int, action: str):
    """产线控制：pause 暂停 / resume 恢复 / stop 停止。

    pause/resume 语义不变（配置保留，恢复后直接继续检测，不重启不重新配置；
    暂停期间可编排数据流队列）。
    stop（停止任务，2026-08-30 新增）：置 stopped 并清空回队队列残留
    （requeued/requeued_at/requeued_ids），防止恢复后暂停期间编排的
    旧回队批次又被优先插队重检；背压自动恢复只作用于 running/auto_paused，
    stopped 不会被误拉起，只能人工 resume 重新启动。
    """
    if action not in ("pause", "resume", "stop"):
        raise HTTPException(status_code=422,
                            detail="action 只能为 pause/resume/stop")
    n_cleared = 0
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        if action == "stop":
            wo.pipeline_status = "stopped"
            qcfg = dict(wo.queue_json or {})
            # 被清空的回队批次数（requeued_ids 的 key 数）
            n_cleared = len(qcfg.get("requeued_ids") or {})
            for k in ("requeued", "requeued_at", "requeued_ids"):
                qcfg.pop(k, None)
            wo.queue_json = qcfg
        else:
            wo.pipeline_status = "paused" if action == "pause" else "running"
        status = wo.pipeline_status
        s.flush()
    log_action("pipeline_control", f"wo={workorder_id} action={action}",
               extra={"workorder_id": workorder_id,
                      "cleared_requeued": n_cleared})
    return {"workorder_id": workorder_id, "pipeline_status": status,
            "cleared_requeued": n_cleared}


class ClaimRecoveryRequest(BaseModel):
    image_id: int
    claim_token: str
    worker_stopped: bool = False
    reset_failed: bool = False


@router.post("/workorders/{workorder_id}/pipeline/recover-claim",
             dependencies=[Depends(require_role("engineer"))])
def api_recover_pipeline_claim(workorder_id: int, req: ClaimRecoveryRequest):
    """人工恢复：必须确认原 worker 已停止；令牌条件更新防止误清新 claim。"""
    if not req.worker_stopped or not req.claim_token:
        raise HTTPException(status_code=422, detail="须确认原 worker 已停止并提供 claim_token")
    with session_scope() as s:
        img = s.get(ImageRow, req.image_id)
        expected_status = "failed" if req.reset_failed else "claiming"
        if (img is None or img.processing_status != expected_status
                or img.claim_workorder_id != workorder_id
                or img.claim_token != req.claim_token):
            raise HTTPException(status_code=409, detail="claim 不匹配或已完成；请重新核查")
        committed = (s.query(Detection.id)
                     .filter(Detection.claim_token == req.claim_token).first())
        if committed is not None:
            raise HTTPException(status_code=409, detail="检测已持久化，不允许重新入队")
        changed = (s.query(ImageRow)
                   .filter(ImageRow.id == req.image_id,
                           ImageRow.processing_status == expected_status,
                           ImageRow.claim_workorder_id == workorder_id,
                           ImageRow.claim_token == req.claim_token)
                   .update({ImageRow.processing_status: "pending",
                            ImageRow.claim_token: None,
                            ImageRow.claim_workorder_id: None,
                            ImageRow.claim_requeued_at: None,
                            ImageRow.last_error: "人工恢复：已确认原 worker 停止"},
                           synchronize_session=False))
        if changed != 1:
            raise HTTPException(status_code=409, detail="claim 状态已变更")
    log_action("pipeline_claim_recover",
               f"wo={workorder_id} image={req.image_id} token={req.claim_token}",
               extra={"workorder_id": workorder_id, "image_id": req.image_id})
    return {"workorder_id": workorder_id, "image_id": req.image_id,
            "processing_status": "pending"}


@router.get("/workorders/{workorder_id}/queue")
def api_workorder_queue(workorder_id: int):
    """数据流队列：工单挂接数据源下的「检测批次（30 图/批）」+ 错检统计 + 回队标记。

    暂停产线后可在此编排：回队（错检批次重新排队，高优）、取消回队。
    """
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        pipeline_status = wo.pipeline_status
        auto_resume = bool(wo.auto_resume)
        qcfg = dict(wo.queue_json or {})
        requeued = {str(x) for x in (qcfg.get("requeued") or [])}
        requeued_at = {str(k): v for k, v in (qcfg.get("requeued_at") or {}).items()}
        items = []
        for b in _workorder_detect_batches(s, wo):
            st = _batch_stats_by_ids(s, b["image_ids"])
            key = b["batch_key"]
            items.append({
                "batch_key": key,
                "source": b["source"], "source_id": b["datasource_id"],
                "category": b["category"], "batch_index": b["batch_index"],
                "name": f"批次 {b['batch_index']}",
                **st,
                "requeued": key in requeued,
                "requeued_at": requeued_at.get(key),
                # 错检批次：已检测且（不良率≥0.5 或 反馈≥3）
                "bad_batch": bool(st["n_detected"]
                                  and (st["bad_rate"] >= 0.5
                                       or st["n_feedback"] >= 3)),
            })
    return {"pipeline_status": pipeline_status, "auto_resume": auto_resume,
            "items": items}


@router.post("/workorders/{workorder_id}/queue/requeue",
             dependencies=[Depends(require_role("engineer"))])
def api_queue_requeue(workorder_id: int, req: QueueDatasetRequest):
    """错检检测批次回队：重新排入检测队列（产线优先消费），允许插队。

    回队时把该批次的 image_ids 固化到 queue_json.requeued_ids，产线据此
    优先消费这批图片（重检口径：无检测 或 最近检测早于回队时刻）。
    """
    keys = [str(k) for k in (req.batch_keys or [])]
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        qcfg = dict(wo.queue_json or {})
        cur = set(str(x) for x in (qcfg.get("requeued") or []))
        at = {str(k): v for k, v in (qcfg.get("requeued_at") or {}).items()}
        ids_map = {str(k): list(v) for k, v in
                   (qcfg.get("requeued_ids") or {}).items()}
        batch_by_key = {b["batch_key"]: b
                        for b in _workorder_detect_batches(s, wo)}
        now = datetime.now()
        for k in keys:
            b = batch_by_key.get(k)
            if b is None:
                continue
            cur.add(k)
            at[k] = now.isoformat()
            ids_map[k] = b["image_ids"]
        qcfg["requeued"] = sorted(cur)
        qcfg["requeued_at"] = {k: v for k, v in at.items()}
        qcfg["requeued_ids"] = ids_map
        wo.queue_json = qcfg
        s.flush()
    log_action("queue_requeue", f"wo={workorder_id} batches={keys}",
               extra={"workorder_id": workorder_id})
    return {"workorder_id": workorder_id, "requeued": sorted(cur)}


@router.delete("/workorders/{workorder_id}/queue/requeue",
               dependencies=[Depends(require_role("engineer"))])
def api_queue_unrequeue(workorder_id: int, req: QueueDatasetRequest):
    """取消回队。"""
    keys = [str(k) for k in (req.batch_keys or [])]
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        qcfg = dict(wo.queue_json or {})
        cur = set(str(x) for x in (qcfg.get("requeued") or []))
        at = {str(k): v for k, v in (qcfg.get("requeued_at") or {}).items()}
        ids_map = {str(k): list(v) for k, v in
                   (qcfg.get("requeued_ids") or {}).items()}
        for k in keys:
            cur.discard(k)
            at.pop(k, None)
            ids_map.pop(k, None)
        qcfg["requeued"] = sorted(cur)
        qcfg["requeued_at"] = {k: v for k, v in at.items()}
        qcfg["requeued_ids"] = ids_map
        wo.queue_json = qcfg
        s.flush()
    return {"workorder_id": workorder_id, "requeued": sorted(cur)}


@router.post("/workorders/{workorder_id}/learn",
             dependencies=[Depends(require_role("engineer"))])
def api_workorder_learn(workorder_id: int):
    """学习提升：把该工单品类的积累反馈固化（consolidate 为新版本），
    学习曲线页可见提升；学完按 auto_resume 配置自动恢复产线。"""
    from ..self_learning import service as sl_service
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        sources = _workorder_sources(s, wo.id)
        cats: list = []
        for src in sources:
            cap = _source_capability(s, src.id)
            cats.extend(cap.get("categories") or [])
        cats = sorted(set(cats))
        name = wo.name
    applied: list[str] = []
    for cat in cats:
        try:
            out = sl_service.consolidate(cat, note=f"工单「{name}」学习提升")
            applied.append(f"{cat}→{out['version']}")
        except Exception:  # noqa: BLE001 未准备品类跳过
            continue
    if applied:
        # 学习固化后登记 Model 表快照行（此前只写磁盘快照，版本列表不更新
        # ——UI 模型管理/学习曲线看不到学习后的新版本）
        from .routes_models import register_engine_snapshots
        try:
            register_engine_snapshots()
        except Exception:  # noqa: BLE001 登记失败不阻塞学习主流程
            pass
    resumed = False
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is not None and wo.auto_resume:
            wo.pipeline_status = "running"
            resumed = True
            s.flush()
    return {"workorder_id": workorder_id, "applied_categories": applied,
            "categories": cats, "resumed": resumed,
            "pipeline_status": "running" if resumed else None,
            "message": f"学习提升完成：{'、'.join(applied) if applied else '无可用反馈/未准备品类'}"
                       + ("；已自动恢复产线" if resumed else "；等待手动恢复产线")}


# ══════════════════ 工单检测报告 ══════════════════
def _decision_of(d: Detection) -> str:
    """检测三态判定：优先 n_tiles.decision，缺失时按 is_anomaly 归并。"""
    nt = d.n_tiles or {}
    dec = str(nt.get("decision") or "")
    if dec not in ("normal", "gray", "anomaly"):
        dec = "anomaly" if d.is_anomaly else "normal"
    return dec


def _p95(values: list) -> float:
    """p95（升序取 ceil(0.95n)-1 位；空列表返回 0）。"""
    import math
    if not values:
        return 0.0
    vs = sorted(float(v) for v in values)
    return vs[max(0, math.ceil(0.95 * len(vs)) - 1)]


def _workorder_report(workorder_id: int) -> dict:
    """工单检测报告数据（JSON / CSV 两接口共用同一口径）。

    汇总：工单/品类/检测时间范围、总检测数、正常/异常/灰区数、不良率
    （口径同工作台 KPI：异常+灰区）、平均/p95 延迟、反馈数、误报/漏检数
    （仅计有效反馈）。明细：每张图的判定/分数/延迟/是否被反馈纠正。
    """
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        dets = (s.query(Detection)
                .filter(Detection.workorder_id == workorder_id)
                .order_by(Detection.id.asc()).all())
        det_ids = [d.id for d in dets]
        fb_map: dict = {}
        if det_ids:
            for f in (s.query(Feedback)
                      .filter(Feedback.detection_id.in_(det_ids),
                              Feedback.invalidated == False)  # noqa: E712
                      .all()):
                fb_map.setdefault(f.detection_id, []).append(f)
        # 品类口径同工单详情：所挂数据源品类并集
        cats: list = []
        for src in _workorder_sources(s, workorder_id):
            cats.extend(_source_capability(s, src.id).get("categories") or [])
        cats = sorted(set(cats))
        details = []
        n_dec = {"normal": 0, "gray": 0, "anomaly": 0}
        latencies: list = []
        n_fp = n_fn = 0
        for d in dets:
            dec = _decision_of(d)
            n_dec[dec] += 1
            # 延迟口径：端到端（含解码/持久化）优先，旧数据回退纯推理耗时
            latencies.append(float(
                d.latency_e2e_ms if d.latency_e2e_ms is not None
                else (d.latency_ms or 0.0)))
            fbs = fb_map.get(d.id, [])
            n_fp += sum(1 for f in fbs if f.feedback_type == "false_positive")
            n_fn += sum(1 for f in fbs if f.feedback_type == "false_negative")
            # 被反馈纠正 = 存在误报/漏检类有效反馈（系统判定被人工推翻）
            corrected = any(f.feedback_type in ("false_positive",
                                                "false_negative")
                            for f in fbs)
            details.append({
                "detection_id": d.id,
                "created_at": (d.created_at.isoformat()
                               if d.created_at else None),
                "image_path": d.image_path,
                "category": d.category,
                "decision": dec,
                "final_score": round(float(d.final_score or 0.0), 4),
                "latency_ms": round(float(d.latency_ms or 0.0), 1),
                "n_feedback": len(fbs),
                "corrected": corrected,
            })
        n_total = len(dets)
        t0 = dets[0].created_at if dets else None
        t1 = dets[-1].created_at if dets else None
        summary = {
            "workorder_id": wo.id,
            "workorder_name": wo.name,
            "categories": cats,
            "time_range": [t0.isoformat() if t0 else None,
                           t1.isoformat() if t1 else None],
            "n_total": n_total,
            "n_normal": n_dec["normal"],
            "n_anomaly": n_dec["anomaly"],
            "n_gray": n_dec["gray"],
            "defect_rate": (round((n_dec["anomaly"] + n_dec["gray"]) / n_total, 4)
                            if n_total else 0.0),
            "avg_latency_ms": (round(sum(latencies) / n_total, 1)
                               if n_total else 0.0),
            "p95_latency_ms": round(_p95(latencies), 1),
            "n_feedback": sum(len(v) for v in fb_map.values()),
            "n_false_positive": n_fp,
            "n_false_negative": n_fn,
        }
    return {"summary": summary, "details": details}


@router.get("/workorders/{workorder_id}/report")
def api_workorder_report(workorder_id: int):
    """工单检测报告（JSON）：汇总指标 + 逐图明细。"""
    return _workorder_report(workorder_id)


@router.get("/workorders/{workorder_id}/report.csv")
def api_workorder_report_csv(workorder_id: int):
    """工单检测报告 CSV 下载（汇总段 + 明细段；带 BOM 供 Excel 直开，
    写法参考 routes_stats 日报导出）。"""
    import csv
    import io

    from fastapi.responses import Response

    rep = _workorder_report(workorder_id)
    sm = rep["summary"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# 工单检测报告"])
    w.writerow(["工单ID", sm["workorder_id"], "工单名称", sm["workorder_name"],
                "品类", "、".join(sm["categories"]) or "-"])
    t0, t1 = sm["time_range"]
    w.writerow(["时间范围", f"{t0 or '-'} ~ {t1 or '-'}"])
    w.writerow(["总检测数", sm["n_total"], "正常", sm["n_normal"],
                "异常", sm["n_anomaly"], "灰区", sm["n_gray"],
                "不良率", sm["defect_rate"]])
    w.writerow(["平均延迟ms", sm["avg_latency_ms"],
                "p95延迟ms", sm["p95_latency_ms"],
                "反馈数", sm["n_feedback"],
                "误报", sm["n_false_positive"],
                "漏检", sm["n_false_negative"]])
    w.writerow([])
    w.writerow(["detection_id", "created_at", "image_path", "category",
                "decision", "final_score", "latency_ms",
                "n_feedback", "corrected"])
    for it in rep["details"]:
        w.writerow([it["detection_id"], it["created_at"], it["image_path"],
                    it["category"], it["decision"], it["final_score"],
                    it["latency_ms"], it["n_feedback"],
                    "是" if it["corrected"] else "否"])
    return Response(
        content="﻿" + buf.getvalue(),  # BOM：Excel 直接打开不乱码
        media_type="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=workorder_{workorder_id}_report.csv"})
