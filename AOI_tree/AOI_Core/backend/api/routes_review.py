"""灰区复核队列接口（M5a：灰区→复核→自动反馈即学闭环）。

decision=="gray" 的检测帧不拦产线也不放过，进人工复核队列；
复核结论（label 0/1）自动走 submit_feedback_record 即学路径
（feedback_type=review，consumed=True）——复核样本是边界样本，
是在线学习价值最高的反馈。

不加新表：已复核判定 = 存在 feedback.detection_id 关联记录。
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import String, cast

from ..db.database import log_action, session_scope
from ..db.models import (Dataset, Detection, Feedback, WorkOrder,
                         WorkOrderSource)
from ..self_learning import service as sl_service
from ._wo_shared import archived_workorder_ids

router = APIRouter()

_GRAY_LIKE = '%"decision": "gray"%'
# 2026-08-29 复核口径修正：复核队列只收产线路径（pipeline/stream/plc）的检测，
# 排除试检旁路（upload/image）——试检是调试动作，不进人工复核工作流
_LIVE_TYPES = ("pipeline", "stream", "plc")


def _review_scope(s) -> tuple[list, set]:
    """开启人工复判的工单 → (工单 id 列表, 挂接数据源的品类并集)。

    前端反馈 v4：复判工单的**全部**检测进待复核队列（不只灰区），
    复核后才计入统计与持续学习。2026-08-29 起检测落库带 workorder_id，
    按工单精确归属；历史行（workorder_id 为空）回退品类近似匹配。
    """
    # 存档排除（2026-10-09）：已存档工单不开启复判口径
    wo_ids = [r[0] for r in s.query(WorkOrder.id)
              .filter(WorkOrder.review_enabled.is_(True),
                      WorkOrder.archived.is_(False)).all()]
    if not wo_ids:
        return [], set()
    ds_ids = [r[0] for r in s.query(WorkOrderSource.datasource_id)
              .filter(WorkOrderSource.workorder_id.in_(wo_ids)).all()]
    if not ds_ids:
        return wo_ids, set()
    cats = [r[0] for r in s.query(Dataset.category)
            .filter(Dataset.datasource_id.in_(ds_ids)).distinct().all()]
    return wo_ids, {str(c) for c in cats if c}


def _in_review_scope(wo_ids: list, rev_cats: set):
    """复判工单归属过滤：workorder_id 精确匹配，历史空值行回退品类。"""
    cond = Detection.workorder_id.in_(wo_ids)
    if rev_cats:
        cond = cond | (Detection.workorder_id.is_(None)
                       & Detection.category.in_(sorted(rev_cats)))
    return cond


def _feedback_det_ids(s) -> set:
    """已有有效反馈的 detection_id 集合（一次查询，替代逐行 _has_feedback
    的 N+1）；已作废反馈不占用——对应检测会重新进入复核队列。"""
    return {r[0] for r in s.query(Feedback.detection_id)
            .filter(Feedback.invalidated == False).all()}  # noqa: E712


@router.get("/review/active_suggestions")
def api_active_suggestions(category: str, top_n: int = 10):
    """主动选样清单（M15b，引擎设计 §6，赛题点名"主动学习"）：

    引擎从灰区积累队列中挑"最不确定 × 最有代表性"的 top-N 返回，
    每张只问一次。返回项带 detection_id（关联最新检测记录），
    可直接进复核/反馈流程。处理清单 = 对每条提交复核或反馈即可。
    """
    from ..engine import get_engine
    return {"category": category,
            "items": get_engine().active_suggestions(category, top_n)}


@router.get("/review/incomplete_report")
def api_incomplete_report(window: int = 100, min_count: int = 3,
                          ratio_thresh: float = 0.15):
    """不完备报告（M14c，引擎设计 §3.3 J 特征演化前半段）：

    E_open 哨兵单帧徽标是"点"，本端点聚合成"面"——每品类最近 window 帧内
    open_alert 占比 ≥ ratio_thresh 且命中 ≥ min_count 时出具报告：
    "疑似体系外缺陷信号持续出现，现有特征组无法解释"。
    只读 DB（n_tiles JSON 内的 open_alert 落库字段），不触碰引擎。
    """
    with session_scope() as s:
        rows = (s.query(Detection)
                .order_by(Detection.id.desc()).limit(20000).all())
    per_cat: dict = {}
    for d in rows:
        bucket = per_cat.setdefault(d.category, {"n": 0, "hits": 0})
        if bucket["n"] >= window:
            continue
        bucket["n"] += 1
        if (d.n_tiles or {}).get("open_alert"):
            bucket["hits"] += 1
    reports = []
    for cat, b in sorted(per_cat.items()):
        ratio = b["hits"] / b["n"] if b["n"] else 0.0
        reports.append({
            "category": cat, "window_n": b["n"], "open_hits": b["hits"],
            "ratio": round(ratio, 3),
            "alert": bool(b["hits"] >= min_count and ratio >= ratio_thresh),
        })
    return {"window": window, "ratio_thresh": ratio_thresh,
            "reports": reports,
            "any_alert": any(r["alert"] for r in reports)}


def _gray_filter():
    return cast(Detection.n_tiles, String).like(_GRAY_LIKE)


class ReviewSubmitRequest(BaseModel):
    label: int                        # 0=正常 1=异常（操作员复核结论）
    comment: Optional[str] = None
    operator: str = "operator"
    region: Optional[List[float]] = None  # M6a 框选缺陷区 [x,y,w,h]（像素），
    #   仅 label=1 时换算归一化 box 喂给缺陷拦截通道；label=0 忽略


def _is_gray(det: Detection) -> bool:
    return (det.n_tiles or {}).get("decision") == "gray"


def _has_feedback(s, detection_id: int) -> bool:
    return (s.query(Feedback.id)
            .filter(Feedback.detection_id == detection_id,
                    Feedback.invalidated == False)  # noqa: E712
            .first()) is not None


@router.get("/review/queue")
def api_review_queue(category: Optional[str] = None, page: int = 1,
                     page_size: int = 20):
    """待复核队列：灰区检测 或 人工复判工单品类的全部检测，且尚无 feedback。

    复判工单（review_enabled）的检测全部进队列（不只灰区），复核提交后
    自动生成 feedback（feedback_type=review）即学。
    仅产线路径（pipeline/stream/plc）检测——试检旁路（upload/image）不进队列。
    """
    with session_scope() as s:
        rev_wo_ids, rev_cats = _review_scope(s)
        fb_ids = _feedback_det_ids(s)
        seen: dict = {}
        # 存档排除（2026-10-09）：灰区队列排除存档工单检测，
        # 保留 workorder_id 为 NULL 的孤儿记录
        arch = archived_workorder_ids(s)
        q = (s.query(Detection)
             .filter(_gray_filter(), Detection.target_type.in_(_LIVE_TYPES)))
        if arch:
            q = q.filter(Detection.workorder_id.is_(None)
                         | ~Detection.workorder_id.in_(arch))
        if category:
            q = q.filter(Detection.category == category)
        for d in q.order_by(Detection.id.desc()).all():
            if d.id not in fb_ids and _is_gray(d):
                seen[d.id] = d
        if rev_wo_ids:
            # 人工复判工单的全部检测（含非灰区）也进队列：
            # workorder_id 精确归属，历史空值行回退品类匹配
            q2 = (s.query(Detection)
                  .filter(_in_review_scope(rev_wo_ids, rev_cats),
                          Detection.target_type.in_(_LIVE_TYPES)))
            if category:
                q2 = q2.filter(Detection.category == category)
            for d in q2.order_by(Detection.id.desc()).all():
                if d.id not in fb_ids and not _is_gray(d):
                    seen[d.id] = d
        items = [{"id": d.id, "image_path": d.image_path,
                  "category": d.category, "final_score": d.final_score,
                  "decision": (d.n_tiles or {}).get("decision")
                  or ("anomaly" if d.is_anomaly else "normal"),
                  "slots": (d.n_tiles or {}).get("slots"),
                  "overlay_path": d.overlay_path,
                  "created_at": d.created_at.isoformat() if d.created_at else None}
                 for d in seen.values()]
        items.sort(key=lambda x: x["id"], reverse=True)
        total = len(items)
        start = (page - 1) * page_size
        return {"total": total, "page": page, "page_size": page_size,
                "items": items[start:start + page_size]}


@router.post("/review/{detection_id}")
def api_review_submit(detection_id: int, req: ReviewSubmitRequest):
    """提交复核结论：自动落 Feedback(feedback_type=review) + 引擎即学。

    404=检测记录不存在；409=该记录已复核/已有反馈。
    """
    if req.label not in (0, 1):
        raise HTTPException(status_code=422, detail="label 只能为 0（正常）或 1（异常）")
    with session_scope() as s:
        det = s.get(Detection, detection_id)
        if det is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")
        if _has_feedback(s, detection_id):
            raise HTTPException(status_code=409, detail="该记录已复核/已有反馈")
        decision = (det.n_tiles or {}).get("decision")

    out = sl_service.submit_feedback_record(
        detection_id, "review", req.label,
        comment=req.comment, operator=req.operator,
        region=req.region)
    log_action("review_submit",
               f"detection_id={detection_id} label={req.label} "
               f"origin_decision={decision}", user=req.operator)
    return {"feedback_id": out["feedback_id"],
            "engine_update": out["engine_update"],
            "note": out["note"], "category": out["category"],
            "pre": out["pre"], "post": out.get("post"),
            "learning_trace": out.get("learning_trace"),
            "model_version_before": out.get("model_version_before"),
            "model_version_after": out.get("model_version_after")}


@router.get("/review/stats")
def api_review_stats():
    """复核队列规模：pending_total（按品类分组）/ reviewed_today / reviewed_total。

    含人工复判工单品类的全部未复核检测（v4）。
    """
    with session_scope() as s:
        rev_wo_ids, rev_cats = _review_scope(s)
        fb_ids = _feedback_det_ids(s)
        # 存档排除（2026-10-09）：灰区统计排除存档工单检测，
        # 保留 workorder_id 为 NULL 的孤儿记录
        arch = archived_workorder_ids(s)
        rows_q = (s.query(Detection)
                  .filter(_gray_filter(),
                          Detection.target_type.in_(_LIVE_TYPES)))
        if arch:
            rows_q = rows_q.filter(Detection.workorder_id.is_(None)
                                   | ~Detection.workorder_id.in_(arch))
        rows = rows_q.all()
        pending: dict = {}
        for d in rows:
            if _is_gray(d) and d.id not in fb_ids:
                pending[d.category] = pending.get(d.category, 0) + 1
        if rev_wo_ids:
            q2 = (s.query(Detection)
                  .filter(_in_review_scope(rev_wo_ids, rev_cats),
                          Detection.target_type.in_(_LIVE_TYPES)).all())
            for d in q2:
                if d.id not in fb_ids and not _is_gray(d):
                    pending[d.category] = pending.get(d.category, 0) + 1
        today0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        q = s.query(Feedback).filter(Feedback.feedback_type == "review")
        reviewed_total = q.count()
        reviewed_today = q.filter(Feedback.created_at >= today0).count()
    return {"pending_total": sum(pending.values()),
            "pending_by_category": pending,
            "categories": sorted(pending),  # M6a：有积压的品类列表（UI 下拉直用）
            "reviewed_today": reviewed_today,
            "reviewed_total": reviewed_total}
