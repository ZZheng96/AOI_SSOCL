"""操作员反馈与自学习闭环接口（赛题问题三）。

M2 起反馈即学：POST /feedback 落库后同步调 engine.submit_feedback
（秒级生效，实测单条 ~130ms，满足 1s 红线）；/self_learning/update
语义改为"巩固落盘"（engine.consolidate → 新快照版本）。
M5a 起即学路径抽为 self_learning.service.submit_feedback_record，
与 routes_review（灰区复核）共用。
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.security import require_role
from ..core.tasks import task_manager
from ..db.database import log_action, session_scope
from ..db.models import Detection, Feedback
from ..self_learning import service as sl_service
from .schemas import FeedbackCreateRequest, SelfUpdateRequest, to_dict

router = APIRouter()


# ── 反馈 ─────────────────────────────────────────────────────
@router.post("/feedback")
def api_create_feedback(req: FeedbackCreateRequest):
    """提交反馈：写 Feedback 表 + 引擎即学（submit_feedback）+ 今日统计。

    引擎未 prepare / 品类无快照时不报错中断：落库照常，
    响应 engine_update=None 并注明"模型未就绪，反馈仅落库"。
    """
    try:
        out = sl_service.submit_feedback_record(
            req.detection_id, req.feedback_type, req.operator_label,
            defect_type=req.defect_type, comment=req.comment,
            region=req.region, operator=req.operator)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"id": out["feedback_id"], "engine_update": out["engine_update"],
            "note": out["note"], "pre": out["pre"]}


@router.get("/feedback")
def api_list_feedback(consumed: Optional[bool] = None,
                      feedback_type: Optional[str] = None,
                      category: Optional[str] = None,
                      page: int = 1, page_size: int = 50):
    """反馈列表，附带关联检测记录的展示字段；可按类型/品类过滤。"""
    with session_scope() as s:
        q = s.query(Feedback)
        if consumed is not None:
            q = q.filter(Feedback.consumed == consumed)
        if feedback_type:
            q = q.filter(Feedback.feedback_type == feedback_type)
        if category:
            q = q.join(Detection, Feedback.detection_id == Detection.id) \
                 .filter(Detection.category == category)
        total = q.count()
        rows = (q.order_by(Feedback.id.desc())
                 .offset((page - 1) * page_size).limit(page_size).all())
        # 批量取关联检测（一次 IN 查询），避免逐条 s.get 的 N+1
        det_ids = [fb.detection_id for fb in rows
                   if fb.detection_id is not None]
        det_map = {}
        if det_ids:
            det_map = {det.id: det for det in
                       s.query(Detection)
                       .filter(Detection.id.in_(det_ids)).all()}
        items = []
        for fb in rows:
            d = to_dict(fb)
            det = det_map.get(fb.detection_id)
            if det is not None:
                d.update(image_path=det.image_path,
                         overlay_path=det.overlay_path,
                         final_score=det.final_score,
                         is_anomaly=det.is_anomaly,
                         category=det.category)
            items.append(d)
        return {"total": total, "items": items}


@router.post("/feedback/{feedback_id}/invalidate")
def api_invalidate_feedback(feedback_id: int, operator: str = "operator"):
    """作废反馈（标错撤回）：标记 invalidated。

    作废后：不计入反馈统计/翻案曲线/错检集，不占用复核队列（对应检测
    重新可复核），巩固时不再作为未消费反馈计数。
    注意：引擎"即学"是单向的——已被学习的反馈对模型的影响无法在线撤销，
    需到模型管理回滚到反馈前版本（响应 was_consumed 提示该情况）。
    """
    with session_scope() as s:
        fb = s.get(Feedback, feedback_id)
        if fb is None:
            raise HTTPException(status_code=404, detail="反馈不存在")
        if fb.invalidated:
            return {"id": fb.id, "invalidated": True, "was_consumed": fb.consumed,
                    "note": "该反馈已是作废状态"}
        fb.invalidated = True
        was_consumed = bool(fb.consumed)
    log_action("feedback_invalidate", f"feedback_id={feedback_id}",
               user=operator)
    note = "已作废：不计入统计与曲线，对应检测重新进入复核队列。"
    if was_consumed:
        note += "该反馈已被引擎即学消费——模型内的影响需回滚到反馈前版本消除。"
    return {"id": feedback_id, "invalidated": True,
            "was_consumed": was_consumed, "note": note}


@router.get("/feedback/summary")
def api_feedback_summary(workorder_id: Optional[int] = None):
    """反馈统计 + 各品类快照/未巩固反馈状态。

    workorder_id 给定则 summary 按工单口径（关联检测图属工单数据源批次）；
    缺省为全局口径（向后兼容）。pending 段是品类级模型状态，不随工单过滤。
    """
    return {"summary": sl_service.feedback_summary(workorder_id=workorder_id),
            "pending": sl_service.pending_update_status()}


# ── 自学习 ───────────────────────────────────────────────────
@router.get("/self_learning/status")
def api_self_learning_status():
    return sl_service.pending_update_status()


@router.get("/self_learning/insight")
def api_self_learning_insight(category: str):
    """M14b：在线学习引擎内部状态（双库/缺陷样例库/孵育头/权重门控）。"""
    from ..engine import get_engine
    return get_engine().learning_insight(category)


@router.post("/self_learning/update",
             dependencies=[Depends(require_role("engineer"))])
def api_self_learning_update(req: SelfUpdateRequest):
    """后台任务：巩固落盘（engine.consolidate → 新快照版本并登记 Model）。"""
    from . import routes_models

    def fn(progress_cb):
        progress_cb(0, 2, "巩固落盘（快照固化）...")
        out = sl_service.consolidate(req.category, note=req.note or "")
        progress_cb(1, 2, "登记 Model 行...")
        routes_models.register_demo5_snapshots()
        progress_cb(2, 2, "完成")
        return out

    task_id = task_manager.submit("self_update", req.model_dump(), fn)
    log_action("self_update", f"category={req.category}")
    return {"task_id": task_id}


# ── 2026-08-31 双系统联合：外部传统 CV 复判接口 ──────────────
# 场景：树枝系统（如 PCB_Dual）用传统 CV 引擎对同一张图判定后，
# 把判定作为"机器复判"结论回传主干 AOI_Core，驱动反馈即学。
# 设计口径：主干保持通用——不耦合任何具体外部系统，source 字段
# 记录来源（如 "PCB_Dual:shift_smt"），operator 记 "external_cv:<source>"。
# 路径选 /feedback/traditional（routes_review 的 /review/{detection_id}
# 是 int 路径参数，避免路由冲突）。
class TraditionalReviewRequest(BaseModel):
    image_path: str                      # 待复判图片路径（必须已登记或可读）
    category: str                        # 品类（未检测过时先检测落库用）
    label: int                           # 传统 CV 判定：0=正常 1=异常
    defect_type: Optional[str] = None    # 传统 CV 检出的具体缺陷类型（如"移位"）
    source: str = "traditional_cv"       # 来源描述（如 PCB_Dual:shift_smt）
    comment: Optional[str] = None
    region: Optional[list] = None        # 传统 CV 缺陷框 [x,y,w,h]（可选，用于漏检拦截）


@router.post("/feedback/traditional")
def api_feedback_traditional(req: TraditionalReviewRequest):
    """外部传统 CV 判定回传 → 作为复判依据驱动 AOI 反馈即学。

    流程：
    1. 图片路径校验（is_path_allowed 白名单 或 已登记路径）
    2. 找该图最近检测记录；无则先检测落库（target_type="review"，
       走产线一致的可视化/归档路径）
    3. 自动映射反馈类型：
       - AOI 判 NG + 传统 OK → false_positive（误检反馈，抑制虚高）
       - AOI 判 OK + 传统 NG → false_negative（漏检反馈，进缺陷拦截）
       - 两侧一致 → confirmed（确认正确，强化）
    4. submit_feedback_record 即学，返回 engine_update
    """
    if req.label not in (0, 1):
        raise HTTPException(status_code=422, detail="label 只能为 0（正常）或 1（异常）")
    from ..core.security import is_path_allowed
    from ..db.models import Detection as DetectionRow
    from ..pipeline.service import get_detection_service

    img = req.image_path
    if not is_path_allowed(img):
        raise HTTPException(status_code=403,
                            detail="图片路径不在白名单（storage 内或已登记）")

    # 1) 找最近检测记录（该图）
    with session_scope() as s:
        det = (s.query(DetectionRow)
               .filter(DetectionRow.image_path == img)
               .order_by(DetectionRow.id.desc()).first())
        det_id = det.id if det is not None else None
        aoi_ng = bool(det.is_anomaly) if det is not None else None

    # 2) 无检测记录 → 先检测落库（target_type=review 走产线口径）
    if det_id is None:
        try:
            out = get_detection_service().detect_image(
                img, req.category, target_type="review", persist=True)
            det_id = out.get("detection_id")
            aoi_ng = out.get("is_anomaly")
        except Exception as e:  # noqa: BLE001 品类未准备等
            raise HTTPException(status_code=400,
                                detail=f"先检测失败（品类未准备？）：{e}")
    else:
        # 已有检测记录 → 幂等：该记录已有有效反馈则拒绝重复复判
        with session_scope() as s:
            has_fb = (s.query(Feedback.id)
                      .filter(Feedback.detection_id == det_id,
                              Feedback.invalidated == False)  # noqa: E712
                      .first()) is not None
        if has_fb:
            raise HTTPException(
                status_code=409,
                detail=f"该图检测记录 {det_id} 已有反馈（可先作废旧反馈再重判）")

    # 3) 映射反馈类型（以传统 CV 判定为复判结论，与 AOI 判定对比）
    trad_ng = bool(req.label == 1)
    if aoi_ng is None:
        fb_type = "review"
    elif aoi_ng and not trad_ng:
        fb_type = "false_positive"      # AOI 报 NG 但传统 CV 判 OK → 疑似误检
    elif not aoi_ng and trad_ng:
        fb_type = "false_negative"      # AOI 判 OK 但传统 CV 判 NG → 漏检
    else:
        fb_type = "confirmed"           # 两侧一致 → 确认正确

    operator = f"external_cv:{req.source}"
    try:
        out = sl_service.submit_feedback_record(
            det_id, fb_type, req.label,
            defect_type=req.defect_type,
            comment=req.comment or f"传统CV复判 source={req.source}",
            region=req.region, operator=operator)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    log_action("feedback_traditional",
               f"detection_id={det_id} image={img} trad_label={req.label} "
               f"fb_type={fb_type} aoi_ng={aoi_ng} source={req.source}",
               user=operator)
    return {"detection_id": det_id, "feedback_id": out["feedback_id"],
            "feedback_type": fb_type, "aoi_ng": aoi_ng,
            "trad_ng": trad_ng, "engine_update": out["engine_update"],
            "pre": out["pre"], "note": out["note"]}
