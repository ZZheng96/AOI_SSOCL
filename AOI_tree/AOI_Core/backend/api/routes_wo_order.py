"""工单制接口 - 工单 CRUD / 数据源挂接 / 体检 / 任务模板（自 routes_workorder 拆分）。"""
from fastapi import APIRouter, Depends, HTTPException

from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import (ConsolidationFeedback, DataSource, Dataset, Detection,
                         Feedback, Image, WorkOrder, WorkOrderSource,
                         WorkOrderTemplate)
from ._wo_shared import (_attach_source, _n_pending_review, _pipeline_summary,
                         _workorder_item, _workorder_sources, check_datasource,
                         check_workorder)
from .schemas import (WorkOrderCreateRequest, WorkOrderSourceRequest,
                      WorkOrderTemplateRequest, WorkOrderUpdateRequest, to_dict)

router = APIRouter()


# ══════════════════ 工单 ══════════════════
@router.get("/workorders")
def api_list_workorders(range: str = "all", archived: bool = False):
    """工单列表：每工单附数据条件、挂接数据源、品类集、
    统计（时间范围聚合，工单为统计基础；range=all/today/7d）与体检状态。
    archived=false 只列进行中工单（默认）；archived=true 只列已存档工单
    （存档统计页历史管理用）。"""
    days = {"today": 0, "7d": 7, "all": None}.get(range, None)
    with session_scope() as s:
        rows = (s.query(WorkOrder)
                .filter(WorkOrder.archived.is_(archived))
                .order_by(WorkOrder.id.desc()).all())
        items = []
        for r in rows:
            items.append(_workorder_item(s, r, range_days=days))
    return {"items": items}


@router.post("/workorders",
             dependencies=[Depends(require_role("engineer"))])
def api_create_workorder(req: WorkOrderCreateRequest):
    """新建工单（数据源必选；数据条件由所挂数据源属性自动聚合，
    工单不再手填；from_template 复制模板的数据源挂接 + 复判开关）。"""
    review = bool(req.review_enabled)
    ds_ids = [int(d) for d in (req.datasource_ids or [])]
    tpl_note = ""
    if req.from_template is not None:
        with session_scope() as s:
            tpl = s.get(WorkOrderTemplate, req.from_template)
        if tpl is None:
            raise HTTPException(status_code=404, detail="任务模板不存在")
        cfg = tpl.config or {}
        if not ds_ids:
            ds_ids = [int(d) for d in (cfg.get("datasource_ids") or [])]
        review = bool(cfg.get("review_enabled", review))
        tpl_note = f"（配置来自模板：{tpl.name}）"
    with session_scope() as s:
        if s.query(WorkOrder).filter(WorkOrder.name == req.name).first():
            raise HTTPException(status_code=409, detail=f"工单已存在：{req.name}")
        row = WorkOrder(name=req.name, label_tier="L0",
                        per_category=False, has_template=False,
                        review_enabled=review, auto_resume=bool(req.auto_resume),
                        note=(req.note or "") + tpl_note,
                        template_id=req.from_template)
        # 2026-09-13（前端反馈 #7）：产线一律手动启停——新建工单默认
        # 「已暂停」，模型准备好/激活投产也不自动开线，避免未验收模型
        # 或操作员未就绪时自动检测污染统计。
        row.pipeline_status = "paused"
        s.add(row)
        s.flush()
        wid = row.id
        for did in ds_ids:
            _attach_source(s, wid, int(did))
        # 2026-08-30：自动检测前置条件——所挂数据源的品类无激活模型时
        # 提示需先准备模型（产线默认 paused，准备并验收后手动启动）。
        cats = ({r[0] for r in
                 s.query(Dataset.category)
                 .filter(Dataset.datasource_id.in_(ds_ids)).distinct().all()}
                if ds_ids else set())
        from ..engine import get_engine
        engine = get_engine()
        not_ready = sorted(c for c in cats
                           if engine.current_version(c) is None)
        if not_ready:
            row.note = (row.note or "") + (
                f"（产线待命：品类 {'/'.join(not_ready)} 未准备模型，"
                f"准备并验收后请在「实时监控」手动启动产线）")
        else:
            row.note = (row.note or "") + (
                "（产线默认暂停：确认模型已验收后到「实时监控」手动启动）")
    log_action("create_workorder",
               f"name={req.name} review={review} sources={ds_ids}",
               extra={"workorder_id": wid})
    # 新建即体检：数据源声明档位按实际支撑自动校正（check.applied 记录调整项）
    return api_check_workorder(wid, apply=True)


@router.get("/workorders/{workorder_id}")
def api_get_workorder(workorder_id: int):
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        item = _workorder_item(s, wo)
        item["pipeline_summary"] = _pipeline_summary(s, wo)
        # 复判积压（前端反馈 v6-3）：开启人工复判的工单才计算，
        # 供监控页显示「待复核 N 条 · 积压限流已启动」
        if wo.review_enabled:
            item["review_pending"] = _n_pending_review(s, workorder_id)
        return item


@router.put("/workorders/{workorder_id}",
            dependencies=[Depends(require_role("engineer"))])
def api_update_workorder(workorder_id: int, req: WorkOrderUpdateRequest):
    """修改工单（全可选，只改传入字段；datasource_ids 传入时整体替换挂接）。"""
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        if req.name:
            dup = (s.query(WorkOrder)
                   .filter(WorkOrder.name == req.name,
                           WorkOrder.id != workorder_id)
                   .first())
            if dup is not None:
                raise HTTPException(status_code=409,
                                    detail=f"工单名已存在：{req.name}")
            wo.name = req.name
        if req.review_enabled is not None:
            wo.review_enabled = bool(req.review_enabled)
        if req.auto_resume is not None:
            wo.auto_resume = bool(req.auto_resume)
        if req.status:
            wo.status = req.status
        if req.note is not None:
            wo.note = req.note
        s.flush()
        if req.datasource_ids is not None:
            s.query(WorkOrderSource).filter(
                WorkOrderSource.workorder_id == workorder_id).delete(
                synchronize_session=False)
            for did in req.datasource_ids:
                _attach_source(s, workorder_id, int(did))
    log_action("update_workorder", f"id={workorder_id} name={wo.name}",
               extra={"workorder_id": workorder_id})
    return api_get_workorder(workorder_id)


@router.post("/workorders/{workorder_id}/datasources",
             dependencies=[Depends(require_role("engineer"))])
def api_set_workorder_sources(workorder_id: int, req: WorkOrderSourceRequest):
    """工单挂接数据源（整体替换列表；可导入新源也可选择已有源）。"""
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        for did in req.datasource_ids:
            if s.get(DataSource, int(did)) is None:
                raise HTTPException(status_code=404, detail=f"数据源不存在: {did}")
        s.query(WorkOrderSource).filter(
            WorkOrderSource.workorder_id == workorder_id).delete(
            synchronize_session=False)
        for did in req.datasource_ids:
            _attach_source(s, workorder_id, int(did))
    log_action("workorder_attach_sources",
               f"workorder={workorder_id} sources={req.datasource_ids}",
               extra={"workorder_id": workorder_id,
                      "datasource_ids": req.datasource_ids})
    # 挂接后体检 + 自动调整（用户澄清：导入数据后检查条件是否符合）
    return api_get_workorder(workorder_id)


@router.delete("/workorders/{workorder_id}",
               dependencies=[Depends(require_role("engineer"))])
def api_delete_workorder(workorder_id: int):
    """删除工单（2026-10-09 起为级联彻底删除）：

    用户裁决——「只要是删除，就不保留相关记录和数据」：级联删除该工单
    名下的检测记录及其反馈（含 SSOCL 巩固血缘），并释放产线认领的图片
    认领状态。想保留数据请改用 /archive 存档。
    """
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        name = wo.name
        # ① 该工单名下检测 → 反馈 → 巩固血缘（旧库 FK 为 NO ACTION，
        #    须手工先清子行）
        det_ids = [r[0] for r in s.query(Detection.id).filter(
            Detection.workorder_id == workorder_id).all()]
        if det_ids:
            fb_ids = [r[0] for r in s.query(Feedback.id).filter(
                Feedback.detection_id.in_(det_ids)).all()]
            if fb_ids:
                s.query(ConsolidationFeedback).filter(
                    ConsolidationFeedback.feedback_id.in_(fb_ids)).delete(
                    synchronize_session=False)
                s.query(Feedback).filter(
                    Feedback.id.in_(fb_ids)).delete(
                    synchronize_session=False)
            s.query(Detection).filter(
                Detection.id.in_(det_ids)).delete(
                synchronize_session=False)
        # ② 释放产线认领中的图片（未完成的回 pending，可重新认领）
        s.query(Image).filter(
            Image.claim_workorder_id == workorder_id,
            Image.processing_status != "done").update(
            {"claim_workorder_id": None, "claim_token": None,
             "claimed_at": None, "processing_status": "pending"},
            synchronize_session=False)
        s.query(Image).filter(
            Image.claim_workorder_id == workorder_id).update(
            {"claim_workorder_id": None}, synchronize_session=False)
        # ③ 挂接关系 + 工单本体
        s.query(WorkOrderSource).filter(
            WorkOrderSource.workorder_id == workorder_id).delete(
            synchronize_session=False)
        s.delete(wo)
    log_action("delete_workorder",
               f"id={workorder_id} name={name} cascade_detections={len(det_ids)}",
               extra={"workorder_id": workorder_id})
    return {"deleted": True, "name": name}


@router.post("/workorders/{workorder_id}/archive",
             dependencies=[Depends(require_role("engineer"))])
def api_archive_workorder(workorder_id: int):
    """存档工单：数据全部保留，转入历史管理（存档统计页）。

    存档工单不参与全局统计/复核队列/反馈列表；产线运行中/暂停的
    工单存档时一并停线（pipeline_status=stopped）。
    """
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        wo.archived = True
        if wo.pipeline_status in ("running", "paused"):
            wo.pipeline_status = "stopped"
        name = wo.name
    log_action("archive_workorder", f"id={workorder_id} name={name}",
               extra={"workorder_id": workorder_id})
    return {"archived": True, "id": workorder_id, "name": name}


@router.post("/workorders/{workorder_id}/unarchive",
             dependencies=[Depends(require_role("engineer"))])
def api_unarchive_workorder(workorder_id: int):
    """还原存档工单：回到进行中列表（产线保持 stopped，需手动启动）。"""
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        wo.archived = False
        name = wo.name
    log_action("unarchive_workorder", f"id={workorder_id} name={name}",
               extra={"workorder_id": workorder_id})
    return {"archived": False, "id": workorder_id, "name": name}


# ══════════════════ 工单体检（条件 vs 数据支撑）══════════════════
@router.post("/workorders/{workorder_id}/check",
             dependencies=[Depends(require_role("engineer"))])
def api_check_workorder(workorder_id: int, apply: bool = True):
    """体检工单（聚合条件）。apply=true 时对数据源自动校正档位。"""
    log_payload = None
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        check = check_workorder(s, wo)
        applied = None
        if apply and not check["matched"]:
            applied = {}
            for src in _workorder_sources(s, wo.id):
                dchk = check_datasource(s, src)
                if not dchk["matched"]:
                    src.label_tier = dchk["suggestion"]["label_tier"]
                    applied[f"source_{src.id}"] = {
                        "label_tier": src.label_tier}
            if applied:
                s.flush()
            check = check_workorder(s, wo)
            check["applied"] = applied
            # log_action 移到事务提交后：SQLite 单写锁，嵌套写会
            # 等 5s 报 "database is locked"（复现：体检自动校正档位路径）
            log_payload = applied
        out = _workorder_item(s, wo)
    if log_payload:
        log_action("workorder_check_apply",
                   f"id={workorder_id} {log_payload}",
                   extra={"workorder_id": workorder_id,
                          "applied": log_payload})
    out["check"] = check
    return out


# ══════════════════ 任务模板 ══════════════════
@router.get("/workorder-templates")
def api_list_templates():
    with session_scope() as s:
        rows = (s.query(WorkOrderTemplate)
                .order_by(WorkOrderTemplate.id.desc()).all())
        return {"items": [to_dict(r) for r in rows]}


@router.post("/workorder-templates",
             dependencies=[Depends(require_role("engineer"))])
def api_create_template(req: WorkOrderTemplateRequest):
    """把工单配置存为任务模板（v4：存数据源挂接 + 复判开关；条件随源自动算）。"""
    with session_scope() as s:
        if (s.query(WorkOrderTemplate)
                .filter(WorkOrderTemplate.name == req.name).first()):
            raise HTTPException(status_code=409, detail=f"模板已存在：{req.name}")
        row = WorkOrderTemplate(
            name=req.name, note=req.note or "",
            config={"datasource_ids": [int(d) for d in (req.datasource_ids or [])],
                    "review_enabled": bool(req.review_enabled)})
        s.add(row)
        s.flush()
        out = to_dict(row)
    log_action("create_workorder_template", f"name={req.name}",
               extra={"template_id": out["id"]})
    return out


@router.delete("/workorder-templates/{template_id}",
               dependencies=[Depends(require_role("engineer"))])
def api_delete_template(template_id: int):
    with session_scope() as s:
        row = s.get(WorkOrderTemplate, template_id)
        if row is None:
            raise HTTPException(status_code=404, detail="模板不存在")
        (s.query(WorkOrder)
         .filter(WorkOrder.template_id == template_id)
         .update({"template_id": None}, synchronize_session=False))
        s.delete(row)
    return {"deleted": True}
