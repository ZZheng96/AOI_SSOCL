"""工单制接口 - 数据源 / 分组方案 / 数据清理与体检（自 routes_workorder 拆分）。"""
from fastapi import APIRouter, Depends, HTTPException

from ..core.groups import compute_groups
from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import (DataSource, DataSourcePlan, Dataset, Detection,
                         Feedback, Image as ImageRow, PseudoAnomaly, Video,
                         VideoFrame, WorkOrderSource)
from ._wo_shared import TIER_CN, _int_or, _source_capability, check_datasource
from .schemas import (DataSourceCreateRequest, DataSourceUpdateRequest,
                      PlanImportRequest, PlanSaveRequest, to_dict)

router = APIRouter()


# ══════════════════ 数据源 ══════════════════
@router.get("/datasources")
def api_list_datasources():
    """数据源列表（附品类/图片/视频统计、支撑能力与体检状态）。"""
    with session_scope() as s:
        rows = s.query(DataSource).order_by(DataSource.id.desc()).all()
        items = []
        for r in rows:
            d = to_dict(r)
            d["capability"] = _source_capability(s, r.id)
            d["check"] = check_datasource(s, r)
            items.append(d)
    return {"items": items}


@router.post("/datasources",
             dependencies=[Depends(require_role("engineer"))])
def api_create_datasource(req: DataSourceCreateRequest):
    """新建数据源：声明模态 + 标注档位 + 采样配置 + 数据流类型 + 分组配置。"""
    modality = req.modality if req.modality in ("image", "video") else "image"
    tier = req.label_tier if req.label_tier in TIER_CN else "L0"
    s_type = req.source_type if req.source_type in ("local", "rtsp") else "local"
    plan_json = {}
    if req.plan_id is not None:
        with session_scope() as s:
            plan = s.get(DataSourcePlan, req.plan_id)
            if plan is not None:
                plan_json = dict(plan.config or {})
                plan_json["plan_id"] = plan.id
    with session_scope() as s:
        if s.query(DataSource).filter(DataSource.name == req.name).first():
            raise HTTPException(status_code=409, detail=f"数据源已存在：{req.name}")
        row = DataSource(name=req.name, modality=modality,
                         label_tier=tier,
                         source_type=s_type,
                         stream_config=req.stream_config or {},
                         per_category=bool(req.per_category),
                         has_template=bool(req.has_template),
                         sampling=req.sampling or {}, note=req.note or "",
                         pretrain_normal=_int_or(100, req.pretrain_normal),
                         pretrain_anomaly=_int_or(30, req.pretrain_anomaly),
                         batch_size=_int_or(30, req.batch_size, lo=1),
                         plan_json=plan_json)
        s.add(row)
        s.flush()
        out = to_dict(row)
    log_action("create_datasource",
               f"name={req.name} modality={modality} tier={tier} type={s_type}",
               extra={"datasource_id": out["id"], "modality": modality})
    return out


@router.delete("/datasources/{datasource_id}",
               dependencies=[Depends(require_role("engineer"))])
def api_delete_datasource(datasource_id: int):
    """删除数据源：解除工单挂接，并级联删除其下批次/图片（含检测/反馈/
    伪异常/视频帧关联清理），不留"未挂源"数据（前端反馈 v10：无未挂源概念）。"""
    with session_scope() as s:
        row = s.get(DataSource, datasource_id)
        if row is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        name = row.name
        s.query(WorkOrderSource).filter(
            WorkOrderSource.datasource_id == datasource_id).delete(
            synchronize_session=False)
        ds_ids = [r[0] for r in s.query(Dataset.id)
                  .filter(Dataset.datasource_id == datasource_id).all()]
        img_ids = []
        if ds_ids:
            # A11（2026-10-03）：批次下视频资产级联——此前删除路径漏
            # Video/VideoFrame，FK 强制（PRAGMA foreign_keys=ON）后删批次
            # 会 IntegrityError。顺序：帧行（引用 images/videos）→ 检测
            # video_id 断联（记录保留）→ 视频行。
            vid_ids = [r[0] for r in s.query(Video.id)
                       .filter(Video.dataset_id.in_(ds_ids)).all()]
            if vid_ids:
                s.query(VideoFrame).filter(
                    VideoFrame.video_id.in_(vid_ids)).delete(
                    synchronize_session=False)
                s.query(Detection).filter(
                    Detection.video_id.in_(vid_ids)).update(
                    {"video_id": None}, synchronize_session=False)
                s.query(Video).filter(Video.id.in_(vid_ids)).delete(
                    synchronize_session=False)
            img_ids = [r[0] for r in s.query(ImageRow.id)
                       .filter(ImageRow.dataset_id.in_(ds_ids)).all()]
            if img_ids:
                # 级联清理：反馈 → 检测 → 伪异常 → 视频帧关联（同 api_delete_image）
                det_ids = [r[0] for r in s.query(Detection.id)
                           .filter(Detection.image_id.in_(img_ids)).all()]
                if det_ids:
                    s.query(Feedback).filter(
                        Feedback.detection_id.in_(det_ids)).delete(
                        synchronize_session=False)
                s.query(Detection).filter(
                    Detection.image_id.in_(img_ids)).delete(
                    synchronize_session=False)
                s.query(PseudoAnomaly).filter(
                    PseudoAnomaly.base_image_id.in_(img_ids)).delete(
                    synchronize_session=False)
                s.query(VideoFrame).filter(
                    VideoFrame.image_id.in_(img_ids)).update(
                    {"image_id": None}, synchronize_session=False)
                s.query(ImageRow).filter(
                    ImageRow.id.in_(img_ids)).delete(
                    synchronize_session=False)
            s.query(Dataset).filter(Dataset.id.in_(ds_ids)).delete(
                synchronize_session=False)
        s.delete(row)
    log_action("delete_datasource", f"id={datasource_id} name={name}",
               extra={"datasource_id": datasource_id,
                      "datasets": len(ds_ids), "images": len(img_ids)})
    return {"deleted": True, "name": name,
            "datasets": len(ds_ids), "images": len(img_ids)}


@router.put("/datasources/{datasource_id}",
            dependencies=[Depends(require_role("engineer"))])
def api_update_datasource(datasource_id: int, req: DataSourceUpdateRequest):
    """编辑数据源：名称/标注档位/备注/采样配置可改，模态锁定（v3 CRUD）。"""
    with session_scope() as s:
        row = s.get(DataSource, datasource_id)
        if row is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        if req.name is not None and req.name.strip():
            name = req.name.strip()
            if name != row.name and \
                    s.query(DataSource).filter(DataSource.name == name).first():
                raise HTTPException(status_code=409,
                                    detail=f"数据源已存在：{name}")
            row.name = name
        if req.label_tier in TIER_CN:
            row.label_tier = req.label_tier
        if req.source_type in ("local", "rtsp"):
            row.source_type = req.source_type
        if req.stream_config is not None:
            row.stream_config = req.stream_config
        if req.per_category is not None:
            row.per_category = bool(req.per_category)
        if req.has_template is not None:
            row.has_template = bool(req.has_template)
        if req.note is not None:
            row.note = req.note
        if req.sampling is not None:
            row.sampling = req.sampling
        # 前端反馈 v9：预训练组/检测组配置（调整即动态重算分组）
        if req.pretrain_normal is not None:
            row.pretrain_normal = max(int(req.pretrain_normal), 0)
        if req.pretrain_anomaly is not None:
            row.pretrain_anomaly = max(int(req.pretrain_anomaly), 0)
        if req.batch_size is not None:
            row.batch_size = max(int(req.batch_size), 1)
        if req.plan_id is not None:
            plan = s.get(DataSourcePlan, req.plan_id)
            if plan is not None:
                pj = dict(plan.config or {})
                pj["plan_id"] = plan.id
                row.plan_json = pj
        s.flush()
        out = to_dict(row)
    log_action("update_datasource", f"id={datasource_id}",
               extra={"datasource_id": datasource_id,
                      "fields": req.dict(exclude_none=True)})
    return out


# ═══════ 前端反馈 v9：预训练组 / 检测组 + 分组方案 ═══════
@router.get("/datasources/{datasource_id}/groups")
def api_datasource_groups(datasource_id: int):
    """数据源分组（动态计算，按当前 N/M/批次量）：
    {cat: {pretrain:[ids], detect_batches:[[ids]...]}}；
    数据源已套用方案（plan_json.pretrain_ids）时按方案固定分配预训练组。"""
    with session_scope() as s:
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        pj = src.plan_json or {}
        groups = compute_groups(
            s, datasource_id, _int_or(100, src.pretrain_normal),
            _int_or(30, src.pretrain_anomaly), _int_or(30, src.batch_size, lo=1),
            pj.get("pretrain_ids"))
        return {"datasource_id": datasource_id,
                "pretrain_normal": src.pretrain_normal,
                "pretrain_anomaly": src.pretrain_anomaly,
                "batch_size": src.batch_size,
                "using_plan": bool(pj.get("pretrain_ids")),
                "plan_id": pj.get("plan_id"),
                "categories": groups}


@router.post("/datasources/{datasource_id}/plan",
             dependencies=[Depends(require_role("engineer"))])
def api_save_plan(datasource_id: int, req: PlanSaveRequest):
    """把当前分组分配另存为可复用方案（预训练组图片清单固化）。"""
    with session_scope() as s:
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        groups = compute_groups(
            s, datasource_id, _int_or(100, src.pretrain_normal),
            _int_or(30, src.pretrain_anomaly), _int_or(30, src.batch_size, lo=1))
        pretrain_ids: list = []
        for g in groups.values():
            pretrain_ids.extend(g.get("pretrain") or [])
        root = (s.query(Dataset.source_path)
                .filter(Dataset.datasource_id == datasource_id).first())
        config = {
            "pretrain_normal": src.pretrain_normal,
            "pretrain_anomaly": src.pretrain_anomaly,
            "batch_size": src.batch_size,
            "pretrain_ids": pretrain_ids,
            "source_root": (root[0] if root else None),
        }
        plan = DataSourcePlan(name=req.name, note=req.note or "",
                              source_root=config["source_root"],
                              config=config)
        s.add(plan)
        s.flush()
        # 回写数据源：套用刚保存的方案
        pj = dict(config)
        pj["plan_id"] = plan.id
        src.plan_json = pj
        out = to_dict(plan)
    log_action("datasource_save_plan", f"src={datasource_id} plan={out['id']}",
               extra={"datasource_id": datasource_id, "plan_id": out["id"]})
    return out


@router.get("/datasource-plans")
def api_list_plans():
    """分组方案列表（新建数据源/编辑数据源时可选复用）。"""
    with session_scope() as s:
        rows = (s.query(DataSourcePlan)
                .order_by(DataSourcePlan.id.desc()).all())
        items = []
        for r in rows:
            d = to_dict(r)
            cfg = r.config or {}
            d["pretrain_normal"] = cfg.get("pretrain_normal")
            d["pretrain_anomaly"] = cfg.get("pretrain_anomaly")
            d["batch_size"] = cfg.get("batch_size")
            d["n_pretrain"] = len(cfg.get("pretrain_ids") or [])
            items.append(d)
        return {"items": items}


@router.get("/datasource-plans/{plan_id}/export")
def api_export_plan(plan_id: int):
    """导出分组方案为本地 JSON 文件（可拷贝/分享/再导入）。"""
    import json as _json

    from fastapi.responses import Response
    with session_scope() as s:
        plan = s.get(DataSourcePlan, plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="方案不存在")
        body = {"name": plan.name, "note": plan.note,
                "source_root": plan.source_root, "config": plan.config}
    return Response(content=_json.dumps(body, ensure_ascii=False, indent=2),
                    media_type="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="plan_{plan_id}.json"'})


@router.post("/datasource-plans/import",
             dependencies=[Depends(require_role("engineer"))])
def api_import_plan(req: PlanImportRequest):
    """从 JSON 内容导入分组方案（复用预训练组分配）。"""
    from datetime import datetime
    cfg = req.config or {}
    name = (req.name or cfg.get("name")
            or f"方案_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with session_scope() as s:
        if s.query(DataSourcePlan).filter(
                DataSourcePlan.name == name).first():
            raise HTTPException(status_code=409, detail=f"方案已存在：{name}")
        plan = DataSourcePlan(name=name, note=req.note or "",
                              source_root=cfg.get("source_root"),
                              config=cfg)
        s.add(plan)
        s.flush()
        out = to_dict(plan)
    return out


@router.delete("/datasource-plans/{plan_id}",
               dependencies=[Depends(require_role("engineer"))])
def api_delete_plan(plan_id: int):
    """删除分组方案。"""
    with session_scope() as s:
        plan = s.get(DataSourcePlan, plan_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="方案不存在")
        s.delete(plan)
    return {"deleted": True, "plan_id": plan_id}


@router.post("/datasources/{datasource_id}/attach-legacy",
             dependencies=[Depends(require_role("engineer"))])
def api_attach_legacy(datasource_id: int):
    """存量一键归入：把所有未挂源批次挂到该数据源（前端反馈 v9）。"""
    with session_scope() as s:
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        n = (s.query(Dataset)
             .filter(Dataset.datasource_id.is_(None))
             .update({"datasource_id": datasource_id},
                     synchronize_session=False))
        s.flush()
    log_action("datasource_attach_legacy",
               f"src={datasource_id} batches={n}",
               extra={"datasource_id": datasource_id, "batches": n})
    return {"attached": n, "datasource_id": datasource_id}


@router.post("/datasets/{dataset_id}/attach",
             dependencies=[Depends(require_role("engineer"))])
def api_attach_dataset(dataset_id: int, datasource_id: int):
    """把批次挂到数据源（未挂源批次 → 工单可用的前提；用户实测：
    数据树有 mvtec 批次但工单对话框无数据源可选）。"""
    with session_scope() as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail="批次不存在")
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        ds.datasource_id = datasource_id
        name = ds.name
        s.flush()
    log_action("dataset_attach", f"dataset={dataset_id} -> source={datasource_id}",
               extra={"dataset_id": dataset_id, "datasource_id": datasource_id})
    return {"attached": True, "dataset_id": dataset_id,
            "datasource_id": datasource_id, "name": name}


@router.delete("/datasources/{datasource_id}/data",
               dependencies=[Depends(require_role("engineer"))])
def api_clear_datasource_data(datasource_id: int):
    """清空数据源下全部批次与图片（保留源登记；v8 更新数据→重建用）。"""
    with session_scope() as s:
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        ds_ids = [r[0] for r in s.query(Dataset.id)
                  .filter(Dataset.datasource_id == datasource_id).all()]
        n_img = 0
        if ds_ids:
            img_ids = [r[0] for r in s.query(ImageRow.id)
                       .filter(ImageRow.dataset_id.in_(ds_ids)).all()]
            if img_ids:
                n_img = len(img_ids)
                (s.query(Detection)
                 .filter(Detection.image_id.in_(img_ids))
                 .delete(synchronize_session=False))
            (s.query(ImageRow)
             .filter(ImageRow.dataset_id.in_(ds_ids))
             .delete(synchronize_session=False))
            (s.query(Dataset)
             .filter(Dataset.datasource_id == datasource_id)
             .delete(synchronize_session=False))
    log_action("clear_datasource_data", f"id={datasource_id} images={n_img}",
               extra={"datasource_id": datasource_id})
    return {"cleared": True, "images": n_img}


@router.post("/datasources/{datasource_id}/check",
             dependencies=[Depends(require_role("engineer"))])
def api_check_datasource(datasource_id: int, apply: bool = True):
    """数据源体检：声明档位 vs 数据支撑；apply=true 时自动校正档位。"""
    with session_scope() as s:
        src = s.get(DataSource, datasource_id)
        if src is None:
            raise HTTPException(status_code=404, detail="数据源不存在")
        check = check_datasource(s, src)
        applied = None
        if apply and not check["matched"]:
            src.label_tier = check["suggestion"]["label_tier"]
            applied = {"label_tier": src.label_tier}
            s.flush()
            check = check_datasource(s, src)
            check["applied"] = applied
        out = to_dict(src)
    out["check"] = check
    return out
