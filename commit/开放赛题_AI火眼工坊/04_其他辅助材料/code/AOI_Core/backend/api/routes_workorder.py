"""工单制核心接口（W-workorder，2026-08-27 前端反馈 v2 重构）。

契约（与用户澄清对齐）：
- 工单 = 产线任务实例（如"3号线 SMT 质检"），下挂数据源，可含多品类，
  可保存为任务模板 / 从模板导入配置；
- 数据源 = 独立登记的数据实体（名称 + 模态图片/视频 + 采样配置），
  可被多个工单复用；数据按数据源管理（数据源 -> 品类 -> 批次）；
- 数据条件在工单级声明，结构为"三档 + 两开关"（非平铺五选一）：
  * 标注档位（信息量递进，选高档自动兼容低档）：
    L0 仅正常图 / L1a 图像级标注 / L1b 缺陷位置标注
  * 两个可叠加开关：品类独立（数据已分品类）/ 模板比对（提供模板图）
- 导入数据后体检：声明条件 vs 数据实际支撑能力，不符给警告并
  自动调整为合适条件（可修复数据后再改工单、再次体检）；
- 统计以工单为基础：工单行汇总 + 选中切换 + 时间范围（today/7d/all）。
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func

from ..core.groups import compute_groups
from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import (DataSource, DataSourcePlan, Dataset, Detection,
                         Feedback, Image as ImageRow, Model, PseudoAnomaly,
                         Video, VideoFrame, WorkOrder, WorkOrderSource,
                         WorkOrderTemplate)
from .schemas import (DataSourceCreateRequest, DataSourceUpdateRequest,
                      PlanImportRequest, PlanSaveRequest,
                      QueueDatasetRequest, WorkOrderCreateRequest,
                      WorkOrderSourceRequest, WorkOrderTemplateRequest,
                      WorkOrderUpdateRequest, to_dict)

router = APIRouter()

# 标注档位（递进）与开关（可叠加）的中文名 -- UI 不露 L 代号
TIER_CN = {"L0": "仅正常图", "L1a": "图像级标注", "L1b": "缺陷位置标注"}
TIER_RANK = {"L0": 0, "L1a": 1, "L1b": 2}
GENERIC_CATEGORY = "通用"   # 无品类说明的数据兜底品类


def _int_or(default: int, v, lo: int = 0) -> int:
    """配置值解析：None → 默认；否则 clamp 下界 lo。0 是合法值
    （0 正常图 + 0 异常图 = 全部进检测组，不进预训练组）。"""
    if v is None:
        return default
    return max(int(v), lo)


def condition_text(cond) -> str:
    """数据条件 -> 中文串（三档 + 两开关，不露代号；吃 dict 或 ORM）。"""
    tier = cond.get("label_tier") if isinstance(cond, dict) \
        else getattr(cond, "label_tier", None)
    per = bool(cond.get("per_category")) if isinstance(cond, dict) \
        else bool(getattr(cond, "per_category", False))
    tpl = bool(cond.get("has_template")) if isinstance(cond, dict) \
        else bool(getattr(cond, "has_template", False))
    parts = [TIER_CN.get(str(tier or ""), str(tier or ""))]
    if per:
        parts.append("品类独立")
    if tpl:
        parts.append("模板比对")
    return "＋".join(p for p in parts if p)


def _workorder_sources(s, workorder_id: int) -> list:
    """工单挂接的数据源列表。"""
    return (s.query(DataSource)
            .join(WorkOrderSource, WorkOrderSource.datasource_id == DataSource.id)
            .filter(WorkOrderSource.workorder_id == workorder_id)
            .order_by(DataSource.id).all())


def _aggregate_conditions(s, sources: list) -> dict:
    """工单数据条件 = 所挂数据源属性自动聚合（前端反馈 v4：工单不再手填）：
    - 标注档位取源声明最高档；品类并集（有品类即「品类独立」）；
    - 模板比对（任一源有模板图）。"""
    tiers = [str(src.label_tier or "") for src in sources
             if str(src.label_tier or "") in TIER_RANK]
    label_tier = max(tiers, key=lambda t: TIER_RANK[t]) if tiers else "L0"
    cats: list = []
    has_template = False
    for src in sources:
        cap = _source_capability(s, src.id)
        cats.extend(cap.get("categories") or [])
        if (cap.get("n_templates") or 0) > 0:
            has_template = True
    cats = sorted(set(cats))
    # 前端反馈 v6：品类独立/模板比对 = 数据源声明 或 数据实际支撑
    per_category = bool(cats) or any(bool(src.per_category) for src in sources)
    has_template = has_template or any(bool(src.has_template) for src in sources)
    return {"label_tier": label_tier, "per_category": per_category,
            "has_template": has_template, "categories": cats}


def _wo_conditions(wo) -> dict:
    # 兼容旧调用：ORM 工单的三档两开关（已废弃，聚合取代）
    return {"label_tier": wo.label_tier, "per_category": bool(wo.per_category),
            "has_template": bool(wo.has_template)}


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


def check_datasource(s, src: DataSource) -> dict:
    """数据源体检：声明（档位/分品类/模板）vs 数据实际支撑。"""
    cap = _source_capability(s, src.id)
    declared = {"label_tier": src.label_tier,
                "per_category": bool(src.per_category),
                "has_template": bool(src.has_template)}
    sugg_tier = cap["tier"] or "L0"
    matched = TIER_RANK.get(src.label_tier, 0) <= TIER_RANK.get(sugg_tier, -1)
    warnings = []
    if not matched:
        warnings.append(
            f"「{src.name}」声明档位「{TIER_CN.get(src.label_tier, src.label_tier)}」"
            f"高于数据实际支撑，建议调整为「{TIER_CN.get(sugg_tier, sugg_tier)}」")
    if src.has_template and (cap.get("n_templates") or 0) == 0:
        warnings.append(
            f"「{src.name}」声明提供模板图，但尚无模板图：可把模板图放入 "
            f"template 子目录或按命名规则 A_tpl.png 重新导入")
    if src.per_category and not (cap.get("categories") or []):
        warnings.append(
            f"「{src.name}」声明已分品类，但数据尚无品类说明（将归入「通用」）；"
            "按 {数据源根}/{品类}/ 结构导入即可自动分品类")
    return {"capability": cap, "declared": declared,
            "suggestion": {"label_tier": sugg_tier,
                           "per_category": bool(cap.get("categories")),
                           "has_template": (cap.get("n_templates") or 0) > 0},
            "warnings": warnings, "matched": matched}


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


def _source_capability(s, datasource_id: int) -> dict:
    """单数据源数据支撑能力（体检原料，纯 DB 统计）。

    - tier：L1b（有掩码+缺陷标注）/ L1a（有缺陷标注）/ L0（有正常图）/ None（空）
    - categories：数据源内品类集合（仅"通用"= 无品类说明）
    - n_templates：模板图数（split=template）
    """
    ds_ids = [r[0] for r in s.query(Dataset.id).filter(
        Dataset.datasource_id == datasource_id).all()]
    cap = {"n_normal": 0, "n_anomaly": 0, "n_masks": 0, "n_images": 0,
           "n_videos": 0, "categories": [], "n_templates": 0, "tier": None}
    if not ds_ids:
        return cap
    rows = (s.query(ImageRow.label, func.count(ImageRow.id))
            .filter(ImageRow.dataset_id.in_(ds_ids)).group_by(ImageRow.label).all())
    for lbl, n in rows:
        if lbl == "normal":
            cap["n_normal"] = int(n)
        elif lbl == "anomaly":
            cap["n_anomaly"] = int(n)
    cap["n_images"] = cap["n_normal"] + cap["n_anomaly"] + sum(
        n for lbl, n in rows if lbl not in ("normal", "anomaly"))
    for (params,) in s.query(Dataset.params).filter(
            Dataset.id.in_(ds_ids)).all():
        cap["n_masks"] += int((params or {}).get("n_masks", 0) or 0)
    cats = [r[0] for r in s.query(ImageRow.category).filter(
        ImageRow.dataset_id.in_(ds_ids)).distinct().all()]
    cap["categories"] = sorted(cats)
    cap["n_templates"] = (s.query(func.count(ImageRow.id))
                          .filter(ImageRow.dataset_id.in_(ds_ids),
                                  ImageRow.split == "template").scalar() or 0)
    cap["n_videos"] = (s.query(func.count(Video.id))
                       .filter(Video.dataset_id.in_(ds_ids)).scalar() or 0)
    if cap["n_masks"] > 0 and cap["n_anomaly"] > 0:
        cap["tier"] = "L1b"
    elif cap["n_anomaly"] > 0:
        cap["tier"] = "L1a"
    elif cap["n_normal"] > 0:
        cap["tier"] = "L0"
    return cap


# ══════════════════ 工单 ══════════════════
@router.get("/workorders")
def api_list_workorders(range: str = "all"):
    """工单列表：每工单附数据条件、挂接数据源、品类集、
    统计（时间范围聚合，工单为统计基础；range=all/today/7d）与体检状态。"""
    days = {"today": 0, "7d": 7, "all": None}.get(range, None)
    with session_scope() as s:
        rows = s.query(WorkOrder).order_by(WorkOrder.id.desc()).all()
        items = []
        for r in rows:
            items.append(_workorder_item(s, r, range_days=days))
    return {"items": items}


def _workorder_item(s, wo: WorkOrder, range_days: Optional[int] = None) -> dict:
    """单工单聚合 dict（挂接源/聚合条件/品类/统计/体检）。
    range_days=None 全量，0 今日，7 近7天。"""
    sources = _workorder_sources(s, wo.id)
    cats = []
    n_images = 0
    for src in sources:
        cap = _source_capability(s, src.id)
        cats.extend(cap["categories"])
        n_images += cap["n_images"]
    cats = sorted(set(cats))
    # 数据条件 = 数据源属性自动聚合（前端反馈 v4，工单不再手填）
    conditions = _aggregate_conditions(s, sources)
    det_q = s.query(func.count(Detection.id))
    an_q = (s.query(func.count(Detection.id))
            .filter(Detection.is_anomaly.is_(True)))
    if wo.review_enabled:
        # 人工复判：统计只认「已复核」的检测（复核提交自动生成 feedback）
        det_q = det_q.filter(
            s.query(Feedback.id).filter(
                Feedback.detection_id == Detection.id).exists())
        an_q = an_q.filter(
            s.query(Feedback.id).filter(
                Feedback.detection_id == Detection.id).exists())
    if cats:
        det_q = det_q.filter(Detection.category.in_(cats))
        an_q = an_q.filter(Detection.category.in_(cats))
    else:
        det_q = det_q.filter(Detection.id < 0)   # 无品类工单恒为 0
        an_q = an_q.filter(Detection.id < 0)
    if range_days == 0:
        day_start = datetime.combine(date.today(), dtime.min)
        det_q = det_q.filter(Detection.created_at >= day_start)
        an_q = an_q.filter(Detection.created_at >= day_start)
    elif range_days:
        since = datetime.combine(date.today() - timedelta(days=range_days - 1),
                                 dtime.min)
        det_q = det_q.filter(Detection.created_at >= since)
        an_q = an_q.filter(Detection.created_at >= since)
    n_ins = int(det_q.scalar() or 0)
    n_an = int(an_q.scalar() or 0)
    check = check_workorder(s, wo)
    return {
        "id": wo.id, "name": wo.name, "note": wo.note, "status": wo.status,
        "review_enabled": bool(wo.review_enabled),
        "pipeline_status": wo.pipeline_status,
        "auto_resume": bool(wo.auto_resume),
        "conditions": conditions,
        "conditions_cn": condition_text(conditions),
        "datasources": [{"id": x.id, "name": x.name, "modality": x.modality}
                        for x in sources],
        "categories": cats, "n_images": n_images,
        "stats": {"inspected": n_ins, "anomaly": n_an,
                  "rate": round(n_an / n_ins, 4) if n_ins else 0.0},
        "check": check,
        "template_id": wo.template_id,
        "created_at": wo.created_at,
    }


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
        s.add(row)
        s.flush()
        wid = row.id
        for did in ds_ids:
            _attach_source(s, wid, int(did))
        # 2026-08-30：自动检测前置条件——所挂数据源的品类无激活模型时
        # 产线默认"待命"（paused），防未准备即自动检测产生高误报并污染
        # 不良率统计/熔断窗口（MPDD 评测 P1）。准备模型后手动启动。
        cats = ({r[0] for r in
                 s.query(Dataset.category)
                 .filter(Dataset.datasource_id.in_(ds_ids)).distinct().all()}
                if ds_ids else set())
        from ..engine import get_engine
        engine = get_engine()
        not_ready = sorted(c for c in cats
                           if engine.current_version(c) is None)
        if not_ready:
            row.pipeline_status = "paused"
            row.note = (row.note or "") + (
                f"（产线待命：品类 {'/'.join(not_ready)} 未准备模型，"
                f"准备并验收后请手动启动产线）")
    log_action("create_workorder",
               f"name={req.name} review={review} sources={ds_ids}",
               extra={"workorder_id": wid})
    # 新建即体检：数据源声明档位按实际支撑自动校正（check.applied 记录调整项）
    return api_check_workorder(wid, apply=True)


def _attach_source(s, workorder_id: int, datasource_id: int) -> None:
    exists = (s.query(WorkOrderSource)
              .filter(WorkOrderSource.workorder_id == workorder_id,
                      WorkOrderSource.datasource_id == datasource_id).first())
    if exists is None:
        s.add(WorkOrderSource(workorder_id=workorder_id,
                              datasource_id=datasource_id))


def _n_undetected_by_ids(s, image_ids: list) -> int:
    """图片 id 列表内尚无任何检测记录的图数（普通检测批次的积压口径）。"""
    if not image_ids:
        return 0
    det_exists = (s.query(Detection.id)
                  .filter(Detection.image_id == ImageRow.id).exists())
    return int(s.query(func.count(ImageRow.id))
               .filter(ImageRow.id.in_(image_ids), ~det_exists).scalar() or 0)


def _n_pending_requeue_by_ids(s, image_ids: list, t_req) -> int:
    """回队检测批次待重检图数：无检测 或 最近一次检测早于回队时刻。

    与 pipeline_service._consume_one 的挑选口径一致。
    """
    if not image_ids:
        return 0
    latest = dict(s.query(Detection.image_id, func.max(Detection.created_at))
                  .filter(Detection.image_id.in_(image_ids))
                  .group_by(Detection.image_id).all())
    n = 0
    for img_id in image_ids:
        t = latest.get(img_id)
        if t is None or (t_req is not None and t < t_req):
            n += 1
    return n


def _batch_stats_by_ids(s, image_ids: list) -> dict:
    """检测批次统计（按图片 id 列表，而非导入批次 Dataset）。"""
    if not image_ids:
        return {"n_images": 0, "n_detected": 0, "n_anomaly": 0,
                "n_feedback": 0, "bad_rate": 0.0}
    dets = (s.query(Detection)
            .filter(Detection.image_id.in_(image_ids)).all())
    det_ids = [d.id for d in dets]
    n_fb = 0
    if det_ids:
        n_fb = int(s.query(func.count(Feedback.id))
                   .filter(Feedback.detection_id.in_(det_ids)).scalar() or 0)
    n_an = sum(1 for d in dets if d.is_anomaly)
    n_det = len(dets)
    return {"n_images": len(image_ids), "n_detected": n_det,
            "n_anomaly": n_an, "n_feedback": n_fb,
            "bad_rate": round(n_an / n_det, 3) if n_det else 0.0}


def _workorder_detect_batches(s, wo: WorkOrder) -> list:
    """工单全部检测批次（30 图/批，来自各数据源 compute_groups）。

    返回 [{batch_key, datasource_id, source, category, batch_index, image_ids}]。
    batch_key = "{datasource_id}:{category}:{batch_index}"（1 起始），
    与 compute_groups 的 detect_batches 顺序一致。
    """
    out: list = []
    for src in _workorder_sources(s, wo.id):
        pj = src.plan_json or {}
        groups = compute_groups(
            s, src.id, _int_or(100, src.pretrain_normal),
            _int_or(30, src.pretrain_anomaly), _int_or(30, src.batch_size, lo=1),
            pj.get("pretrain_ids"))
        for cat, g in (groups or {}).items():
            for bi, ids in enumerate(g.get("detect_batches") or [], 1):
                out.append({"batch_key": f"{src.id}:{cat}:{bi}",
                            "datasource_id": src.id, "source": src.name,
                            "category": cat, "batch_index": bi,
                            "image_ids": list(ids)})
    return out


def _pipeline_summary(s, wo: WorkOrder) -> dict:
    """产线空转体检（UI 走查 2026-08-29）：待检测图数 + 未准备品类。

    监控页用它提示「产线运行中但无可检数据」并引导去准备剩余品类：
    - n_pending：已准备品类批次的可消费积压（普通=未检测；回队=待重检）；
    - unprepared_categories：挂接数据源下无引擎快照的品类（整批跳过送检）；
    - n_images_blocked：未准备品类名下的图数。
    仅用于 GET /workorders/{id}（详情），列表接口不加（避免逐工单读快照）。
    """
    from ..engine import get_engine
    engine = get_engine()
    out = {"n_pending": 0, "unprepared_categories": [], "n_images_blocked": 0}
    batches = _workorder_detect_batches(s, wo)
    if not batches:
        return out
    unprepared = sorted({b["category"] for b in batches
                         if b["category"]
                         and engine.current_version(b["category"]) is None})
    out["unprepared_categories"] = unprepared
    qcfg = wo.queue_json or {}
    requeued_ids = {k: list(v) for k, v in
                    (qcfg.get("requeued_ids") or {}).items()}
    requeued_at = {k: v for k, v in (qcfg.get("requeued_at") or {}).items()}
    n_pending = 0
    for b in batches:
        if not b["category"]:
            continue
        if b["category"] in unprepared:
            out["n_images_blocked"] += len(b["image_ids"])
            continue
        key = b["batch_key"]
        if key in requeued_ids:
            try:
                t_req = datetime.fromisoformat(str(requeued_at.get(key)))
            except (TypeError, ValueError):
                t_req = None
            n_pending += _n_pending_requeue_by_ids(b["image_ids"], t_req)
        else:
            n_pending += _n_undetected_by_ids(s, b["image_ids"])
    out["n_pending"] = n_pending
    return out


def _n_pending_review(s, workorder_id: int) -> int:
    """该工单名下待复核检测数（产线路径 + 无有效反馈）。

    与 pipeline_service._apply_review_backpressure 口径一致，
    供监控页实时显示复判积压。
    """
    fb_ids = {r[0] for r in s.query(Feedback.detection_id)
              .filter(Feedback.invalidated == False).all()}  # noqa: E712
    rows = (s.query(Detection.id)
            .filter(Detection.workorder_id == workorder_id,
                    Detection.target_type.in_(("pipeline", "stream", "plc")))
            .all())
    return sum(1 for (did,) in rows if did not in fb_ids)


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
    with session_scope() as s:
        wo = s.get(WorkOrder, workorder_id)
        if wo is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        name = wo.name
        s.query(WorkOrderSource).filter(
            WorkOrderSource.workorder_id == workorder_id).delete(
            synchronize_session=False)
        s.delete(wo)
    log_action("delete_workorder", f"id={workorder_id} name={name}",
               extra={"workorder_id": workorder_id})
    return {"deleted": True, "name": name}


# ══════════════════ 工单体检（条件 vs 数据支撑）══════════════════
def check_workorder(s, wo: WorkOrder) -> dict:
    """体检：工单聚合条件 vs 数据实际支撑（v4：条件由数据源属性聚合）。

    declared = 数据源聚合条件；capability = 全源支撑合并；
    suggestion = 按支撑建议；warnings 含「数据源层体检」警告。
    """
    sources = _workorder_sources(s, wo.id)
    cap = {"n_normal": 0, "n_anomaly": 0, "n_masks": 0, "n_images": 0,
           "n_videos": 0, "categories": [], "n_templates": 0, "tier": None}
    for src in sources:
        c = _source_capability(s, src.id)
        for k in ("n_normal", "n_anomaly", "n_masks", "n_images",
                  "n_videos", "n_templates"):
            cap[k] += c[k]
        cap["categories"] = sorted(set(cap["categories"]) | set(c["categories"]))
    if cap["n_masks"] > 0 and cap["n_anomaly"] > 0:
        cap["tier"] = "L1b"
    elif cap["n_anomaly"] > 0:
        cap["tier"] = "L1a"
    elif cap["n_normal"] > 0:
        cap["tier"] = "L0"

    cats = cap["categories"]
    no_category_info = (not cats) or cats == [GENERIC_CATEGORY]
    sup_pc = not no_category_info          # 数据已分品类
    sup_tpl = cap["n_templates"] > 0       # 有模板图
    declared = _aggregate_conditions(s, sources)
    warnings: list[str] = []
    if not sources:
        warnings.append("尚未挂接数据源：可导入新数据源或选择已有数据源")
    for src in sources:
        dchk = check_datasource(s, src)
        warnings.extend(dchk["warnings"])
    if TIER_RANK.get(declared["label_tier"], 0) > TIER_RANK.get(
            cap["tier"] or "", -1):
        warnings.append(
            f"聚合条件「{TIER_CN.get(declared['label_tier'], declared['label_tier'])}」"
            f"超出数据支撑（当前支撑：{TIER_CN.get(cap['tier'], '空')}）；"
            "可在数据源体检中自动校正")
    if declared["has_template"] and not sup_tpl:
        warnings.append("模板比对：数据源声明了模板能力但尚无模板图，"
                        "可把模板图放入 template 子目录重新导入")
    suggestion = {
        "label_tier": cap["tier"] or "L0",
        "per_category": sup_pc,
        "has_template": sup_tpl,
    }
    matched = not warnings
    return {"capability": cap, "declared": declared,
            "suggestion": suggestion, "warnings": warnings,
            "matched": matched}


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
        from .routes_models import register_demo5_snapshots
        try:
            register_demo5_snapshots()
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
