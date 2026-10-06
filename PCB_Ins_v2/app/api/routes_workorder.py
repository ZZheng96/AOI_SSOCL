"""工单 / 数据源 API：产线任务实例 + 数据资产管理 + 批次导入。

工单 = 产线任务实例（绑定数据源、配置检测条件、产线状态机 running/paused）。
数据源 = 独立登记的数据实体（本地目录 / 实时流），可被多工单复用。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.dataset_adapter import import_adapted, probe_dataset
from app.core.ingest import import_folder
from app.db.database import session_scope
from app.db.models import (Dataset, DataSource, Image as ImageRow,
                           WorkOrder, WorkOrderSource)

router = APIRouter()


def _ds_dict(s, ds: DataSource) -> dict:
    batches = (s.query(Dataset)
               .filter(Dataset.datasource_id == ds.id).all())
    n_images = (s.query(ImageRow)
                .filter(ImageRow.dataset_id.in_([b.id for b in batches]))
                .count()) if batches else 0
    return {
        "id": ds.id, "name": ds.name, "modality": ds.modality,
        "source_type": ds.source_type, "stream_config": ds.stream_config,
        "label_tier": ds.label_tier, "per_category": ds.per_category,
        "has_template": ds.has_template,
        "pretrain_normal": ds.pretrain_normal, "pretrain_anomaly": ds.pretrain_anomaly,
        "batch_size": ds.batch_size, "note": ds.note,
        "n_batches": len(batches), "n_images": n_images,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


# ── 工单 ───────────────────────────────────────────────────
@router.post("/workorders")
def create_workorder(payload: dict) -> dict:
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="需要 name")
    with session_scope() as s:
        if s.query(WorkOrder).filter(WorkOrder.name == name).first():
            raise HTTPException(status_code=400, detail=f"工单已存在: {name}")
        wo = WorkOrder(
            name=name,
            label_tier=payload.get("label_tier", "L1a"),
            per_category=bool(payload.get("per_category", True)),
            has_template=bool(payload.get("has_template", False)),
            review_enabled=bool(payload.get("review_enabled", False)),
            pipeline_status=payload.get("pipeline_status", "running"),
            auto_resume=bool(payload.get("auto_resume", False)),
            template_ref=payload.get("template_ref"),
            note=payload.get("note", ""),
        )
        s.add(wo)
        s.flush()
        for ds_id in payload.get("sources") or []:
            if s.get(DataSource, int(ds_id)):
                s.add(WorkOrderSource(workorder_id=wo.id, datasource_id=int(ds_id)))
        return _wo_dict(s, wo)


@router.get("/workorders")
def list_workorders() -> dict:
    with session_scope() as s:
        rows = s.query(WorkOrder).order_by(WorkOrder.id.desc()).all()
        return {"items": [_wo_dict(s, wo) for wo in rows]}


@router.get("/workorders/{wo_id}")
def get_workorder(wo_id: int) -> dict:
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        return _wo_dict(s, wo)


@router.delete("/workorders/{wo_id}")
def delete_workorder(wo_id: int) -> dict:
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        s.query(WorkOrderSource).filter(WorkOrderSource.workorder_id == wo_id).delete()
        s.delete(wo)
    return {"ok": True}


@router.post("/workorders/{wo_id}/sources")
def attach_source(wo_id: int, payload: dict) -> dict:
    ds_id = payload.get("datasource_id")
    if not ds_id:
        raise HTTPException(status_code=400, detail="需要 datasource_id")
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        if s.get(DataSource, int(ds_id)) is None:
            raise HTTPException(status_code=404, detail=f"数据源不存在: {ds_id}")
        exists = (s.query(WorkOrderSource)
                  .filter(WorkOrderSource.workorder_id == wo_id,
                          WorkOrderSource.datasource_id == int(ds_id)).first())
        if exists is None:
            s.add(WorkOrderSource(workorder_id=wo_id, datasource_id=int(ds_id)))
        return _wo_dict(s, wo)


@router.delete("/workorders/{wo_id}/sources/{ds_id}")
def detach_source(wo_id: int, ds_id: int) -> dict:
    with session_scope() as s:
        (s.query(WorkOrderSource)
         .filter(WorkOrderSource.workorder_id == wo_id,
                 WorkOrderSource.datasource_id == ds_id).delete())
        wo = s.get(WorkOrder, wo_id)
        return _wo_dict(s, wo) if wo else {"ok": True}


@router.post("/workorders/{wo_id}/pause")
def pause_workorder(wo_id: int) -> dict:
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        wo.pipeline_status = "paused"
        return _wo_dict(s, wo)


@router.post("/workorders/{wo_id}/resume")
def resume_workorder(wo_id: int) -> dict:
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        wo.pipeline_status = "running"
        return _wo_dict(s, wo)


@router.post("/workorders/{wo_id}/requeue")
def requeue_batch(wo_id: int, payload: dict) -> dict:
    """批次回队重检：把某批次全部图标记为待重检（产线消费时优先）。

    body: {dataset_id, batch_key?}  batch_key 缺省为 dataset_id 字符串。
    """
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="需要 dataset_id")
    with session_scope() as s:
        wo = s.get(WorkOrder, wo_id)
        if wo is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {wo_id}")
        ds = s.get(Dataset, int(dataset_id))
        if ds is None:
            raise HTTPException(status_code=404, detail=f"批次不存在: {dataset_id}")
        img_ids = [r[0] for r in
                   s.query(ImageRow.id).filter(ImageRow.dataset_id == int(dataset_id)).all()]
        qcfg = dict(wo.queue_json or {})
        requeued = qcfg.get("requeued") or []
        if str(dataset_id) not in requeued:
            requeued.append(str(dataset_id))
        qcfg["requeued"] = requeued
        from datetime import datetime
        qcfg.setdefault("requeued_at", {})[str(dataset_id)] = datetime.now().isoformat()
        qcfg["requeued_ids"] = {str(dataset_id): img_ids}
        wo.queue_json = qcfg
        return {"ok": True, "workorder": wo_id, "dataset_id": int(dataset_id),
                "n_images": len(img_ids)}


def _wo_dict(s, wo: WorkOrder) -> dict:
    srcs = (s.query(DataSource)
            .join(WorkOrderSource, WorkOrderSource.datasource_id == DataSource.id)
            .filter(WorkOrderSource.workorder_id == wo.id).all())
    return {
        "id": wo.id, "name": wo.name, "label_tier": wo.label_tier,
        "per_category": wo.per_category, "has_template": wo.has_template,
        "status": wo.status, "pipeline_status": wo.pipeline_status,
        "auto_resume": wo.auto_resume, "review_enabled": wo.review_enabled,
        "template_ref": wo.template_ref, "note": wo.note,
        "queue": wo.queue_json or {},
        "sources": [_ds_dict(s, ds) for ds in srcs],
        "created_at": wo.created_at.isoformat() if wo.created_at else None,
    }


# ── 数据源 ─────────────────────────────────────────────────
@router.post("/datasources")
def create_datasource(payload: dict) -> dict:
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="需要 name")
    with session_scope() as s:
        if s.query(DataSource).filter(DataSource.name == name).first():
            raise HTTPException(status_code=400, detail=f"数据源已存在: {name}")
        ds = DataSource(
            name=name,
            modality=payload.get("modality", "image"),
            source_type=payload.get("source_type", "local"),
            stream_config=payload.get("stream_config") or {},
            label_tier=payload.get("label_tier", "L0"),
            per_category=bool(payload.get("per_category", False)),
            has_template=bool(payload.get("has_template", False)),
            pretrain_normal=int(payload.get("pretrain_normal", 100)),
            pretrain_anomaly=int(payload.get("pretrain_anomaly", 30)),
            batch_size=int(payload.get("batch_size", 30)),
            note=payload.get("note", ""),
        )
        s.add(ds)
        s.flush()
        return _ds_dict(s, ds)


@router.get("/datasources")
def list_datasources() -> dict:
    with session_scope() as s:
        rows = s.query(DataSource).order_by(DataSource.id.desc()).all()
        return {"items": [_ds_dict(s, ds) for ds in rows]}


@router.get("/datasources/{ds_id}")
def get_datasource(ds_id: int) -> dict:
    with session_scope() as s:
        ds = s.get(DataSource, ds_id)
        if ds is None:
            raise HTTPException(status_code=404, detail=f"数据源不存在: {ds_id}")
        d = _ds_dict(s, ds)
        batches = (s.query(Dataset).filter(Dataset.datasource_id == ds_id)
                   .order_by(Dataset.id.desc()).all())
        d["batches"] = [{"id": b.id, "name": b.name, "category": b.category,
                         "source_type": b.source_type, "n_images": b.n_images,
                         "source_path": b.source_path,
                         "params": b.params, "note": b.note}
                        for b in batches]
        return d


@router.delete("/datasources/{ds_id}")
def delete_datasource(ds_id: int, purge_files: bool = False) -> dict:
    """删除数据源（级联，2026-08-30 补强）。

    旧实现只删 data_sources 行 -> 批次/图片成孤儿、盘文件不动（不足清单 #6）。
    现在：批次/图片/检测/反馈/缺陷明细/伪异常/视频一并清理；
    purge_files=true 时同步删除 storage 根内的系统文件副本（热力图/伪
    异常/批次目录等），外部原始目录（用户数据）永不自动删除。
    """
    from app.core.datasets import delete_datasource_cascade
    try:
        return delete_datasource_cascade(ds_id, purge_files=purge_files)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/datasources/{ds_id}/import")
def import_into_datasource(ds_id: int, payload: dict) -> dict:
    """从目录导入图片到数据源（建批次 + 登记 Image）。

    body: {folder, category?, name?, note?}
    子目录约定：good/normal → normal；其余子目录 → anomaly（子目录名作缺陷类型）。
    """
    from app.core.ingest import import_folder
    folder = payload.get("folder")
    if not folder:
        raise HTTPException(status_code=400, detail="需要 folder")
    if not Path(folder).is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {folder}")
    try:
        result = import_folder(
            folder,
            category=payload.get("category", "default"),
            source="datasource",
            copy_to_storage=False,
            datasource_id=ds_id,
            name=payload.get("name"),
            note=payload.get("note"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"导入失败: {exc}")
    return result


# ── 数据集适配（识别 + 确认导入，与 AOI_sys 同步 2026-08-31）──
@router.post("/datasources/probe")
def probe_datasource(payload: dict) -> dict:
    """只读扫描：识别数据集格式与内容构成，返回确认报告（不写库）。

    body: {root}
    格式覆盖：mvtec_like（MVTec/MPDD/BTAD/data_local，含包装目录下钻）、
    yolo_split（GYU-DET）、dir_rules（OK/NG/模板/良品/误报 命名约定）、plain。
    """
    root = (payload.get("root") or "").strip()
    if not root or not Path(root).is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {root}")
    try:
        return probe_dataset(root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"目录扫描失败: {e}")


@router.post("/datasources/adapt_import")
def adapt_import_into_datasource(payload: dict) -> dict:
    """按确认后的映射导入（与 AOI_sys dataset_adapter 同口径）。

    body: {root, format?, category?, ok_role?, group_overrides?,
           categories?, dataset_name?, name?, note?, datasource_id?}
    datasource_id 缺失时只建批次不挂源（后续可手动挂）。
    """
    root = (payload.get("root") or "").strip()
    if not root or not Path(root).is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {root}")
    fmt = str(payload.get("format") or "dir_rules")
    try:
        return import_adapted(
            root, fmt,
            category=payload.get("category"),
            ok_role=str(payload.get("ok_role") or "train"),
            group_overrides=payload.get("group_overrides") or None,
            categories=payload.get("categories"),
            dataset_name=payload.get("dataset_name"),
            name=payload.get("name"),
            note=payload.get("note"),
            datasource_id=payload.get("datasource_id"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"导入失败: {exc}")
