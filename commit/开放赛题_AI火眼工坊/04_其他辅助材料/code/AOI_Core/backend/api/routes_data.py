"""数据管理接口：图片 / 视频 / 数据增强配置 / 数据集批次（M8a）。"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..core.security import require_role
from ..core.config import AOI_ROOT, get_settings
from ..core import datasets as ds_helper
from ..core.dataset_adapter import import_adapted, probe_dataset
from ..core.ingest import import_folder, import_mvtec, register_image, register_video
from ..core.tasks import task_manager
from ..db.database import log_action, session_scope
from ..db.models import (Dataset, Detection, Feedback, Image, Model,
                         PseudoAnomaly, Video, VideoFrame)
from .schemas import (AugmentConfigUpdateRequest, DatasetAdaptImportRequest,
                      DatasetProbeRequest, ImageAnnotateRequest,
                      ImageImportRequest, MvtecImportRequest, OrphanCleanRequest,
                      PseudoPreviewRequest, VideoImportRequest, to_dict)

router = APIRouter()


def _get_image_path_or_404(image_id: int) -> str:
    """按 id 取图片路径，不存在则 404。"""
    with session_scope() as s:
        row = s.get(Image, image_id)
        if row is None:
            raise HTTPException(status_code=404, detail="图片不存在")
        return row.path


def _resolve_image_paths(image_ids: Optional[List[int]],
                         category: Optional[str],
                         split: Optional[str] = None,
                         label: Optional[str] = None) -> List[str]:
    """解析基底图路径列表：image_ids 优先，否则按 category/split/label 过滤。"""
    with session_scope() as s:
        q = s.query(Image)
        if image_ids:
            q = q.filter(Image.id.in_(image_ids))
        else:
            if category:
                q = q.filter(Image.category == category)
            if split:
                q = q.filter(Image.split == split)
            if label:
                q = q.filter(Image.label == label)
        return [r.path for r in q.order_by(Image.id).all()]


# ── 图片 ─────────────────────────────────────────────────────
@router.post("/images/import")
def api_import_images(req: ImageImportRequest):
    """后台任务：递归导入文件夹内全部图片。

    M8a：同步先建 Dataset 批次行（name 默认 folder_{时间戳}），返回 dataset_id。
    """
    if not Path(req.folder).is_dir():
        raise HTTPException(status_code=400, detail=f"文件夹不存在: {req.folder}")
    dataset_id = ds_helper.create_dataset(
        req.category, "folder", source_path=req.folder,
        params={"split": req.split, "label": req.label, "source": req.source,
                "copy_to_storage": req.copy_to_storage},
        name=req.name, note=req.note,
        dataset_name=req.dataset_name or Path(req.folder).name,
        datasource_id=req.datasource_id)

    def fn(progress_cb):
        return import_folder(req.folder, req.category, req.split, req.label,
                             req.source, req.copy_to_storage,
                             progress_cb=progress_cb, dataset_id=dataset_id,
                             datasource_id=req.datasource_id)

    task_id = task_manager.submit("import", req.model_dump(), fn)
    log_action("import_images", f"folder={req.folder} category={req.category}",
               extra={"dataset_id": dataset_id, "category": req.category})
    return {"task_id": task_id, "dataset_id": dataset_id}


@router.post("/images/import_mvtec")
def api_import_mvtec(req: MvtecImportRequest):
    """后台任务：导入 MVTec AD 结构数据集。

    M8a：按品类各建一个 Dataset 批次行，返回 dataset_ids（{品类: id}）。
    """
    root = Path(req.root)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {req.root}")
    cats = req.categories or [d.name for d in root.iterdir() if d.is_dir()]
    ts = time.strftime("%Y%m%d_%H%M%S")
    dataset_ids = {}
    for cat in cats:
        if (root / cat).is_dir():
            dataset_ids[cat] = ds_helper.create_dataset(
                cat, "mvtec", source_path=req.root,
                name=req.name or f"{cat}_{ts}", note=req.note,
                dataset_name=req.dataset_name or root.name,
                datasource_id=req.datasource_id)

    def fn(progress_cb):
        return import_mvtec(req.root, req.categories, progress_cb=progress_cb,
                            dataset_ids=dataset_ids,
                            datasource_id=req.datasource_id)

    task_id = task_manager.submit("import", req.model_dump(), fn)
    log_action("import_images", f"mvtec root={req.root}",
               extra={"dataset_ids": dataset_ids})
    return {"task_id": task_id, "dataset_ids": dataset_ids}


@router.get("/images")
def api_list_images(category: Optional[str] = None, split: Optional[str] = None,
                    label: Optional[str] = None,
                    dataset_id: Optional[int] = None, page: int = 1,
                    page_size: int = 50,
                    datasource_id: Optional[int] = None,
                    ids: Optional[str] = None):
    """图片分页查询；ids=逗号分隔的 image id（前端反馈 v9：预训练组/检测组批次查看）；
    datasource_id=按数据源过滤（v10 查询栏数据源维度）。"""
    with session_scope() as s:
        q = s.query(Image)
        if category:
            q = q.filter(Image.category == category)
        if split:
            q = q.filter(Image.split == split)
        if label:
            q = q.filter(Image.label == label)
        if dataset_id is not None:
            q = q.filter(Image.dataset_id == dataset_id)
        if datasource_id is not None:
            # 数据源过滤：图片所属批次归属该数据源
            from ..db.models import Dataset
            q = q.filter(Image.dataset_id.in_(
                s.query(Dataset.id).filter(
                    Dataset.datasource_id == datasource_id)))
        if ids:
            id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
            if id_list:
                q = q.filter(Image.id.in_(id_list))
        total = q.count()
        rows = (q.order_by(Image.id.desc())
                 .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [to_dict(r) for r in rows]}


@router.get("/images/tree")
def api_images_tree():
    """数据树：{category: {split: {label: count}}}，供数据管理页树状导航。

    M8a：新增第四维 datasets——{category: {dataset_id: {name, splits: {split: {label: count}}}}}，
    原三维 tree 字段保持不变（向后兼容）。
    """
    from sqlalchemy import func
    from ..db.models import DataSource
    with session_scope() as s:
        rows = (s.query(Image.category, Image.split, Image.label,
                        Image.dataset_id, Dataset.dataset_name,
                        Dataset.datasource_id,
                        func.count(Image.id))
                .outerjoin(Dataset, Image.dataset_id == Dataset.id)
                .group_by(Image.category, Image.split, Image.label,
                          Image.dataset_id, Dataset.dataset_name,
                          Dataset.datasource_id).all())
        ds_names = {r.id: r.name for r in s.query(Dataset).all()}
        src_names = {r.id: r.name for r in s.query(DataSource).all()}
    tree: Dict[str, Dict[str, Dict[str, int]]] = {}
    datasets: Dict[str, Dict[str, Dict]] = {}
    dataset_groups: Dict[str, Dict[str, Dict]] = {}
    source_groups: Dict[str, Dict[str, Dict]] = {}
    for cat, split, label, ds_id, dname, src_id, cnt in rows:
        tree.setdefault(cat, {}).setdefault(split, {})[label] = cnt
        key = str(ds_id) if ds_id is not None else "none"
        node = datasets.setdefault(cat, {}).setdefault(
            key, {"name": ds_names.get(ds_id, "未分组"), "splits": {}})
        node["splits"].setdefault(split, {})[label] = cnt
        # U-opsflow 数据集层级：数据集 → 品类 → 批次
        gname = dname or "未分组数据集"
        gnode = dataset_groups.setdefault(gname, {}).setdefault(
            cat, {}).setdefault(
            key, {"name": ds_names.get(ds_id, "未分组"), "splits": {}})
        gnode["splits"].setdefault(split, {})[label] = cnt
        # 工单制：数据源 → 品类 → 批次（未挂源批次不进 source_groups，
        # 由 dataset_groups 承载）
        sname = src_names.get(src_id)
        if sname is None:
            continue
        snode = source_groups.setdefault(sname, {}).setdefault(
            cat, {}).setdefault(
            key, {"name": ds_names.get(ds_id, "未分组"), "splits": {}})
        snode["splits"].setdefault(split, {})[label] = cnt
    # 预填全部数据源名：无数据源（如残留空壳"1"）也要在数据树可见、
    # 可右键编辑/删除——否则用户无法管理它（工单对话框却列得出，两边不对齐）
    for _sid, _sname in src_names.items():
        source_groups.setdefault(_sname, {})
    return {"tree": tree, "datasets": datasets,
            "dataset_groups": dataset_groups,
            "source_groups": source_groups}


@router.get("/images/{image_id}")
def api_get_image(image_id: int):
    with session_scope() as s:
        row = s.get(Image, image_id)
        if row is None:
            raise HTTPException(status_code=404, detail="图片不存在")
        return to_dict(row)


@router.delete("/images/{image_id}",
               dependencies=[Depends(require_role("engineer"))])
def api_delete_image(image_id: int, delete_file: bool = False):
    """删除 DB 记录；delete_file=true 时同时删除磁盘文件（M8a，默认保持旧行为）。
    先清关联（检测/反馈/伪异常/视频帧），避免外键冲突导致删除静默失败。"""
    with session_scope() as s:
        row = s.get(Image, image_id)
        if row is None:
            raise HTTPException(status_code=404, detail="图片不存在")
        path, category, ds_id = row.path, row.category, row.dataset_id
        det_ids = [r[0] for r in s.query(Detection.id).filter(
            Detection.image_id == image_id).all()]
        if det_ids:
            s.query(Feedback).filter(
                Feedback.detection_id.in_(det_ids)).delete(
                synchronize_session=False)
        s.query(Detection).filter(
            Detection.image_id == image_id).delete(synchronize_session=False)
        s.query(PseudoAnomaly).filter(
            PseudoAnomaly.base_image_id == image_id).delete(
            synchronize_session=False)
        s.query(VideoFrame).filter(
            VideoFrame.image_id == image_id).update(
            {"image_id": None}, synchronize_session=False)
        s.delete(row)
    file_deleted = False
    if delete_file:
        try:
            os.remove(path)
            file_deleted = True
        except OSError:
            pass
    if ds_id is not None:
        ds_helper.refresh_dataset_count(ds_id)
    log_action("delete_image", f"image_id={image_id} delete_file={delete_file}",
               extra={"image_id": image_id, "category": category,
                      "file_deleted": file_deleted})
    return {"deleted": True, "file_deleted": file_deleted}


@router.post("/images/{image_id}/annotate")
def api_annotate_image(image_id: int, req: ImageAnnotateRequest):
    """标注流转（M8a）：改 label/split/defect_type，变更记 OperationLog(extra)。"""
    with session_scope() as s:
        row = s.get(Image, image_id)
        if row is None:
            raise HTTPException(status_code=404, detail="图片不存在")
        old = {"label": row.label, "split": row.split,
               "defect_type": row.defect_type}
        new = dict(old)
        if req.label is not None:
            row.label = req.label
            new["label"] = req.label
        if req.split is not None:
            row.split = req.split
            new["split"] = req.split
        if req.defect_type is not None:
            row.defect_type = req.defect_type
            new["defect_type"] = req.defect_type
        category = row.category
        out = to_dict(row)
    log_action("annotate_image",
               f"image_id={image_id} {old} -> {new}",
               extra={"image_id": image_id, "category": category,
                      "old": old, "new": new})
    return out


@router.post("/images/upload")
async def api_upload_image(file: UploadFile = File(...),
                           category: str = Form("default"),
                           split: str = Form("unlabeled"),
                           label: str = Form("unknown"),
                           name: Optional[str] = Form(None),
                           note: Optional[str] = Form(None),
                           datasource_id: Optional[int] = Form(None)):
    """multipart 上传单张图片，落盘 storage/uploads 后登记入库，返回 Image 行。

    M8a：默认挂"uploads"批次（按品类 get_or_create）；传 name 时新建命名批次。
    v8：传 datasource_id 时挂到该数据源。
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if name:
        dataset_id = ds_helper.create_dataset(
            category, "upload", name=name, note=note,
            datasource_id=datasource_id)
    else:
        dataset_id = ds_helper.get_or_create_dataset(
            ds_helper.UPLOAD_DATASET_NAME, category, "upload",
            datasource_id=datasource_id)
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    dst = (get_settings().storage("uploads")
           / f"up_{int(time.time() * 1000)}{suffix}")
    dst.write_bytes(data)
    image_id = register_image(dst, category=category, split=split, label=label,
                              source="upload", copy_to_storage=False,
                              dataset_id=dataset_id)
    ds_helper.refresh_dataset_count(dataset_id)
    log_action("upload_image", f"image_id={image_id} category={category}",
               extra={"image_id": image_id, "category": category,
                      "dataset_id": dataset_id})
    with session_scope() as s:
        return to_dict(s.get(Image, image_id))


# ── 视频 ─────────────────────────────────────────────────────
@router.post("/videos/import")
def api_import_video(req: VideoImportRequest):
    """同步登记视频并读取元信息。M8a：建 source_type=video 批次，抽帧图挂该批次。"""
    if not Path(req.path).is_file():
        raise HTTPException(status_code=400, detail=f"视频不存在: {req.path}")
    dataset_id = ds_helper.create_dataset(
        req.category, "video", source_path=req.path,
        name=req.name, note=req.note, datasource_id=req.datasource_id)
    try:
        video_id = register_video(req.path, category=req.category,
                                  label=req.label, dataset_id=dataset_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"视频登记失败: {e}")
    log_action("import_video", f"path={req.path} video_id={video_id}",
               extra={"video_id": video_id, "dataset_id": dataset_id,
                      "category": req.category})
    return {"video_id": video_id, "dataset_id": dataset_id}


@router.get("/videos")
def api_list_videos(page: int = 1, page_size: int = 50):
    with session_scope() as s:
        q = s.query(Video)
        total = q.count()
        rows = (q.order_by(Video.id.desc())
                 .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [to_dict(r) for r in rows]}


@router.get("/videos/{video_id}")
def api_get_video(video_id: int):
    with session_scope() as s:
        row = s.get(Video, video_id)
        if row is None:
            raise HTTPException(status_code=404, detail="视频不存在")
        return to_dict(row)


# ── 数据增强配置（§15.29：伪异常合成 / 图像预处理 / 训练增强）────
@router.get("/augment/config")
def api_get_augment_config():
    """读取数据增强配置（基础配置 yaml 的 augment 节）。

    页面按引擎实际实现呈现三块：pseudo（伪异常合成方式，准备模型时自动执行）、
    preprocess（图像预处理，训练与检测同口径）、enhance（训练增强）。
    """
    base_cfg = get_settings().demo5_base_cfg
    if base_cfg is None:
        raise HTTPException(status_code=404, detail="基础配置不存在")
    import yaml
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    aug = cfg.get("augment", {}) or {}
    disc = (cfg.get("slots", {}) or {}).get("disc", {}) or {}
    return {
        "config_path": str(base_cfg),
        "pseudo": {
            "enabled": bool(disc.get("enabled", True)),
            "methods": list(aug.get("pseudo", {}).get("methods")
                            or ["defect_transplant", "cutpaste", "color_blot"]),
            "pseudo_per_image": disc.get("pseudo_per_image", 8),
            "transplant_scale": list(aug.get("pseudo", {}).get("transplant_scale",
                                                               [0.35, 0.9])),
            "feather": bool(aug.get("pseudo", {}).get("feather", True)),
        },
        "preprocess": dict(aug.get("preprocess", {}) or {"enabled": False}),
        "enhance": dict(aug.get("enhance", {}) or {}),
        "note": "伪异常在准备模型时由引擎自动合成（不生成图片文件），"
                "保存后下一次准备模型生效。",
    }


@router.put("/augment/config",
            dependencies=[Depends(require_role("engineer"))])
def api_update_augment_config(req: AugmentConfigUpdateRequest):
    """写回数据增强配置（augment 节白名单字段），保存后 reload 引擎配置。

    只允许改数据增强相关字段；disc 槽的 pseudo_per_image（每图伪异常数）
    属伪异常合成强度，一并在此维护。不涉及学习率/轮数等训练参数。
    """
    base_cfg = get_settings().demo5_base_cfg
    if base_cfg is None:
        raise HTTPException(status_code=404, detail="基础配置不存在")
    import yaml
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    aug = cfg.setdefault("augment", {})
    if req.pseudo is not None:
        p = aug.setdefault("pseudo", {})
        for k in ("methods", "transplant_scale", "feather"):
            if k in req.pseudo:
                p[k] = req.pseudo[k]
        if "pseudo_per_image" in req.pseudo:
            (cfg.setdefault("slots", {})
                .setdefault("disc", {})["pseudo_per_image"]
             ) = req.pseudo["pseudo_per_image"]
    if req.preprocess is not None:
        pp = aug.setdefault("preprocess", {})
        for k in ("enabled", "gray", "clahe", "median"):
            if k in req.preprocess:
                pp[k] = req.preprocess[k]
    if req.enhance is not None:
        en = aug.setdefault("enhance", {})
        for k in ("flip", "rot90", "brightness"):
            if k in req.enhance:
                en[k] = req.enhance[k]
    with open(base_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    # 引擎单例内存配置重载（下一次 prepare 生效）
    try:
        from ..engine import get_engine
        get_engine().reload_config()
    except Exception:  # noqa: BLE001 引擎未初始化则下次自然读取新 yaml
        pass
    log_action("augment_config_update",
               f"pseudo={req.pseudo or {}} preprocess={req.preprocess or {}} "
               f"enhance={req.enhance or {}}")
    return {"ok": True, "config_path": str(base_cfg),
            "pseudo": dict(aug.get("pseudo", {})),
            "preprocess": dict(aug.get("preprocess", {})),
            "enhance": dict(aug.get("enhance", {}))}


@router.post("/pseudo/preview")
def api_preview_pseudo(req: PseudoPreviewRequest):
    """单张伪异常合成预览（同步，不入库）。

    defect_transplant（真实缺陷移植）：取同品类已标注缺陷图整图作移植源，
    缩放到配置比例后随机贴入该正常图；该品类无缺陷图时自动回退 CutPaste。
    """
    path = _get_image_path_or_404(req.image_id)
    try:
        return _preview_pseudo_synth(path, req.image_id, req.method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"预览失败: {e}")


def _preview_pseudo_synth(path: str, image_id: int, method: str) -> dict:
    """用引擎同款 synth_pseudo 生成预览（1 张 + 掩码，落 storage/pseudo/preview_*）。

    与 algo/slots/disc.py 的合成保持一致：合成时基底=该图本身；
    defect_transplant 时按该图品类查已标注缺陷图构建移植源池。
    """
    import numpy as np
    from ..core.imaging import imread, imwrite

    img = imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    from algo.slots.disc import synth_pseudo
    # 读当前数据增强配置（transplant_scale/feather）
    pcfg = {}
    base_cfg = get_settings().demo5_base_cfg
    if base_cfg:
        import yaml
        try:
            with open(base_cfg, "r", encoding="utf-8") as f:
                pcfg = ((yaml.safe_load(f).get("augment", {}) or {})
                        .get("pseudo", {}) or {})
        except Exception:  # noqa: BLE001
            pass
    # 预览强制使用下拉选定的合成方式（synth_pseudo 默认从配置 methods 里随机挑，
    # 不锁定则三种预览效果相同）
    pcfg["methods"] = [method]
    # 同品类缺陷图移植源池（label=anomaly）：按缺陷位置掩码抠缺陷（§15.30），
    # 只收带有效掩码的 (图, 掩码) 对；无位置标注的图不参与移植（整块贴入无意义），
    # 池空时 synth_pseudo 自动回退 CutPaste
    defect_pool = None
    if method == "defect_transplant":
        from algo.slots.disc import load_defect_mask
        import numpy as _np
        tiles = []
        with session_scope() as s:
            row = s.get(Image, image_id)
            if row is not None:
                q = s.query(Image).filter(
                    Image.category == row.category,
                    Image.label == "anomaly",
                    Image.id != image_id)
                for p in [r.path for r in q.limit(10).all()]:
                    t = imread(p)
                    if t is None:
                        continue
                    m = load_defect_mask(p, t.shape[:2])
                    if m is not None and _np.count_nonzero(m):
                        tiles.append((t, m))
        defect_pool = tiles or None
    rng = np.random.default_rng(20260828)
    out, mask = synth_pseudo(img, rng, defect_pool, pcfg)
    out_dir = get_settings().storage("pseudo")
    stem = Path(path).stem
    img_path = out_dir / f"preview_{stem}_{method}.png"
    mask_path = out_dir / f"preview_{stem}_{method}_mask.png"
    imwrite(img_path, out)
    imwrite(mask_path, (mask * 255).astype("uint8"))
    return {"image_path": str(img_path), "mask_path": str(mask_path)}


# ── 数据集适配（识别 + 确认导入，前端反馈 2026-08-31）────────
@router.post("/datasets/probe")
def api_probe_dataset(req: DatasetProbeRequest):
    """只读扫描：识别数据集格式与内容构成，返回确认报告（不写库）。

    供前端「数据集识别与确认」弹窗展示：格式/品类/分组计数/告警，
    用户确认后再调 /datasets/adapt_import 导入。
    """
    try:
        return probe_dataset(req.root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"目录扫描失败: {e}")


@router.post("/datasets/adapt_import")
def api_adapt_import(req: DatasetAdaptImportRequest):
    """后台任务：按确认后的映射导入（逐品类建批次 + 批量入库）。

    返回 {task_id}；导入结果在 Task.result（imported/total/dataset_ids）。
    """
    if not Path(req.root).is_dir():
        raise HTTPException(status_code=400, detail=f"目录不存在: {req.root}")

    def fn(progress_cb):
        return import_adapted(
            req.root, req.format, category=req.category,
            ok_role=req.ok_role, group_overrides=req.group_overrides,
            categories=req.categories, dataset_name=req.dataset_name,
            name=req.name, note=req.note,
            datasource_id=req.datasource_id, progress_cb=progress_cb)

    task_id = task_manager.submit("adapt_import", req.model_dump(), fn)
    log_action("adapt_import",
               f"root={req.root} format={req.format} category={req.category}",
               extra={"root": req.root, "format": req.format,
                      "datasource_id": req.datasource_id})
    return {"task_id": task_id}


# ── 数据集批次（M8a）─────────────────────────────────────────
@router.get("/datasets")
def api_list_datasets(category: Optional[str] = None):
    """批次列表：{items:[{id,name,category,source_type,source_path,n_images,note,created_at,layers}]}。

    layers（M14d，demo5 §0 分层契约）：由批次图像构成推断"可支撑层级"——
    L0 有正常图 / L1a 有缺陷标注 / L1b 有掩膜图（ground_truth/*_mask）/
    L2 有品类标识 / L3 有模板图（split=template）。UI 树节点直接展示。"""
    with session_scope() as s:
        from ..db.models import DataSource
        q = s.query(Dataset)
        if category:
            q = q.filter(Dataset.category == category)
        rows = q.order_by(Dataset.id.desc()).all()
        src_names = {r.id: r.name for r in s.query(DataSource).all()}
        items = []
        for r in rows:
            d = to_dict(r)
            d["layers"] = _dataset_layers(s, r)
            d["datasource_name"] = src_names.get(r.datasource_id, "")
            items.append(d)
        return {"items": items}


def _dataset_layers(s, ds: Dataset) -> list:
    """批次可支撑层级推断（纯 DB 统计，小表多次点查）。"""
    q = s.query(Image.id).filter(Image.dataset_id == ds.id)
    layers = []
    if q.filter(Image.label == "normal").first():
        layers.append("L0")
    if q.filter(Image.label == "anomaly").first():
        layers.append("L1a")
    if q.filter(Image.path.like("%ground_truth%") |
                Image.path.like("%_mask.%")).first():
        layers.append("L1b")
    if ds.category:
        layers.append("L2")
    if q.filter(Image.split == "template").first():
        layers.append("L3")
    return layers


@router.get("/datasets/{dataset_id}")
def api_get_dataset(dataset_id: int, page: int = 1, page_size: int = 50):
    """批次详情 + 图片分页 + M16f 数据谱系（该批次参与训练的模型版本）。"""
    with session_scope() as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail="批次不存在")
        q = s.query(Image).filter(Image.dataset_id == dataset_id)
        total = q.count()
        rows = (q.order_by(Image.id)
                .offset((page - 1) * page_size).limit(page_size).all())
        # M16f 谱系反查（review R6）：Model.metrics.dataset_ids 含本批次
        used_by = []
        for m in s.query(Model).filter(Model.category == ds.category).all():
            ids = (m.metrics or {}).get("dataset_ids") or []
            if dataset_id in ids:
                used_by.append({"model_id": m.id, "name": m.name,
                                "version": m.version,
                                "is_active": bool(m.is_active)})
        return {"dataset": to_dict(ds), "total": total, "page": page,
                "page_size": page_size, "items": [to_dict(r) for r in rows],
                "used_by_models": used_by}


@router.delete("/datasets/{dataset_id}",
               dependencies=[Depends(require_role("admin"))])
def api_delete_dataset(dataset_id: int, delete_files: bool = False,
                       keep_detections: bool = True):
    """级联删除批次：删 Image 行 +（delete_files=true 时）磁盘文件；
    keep_detections=true（默认）时检测记录保留、image_id 置空，
    否则连带删除该批次图片的检测记录及其反馈。
    返回 {deleted_images, deleted_files, affected_detections}。"""
    with session_scope() as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail="批次不存在")
        rows = s.query(Image).filter(Image.dataset_id == dataset_id).all()
        paths = [r.path for r in rows]
        img_ids = [r.id for r in rows]
        affected = 0
        if img_ids:
            det_q = s.query(Detection).filter(Detection.image_id.in_(img_ids))
            affected = det_q.count()
            if keep_detections:
                det_q.update({"image_id": None}, synchronize_session=False)
            else:
                from ..db.models import Feedback
                det_ids = [d.id for d in
                           det_q.with_entities(Detection.id).all()]
                if det_ids:
                    s.query(Feedback).filter(
                        Feedback.detection_id.in_(det_ids)).delete(
                        synchronize_session=False)
                det_q.delete(synchronize_session=False)
            s.query(Image).filter(Image.id.in_(img_ids)).delete(
                synchronize_session=False)
        s.delete(ds)
        category = ds.category
    deleted_files = 0
    if delete_files:
        for p in paths:
            try:
                os.remove(p)
                deleted_files += 1
            except OSError:
                pass
        # 删除批次文件后清理空目录，避免数据管理留下不可见的空批次目录。
        for p in paths:
            if not p:
                continue
            try:
                # 从 split/label 子目录逐级删除到批次目录；有其他文件时保留目录。
                os.removedirs(os.path.dirname(p))
            except OSError:
                pass
    log_action("delete_dataset",
               f"dataset_id={dataset_id} images={len(img_ids)} "
               f"files={deleted_files} detections={affected}",
               extra={"dataset_id": dataset_id, "category": category,
                      "deleted_images": len(img_ids)})
    return {"deleted_images": len(img_ids), "deleted_files": deleted_files,
            "affected_detections": affected}


# ── 孤儿文件治理（M8a）───────────────────────────────────────
_ORPHAN_SCAN_DIRS = ("raw_images", "uploads", "pseudo", "heatmaps",
                     "images", "raw_videos")


def _referenced_paths() -> set:
    """DB 中所有被引用的文件路径集合（原始串 + 归一化绝对路径两种形态）。"""
    refs = set()
    with session_scope() as s:
        queries = [s.query(Image.path), s.query(Detection.image_path),
                   s.query(Detection.heatmap_path), s.query(Detection.overlay_path),
                   s.query(PseudoAnomaly.image_path),
                   s.query(PseudoAnomaly.mask_path), s.query(Video.path)]
        for q in queries:
            for (p,) in q.all():
                if p:
                    refs.add(p)
    out = set(refs)
    for p in refs:
        pp = Path(p)
        if not pp.is_absolute():
            pp = AOI_ROOT / pp  # 相对路径（如 storage/...）按仓库根解析
        out.add(os.path.normcase(str(pp)))
    return out


def _scan_orphans(progress_cb=None) -> Dict:
    """遍历 storage 各产物目录，对照 DB 引用列出无引用文件清单。"""
    settings = get_settings()
    refs = _referenced_paths()
    orphans: List[str] = []
    scanned = 0
    for i, sub in enumerate(_ORPHAN_SCAN_DIRS):
        d = settings.storage_dir / sub
        if d.is_dir():
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                scanned += 1
                if str(f) not in refs \
                        and os.path.normcase(str(f)) not in refs:
                    orphans.append(str(f))
        if progress_cb:
            progress_cb(i + 1, len(_ORPHAN_SCAN_DIRS), f"扫描 {sub}")
    return {"orphans": sorted(orphans), "n_scanned": scanned,
            "n_referenced": len(refs)}


@router.post("/maintenance/orphan_scan")
def api_orphan_scan():
    """后台任务：扫描 storage 下无 DB 引用的孤儿文件，清单写 Task.result。"""
    task_id = task_manager.submit("orphan_scan", {}, _scan_orphans)
    log_action("orphan_scan", "启动孤儿文件扫描")
    return {"task_id": task_id}


@router.post("/maintenance/orphan_clean",
             dependencies=[Depends(require_role("engineer"))])
def api_orphan_clean(req: OrphanCleanRequest):
    """删除孤儿清单内的文件（安全约束：仅允许 storage 目录内的路径）。"""
    storage_root = os.path.normcase(str(get_settings().storage_dir))
    deleted, missing, skipped = 0, 0, []
    for p in req.paths:
        norm = os.path.normcase(str(Path(p)))
        if not norm.startswith(storage_root + os.sep):
            skipped.append(p)  # 越界路径拒绝删除
            continue
        try:
            os.remove(p)
            deleted += 1
        except FileNotFoundError:
            missing += 1
        except OSError:
            skipped.append(p)
    log_action("orphan_clean",
               f"deleted={deleted} missing={missing} skipped={len(skipped)}",
               extra={"deleted": deleted, "skipped": skipped})
    return {"deleted": deleted, "missing": missing, "skipped": skipped}
