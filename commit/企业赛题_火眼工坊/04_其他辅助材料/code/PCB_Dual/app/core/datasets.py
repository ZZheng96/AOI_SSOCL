"""数据集批次助手（M8a 数据资产工业化治理）。

所有导入/生成入口强制挂 Dataset 批次；老数据（dataset_id IS NULL）
在服务启动时由 assign_legacy_datasets() 按品类归入"历史未分组"批次（幂等）。
"""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.db.database import log_action, session_scope
from app.db.models import Dataset, Image

LEGACY_DATASET_NAME = "历史未分组"
FEEDBACK_DATASET_NAME = "反馈回流"
UPLOAD_DATASET_NAME = "uploads"


def _default_name(source_type: str) -> str:
    return f"{source_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_dataset(category: str, source_type: str,
                   source_path: Optional[str] = None,
                   params: Optional[Dict] = None,
                   name: Optional[str] = None,
                   note: Optional[str] = None,
                   dataset_name: Optional[str] = None,
                   datasource_id: Optional[int] = None) -> int:
    """新建批次行，返回 dataset_id。dataset_name=数据集组名（数据集->品类->批次）。

    W-source：datasource_id 指定批次归属的数据源（数据按数据源管理）。
    """
    with session_scope() as s:
        ds = Dataset(name=name or _default_name(source_type),
                     category=category, source_type=source_type,
                     source_path=source_path, params=params or {},
                     note=note or "", dataset_name=dataset_name or "",
                     datasource_id=datasource_id)
        s.add(ds)
        s.flush()
        return ds.id


def update_dataset_params(dataset_id: Optional[int],
                          extra: Optional[Dict]) -> None:
    """把额外统计并入批次 params（W-check：掩码数等体检原料）。"""
    if dataset_id is None or not extra:
        return
    with session_scope() as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None:
            return
        params = dict(ds.params or {})
        params.update(extra)
        ds.params = params


def get_or_create_dataset(name: str, category: str, source_type: str,
                          source_path: Optional[str] = None,
                          dataset_name: Optional[str] = None,
                          datasource_id: Optional[int] = None) -> int:
    """按 (name, category, source_type) 复用或新建批次（系统批次用）。

    v8：datasource_id 指定新批次的归属数据源（仅新建时生效）。
    """
    with session_scope() as s:
        ds = (s.query(Dataset)
              .filter(Dataset.name == name, Dataset.category == category,
                      Dataset.source_type == source_type).first())
        if ds is not None:
            # v8：复用已有系统批次时补挂数据源（此前上传批次未挂源）
            if datasource_id is not None and ds.datasource_id is None:
                ds.datasource_id = datasource_id
                s.flush()
            return ds.id
        ds = Dataset(name=name, category=category, source_type=source_type,
                     source_path=source_path, dataset_name=dataset_name or "",
                     datasource_id=datasource_id)
        s.add(ds)
        s.flush()
        return ds.id


def refresh_dataset_count(dataset_id: Optional[int]) -> None:
    """按 Image 行数回填 n_images（导入/生成/删除后调用）。"""
    if dataset_id is None:
        return
    with session_scope() as s:
        n = s.query(Image).filter(Image.dataset_id == dataset_id).count()
        ds = s.get(Dataset, dataset_id)
        if ds is not None:
            ds.n_images = n


def assign_legacy_datasets() -> int:
    """把 dataset_id IS NULL 的存量 Image 按品类归入"历史未分组"批次。

    幂等：归组后不再有 NULL 行，重复执行返回 0。返回归组图片数。
    """
    with session_scope() as s:
        cats = [r[0] for r in
                s.query(Image.category)
                .filter(Image.dataset_id.is_(None)).distinct().all()]
        total = 0
        for cat in cats:
            ds = (s.query(Dataset)
                  .filter(Dataset.name == LEGACY_DATASET_NAME,
                          Dataset.category == cat,
                          Dataset.source_type == "legacy").first())
            if ds is None:
                ds = Dataset(name=LEGACY_DATASET_NAME, category=cat,
                             source_type="legacy",
                             note="M8a 前的存量图片自动归组")
                s.add(ds)
                s.flush()
            n = (s.query(Image)
                 .filter(Image.dataset_id.is_(None), Image.category == cat)
                 .update({"dataset_id": ds.id}, synchronize_session=False))
            ds.n_images = (s.query(Image)
                           .filter(Image.dataset_id == ds.id).count())
            total += n
        return total


# ── 数据源级联删除（不足清单 #6 补强，2026-08-30）────────────
def _chunks(seq: list, size: int = 900) -> Iterator[list]:
    """SQLite IN 子句安全分块（变量数上限，大批次数据源删除必需）。"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def delete_datasource_cascade(ds_id: int, purge_files: bool = False) -> dict:
    """级联删除数据源：批次/图片/检测/反馈/缺陷明细/伪异常/视频一并清理。

    旧 delete_datasource 只删 data_sources 行，批次/图片成孤儿行
    （SQLite 未启用外键，静默残留），盘上文件完全不动。

    - DB 级联（显式按序删，子表在前）：
      工单挂接 -> 反馈/缺陷明细 -> 检测 -> 伪异常 -> 图片 ->
      视频帧关联置空 -> 视频 -> 批次 -> 数据源
    - 文件级联（purge_files=True）：只删 storage 根内的系统文件
      （图片/视频副本、热力图、伪异常文件、批次目录 ds{id}）；
      **外部原始目录属用户数据，永不自动删除**（生命周期策略：避免误删），
      计入 external_skipped 供手动清理。
    """
    from app.config import get_settings
    from app.db.models import (DataSource, DefectRecord, Detection, Feedback,
                               Image as ImageRow, PseudoAnomaly, Video,
                               VideoFrame, WorkOrderSource)

    # ── 阶段 1：删行前收集 ID 与文件路径（删后取不到）──
    with session_scope() as s:
        ds = s.get(DataSource, ds_id)
        if ds is None:
            raise ValueError(f"数据源不存在: {ds_id}")
        name = ds.name
        ds_cats: dict[int, str] = {r[0]: r[1] for r in
                                   s.query(Dataset.id, Dataset.category)
                                   .filter(Dataset.datasource_id == ds_id).all()}
        ds_ids = list(ds_cats)
        img_ids: list[int] = []
        image_paths: list[str] = []
        for part in _chunks(ds_ids):
            for iid, p in (s.query(ImageRow.id, ImageRow.path)
                           .filter(ImageRow.dataset_id.in_(part)).all()):
                img_ids.append(iid)
                image_paths.append(p)
        video_ids: list[int] = []
        video_paths: list[str] = []
        for part in _chunks(ds_ids):
            for vid, p in (s.query(Video.id, Video.path)
                           .filter(Video.dataset_id.in_(part)).all()):
                video_ids.append(vid)
                video_paths.append(p)
        det_ids: list[int] = []
        det_files: list[str] = []
        pseudo_files: list[str] = []
        n_feedback = n_pseudo = 0
        for part in _chunks(img_ids):
            for did, hm, ov in (s.query(Detection.id, Detection.heatmap_path,
                                        Detection.overlay_path)
                                .filter(Detection.image_id.in_(part)).all()):
                det_ids.append(did)
                det_files.extend(x for x in (hm, ov) if x)
            for r in (s.query(PseudoAnomaly)
                      .filter(PseudoAnomaly.base_image_id.in_(part)).all()):
                n_pseudo += 1
                pseudo_files.extend(x for x in (r.image_path, r.mask_path) if x)
        for part in _chunks(video_ids):
            for did, hm, ov in (s.query(Detection.id, Detection.heatmap_path,
                                        Detection.overlay_path)
                                .filter(Detection.video_id.in_(part)).all()):
                det_ids.append(did)
                det_files.extend(x for x in (hm, ov) if x)
        if det_ids:
            n_feedback = (s.query(Feedback)
                          .filter(Feedback.detection_id.in_(det_ids)).count())

        # ── 阶段 2：按序删行（子表在前，不留孤儿）──
        (s.query(WorkOrderSource)
         .filter(WorkOrderSource.datasource_id == ds_id)
         .delete(synchronize_session=False))
        for part in _chunks(det_ids):
            (s.query(Feedback)
             .filter(Feedback.detection_id.in_(part))
             .delete(synchronize_session=False))
            (s.query(DefectRecord)
             .filter(DefectRecord.detection_id.in_(part))
             .delete(synchronize_session=False))
            (s.query(Detection)
             .filter(Detection.id.in_(part))
             .delete(synchronize_session=False))
        for part in _chunks(img_ids):
            (s.query(PseudoAnomaly)
             .filter(PseudoAnomaly.base_image_id.in_(part))
             .delete(synchronize_session=False))
            (s.query(VideoFrame)
             .filter(VideoFrame.image_id.in_(part))
             .update({"image_id": None}, synchronize_session=False))
            (s.query(ImageRow)
             .filter(ImageRow.id.in_(part))
             .delete(synchronize_session=False))
        for part in _chunks(video_ids):
            (s.query(VideoFrame)
             .filter(VideoFrame.video_id.in_(part))
             .delete(synchronize_session=False))
            (s.query(Video)
             .filter(Video.id.in_(part))
             .delete(synchronize_session=False))
        if ds_ids:
            (s.query(Dataset)
             .filter(Dataset.id.in_(ds_ids))
             .delete(synchronize_session=False))
        s.delete(ds)

    out = {"deleted": True, "datasource_id": ds_id, "name": name,
           "datasets": len(ds_ids), "images": len(img_ids),
           "videos": len(video_ids), "detections": len(det_ids),
           "feedback": n_feedback, "pseudo": n_pseudo,
           "purge_files": bool(purge_files),
           "files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0,
           "external_skipped": []}

    # ── 阶段 3：文件清理（DB 提交后执行；仅 storage 根内）──
    if purge_files:
        storage_root = get_settings().storage("").resolve()

        def _in_storage(p: str) -> bool:
            try:
                return Path(p).resolve().is_relative_to(storage_root)
            except OSError:
                return False

        for p in dict.fromkeys(image_paths + video_paths + det_files
                               + pseudo_files):
            if not p:
                continue
            fp = Path(p)
            if not fp.exists():
                continue
            if _in_storage(p) and fp.is_file():
                out["bytes_freed"] += fp.stat().st_size
                fp.unlink(missing_ok=True)
                out["files_deleted"] += 1
            else:
                out["external_skipped"].append(str(p))
        for did, cat in ds_cats.items():
            d = get_settings().storage("images") / str(cat) / f"ds{did}"
            if d.is_dir():
                out["bytes_freed"] += sum(
                    f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
                out["dirs_deleted"] += 1
                try:  # 顺手清空品类目录（仅当已空）
                    d.parent.rmdir()
                except OSError:
                    pass
    log_action("delete_datasource_cascade",
               f"id={ds_id} name={name} datasets={len(ds_ids)} "
               f"images={len(img_ids)} purge_files={purge_files}",
               extra=out)
    return out
