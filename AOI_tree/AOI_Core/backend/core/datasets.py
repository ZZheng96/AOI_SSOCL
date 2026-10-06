"""数据集批次助手（M8a 数据资产工业化治理）。

所有导入/生成入口强制挂 Dataset 批次；老数据（dataset_id IS NULL）
在服务启动时由 assign_legacy_datasets() 按品类归入"历史未分组"批次（幂等）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..db.database import session_scope
from ..db.models import Dataset, Image

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
