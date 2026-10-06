"""数据导入：文件夹 / MVTec AD / DAGM / 视频 → DB + storage。"""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
from PIL import Image

from .config import get_settings
from .datasets import create_dataset, refresh_dataset_count, update_dataset_params
from ..db.database import session_scope
from ..db.models import Image as ImageRow, Video

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fingerprint(path: Path) -> str:
    """快速文件指纹（大小 + 修改时间 + 头部 64KB）：
    导入幂等去重用，避免对 GB 级数据集做全文 SHA1 导致导入极慢。"""
    st = path.stat()
    h = hashlib.sha1()
    h.update(str(st.st_size).encode())
    h.update(str(st.st_mtime_ns).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 16))
    return h.hexdigest()


def _dataset_image_dir(category: str, dataset_id: int) -> Path:
    """M8a 批次图片目录：storage/images/{category}/ds{dataset_id}/。"""
    d = get_settings().storage("images") / category / f"ds{dataset_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_image(path: str | Path, category: str = "default",
                   split: str = "unlabeled", label: str = "unknown",
                   defect_type: Optional[str] = None, source: str = "manual",
                   copy_to_storage: bool = False,
                   dataset_id: Optional[int] = None) -> int:
    """登记单张图片（幂等：path 唯一）。返回 image_id。

    M8a：带 dataset_id 的拷贝走 storage/images/{category}/ds{id}/ds{id}_{原名}
    （批次目录 + 前缀消除同名冲突）；无 dataset_id 保持旧 raw_images 目录。
    """
    p = Path(path)
    if copy_to_storage:
        # 2026-08-31 走查改进：copy 到 storage 时保留 split/label 子目录结构
        # （storage/images/{cat}/ds{n}/train/good/...），否则 guard_bundle 的
        # "init_normal 必须来自 train 域" 红线会把复制的正常图误判为泄漏
        # （shift2 走查复现：copy_to_storage=True 后路径丢 train 域标记）。
        sub = ""
        if dataset_id is not None:
            base = _dataset_image_dir(category, dataset_id)
        else:
            base = get_settings().storage("raw_images")
        if split in ("train", "test", "val", "template", "feedback"):
            sub = str(Path(split) / ("good" if label == "normal" else label))
        dst_dir = base / sub if sub else base
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"ds{dataset_id}_{p.name}" if dataset_id is not None else dst_dir / f"{category}_{p.name}"
        if not dst.exists():
            shutil.copy2(p, dst)
        p = dst
    with session_scope() as s:
        row = s.query(ImageRow).filter(ImageRow.path == str(p)).first()
        if row is not None:
            if dataset_id is not None and row.dataset_id is None:
                row.dataset_id = dataset_id
            return row.id
        w = h = None
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            pass
        row = ImageRow(path=str(p), category=category, split=split, label=label,
                       defect_type=defect_type, source=source, width=w, height=h,
                       sha1=_fingerprint(p), dataset_id=dataset_id)
        s.add(row)
        s.flush()
        return row.id


def _register_batch(records: List[tuple], source: str,
                    progress_cb: Optional[Callable[[int, int], None]] = None,
                    dataset_id: Optional[int] = None) -> Dict:
    """批量登记图片：单事务 + 快速指纹，records=(path, category, split, label, defect_type)。

    v8 修复：图片已存在（此前导入过、换新数据源重导入）时不再跳过——
    改挂到当前批次/数据源（重归属），否则重导入到新源会得到 0 张的空批次。
    """
    total = len(records)
    imported = 0
    reowned = 0
    changed_old: set = set()
    with session_scope() as s:
        # 关 autoflush：循环内逐条 SELECT（查重/重归属）不应触发挂起
        # INSERT/UPDATE 提前落库——否则本会话长期持有写锁，progress_cb
        # 的 task 进度 UPDATE 被自己阻塞 5s 报 "database is locked"
        s.autoflush = False
        for i, (p, cat, split, lbl, defect) in enumerate(records):
            sp = str(p)
            row = s.query(ImageRow).filter(ImageRow.path == sp).first()
            if row is None:
                w = h = None
                try:
                    with Image.open(p) as im:
                        w, h = im.size  # 仅读文件头，不解码像素
                except Exception:  # noqa: BLE001
                    pass
                s.add(ImageRow(path=sp, category=cat, split=split, label=lbl,
                               defect_type=defect, source=source,
                               width=w, height=h, sha1=_fingerprint(p),
                               dataset_id=dataset_id))
                imported += 1
            elif dataset_id is not None and row.dataset_id != dataset_id:
                # 重导入：图片已在库（旧批次/未挂源），改挂新批次
                if row.dataset_id is not None:
                    changed_old.add(row.dataset_id)
                row.dataset_id = dataset_id
                row.category = cat
                row.split = split
                row.label = lbl
                if defect:
                    row.defect_type = defect
                reowned += 1
            if progress_cb:
                progress_cb(i + 1, total)
    # 旧批次计数回刷（图片被移走后 n_images 需更新）
    for old_id in changed_old:
        refresh_dataset_count(old_id)
    return {"imported": imported, "total": total, "reowned": reowned}


def import_folder(folder: str | Path, category: str = "default",
                  split: str = "unlabeled", label: str = "unknown",
                  source: str = "manual", copy_to_storage: bool = True,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  dataset_id: Optional[int] = None,
                  name: Optional[str] = None,
                  note: Optional[str] = None,
                  datasource_id: Optional[int] = None) -> Dict:
    """递归导入文件夹内全部图片。子文件夹约定：
    若存在 good/normal 子目录→normal；其余子目录→anomaly 且子目录名为 defect_type。

    M8a：未传 dataset_id 时自动建 source_type=folder 的批次行，导入完成回填 n_images。
    """
    folder = Path(folder)
    files = [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS
             and not f.name.lower().endswith("_mask.png")]  # 跳过标注掩码图
    n_masks = sum(1 for f in folder.rglob("*")
                  if f.suffix.lower() in IMG_EXTS
                  and f.name.lower().endswith("_mask.png"))
    if dataset_id is None:
        dataset_id = create_dataset(
            category, "folder", source_path=str(folder),
            params={"split": split, "label": label, "source": source,
                    "copy_to_storage": copy_to_storage, "n_masks": n_masks},
            name=name, note=note, dataset_name=folder.name,
            datasource_id=datasource_id)
    records = []
    for f in files:
        rel = f.relative_to(folder)
        lbl, defect = label, None
        parts = [p.lower() for p in rel.parts[:-1]]
        sp = split
        # W-check：template 目录或命名规则（A_tpl.png / A_template.png）
        # 约定为模板图（前端反馈 v6：数据源声明「提供模板图」时按命名识别）
        if "template" in parts:
            sp, lbl = "template", "normal"
        elif rel.name.lower().rsplit(".", 1)[0].endswith(("_tpl", "_template")):
            sp, lbl = "template", "normal"
        elif any(p in ("good", "normal") for p in parts):
            lbl = "normal"
        elif label == "unknown" and parts:
            lbl = "anomaly"
            defect = rel.parts[-2] if len(rel.parts) >= 2 else None
        records.append((f, category, sp, lbl, defect))
    if copy_to_storage:
        # 复制模式下逐张处理（保持原幂等语义）
        n_ok = 0
        for i, (f, cat, sp, lbl, defect) in enumerate(records):
            register_image(f, cat, sp, lbl, defect, source, copy_to_storage,
                           dataset_id=dataset_id)
            n_ok += 1
            if progress_cb:
                progress_cb(i + 1, len(records))
        refresh_dataset_count(dataset_id)
        return {"imported": n_ok, "folder": str(folder),
                "dataset_id": dataset_id}
    result = _register_batch(records, source, progress_cb, dataset_id=dataset_id)
    refresh_dataset_count(dataset_id)
    result["folder"] = str(folder)
    result["dataset_id"] = dataset_id
    return result


def import_mvtec(mvtec_root: str | Path, categories: Optional[List[str]] = None,
                 progress_cb: Optional[Callable[[int, int], None]] = None,
                 dataset_ids: Optional[Dict[str, int]] = None,
                 name: Optional[str] = None,
                 note: Optional[str] = None,
                 datasource_id: Optional[int] = None) -> Dict:
    """导入 MVTec AD 结构：{cat}/train/good/*, {cat}/test/{good|defect}/*。
    单事务批量入库 + 快速指纹 + 预扫描总数（进度条有真实百分比）。

    M8a：按品类各建一个 source_type=mvtec 批次（dataset_ids 可传入复用），
    导入完成回填各批次 n_images，结果带 dataset_ids。
    W-source：datasource_id 归属数据源；各品类 ground_truth 掩码数写入批次
    params（缺陷位置标注条件的体检原料）。"""
    root = Path(mvtec_root)
    cats = categories or [d.name for d in root.iterdir() if d.is_dir()]
    if dataset_ids is None:
        dataset_ids = {}
        for cat in cats:
            if (root / cat).is_dir():
                dataset_ids[cat] = create_dataset(
                    cat, "mvtec", source_path=str(root),
                    name=name or f"{cat}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    note=note, dataset_name=root.name,
                    datasource_id=datasource_id)
    total_imported, total_files = 0, 0
    for cat in cats:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        records = []
        n_masks = 0
        for f in cat_dir.rglob("*"):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            if f.name.lower().endswith("_mask.png"):
                # MVTec 标注掩码图（ground truth），不是缺陷样本：
                # 跳过入库，但计入掩码数（工单体检用）
                n_masks += 1
                continue
            rel = f.relative_to(cat_dir).parts
            split = rel[0] if rel[0] in ("train", "test") else "unlabeled"
            defect = rel[1] if len(rel) > 2 else None
            lbl = "normal" if defect in (None, "good") else "anomaly"
            records.append((f, cat, split, lbl,
                            None if lbl == "normal" else defect))
        r = _register_batch(records, "mvtec", progress_cb,
                            dataset_id=dataset_ids.get(cat))
        if n_masks:
            update_dataset_params(dataset_ids.get(cat), {"n_masks": n_masks})
        total_imported += r["imported"]
        total_files += r["total"]
        refresh_dataset_count(dataset_ids.get(cat))
    return {"imported": total_imported, "total": total_files,
            "categories": cats, "dataset_ids": dataset_ids}


def register_video(path: str | Path, category: str = "default",
                   label: str = "unknown", copy_to_storage: bool = True,
                   dataset_id: Optional[int] = None) -> int:
    p = Path(path)
    if copy_to_storage:
        dst = get_settings().storage("raw_videos") / f"{category}_{p.name}"
        if not dst.exists():
            shutil.copy2(p, dst)
        p = dst
    with session_scope() as s:
        row = s.query(Video).filter(Video.path == str(p)).first()
        if row is not None:
            if dataset_id is not None and row.dataset_id is None:
                row.dataset_id = dataset_id
            return row.id
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        row = Video(path=str(p), category=category, label=label, fps=fps,
                    n_frames=n, duration_s=(n / fps if fps else None),
                    width=w, height=h, dataset_id=dataset_id)
        s.add(row)
        s.flush()
        return row.id
