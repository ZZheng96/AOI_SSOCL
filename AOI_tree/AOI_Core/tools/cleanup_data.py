# -*- coding: utf-8 -*-
"""AOI_sys 历史数据清理工具（数据生命周期管理，2026-08-30）。

用法（仓库根目录下）：
    python tools/cleanup_data.py --report                 # 只出报告（默认，不改任何东西）
    python tools/cleanup_data.py --purge-tmp              # 清空 storage/tmp（评估扰动图等临时产物）
    python tools/cleanup_data.py --purge-heatmaps --days 7   # 清 7 天前的热力图/叠加图（可再生成）
    python tools/cleanup_data.py --snapshots-keep 5       # 每品类保留 激活版+最新5版，其余移入 _archive/
    python tools/cleanup_data.py --db-trim --days 30      # 清 30 天前的操作日志/后台任务记录
    python tools/cleanup_data.py --drop-datasource BTAD   # 删除指定数据源及其批次/工单关联（行级）
    所有动作支持 --dry-run（只打印将删除的内容，不实际删除）

保留/清理策略详见 docs/数据生命周期与清理策略.md。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import get_settings  # noqa: E402


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _fmt(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def report(settings) -> None:
    sd = settings.storage_dir
    print("== storage 目录占用 ==")
    for sub in sorted(sd.iterdir()):
        if sub.is_dir():
            print(f"  {sub.name:<16} {_fmt(_dir_size(sub)):>10}")
        else:
            print(f"  {sub.name:<16} {_fmt(sub.stat().st_size):>10} (file)")
    print("\n== 数据库表行数 ==")
    from backend.db.database import session_scope
    from backend.db import models as m
    with session_scope() as s:
        for cls in (m.Dataset, m.DataSource, m.WorkOrder, m.Image, m.Video,
                    m.PseudoAnomaly, m.Detection, m.Feedback, m.Model,
                    m.EvalRun, m.StatsDaily, m.OperationLog, m.Task):
            print(f"  {cls.__tablename__:<20} {s.query(cls).count():>8}")


def purge_tmp(settings, dry: bool) -> None:
    d = settings.storage_dir / "tmp"
    if not d.is_dir():
        print("tmp 不存在，跳过")
        return
    n, size = 0, _dir_size(d)
    for f in d.iterdir():
        n += 1
        if not dry:
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)
    print(f"{'[dry] ' if dry else ''}tmp 清理：{n} 项，{_fmt(size)}")


def purge_heatmaps(settings, days: int, dry: bool) -> None:
    d = settings.storage_dir / "heatmaps"
    if not d.is_dir():
        print("heatmaps 不存在，跳过")
        return
    cutoff = time.time() - days * 86400
    n, size = 0, 0
    for f in d.iterdir():
        if not f.is_file():
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff:
            n += 1
            size += st.st_size
            if not dry:
                f.unlink(missing_ok=True)
    print(f"{'[dry] ' if dry else ''}heatmaps 清理（>{days}天）：{n} 文件，{_fmt(size)}"
          "（DB 检测记录保留，热力图可由追溯重检再生成）")


def snapshots_keep(settings, keep: int, dry: bool) -> None:
    root = settings.engine_storage / "snapshots"
    if not root.is_dir():
        print("snapshots 不存在，跳过")
        return
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "_archive":
            continue
        active = None
        cur = cat_dir / "current.json"
        if cur.is_file():
            import json
            try:
                active = int(json.loads(cur.read_text(encoding="utf-8"))["version"])
            except Exception:  # noqa: BLE001
                pass
        versions = sorted(
            (int(p.name[1:]) for p in cat_dir.iterdir()
             if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()),
            reverse=True)
        keep_set = set(versions[:keep]) | ({active} if active else set())
        for v in versions:
            if v in keep_set:
                continue
            src = cat_dir / f"v{v}"
            dst = root / "_archive" / f"{cat_dir.name}_v{v}_{time.strftime('%Y%m%d_%H%M%S')}"
            size = _dir_size(src)
            print(f"{'[dry] ' if dry else ''}  归档 {cat_dir.name}/v{v} -> {dst.name} ({_fmt(size)})")
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))


def db_trim(settings, days: int, dry: bool) -> None:
    from datetime import datetime, timedelta
    from backend.db.database import session_scope
    from backend.db.models import OperationLog, Task
    cutoff = datetime.now() - timedelta(days=days)
    with session_scope() as s:
        for cls in (OperationLog, Task):
            q = s.query(cls).filter(cls.created_at < cutoff)
            n = q.count()
            print(f"{'[dry] ' if dry else ''}{cls.__tablename__} 清理（>{days}天）：{n} 行")
            if not dry:
                q.delete(synchronize_session=False)
        s.flush()


def drop_datasource(settings, name: str, dry: bool) -> None:
    """删除数据源及其批次关联；多源工单只解除关联，不删除工单。"""
    from backend.db.database import session_scope
    from backend.db import models as m
    workorder_ids = []
    shared_workorder_ids = set()
    with session_scope() as s:
        srcs = s.query(m.DataSource).filter(m.DataSource.name == name).all()
        if not srcs:
            print(f"未找到数据源：{name}")
            return
        for src in srcs:
            ds_ids = [r.id for r in s.query(m.Dataset).filter(
                m.Dataset.datasource_id == src.id).all()]
            wo_ids = [r.workorder_id for r in s.query(m.WorkOrderSource).filter(
                m.WorkOrderSource.datasource_id == src.id).all()]
            shared_wo_ids = {wo_id for wo_id in wo_ids if s.query(m.WorkOrderSource).filter(
                m.WorkOrderSource.workorder_id == wo_id).count() > 1}
            workorder_ids.extend(wo_ids)
            shared_workorder_ids.update(shared_wo_ids)
            n_img = s.query(m.Image).filter(m.Image.dataset_id.in_(ds_ids)).count() if ds_ids else 0
            print(f"{'[dry] ' if dry else ''}删除数据源 {name}(id={src.id})："
                  f"批次 {len(ds_ids)}、工单关联 {len(wo_ids)}（共享工单不删除）、图片行 {n_img}")
            if dry:
                continue
            if ds_ids:
                s.query(m.Image).filter(m.Image.dataset_id.in_(ds_ids)).delete(
                    synchronize_session=False)
                s.query(m.Video).filter(m.Video.dataset_id.in_(ds_ids)).delete(
                    synchronize_session=False)
                s.query(m.Dataset).filter(m.Dataset.id.in_(ds_ids)).delete(
                    synchronize_session=False)
            s.query(m.WorkOrderSource).filter(
                m.WorkOrderSource.datasource_id == src.id).delete(synchronize_session=False)
            s.delete(src)
        s.flush()
    if not dry:
        from backend.db.database import log_action
        log_action(
            "cleanup_drop_datasource",
            f"datasource_name={name} sources={len(srcs)}",
            user="cleanup_tool",
            extra={"workorder_ids": workorder_ids,
                   "shared_workorder_ids": sorted(shared_workorder_ids)},
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="AOI_sys 历史数据清理工具")
    ap.add_argument("--report", action="store_true", help="只出占用/行数报告")
    ap.add_argument("--purge-tmp", action="store_true")
    ap.add_argument("--purge-heatmaps", action="store_true")
    ap.add_argument("--snapshots-keep", type=int, default=None,
                    metavar="N", help="每品类保留 激活版+最新N版")
    ap.add_argument("--db-trim", action="store_true")
    ap.add_argument("--days", type=int, default=7, help="heatmaps/db-trim 保留天数")
    ap.add_argument("--drop-datasource", type=str, default=None, metavar="NAME")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    acted = any([args.purge_tmp, args.purge_heatmaps,
                 args.snapshots_keep is not None, args.db_trim,
                 args.drop_datasource])
    if args.report or not acted:
        report(settings)
        if not acted:
            print("\n（默认只出报告；加 --purge-* 等选项执行清理，详见 --help）")
            return
    if args.purge_tmp:
        purge_tmp(settings, args.dry_run)
    if args.purge_heatmaps:
        purge_heatmaps(settings, args.days, args.dry_run)
    if args.snapshots_keep is not None:
        snapshots_keep(settings, args.snapshots_keep, args.dry_run)
    if args.db_trim:
        db_trim(settings, args.days, args.dry_run)
    if args.drop_datasource:
        drop_datasource(settings, args.drop_datasource, args.dry_run)


if __name__ == "__main__":
    main()
