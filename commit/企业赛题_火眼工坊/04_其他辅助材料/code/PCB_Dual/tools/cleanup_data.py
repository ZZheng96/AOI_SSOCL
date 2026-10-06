# -*- coding: utf-8 -*-
"""PCB_Ins v2 历史数据清理工具（数据生命周期管理，2026-08-30）。

用法（仓库根目录下）：
    python tools/cleanup_data.py --report                # 只出报告（默认，不改任何东西）
    python tools/cleanup_data.py --reset-train           # 清空 storage/train（prepare 时自动重建）
    python tools/cleanup_data.py --purge-outputs --days 30  # 清 30 天前的检测结果归档
    python tools/cleanup_data.py --snapshots-keep 5      # 每品类保留 激活版+最新5版，其余移入 _archive/
    python tools/cleanup_data.py --db-trim --days 30     # 清 30 天前的操作日志/后台任务记录
    所有动作支持 --dry-run（只打印将删除的内容，不实际删除）

保留/清理策略详见 docs/架构与设计.md「数据生命周期」附录。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402


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
    root = settings.storage("")
    print("== storage 目录占用 ==")
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            print(f"  {sub.name:<16} {_fmt(_dir_size(sub)):>10}")
        else:
            print(f"  {sub.name:<16} {_fmt(sub.stat().st_size):>10} (file)")
    print("\n== 数据库表行数 ==")
    from app.db.database import session_scope
    from app.db import models as m
    with session_scope() as s:
        for name in ("datasets", "data_sources", "work_orders", "images",
                     "detections", "inspects", "defect_records", "feedback",
                     "models", "eval_runs", "operation_logs", "tasks",
                     "alarm_records", "ng_filters"):
            cls = getattr(m, "".join(p.capitalize() for p in name.split("_")), None)
            if cls is None:
                # 类名推断失败时按表名统计
                from sqlalchemy import text
                try:
                    n = s.execute(text(f"SELECT COUNT(1) FROM {name}")).scalar()
                except Exception:  # noqa: BLE001
                    n = "-"
                print(f"  {name:<20} {n}")
            else:
                print(f"  {name:<20} {s.query(cls).count():>8}")


def reset_train(settings, dry: bool) -> None:
    d = settings.storage("train")
    size = _dir_size(d)
    n = len(list(d.iterdir())) if d.is_dir() else 0
    print(f"{'[dry] ' if dry else ''}train 重建清理：{n} 品类目录，{_fmt(size)}"
          "（prepare 时按当前数据自动重建）")
    if not dry:
        shutil.rmtree(d, ignore_errors=True)


def purge_outputs(settings, days: int, dry: bool) -> None:
    d = settings.outputs_dir
    if not d.is_dir():
        print("outputs 不存在，跳过")
        return
    cutoff = time.time() - days * 86400
    n, size = 0, 0
    for day_dir in sorted(d.iterdir()):
        if not day_dir.is_dir():
            continue
        try:
            mt = day_dir.stat().st_mtime
        except OSError:
            continue
        if mt < cutoff:
            n += 1
            size += _dir_size(day_dir)
            if not dry:
                shutil.rmtree(day_dir, ignore_errors=True)
    print(f"{'[dry] ' if dry else ''}outputs 清理（>{days}天）：{n} 个日期目录，{_fmt(size)}")


def snapshots_keep(settings, keep: int, dry: bool) -> None:
    root = settings.snapshots_dir
    if not root.is_dir():
        print("snapshots 不存在，跳过")
        return
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "_archive":
            continue
        active = None
        cur = cat_dir / "current.json"
        if cur.is_file():
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
    from app.db.database import session_scope
    from app.db.models import OperationLog, Task
    cutoff = datetime.now() - timedelta(days=days)
    with session_scope() as s:
        for cls in (OperationLog, Task):
            q = s.query(cls).filter(cls.created_at < cutoff)
            n = q.count()
            print(f"{'[dry] ' if dry else ''}{cls.__tablename__} 清理（>{days}天）：{n} 行")
            if not dry:
                q.delete(synchronize_session=False)
        s.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="PCB_Ins v2 历史数据清理工具")
    ap.add_argument("--report", action="store_true", help="只出占用/行数报告")
    ap.add_argument("--reset-train", action="store_true")
    ap.add_argument("--purge-outputs", action="store_true")
    ap.add_argument("--snapshots-keep", type=int, default=None, metavar="N")
    ap.add_argument("--db-trim", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    acted = any([args.reset_train, args.purge_outputs,
                 args.snapshots_keep is not None, args.db_trim])
    if args.report or not acted:
        report(settings)
        if not acted:
            print("\n（默认只出报告；加 --purge-* 等选项执行清理，详见 --help）")
            return
    if args.reset_train:
        reset_train(settings, args.dry_run)
    if args.purge_outputs:
        purge_outputs(settings, args.days, args.dry_run)
    if args.snapshots_keep is not None:
        snapshots_keep(settings, args.snapshots_keep, args.dry_run)
    if args.db_trim:
        db_trim(settings, args.days, args.dry_run)


if __name__ == "__main__":
    main()
