"""AC-072：保留期清理——预览（dry-run，默认）+ 审批后执行。

规则取 business_budget：
- heatmaps/（热图/overlay/mask/raw .npy）超 heatmap_overlay_days 天；
- raw_images/、uploads/（原图）超 raw_hot_days 天 + raw_cold_months 月（热存 + 冷存）。
快照（snapshot_keep_*，由快照治理负责）与反馈事件 WAL（永久）永不进入计划。
执行要求：approval.plan_id 与预览一致、approver 非空、confirm=True；执行时逐项复核
仍过期且仍在存储根内，否则跳过；每次执行追加审计行 retention_audit.jsonl。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_DAY = 86400.0


def _rules(budget: Dict) -> Dict[str, float]:
    raw_days = float(budget["raw_hot_days"]) + 30.0 * float(budget["raw_cold_months"])
    return {"heatmaps": float(budget["heatmap_overlay_days"]),
            "raw_images": raw_days, "uploads": raw_days}


def _inside(root: Path, p: Path) -> bool:
    try:
        return os.path.commonpath([str(root), str(p)]) == str(root)
    except ValueError:
        return False


def _plan_id(items: List[Dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it['path']}|{it['mtime']}|{it['size']}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def preview_retention(storage_dir, budget: Dict, now: Optional[float] = None,
                      limit: int = 200) -> Dict:
    """只读扫描，返回过期候选清单（不删除任何文件）。"""
    root = Path(storage_dir).resolve()
    now = time.time() if now is None else float(now)
    items: List[Dict] = []
    per_dir: Dict[str, Dict] = {}
    for sub, days in _rules(budget).items():
        d = root / sub
        stat = per_dir.setdefault(sub, {"keep_days": days, "files": 0, "bytes": 0})
        if not d.is_dir():
            continue
        cutoff = now - days * _DAY
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            st = p.stat()
            if st.st_mtime >= cutoff:
                continue
            items.append({"path": str(p.resolve()), "dir": sub, "size": int(st.st_size),
                          "mtime": int(st.st_mtime),
                          "age_days": round((now - st.st_mtime) / _DAY, 1)})
            stat["files"] += 1
            stat["bytes"] += int(st.st_size)
    return {"dry_run": True, "plan_id": _plan_id(items), "storage_dir": str(root),
            "generated_at": datetime.fromtimestamp(now).isoformat(),
            "total_files": len(items), "total_bytes": sum(i["size"] for i in items),
            "per_dir": per_dir, "items": items[:limit], "truncated": len(items) > limit,
            "excluded": ["snapshots（snapshot_keep_* 治理）", "feedback_event_wal（永久）"]}


class RetentionApprovalError(PermissionError):
    pass


def execute_retention(storage_dir, budget: Dict, approval: Optional[Dict] = None,
                      dry_run: bool = True, now: Optional[float] = None) -> Dict:
    """dry_run=True（默认）等价于预览；真删须 approval={plan_id, approver, confirm=True}，
    且 plan_id 与当前预览一致（计划变化 → 拒绝，须重新预览审批）。"""
    plan = preview_retention(storage_dir, budget, now=now, limit=10 ** 9)
    if dry_run:
        plan["items"] = plan["items"][:200]
        return plan
    approval = approval or {}
    if not (approval.get("confirm") is True and str(approval.get("approver") or "").strip()):
        raise RetentionApprovalError("执行清理需审批：approver 非空且 confirm=True")
    if approval.get("plan_id") != plan["plan_id"]:
        raise RetentionApprovalError("计划已变化或 plan_id 不匹配，请重新预览并审批")
    root = Path(storage_dir).resolve()
    rules = _rules(budget)
    now_t = time.time() if now is None else float(now)
    deleted, skipped, freed = [], [], 0
    for it in plan["items"]:
        p = Path(it["path"])
        try:
            st = p.stat()
            still = st.st_mtime < now_t - rules[it["dir"]] * _DAY
            if not (_inside(root, p.resolve()) and still) or p.is_symlink():
                skipped.append(it["path"])
                continue
            p.unlink()
            deleted.append(it["path"])
            freed += int(st.st_size)
        except FileNotFoundError:
            skipped.append(it["path"])
    rec = {"at": datetime.now().isoformat(), "plan_id": plan["plan_id"],
           "approver": approval["approver"], "reason": approval.get("reason"),
           "deleted": len(deleted), "skipped": len(skipped), "freed_bytes": freed}
    with open(root / "retention_audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {**rec, "dry_run": False, "deleted_paths": deleted[:200]}
