"""统计看板 / 操作日志 / 后台任务查询接口。"""
from __future__ import annotations

from datetime import date, timedelta
import io
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from sqlalchemy import func

from ..core.tasks import task_manager
from ..db.database import log_action, session_scope
from ..db.models import Detection, Feedback, Image, OperationLog, StatsDaily, Video
from ._wo_shared import archived_workorder_ids
from .schemas import task_interpretation, to_dict

router = APIRouter()


def _today_stats_dict(stats: StatsDaily | None) -> dict:
    """StatsDaily 行 → 看板用 dict（含平均延迟）。"""
    if stats is None:
        return {"date": date.today().isoformat(), "n_inspected": 0,
                "n_anomaly": 0, "n_feedback": 0, "n_false_positive": 0,
                "n_false_negative": 0, "avg_latency_ms": None}
    # 无数据返回 None（不以 0.0 冒充缺失）
    avg = (stats.total_latency_ms / stats.n_inspected
           if stats.n_inspected else None)
    return {"date": stats.date.isoformat(), "n_inspected": stats.n_inspected,
            "n_anomaly": stats.n_anomaly, "n_feedback": stats.n_feedback,
            "n_false_positive": stats.n_false_positive,
            "n_false_negative": stats.n_false_negative,
            "avg_latency_ms": avg}


# ── 统计 ─────────────────────────────────────────────────────
def _scope_categories(s, workorder_id: Optional[int],
                      category: Optional[str]) -> Optional[list]:
    """仅非工单查询使用品类过滤；工单由各查询按 workorder_id 严格隔离。"""
    if workorder_id is not None:
        return None  # 工单按 Detection.workorder_id 隔离，不从共享品类推断归属
    if category:
        return [category]
    return None


def _range_start(range_name: str):
    """统计时间范围 -> 起始时间（None=全部）。today=今日 0 点，7d/30d=近 N 天。"""
    if range_name == "today":
        from datetime import datetime, time as dtime
        return datetime.combine(date.today(), dtime.min)
    if range_name == "7d":
        from datetime import datetime, time as dtime
        return datetime.combine(date.today() - timedelta(days=6), dtime.min)
    if range_name == "30d":
        from datetime import datetime, time as dtime
        return datetime.combine(date.today() - timedelta(days=29), dtime.min)
    return None


# ── 质检报表（复核后口径，W-report 2026-08-29）────────────────
# 设计依据（docs/重构设计_学习与统计页.md）：系统判定口径含误报，反馈闭环
# 产生的复核真值（Feedback.operator_label × Detection.is_anomaly 四象限）
# 才是质检报表口径；StatsDaily 已累积的 FP/FN 此前零展示。
def _scope_fb_rows(s, cats, start, arch=None, workorder_id=None):
    """口径内有效反馈四象限原料 [(operator_label, is_anomaly, defect_type,
    created_at, dataset_id), ...]（剔除作废反馈）。

    arch=存档工单 id 列表（全局口径传入）：排除存档工单检测，
    保留 workorder_id 为 NULL 的孤儿记录（存档排除，2026-10-09）。"""
    q = (s.query(Feedback.operator_label, Detection.is_anomaly,
                 Feedback.defect_type, Feedback.created_at,
                 Image.dataset_id)
         .join(Detection, Feedback.detection_id == Detection.id)
         .outerjoin(Image, Detection.image_id == Image.id)
         .filter(Feedback.invalidated.is_(False),
                 Feedback.operator_label.in_((0, 1))))
    if workorder_id is not None:
        q = q.filter(Detection.workorder_id == workorder_id)
    if cats is not None:
        q = q.filter(Detection.category.in_(cats))
    if start is not None:
        q = q.filter(Feedback.created_at >= start)
    if arch:
        # 存档排除（2026-10-09）：NULL IN 子句结果为 NULL，
        # 故须显式保留 workorder_id 为 NULL 的孤儿记录
        q = q.filter(Detection.workorder_id.is_(None)
                     | ~Detection.workorder_id.in_(arch))
    return q.all()


def _quadrants(rows):
    """四象限计数：tp/fp/fn/tn（operator_label 真值 × 系统判定）。"""
    tp = sum(1 for r in rows if r[0] == 1 and r[1])
    fp = sum(1 for r in rows if r[0] == 0 and r[1])
    fn = sum(1 for r in rows if r[0] == 1 and not r[1])
    tn = sum(1 for r in rows if r[0] == 0 and not r[1])
    return tp, fp, fn, tn


@router.get("/stats/quality_overview")
def api_stats_quality_overview(category: Optional[str] = None,
                               workorder_id: Optional[int] = None,
                               range_name: str = "all"):
    """质检总览（复核后口径）：检测数/复核覆盖率/真实不良率/误报率/检出率。

    误报率 = FP/(FP+TN)（正常样本被判为缺陷的比例）；
    检出率 = TP/(TP+FN)（真实缺陷被系统检出的比例）；内部兼容字段 fn_rate 仍保留，表示同一混淆矩阵的 FN 侧数据；
    真实不良率 = 复核确认缺陷 / 复核数（区别于系统判定口径 defect_rate）。
    """
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        start = _range_start(range_name)
        # 存档排除（2026-10-09）：全局口径排除存档工单的检测
        arch = archived_workorder_ids(s) if workorder_id is None else None
        det_q = s.query(Detection.is_anomaly, Detection.latency_ms)
        if workorder_id is not None:
            det_q = det_q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            det_q = det_q.filter(Detection.category.in_(cats))
        if start is not None:
            det_q = det_q.filter(Detection.created_at >= start)
        if arch:
            det_q = det_q.filter(Detection.workorder_id.is_(None)
                                 | ~Detection.workorder_id.in_(arch))
        dets = det_q.all()
        n_det = len(dets)
        n_anom = sum(1 for a, _ in dets if a)
        avg_lat = (sum(l for _, l in dets) / n_det) if n_det else None

        rows = _scope_fb_rows(s, cats, start, arch, workorder_id)
        tp, fp, fn, tn = _quadrants(rows)
        n_rev = tp + fp + fn + tn
        # 分母为 0 的比率返回 None（不以 0.0 冒充缺失）
        return {
            "range": range_name,
            "categories": cats if cats is not None else [],
            "n_detections": n_det,
            "n_anomaly": n_anom,
            "system_defect_rate": (n_anom / n_det) if n_det else None,
            "avg_latency_ms": avg_lat,
            "n_reviewed": n_rev,
            "coverage": (n_rev / n_det) if n_det else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "true_defect_rate": ((tp + fn) / n_rev) if n_rev else None,
            "fp_rate": (fp / (fp + tn)) if (fp + tn) else None,
            "recall_defect": (tp / (tp + fn)) if (tp + fn) else None,
            "fn_rate": (fn / (fn + tp)) if (fn + tp) else None,
            "review_accuracy": ((tp + tn) / n_rev) if n_rev else None,
        }


@router.get("/stats/mistake_trend")
def api_stats_mistake_trend(category: Optional[str] = None,
                            workorder_id: Optional[int] = None,
                            range_name: str = "30d"):
    """误判率趋势：按真实检测批次聚合 FP 率 / FN 率 / 复核数。

    优先使用工单数据源的实际检测分组；无工单分组时回退到导入批次。
    未关联图片的反馈单独归组，不按自然日补空点。"""
    from ..db.models import Dataset
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        arch = archived_workorder_ids(s) if workorder_id is None else None
        from ..db.models import WorkOrder
        from ._wo_shared import _workorder_detect_batches
        batch_map = {}
        batch_names = {}
        batch_order = {}
        orders = s.query(WorkOrder).order_by(WorkOrder.id).all()
        for wo in orders:
            if workorder_id is not None and wo.id != workorder_id:
                continue
            for batch in _workorder_detect_batches(s, wo):
                key = f"wo{wo.id}:{batch['batch_key']}"
                batch_names[key] = f"{batch['source']} · 批次 {batch['batch_index']}"
                batch_order[key] = len(batch_order)
                for image_id in batch["image_ids"]:
                    batch_map[(wo.id, image_id)] = key
        q = (s.query(Feedback.operator_label, Detection.is_anomaly,
                     Feedback.defect_type, Feedback.created_at, Image.dataset_id,
                     Detection.workorder_id, Detection.image_id)
             .join(Detection, Feedback.detection_id == Detection.id)
             .outerjoin(Image, Detection.image_id == Image.id)
             .filter(Feedback.invalidated.is_(False), Feedback.operator_label.in_((0, 1))))
        if workorder_id is not None:
            q = q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            q = q.filter(Detection.category.in_(cats))
        if arch:
            q = q.filter(Detection.workorder_id.is_(None) | ~Detection.workorder_id.in_(arch))
        per_batch: dict = {}
        for label, anom, _dt, _created, ds_id, wo_id, image_id in q.all():
            key = batch_map.get((wo_id, image_id), int(ds_id) if ds_id is not None else "unbatched")
            b = per_batch.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            if label == 1 and anom:
                b["tp"] += 1
            elif label == 0 and anom:
                b["fp"] += 1
            elif label == 1 and not anom:
                b["fn"] += 1
            else:
                b["tn"] += 1
        ids = [k for k in per_batch if isinstance(k, int)]
        names = {int(i): n for i, n in s.query(Dataset.id, Dataset.name)
                 .filter(Dataset.id.in_(ids)).all()} if ids else {}
        items = []
        for key in sorted(per_batch, key=lambda x: (
                0 if x in batch_order else 1 if isinstance(x, int) else 2,
                batch_order[x] if x in batch_order else x if isinstance(x, int) else 0)):
            b = per_batch[key]
            n = b["tp"] + b["fp"] + b["fn"] + b["tn"]
            name = ("未关联批次" if key == "unbatched" else
                    batch_names.get(key, names.get(key, f"批次#{key}")))
            items.append({
                "date": name, "batch_id": None if key == "unbatched" else key,
                "batch_name": name, "reviewed": n,
                "tp": b["tp"], "fp": b["fp"], "fn": b["fn"], "tn": b["tn"],
                "fp_rate": (b["fp"] / (b["fp"] + b["tn"]))
                if (b["fp"] + b["tn"]) else None,
                "recall_defect": (b["tp"] / (b["tp"] + b["fn"]))
                if (b["tp"] + b["fn"]) else None,
                "fn_rate": (b["fn"] / (b["fn"] + b["tp"]))
                if (b["fn"] + b["tp"]) else None,
                "mistake_rate": ((b["fp"] + b["fn"]) / n) if n else None,
            })
        return {"items": items, "range": range_name, "basis": "batch"}


@router.get("/stats/defect_types")
def api_stats_defect_types(category: Optional[str] = None,
                           workorder_id: Optional[int] = None,
                           range_name: str = "all"):
    """缺陷类型帕累托：复核确认缺陷（operator_label=1）按 defect_type 分布。
    操作员未标注类型的归入「未标注」。"""
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        start = _range_start(range_name)
        # 存档排除（2026-10-09）：全局口径排除存档工单的检测
        arch = archived_workorder_ids(s) if workorder_id is None else None
        rows = [r for r in _scope_fb_rows(s, cats, start, arch, workorder_id) if r[0] == 1]
        counts: dict = {}
        for _l, _a, dt, _c, _ds in rows:
            key = str(dt).strip() if dt else "未标注"
            counts[key] = counts.get(key, 0) + 1
        items = [{"type": k, "count": v}
                 for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        return {"items": items, "total": len(rows)}


@router.get("/stats/batch_summary")
def api_stats_batch_summary(category: Optional[str] = None,
                            workorder_id: Optional[int] = None,
                            range_name: str = "all"):
    """批次对比：按导入批次（Dataset）聚合检测/系统不良/复核四象限。"""
    from ..db.models import Dataset
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        start = _range_start(range_name)
        # 存档排除（2026-10-09）：全局口径排除存档工单的检测
        arch = archived_workorder_ids(s) if workorder_id is None else None
        det_q = (s.query(Image.dataset_id,
                         func.count(Detection.id),
                         func.sum(Detection.is_anomaly),
                         func.avg(Detection.latency_ms),
                         func.avg(Detection.latency_e2e_ms))
                 .join(Image, Detection.image_id == Image.id))
        if workorder_id is not None:
            det_q = det_q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            det_q = det_q.filter(Detection.category.in_(cats))
        if start is not None:
            det_q = det_q.filter(Detection.created_at >= start)
        if arch:
            det_q = det_q.filter(Detection.workorder_id.is_(None)
                                 | ~Detection.workorder_id.in_(arch))
        det_rows = det_q.group_by(Image.dataset_id).all()

        fb_map: dict = {}
        for label, anom, _dt, _c, ds_id in _scope_fb_rows(s, cats, start, arch, workorder_id):
            if ds_id is None:
                continue
            b = fb_map.setdefault(ds_id, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            if label == 1 and anom:
                b["tp"] += 1
            elif label == 0 and anom:
                b["fp"] += 1
            elif label == 1 and not anom:
                b["fn"] += 1
            else:
                b["tn"] += 1

        ds_ids = [int(d) for d, *_ in det_rows if d is not None]
        name_map = {r.id: r.name for r in
                    s.query(Dataset.id, Dataset.name)
                    .filter(Dataset.id.in_(ds_ids)).all()} if ds_ids else {}
        items = []
        for ds_id, n_det, n_anom, avg_lat, avg_lat_e2e in det_rows:
            if ds_id is None:
                continue
            b = fb_map.get(ds_id, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            n_det = int(n_det or 0)
            items.append({
                "dataset_id": int(ds_id),
                "name": name_map.get(int(ds_id), f"批次#{int(ds_id)}"),
                "n_detections": n_det,
                "n_anomaly": int(n_anom or 0),
                "n_reviewed": b["tp"] + b["fp"] + b["fn"] + b["tn"],
                "fp_rate": (b["fp"] / (b["fp"] + b["tn"]))
                if (b["fp"] + b["tn"]) else None,
                "recall_defect": (b["tp"] / (b["tp"] + b["fn"]))
                if (b["tp"] + b["fn"]) else None,
                "fn_rate": (b["fn"] / (b["fn"] + b["tp"]))
                if (b["fn"] + b["tp"]) else None,
                # AVG 忽略 NULL；全缺时为 None，不补 0.0
                "avg_latency_ms": (float(avg_lat) if avg_lat is not None
                                   else None),
                "avg_latency_e2e_ms": (float(avg_lat_e2e)
                                       if avg_lat_e2e is not None else None),
            })
        items.sort(key=lambda x: -x["n_detections"])
        return {"items": items}


@router.get("/stats/health")
def api_stats_health(category: Optional[str] = None,
                     workorder_id: Optional[int] = None,
                     range_name: str = "7d",
                     limit: int = 3000):
    """产线健康信号（algo 每图产出落库 n_tiles，纯 DB 解析零前向）：
    灰区率（模型不确定→复判工作量）/ open 未解释信号（体系外缺陷预警）/
    对位预警（治具/传送带漂移）。工单 all 扫描全部检测，其余口径扫描最近 limit 条。"""
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        start = _range_start(range_name)
        # 存档排除（2026-10-09）：全局口径排除存档工单的检测
        arch = archived_workorder_ids(s) if workorder_id is None else None
        q = (s.query(Detection.n_tiles, Detection.latency_ms)
             .order_by(Detection.id.desc()))
        if workorder_id is not None:
            q = q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            q = q.filter(Detection.category.in_(cats))
        if start is not None:
            q = q.filter(Detection.created_at >= start)
        if arch:
            q = q.filter(Detection.workorder_id.is_(None)
                         | ~Detection.workorder_id.in_(arch))
        rows = (q.all() if workorder_id is not None and range_name == "all"
                else q.limit(min(max(int(limit), 1), 10000)).all())
        n = len(rows)
        gray = open_hits = align_warns = 0
        for tiles, _lat in rows:
            t = tiles if isinstance(tiles, dict) else {}
            if t.get("decision") == "gray":
                gray += 1
            if t.get("open_alert"):
                open_hits += 1
            if t.get("align_warn"):
                align_warns += 1
        return {"scanned": n,
                "gray_count": gray, "gray_rate": (gray / n) if n else None,
                "open_alerts": open_hits,
                "open_rate": (open_hits / n) if n else None,
                "align_warns": align_warns,
                "align_rate": (align_warns / n) if n else None,
                "range": range_name}


@router.get("/stats/overview")
def api_stats_overview(category: Optional[str] = None,
                       workorder_id: Optional[int] = None,
                       range: str = "all"):
    """统计看板（W-workorder：统计以工单为基础）。

    - workorder_id：按检测记录的工单归属过滤（优先于 category）；
    - range：today/7d/all，作用于 scope_stats；模拟系统页面固定 all；
    - 兼容字段 today/totals/defect_rate 同样遵守工单隔离，保留各自时间语义。
    """
    with session_scope() as s:
        cats = _scope_categories(s, workorder_id, category)
        # 存档排除（2026-10-09）：全局口径排除存档工单的检测
        arch = archived_workorder_ids(s) if workorder_id is None else None
        det_q = s.query(Detection)
        if workorder_id is not None:
            det_q = det_q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            det_q = det_q.filter(Detection.category.in_(cats))
        start = _range_start(range)
        if start is not None:
            det_q = det_q.filter(Detection.created_at >= start)
        if arch:
            det_q = det_q.filter(Detection.workorder_id.is_(None)
                                 | ~Detection.workorder_id.in_(arch))
        rows = det_q.with_entities(Detection.is_anomaly,
                                   Detection.latency_ms,
                                   Detection.latency_e2e_ms,
                                   Detection.target_type).all()
        n_scope = len(rows)
        n_scope_anomaly = sum(1 for a, _l, _e, _t in rows if a)
        lat_scope = (sum(l for _a, l, _e, _t in rows) / n_scope) if n_scope else None
        # AC-058：e2e 仅统计真实写入的记录，不拿推理耗时冒充；缺失计数单列
        e2e_vals = sorted(e for _a, _l, e, _t in rows if e is not None)
        lat_scope_e2e = (sum(e2e_vals) / len(e2e_vals)) if e2e_vals else None
        lat_scope_e2e_p95 = (e2e_vals[min(len(e2e_vals) - 1, int(0.95 * len(e2e_vals)))]
                             if e2e_vals else None)
        by_source = {}
        for _a, _l, _e, t in rows:
            by_source[t or "unknown"] = by_source.get(t or "unknown", 0) + 1

        today = (s.query(StatsDaily)
                 .filter(StatsDaily.date == date.today()).first())
        n_images = s.query(func.count(Image.id)).scalar() or 0
        n_videos = s.query(func.count(Video.id)).scalar() or 0
        n_feedback = s.query(func.count(Feedback.id)).scalar() or 0

        det_q2 = s.query(func.count(Detection.id))
        an_q = (s.query(func.count(Detection.id))
                .filter(Detection.is_anomaly.is_(True)))
        if workorder_id is not None:
            det_q2 = det_q2.filter(Detection.workorder_id == workorder_id)
            an_q = an_q.filter(Detection.workorder_id == workorder_id)
        if cats is not None:
            det_q2 = det_q2.filter(Detection.category.in_(cats))
            an_q = an_q.filter(Detection.category.in_(cats))
        if arch:
            # 存档排除（2026-10-09）：全局口径排除存档工单的检测
            det_q2 = det_q2.filter(Detection.workorder_id.is_(None)
                                   | ~Detection.workorder_id.in_(arch))
            an_q = an_q.filter(Detection.workorder_id.is_(None)
                               | ~Detection.workorder_id.in_(arch))
        n_detections = det_q2.scalar() or 0
        n_anomaly_total = an_q.scalar() or 0
        defect_rate = (n_anomaly_total / n_detections
                       if n_detections else None)
        return {"today": _today_stats_dict(today),
                "totals": {"n_images": n_images, "n_videos": n_videos,
                           "n_detections": n_detections,
                           "n_anomalies": n_anomaly_total,
                           "n_feedback": n_feedback},
                "defect_rate": defect_rate,
                # W-workorder：工单口径 + 时间范围（前端 KPI 用）
                "scope_stats": {"n_inspected": n_scope,
                                "n_anomaly": n_scope_anomaly,
                                "rate": (n_scope_anomaly / n_scope
                                         if n_scope else None),
                                "avg_latency_ms": lat_scope,
                                "latency_basis": "inference_only",
                                "avg_latency_e2e_ms": lat_scope_e2e,
                                "p95_latency_e2e_ms": lat_scope_e2e_p95,
                                "n_e2e_missing": n_scope - len(e2e_vals),
                                "categories": cats if cats is not None else [],
                                "range": range,
                                # AC-058 口径：时间窗口（非最近N）、分母=已落库检测记录（非相机帧）
                                "window_start": start.isoformat() if start is not None else None,
                                "denominator": "persisted_detections",
                                "workorder_id": workorder_id,
                                "excludes_archived_workorders": bool(arch),
                                "by_source": by_source,
                                "receipt_rate": None,
                                "receipt_rate_availability": "unavailable"}}


@router.get("/stats/latency_trend")
def api_stats_latency_trend(workorder_id: int):
    """工单全部检测的推理耗时：批次按首次检测排序，缺批次则按检测顺序。"""
    from ..db.models import Dataset
    with session_scope() as s:
        q = (s.query(Detection.id, Detection.latency_ms, Image.dataset_id,
                     Dataset.name)
             .outerjoin(Image, Detection.image_id == Image.id)
             .outerjoin(Dataset, Image.dataset_id == Dataset.id)
             .filter(Detection.workorder_id == workorder_id))
        has_unbatched = q.filter(Image.dataset_id.is_(None)).first() is not None
        if has_unbatched:
            rows = q.order_by(Detection.id.asc()).all()
            items = [{"label": f"检测 {i}", "detection_id": did,
                      "n_inspected": 1, "avg_latency_ms": latency}
                     for i, (did, latency, _ds, _name) in enumerate(rows, 1)]
            basis = "detection_order"
        else:
            rows = (q.with_entities(Image.dataset_id, Dataset.name,
                                    func.count(Detection.id),
                                    func.avg(Detection.latency_ms))
                    .group_by(Image.dataset_id, Dataset.name)
                    .order_by(func.min(Detection.id).asc()).all())
            items = [{"label": name or f"批次#{ds}", "dataset_id": ds,
                      "n_inspected": int(n), "avg_latency_ms": latency}
                     for ds, name, n, latency in rows]
            basis = "batch"
        return {"workorder_id": workorder_id, "range": "all",
                "basis": basis, "items": items}


@router.get("/stats/timeseries")
def api_stats_timeseries(days: int = 14, category: Optional[str] = None):
    """最近 days 天统计，按日期升序。

    M8a：category 过滤时 StatsDaily（全局表）口径改为实时从 Detection 表
    按日聚合（group by date(created_at)），返回结构不变（反馈类字段置 0）。
    """
    since = date.today() - timedelta(days=days - 1)
    with session_scope() as s:
        if category:
            day_col = func.date(Detection.created_at)
            rows = (s.query(day_col.label("d"),
                            func.count(Detection.id).label("n_inspected"),
                            func.sum(Detection.is_anomaly).label("n_anomaly"),
                            func.sum(Detection.latency_ms).label("total_latency"),
                            # e2e 仅取真实写入值（AVG 忽略 NULL），不回退推理耗时
                            func.avg(Detection.latency_e2e_ms)
                            .label("avg_latency_e2e"))
                    .filter(Detection.category == category,
                            day_col >= since.isoformat())
                    .group_by(day_col).order_by(day_col.asc()).all())
            items = []
            for d, n_ins, n_an, lat, lat_e2e in rows:
                n_ins = int(n_ins or 0)
                items.append({
                    "date": d, "n_inspected": n_ins,
                    "n_anomaly": int(n_an or 0), "n_feedback": 0,
                    "n_false_positive": 0, "n_false_negative": 0,
                    "total_latency_ms": float(lat or 0.0),
                    "avg_latency_ms": (float(lat or 0.0) / n_ins
                                       if n_ins else None),
                    "avg_latency_e2e_ms": (float(lat_e2e)
                                           if lat_e2e is not None else None)})
            return {"items": items}
        rows = (s.query(StatsDaily).filter(StatsDaily.date >= since)
                .order_by(StatsDaily.date.asc()).all())
        items = []
        for r in rows:
            d = to_dict(r)
            d["avg_latency_ms"] = (r.total_latency_ms / r.n_inspected
                                   if r.n_inspected else None)
            # e2e 累计缺失（旧数据为 0/NULL）时返回 None，不回退推理口径
            total_e2e = r.total_latency_e2e_ms
            d["avg_latency_e2e_ms"] = (total_e2e / r.n_inspected
                                       if (r.n_inspected and total_e2e)
                                       else None)
            items.append(d)
        return {"items": items}


# ── 操作日志 ─────────────────────────────────────────────────
@router.get("/logs")
def api_list_logs(limit: int = 100, action: Optional[str] = None,
                  category: Optional[str] = None):
    """操作日志。M8a：action 精确过滤；category 从 extra JSON 里过滤。"""
    with session_scope() as s:
        q = s.query(OperationLog)
        if action:
            q = q.filter(OperationLog.action == action)
        q = q.order_by(OperationLog.id.desc())
        # category 需解析 extra JSON，先多取再在 Python 侧精确过滤
        rows = q.limit(limit * 5 if category else limit).all()
        if category:
            rows = [r for r in rows
                    if isinstance(r.extra, dict)
                    and r.extra.get("category") == category][:limit]
        return {"items": [to_dict(r) for r in rows]}


# ── M10c：产线日报 / 审计日志导出（CSV，评审自检 §9 P2）────────
def _csv_response(filename: str, header: list, rows: list) -> Response:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return Response(
        content="﻿" + buf.getvalue(),  # BOM：Excel 直接打开不乱码
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.get("/stats/export")
def api_stats_export(days: int = 30, category: Optional[str] = None):
    """产线日报导出（CSV）：日期/检测数/异常数/不良率/反馈数/误检/漏检/
    平均延迟。category 过滤时从 Detection 表按日实时聚合（口径同
    /stats/timeseries）。"""
    data = api_stats_timeseries(days=days, category=category)
    rows = []
    for d in data["items"]:
        n_ins = d["n_inspected"]
        rows.append([d["date"], n_ins, d["n_anomaly"],
                     round(d["n_anomaly"] / n_ins, 4) if n_ins else "",
                     d["n_feedback"], d["n_false_positive"],
                     d["n_false_negative"],
                     (round(d["avg_latency_ms"], 1)
                      if d.get("avg_latency_ms") is not None else "")])
    suffix = f"_{category}" if category else ""
    return _csv_response(
        f"aoi_daily_report{suffix}_{date.today().isoformat()}.csv",
        ["date", "n_inspected", "n_anomaly", "defect_rate", "n_feedback",
         "n_false_positive", "n_false_negative", "avg_latency_ms"], rows)


@router.get("/logs/export")
def api_logs_export(action: Optional[str] = None,
                    category: Optional[str] = None,
                    limit: int = 2000):
    """审计日志导出（CSV）：时间/用户/动作/详情/结构化字段(JSON)。
    产线审计口径：谁、何时、对哪个品类/模型做了什么。"""
    import json as _json
    data = api_list_logs(limit=limit, action=action, category=category)
    rows = []
    for r in data["items"]:
        rows.append([r.get("created_at"), r.get("user"), r.get("action"),
                     r.get("detail"),
                     _json.dumps(r.get("extra") or {}, ensure_ascii=False)])
    return _csv_response(
        f"aoi_audit_{date.today().isoformat()}.csv",
        ["created_at", "user", "action", "detail", "extra_json"], rows)


# ── 后台任务 ─────────────────────────────────────────────────
@router.get("/tasks/recent")
def api_recent_tasks():
    items = task_manager.list_recent()
    return {"items": [{**item, **task_interpretation(item["status"])}
                       for item in items]}


@router.get("/tasks/{task_id}")
def api_get_task(task_id: int):
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {**task, **task_interpretation(task["status"], task.get("result"))}


@router.post("/tasks/{task_id}/cancel")
def api_cancel_task(task_id: int):
    """请求取消任务（A15 协作式取消）。

    pending 任务直接标记 cancelled；running 任务置位取消标志，由任务函数
    在循环检查点响应（不可中断的 C 调用如 sklearn/torch fit 会在当前
    阶段完成后于下一检查点终止）。返回 {cancel_requested}。"""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in ("done", "success", "failed", "error", "cancelled"):
        return {"cancel_requested": False, "status": task["status"],
                "detail": "任务已结束",
                **task_interpretation(task["status"], task.get("result"))}
    ok = task_manager.cancel(task_id)
    log_action("cancel_task", f"task_id={task_id} type={task['task_type']}",
               extra={"task_id": task_id, "accepted": ok})
    current = task_manager.get(task_id)
    return {"cancel_requested": ok, "status": current["status"],
            **task_interpretation(current["status"], current.get("result"))}
