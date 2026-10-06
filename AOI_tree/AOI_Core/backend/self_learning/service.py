"""自学习闭环服务（检测引擎版，赛题问题三：用户反馈驱动的优化）。

M2 起闭环流程改为 引擎 SSCL：
  1. 操作员反馈落库 Feedback 表后，routes_feedback 同步调
     engine.submit_feedback 即学（秒级生效），成功即置 consumed=True
  2. /self_learning/update 语义改为"巩固落盘"：engine.consolidate
     把当前内存态（已在线更新）固化为新快照版本，供 A-B 对比与回滚
  3. activate_model 切换激活版本 → engine.activate(category, version)

M5a："落 Feedback + 即学 + consumed"抽为 submit_feedback_record 共用函数，
routes_feedback（操作员反馈）与 routes_review（灰区复核）两边复用。

旧 demo1 EWC 增量微调 / demo4 重拟合路径已随 model_adapter 一并删除。
"""
from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func

from ..core import datasets as ds_helper
from ..core.ingest import register_image
from ..core.notify import notify_event
from ..db.database import log_action, session_scope
from ..db.models import (ConsolidationBatch, ConsolidationFeedback, Detection,
                         Feedback, Model, StatsDaily)
from ..engine import get_engine

logger = logging.getLogger(__name__)


# ── 反馈即学共用路径（M5a：/feedback 与 /review 复用）─────────
def map_verdict_label(feedback_type: str, operator_label: int) -> Tuple[str, int]:
    """false_positive→(wrong,0) false_negative/new_defect→(wrong,1)
    confirmed→(correct, operator_label)；review（灰区复核）系统原判定为
    待复核（未承诺二分类），verdict 取 none、以人工 label 为准学习
    （handler 侧 none+label 分支按真值分流：正常→回流评估 / 缺陷→入样例库）。
    uncertain（M15a 操作员"无法确认"）→(none, None)：不定真值，
    样本进主动选样队列语义（引擎设计 §5 第三类反馈）。
    未知类型（2026-08-30）：记警告日志后按 (none, label) 处理——handler
    会按真值学习，但未知类型多为前后端词汇表失配，必须留痕（原静默
    落入死路零学习，MPDD 评测教训）。"""
    if feedback_type == "false_positive":
        return "wrong", 0
    if feedback_type in ("false_negative", "new_defect"):
        return "wrong", 1
    if feedback_type == "confirmed":
        return "correct", int(operator_label)
    if feedback_type == "uncertain":
        return "none", None
    if feedback_type != "review":
        logger.warning("[feedback] 未知 feedback_type=%r，按 (none, label) "
                       "真值学习处理——请检查前后端类型词汇表", feedback_type)
    return "none", int(operator_label)


def region_box_pre(region) -> Tuple[Optional[list], Optional[dict]]:
    """拆解 Feedback.region（M7a 起为 dict，存量为 list/None）→ (box, pre)。

    旧格式 list → (list, None)；新格式 dict → (region["box"], region["pre"]）。
    """
    if isinstance(region, dict):
        return region.get("box"), region.get("pre")
    if isinstance(region, list):
        return region, None
    return None, None


def _region_to_norm_box(image_path: str, region) -> Optional[List[float]]:
    """操作员框选区域 [x,y,w,h]（像素）→ 归一化 [x0,y0,x1,y1]。
    兼容存量：region 为 dict（M7a 新格式）时先取 ["box"]。"""
    from PIL import Image as PILImage
    region, _ = region_box_pre(region)
    if not region:
        return None
    try:
        with PILImage.open(image_path) as im:
            w, h = im.size
        x, y, rw, rh = [float(v) for v in region[:4]]
        if w <= 0 or h <= 0 or rw <= 0 or rh <= 0:
            return None
        return [max(0.0, min(1.0, x / w)),
                max(0.0, min(1.0, y / h)),
                max(0.0, min(1.0, (x + rw) / w)),
                max(0.0, min(1.0, (y + rh) / h))]
    except Exception:  # noqa: BLE001 图像读不出则不带框
        return None


def submit_feedback_record(detection_id: int, feedback_type: str,
                           operator_label: int,
                           defect_type: Optional[str] = None,
                           comment: Optional[str] = None,
                           region: Optional[list] = None,
                           operator: str = "operator",
                           source_event_id: Optional[str] = None) -> Dict:
    """落 Feedback + 今日统计 + 引擎即学（成功置 consumed=True）。

    M7a：即学前先用当前引擎对该图打一次分，记"反馈前判定" pre
    （score/decision/slots）随 region 一起落库（{"box":.., "pre":..}），
    供翻案曲线/追溯链做 pre→post 对比。引擎未就绪时 pre=None 不中断。

    引擎未 prepare / 品类无快照时不中断：落库照常，engine_update=None。
    返回 {feedback_id, engine_update, note, category, image_path, pre}。

    M8a：反馈图不在 Image 表时自动登记（split=feedback, source=feedback），
    挂"反馈回流"系统批次（按品类 get_or_create）。
    """
    engine = get_engine()
    def _current_version(category: str):
        try:
            return engine.current_version(category)
        except Exception:  # noqa: BLE001
            return None

    with session_scope() as s:
        det = s.get(Detection, detection_id)
        if det is None:
            raise ValueError("检测记录不存在")
        if source_event_id:
            existing = (s.query(Feedback)
                        .filter(Feedback.source_event_id == source_event_id)
                        .first())
            if existing is not None:
                version = _current_version(det.category)
                return {"feedback_id": existing.id,
                        "engine_update": None if not existing.consumed else {"idempotent": True},
                        "note": "已存在相同来源事件",
                        "category": det.category, "image_path": det.image_path,
                        "pre": (existing.region or {}).get("pre"),
                        "post": existing.post,
                        "learning_trace": existing.learning_trace,
                        "model_version_before": existing.model_version_before,
                        "model_version_after": existing.model_version_after,
                        "feedback_status": "stored",
                        "learning_status": "applied" if existing.consumed else "unavailable",
                        "consolidation_status": "not_requested",
                        "model_activation_status": "unchanged",
                        "model_version_before": version,
                        "model_version_after": version}
        image_path, category = det.image_path, det.category
        model_version_before = _current_version(category)
        img_registered = (s.query(Image)
                          .filter(Image.path == image_path).first() is not None)

    if not img_registered:
        ds_id = ds_helper.get_or_create_dataset(
            ds_helper.FEEDBACK_DATASET_NAME, category, "feedback")
        # operator_label=-1（uncertain 无法确认）→ label=unknown，不妄断真值
        fb_label = ("unknown" if int(operator_label) < 0
                    else ("anomaly" if int(operator_label) == 1 else "normal"))
        register_image(image_path, category=category, split="feedback",
                       label=fb_label,
                       defect_type=defect_type, source="feedback",
                       copy_to_storage=False, dataset_id=ds_id)
        ds_helper.refresh_dataset_count(ds_id)

    # M7a：反馈前判定（~60-300ms，即学 ~130ms，合计仍 <1s 红线）
    pre: Optional[dict] = None
    try:
        r = get_engine().predict_image_path(category, image_path)
        pre = {"score": round(float(r.score), 6), "decision": r.decision,
               "slots": {k: round(float(v), 4)
                         for k, v in (r.slot_scores or {}).items()}}
    except Exception:  # noqa: BLE001 引擎未就绪/未 prepare 时 pre=None
        pre = None

    with session_scope() as s:
        fb = Feedback(detection_id=detection_id,
                      feedback_type=feedback_type,
                      operator_label=operator_label,
                      defect_type=defect_type, comment=comment,
                      region={"box": region, "pre": pre}, operator=operator,
                      source_event_id=source_event_id)
        s.add(fb)
        s.flush()
        fb_id = fb.id

        stats = s.query(StatsDaily).filter(StatsDaily.date == date.today()).first()
        if stats is None:
            stats = StatsDaily(date=date.today())
            s.add(stats)
            s.flush()
        stats.n_feedback += 1
        if feedback_type == "false_positive":
            stats.n_false_positive += 1
        elif feedback_type == "false_negative":
            stats.n_false_negative += 1

    verdict, label = map_verdict_label(feedback_type, operator_label)
    box = None
    if region and label == 1:
        box = _region_to_norm_box(image_path, region)
    engine_update: Optional[dict] = None
    note: Optional[str] = None
    def mark_consumed():
        with session_scope() as s:
            row = s.get(Feedback, fb_id)
            if row is not None:
                row.consumed = True

    try:
        engine_update = get_engine().submit_feedback(
            category, image_path, verdict=verdict, label=label, box=box,
            on_complete=mark_consumed, feedback_id=fb_id)
        notify_event("feedback", {
            "category": category, "detection_id": detection_id,
            "feedback_type": feedback_type, "label": label,
            "image_path": image_path,
            "version": get_engine().current_version(category),
        })
    except Exception as e:  # noqa: BLE001 引擎未就绪不中断反馈落库
        note = f"模型未就绪，反馈仅落库（{e}）"

    log_action("feedback",
               f"detection_id={detection_id} type={feedback_type} "
               f"engine_update={'ok' if engine_update else 'skip'}",
               user=operator)
    model_version_after = _current_version(category)
    post: Optional[dict] = None
    if engine_update and not engine_update.get("async"):
        try:
            r = engine.predict_image_path(category, image_path)
            post = {"score": round(float(r.score), 6),
                    "decision": r.decision,
                    "slots": {k: round(float(v), 4)
                              for k, v in (r.slot_scores or {}).items()},
                    "model_version": model_version_after,
                    "status": "captured"}
        except Exception:  # noqa: BLE001 学习完成但反馈后推理不可用
            post = {"model_version": model_version_after,
                    "status": "unavailable"}
    trace = {"feedback_id": fb_id, "verdict": verdict, "label": label,
             "learning_status": "applied" if engine_update else "unavailable",
             "model_version_before": model_version_before,
             "model_version_after": model_version_after,
             "engine_update": engine_update}
    with session_scope() as s:
        row = s.get(Feedback, fb_id)
        if row is not None:
            row.model_version_before = (f"v{model_version_before}"
                                         if isinstance(model_version_before, int)
                                         else model_version_before)
            row.model_version_after = (f"v{model_version_after}"
                                        if isinstance(model_version_after, int)
                                        else model_version_after)
            row.post = post
            row.learning_trace = trace
    return {"feedback_id": fb_id, "engine_update": engine_update,
            "note": note, "category": category, "image_path": image_path,
            "pre": pre, "post": post, "learning_trace": trace,
            "feedback_status": "stored",
            "learning_status": "applied" if engine_update else "unavailable",
            "consolidation_status": "not_requested",
            "model_activation_status": "unchanged",
            "model_version_before": model_version_before,
            "model_version_after": model_version_after}


# ── 反馈判对映射（M5a 在线学习曲线用）────────────────────────
def feedback_correctness(feedback_type: str, operator_label: int,
                         det: Optional[Detection]) -> int:
    """1=系统判对 0=系统误判 -1=不计入（M15a uncertain 无法确认不定真值）。
    false_positive/false_negative/new_defect → 0（人工已指出系统错）；
    confirmed → 1；review → 人工 label 与系统二分类一致为对
    （无关联 detection 时按判对计）。"""
    if feedback_type in ("false_positive", "false_negative", "new_defect"):
        return 0
    if feedback_type == "confirmed":
        return 1
    if feedback_type == "uncertain":
        return -1
    if feedback_type == "review":
        if det is None:
            return 1
        return int(bool(det.is_anomaly) == bool(operator_label))
    return 1


def online_learning_curve(category: str, window: int = 10) -> Dict:
    """真实反馈流的在线学习曲线（纯 DB 计算，不跑模型）。

    按反馈时间升序逐条算判对率（滚动窗口 window），横轴=反馈序号 k。
    附当前引擎融合权重快照（pipe 未加载则为 None，不强加载）。
    """
    with session_scope() as s:
        rows = (s.query(Feedback, Detection)
                .outerjoin(Detection, Feedback.detection_id == Detection.id)
                .filter(Detection.category == category)
                .order_by(Feedback.created_at.asc(), Feedback.id.asc())
                .all())
    points: List[Dict] = []
    corr_hist: List[int] = []
    for fb, det in rows:
        c = feedback_correctness(fb.feedback_type, fb.operator_label, det)
        if c < 0:
            continue                     # M15a：uncertain 不计入判对率曲线
        corr_hist.append(c)
        k = len(corr_hist)
        win = corr_hist[-window:]
        points.append({"k": k, "rolling_acc": sum(win) / len(win),
                       "cum_feedback": k,
                       "correct": c, "feedback_type": fb.feedback_type})

    total = len(corr_hist)
    cutoff = datetime.now() - timedelta(days=7)
    last7d = sum(1 for fb, _ in rows if fb.created_at and fb.created_at >= cutoff)

    weights: Optional[Dict] = None
    loaded = get_engine()._pipes.get(category)  # 只读内存态，不触发装载
    if loaded is not None:
        w = getattr(loaded[0], "weights", None)
        if w:
            weights = dict(w)

    return {
        "category": category,
        "window": window,
        "points": points,
        "summary": {
            "total_feedback": total,
            "correct_rate": (sum(corr_hist) / total) if total else None,
            "last7d_feedback": last7d,
        },
        "weights": weights,
    }


# ── 翻案曲线（M7a：反馈前判定 vs 当前引擎重打分）─────────────
FLIP_CURVE_SYNC_MAX = 50   # 有 pre 的反馈数超过该值时走后台任务


def _judge_match(decision: Optional[str], label: int) -> bool:
    """引擎判定与人工 label 是否一致（gray 视为未承诺二分类 → 不一致）。"""
    if decision is None:
        return False
    return (decision == "anomaly") if int(label) == 1 else (decision == "normal")


def flip_curve(category: str, progress_cb=None) -> Dict:
    """翻案曲线：该品类有 pre 的反馈按时间升序，逐条用当前引擎重打分 → post。

    flipped = post 判定与人工 label 一致 且 pre 判定与人工不一致。
    引擎未加载该品类时抛 RuntimeError（路由层转 409）。
    """
    engine = get_engine()
    if engine.current_version(category) is None:
        raise RuntimeError(f"品类 '{category}' 未准备模型，无法重打分")

    with session_scope() as s:
        rows = (s.query(Feedback, Detection)
                .join(Detection, Feedback.detection_id == Detection.id)
                .filter(Detection.category == category,
                        Feedback.invalidated == False)  # noqa: E712
                .order_by(Feedback.created_at.asc(), Feedback.id.asc())
                .all())

    with_pre: List[Tuple[Feedback, Detection, dict]] = []
    legacy_count = 0
    for fb, det in rows:
        _, pre = region_box_pre(fb.region)
        if pre:
            with_pre.append((fb, det, pre))
        else:
            legacy_count += 1

    points: List[Dict] = []
    # M15a：uncertain（operator_label=-1/None）无确定真值，不纳入翻案率
    with_pre = [(fb, det, pre) for fb, det, pre in with_pre
                if fb.operator_label is not None and int(fb.operator_label) >= 0]
    flipped_count = 0
    total = len(with_pre)
    for k, (fb, det, pre) in enumerate(with_pre, start=1):
        post_score: Optional[float] = None
        post_decision: Optional[str] = None
        post_status = "unavailable"
        stored_post = fb.post if isinstance(fb.post, dict) else None
        if stored_post and stored_post.get("status") == "captured":
            post_score = stored_post.get("score")
            post_decision = stored_post.get("decision")
            post_status = "captured"
        else:
            try:
                r = engine.predict_image_path(category, det.image_path)
                post_score = round(float(r.score), 6)
                post_decision = r.decision
                post_status = "dynamic_legacy"
            except Exception:  # noqa: BLE001 图像丢失/推理失败 → post 置空
                pass
        flipped = (_judge_match(post_decision, fb.operator_label)
                   and not _judge_match(pre.get("decision"), fb.operator_label))
        flipped_count += int(flipped)
        points.append({
            "k": k, "image_path": det.image_path,
            "feedback_type": fb.feedback_type, "label": fb.operator_label,
            "pre_score": pre.get("score"), "post_score": post_score,
            "pre_decision": pre.get("decision"), "post_decision": post_decision,
            "post_model_version": (fb.post or {}).get("model_version"),
            "post_status": post_status,
            "flipped": flipped,
        })
        if progress_cb:
            progress_cb(k, total, f"重打分 {k}/{total}")

    return {
        "category": category,
        "points": points,
        "summary": {
            "total": total,
            "flipped_count": flipped_count,
            "flip_rate": (flipped_count / total) if total else None,
            "legacy_count": legacy_count,
        },
    }


def flip_curve_feedback_count(category: str) -> int:
    """该品类有 pre 的有效反馈数（路由层据此决定同步/后台；作废反馈不计）。"""
    with session_scope() as s:
        rows = (s.query(Feedback.region)
                .join(Detection, Feedback.detection_id == Detection.id)
                .filter(Detection.category == category,
                        Feedback.invalidated == False)  # noqa: E712
                .all())
    return sum(1 for (region,) in rows if region_box_pre(region)[1])


# ── 反馈统计 ─────────────────────────────────────────────────
def _feedback_query(s, category: Optional[str]):
    """有效反馈查询（可按 Detection.category 过滤）；已作废反馈不参与统计。"""
    q = (s.query(Feedback, Detection)
         .join(Detection, Feedback.detection_id == Detection.id)
         .filter(Feedback.invalidated == False))  # noqa: E712
    if category:
        q = q.filter(Detection.category == category)
    return q


def feedback_summary(category: Optional[str] = None,
                     workorder_id: Optional[int] = None) -> Dict:
    """反馈统计：total/fp/fn/confirmed/new_defect/reviewed/unconsumed/misclassification_rate。

    workorder_id 给定则按工单口径过滤：反馈关联检测的图像须属于该工单
    挂接数据源的批次（Feedback→Detection.image_id→Image.dataset_id→
    Dataset.datasource_id）；无图像关联的反馈（RTSP 流帧等）不计入。
    """
    with session_scope() as s:
        q = _feedback_query(s, category)
        if workorder_id is not None:
            from ..db.models import Dataset, Image as ImageRow, WorkOrderSource
            ds_ids = [r[0] for r in s.query(WorkOrderSource.datasource_id)
                      .filter(WorkOrderSource.workorder_id == workorder_id).all()]
            if not ds_ids:
                rows = []
            else:
                rows = (q.join(ImageRow, Detection.image_id == ImageRow.id)
                        .join(Dataset, ImageRow.dataset_id == Dataset.id)
                        .filter(Dataset.datasource_id.in_(ds_ids)).all())
        else:
            rows = q.all()
    total = len(rows)
    n_fp = sum(1 for fb, _ in rows if fb.feedback_type == "false_positive")
    n_fn = sum(1 for fb, _ in rows if fb.feedback_type == "false_negative")
    n_confirmed = sum(1 for fb, _ in rows if fb.feedback_type == "confirmed")
    n_new = sum(1 for fb, _ in rows if fb.feedback_type == "new_defect")
    n_reviewed = sum(1 for fb, _ in rows if fb.feedback_type == "review")
    n_unconsumed = sum(1 for fb, _ in rows if not fb.consumed)
    return {
        "total": total,
        "false_positives": n_fp,
        "false_negatives": n_fn,
        "confirmed": n_confirmed,
        "new_defects": n_new,
        "reviewed": n_reviewed,
        "unconsumed": n_unconsumed,
        "misclassification_rate": (n_fp + n_fn) / max(total, 1),
    }


def pending_update_status() -> Dict:
    """各品类快照版本 + 未巩固反馈数。

    未巩固反馈数 = 引擎 pipe 已加载时 handler.feedback_log 长度
    （自上次 consolidate 起累积；consolidate 后快照固化为新版本）；
    pipe 未加载（服务重启后未检测过该品类）时为 None。
    """
    engine = get_engine()
    cats = set()
    if os.path.isdir(engine.snap_root):
        cats |= {d for d in os.listdir(engine.snap_root)
                 if os.path.isdir(os.path.join(engine.snap_root, d))}
    with session_scope() as s:
        cats |= {r[0] for r in s.query(Detection.category).distinct().all()}
        # DB 中尚未即学消费的反馈（引擎未就绪时仅落库的兜底）
        pending_rows = (s.query(Detection.category, func.count(Feedback.id))
                        .join(Feedback, Feedback.detection_id == Detection.id)
                        .filter(Feedback.consumed == False,  # noqa: E712
                                Feedback.invalidated == False)  # noqa: E712
                        .group_by(Detection.category).all())
    db_pending = {cat: int(cnt) for cat, cnt in pending_rows}

    categories: List[Dict] = []
    for cat in sorted(cats):
        versions = engine.list_versions(cat)
        cur = engine.current_version(cat)
        feedback_log_len: Optional[int] = None
        loaded = engine._pipes.get(cat)  # 不触发装载，仅读内存态
        if loaded is not None:
            handler = getattr(loaded[0], "handler", None)
            flog = getattr(handler, "feedback_log", None)
            if flog is not None:
                feedback_log_len = len(flog)
        categories.append({
            "category": cat,
            "versions": versions,
            "current_version": cur,
            "prepared": cur is not None,
            "feedback_log_len": feedback_log_len,
            "unconsumed_db": db_pending.get(cat, 0),
        })
    return {"categories": categories}


# ── 巩固落盘（旧"增量更新"语义的新实现）────────────────────
def consolidate(category: str, note: str = "") -> Dict:
    """固化在线学习，并记录本次快照包含的反馈集合与版本血缘。"""
    engine = get_engine()
    before = engine.current_version(category)
    with session_scope() as s:
        rows = (s.query(Feedback.id)
                .join(Detection, Feedback.detection_id == Detection.id)
                .filter(Detection.category == category,
                        Feedback.consumed.is_(True),
                        Feedback.invalidated.is_(False)).all())
        feedback_ids = [int(row[0]) for row in rows]
    try:
        new_v = engine.consolidate(category, note=note or "consolidate via API")
    except RuntimeError as e:
        raise ValueError(f"品类 '{category}' 未准备，无法巩固（{e}）")
    with session_scope() as s:
        batch = ConsolidationBatch(
            category=category, from_version=before, to_version=new_v,
            feedback_count=len(feedback_ids), note=note,
            status="activated")
        s.add(batch)
        s.flush()
        for feedback_id in feedback_ids:
            s.add(ConsolidationFeedback(
                consolidation_id=batch.id, feedback_id=feedback_id))
        consolidation_id = batch.id
    trace = {"consolidation_id": consolidation_id, "category": category,
             "from_version": before, "to_version": new_v,
             "feedback_ids": feedback_ids,
             "feedback_count": len(feedback_ids), "note": note}
    log_action("self_update_consolidate",
               f"category={category} from=v{before} new_version=v{new_v} "
               f"feedback_count={len(feedback_ids)} note={note}",
               extra=trace)
    notify_event("consolidate", {"category": category, "version": new_v})
    return {"category": category, "version": f"v{new_v}",
            "version_num": new_v, "from_version": before,
            "consolidation_id": consolidation_id,
            "feedback_ids": feedback_ids, "feedback_count": len(feedback_ids),
            "activated": True, "message": "巩固落盘完成，已激活为新版本"}


# ── 模型切换 ────────────────────────────────────────────────
def resolve_model_version(model_id: int) -> Tuple[str, int]:
    """解析 Model 行 → (category, version_num)；非 引擎快照/版本不可解析时抛 ValueError。"""
    with session_scope() as s:
        model = s.get(Model, model_id)
        if model is None:
            raise ValueError(f"模型不存在: id={model_id}")
        category, version_str, fmt = model.category, model.version, model.format
    if fmt != "engine_snapshot":
        raise ValueError(f"模型 id={model_id} 非 引擎快照（format={fmt}），"
                         f"旧格式模型已随 demo1/demo4 引擎下线")
    m = re.search(r"v(\d+)", version_str or "")
    if m is None:
        raise ValueError(f"模型版本号无法解析: {version_str}")
    return category, int(m.group(1))


def activate_model(model_id: int) -> Dict:
    """按 Model 行的 (category, version) 调 engine.activate 热切换，并记录前后版本。"""
    category, version = resolve_model_version(model_id)
    engine = get_engine()
    before = engine.current_version(category)

    engine.activate(category, version)  # 版本不存在时抛 ValueError

    with session_scope() as s:
        s.query(Model).filter(Model.category == category,
                              Model.id != model_id).update(
            {"is_active": False}, synchronize_session=False)
        s.query(Model).filter(Model.id == model_id).update(
            {"is_active": True}, synchronize_session=False)

    trace = {"model_id": model_id, "category": category,
             "from_version": before, "to_version": version,
             "feedback_ids": [], "reason": "manual_activate"}
    log_action("activate_model", f"model_id={model_id} category={category} "
                                 f"from=v{before} version=v{version}",
               extra=trace)
    return {"ok": True, "activated": True, "model_id": model_id,
            "category": category, "from_version": before,
            "version": f"v{version}", "trace": trace}
