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
import math
import os
from numbers import Real, Integral
import re
import threading
from functools import wraps
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from algo.decision.localization import consumable_regions, record_consumption

from ..api._wo_shared import archived_workorder_ids
from ..core import datasets as ds_helper
from ..core.ingest import register_image
from ..core.notify import notify_event
from ..db.database import log_action, session_scope
from ..db.models import (ConsolidationBatch, ConsolidationFeedback, Detection,
                         Feedback, Image, Model, StatsDaily)
from ..engine import get_engine

logger = logging.getLogger(__name__)


def feedback_json(value):
    """仅包装反馈证据：非有限/非JSON值置未知，不修改算法工作态。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): feedback_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [feedback_json(v) for v in value]
    return None


def feedback_details(trace):
    """只读持久反馈结果；缺失归因不使用当前模型补推。"""
    trace = trace if isinstance(trace, dict) else {}
    update = trace.get("engine_update")
    update = feedback_json(update) if isinstance(update, dict) else {}
    attribution = update.get("attribution")
    evidence = update.get("evidence")
    return {"engine_update": update or None,
            "learning_status": trace.get("learning_status", "unknown"),
            "attribution": attribution,
            "attribution_availability": "available" if attribution else "unavailable",
            "evidence": evidence,
            "evidence_availability": "available" if evidence else "unavailable"}


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


class FeedbackConflict(ValueError):
    pass


_feedback_submit_lock = threading.RLock()


def _serialized_feedback(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _feedback_submit_lock:
            try:
                return fn(*args, **kwargs)
            except IntegrityError:
                if kwargs.get("source_event_id"):
                    with session_scope() as s:
                        found = s.query(Feedback.id).filter(
                            Feedback.source_event_id == kwargs["source_event_id"]).first()
                    if found:
                        return fn(*args, **kwargs)
                raise
    return wrapped


@_serialized_feedback
def submit_feedback_record(detection_id: int, feedback_type: str,
                           operator_label: int,
                           defect_type: Optional[str] = None,
                           comment: Optional[str] = None,
                           region: Optional[list] = None,
                           operator: str = "operator",
                           source_event_id: Optional[str] = None,
                           candidate_only: bool = True,
                           allow_legacy_online: bool = False) -> Dict:
    """落库反馈；默认仅排队候选，不改变生产工作态。

    在线即学仅供显式隔离的 legacy 实验调用（allow_legacy_online=True）。

    M7a：即学前先用当前引擎对该图打一次分，记"反馈前判定" pre
    （score/decision/slots）随 region 一起落库（{"box":.., "pre":..}），
    供翻案曲线/追溯链做 pre→post 对比。引擎未就绪时 pre=None 不中断。

    引擎未 prepare / 品类无快照时不中断：落库照常，engine_update=None。
    返回 {feedback_id, engine_update, note, category, image_path, pre}。

    M8a：反馈图不在 Image 表时自动登记（split=feedback, source=feedback），
    挂"反馈回流"系统批次（按品类 get_or_create）。
    """
    if not candidate_only and not allow_legacy_online:
        raise FeedbackConflict("生产反馈只能排队候选；legacy 在线学习必须显式隔离")
    engine = get_engine()
    def _current_version(category: str):
        try:
            return engine.current_version(category)
        except Exception:  # noqa: BLE001
            return None

    existing = None
    source_event_id = source_event_id or None
    with session_scope() as s:
        det = s.get(Detection, detection_id)
        if det is None:
            raise ValueError("检测记录不存在")
        if source_event_id:
            existing = (s.query(Feedback)
                        .filter(Feedback.source_event_id == source_event_id)
                        .first())
            if existing is not None:
                stored = (existing.detection_id, existing.feedback_type,
                          existing.operator_label, existing.defect_type,
                          existing.comment, region_box_pre(existing.region)[0], existing.operator)
                incoming = (detection_id, feedback_type, operator_label, defect_type,
                            comment, region, operator)
                stored_candidate = bool((existing.learning_trace or {}).get("candidate_only"))
                if stored_candidate != candidate_only:
                    raise FeedbackConflict("来源事件学习路径不一致")
                if stored != incoming or existing.invalidated:
                    raise FeedbackConflict("来源事件ID对应不同payload或已作废")
            if existing is not None and (candidate_only or existing.consumed or
                    (existing.learning_trace or {}).get("learning_status") == "pending"):
                version = _current_version(det.category)
                return {"feedback_id": existing.id,
                        "engine_update": feedback_details(existing.learning_trace)["engine_update"],
                        "note": "已存在相同来源事件",
                        "category": det.category, "image_path": det.image_path,
                        "pre": region_box_pre(existing.region)[1],
                        "post": existing.post,
                        "learning_trace": existing.learning_trace,
                        "feedback_status": "stored",
                        "learning_status": (existing.learning_trace or {}).get(
                            "learning_status", "applied" if existing.consumed else "unknown"),
                        "consolidation_status": "not_requested",
                        "model_activation_status": "unchanged",
                        "model_version_before": version,
                        "model_version_after": version,
                        "idempotent": True}
        image_path, category = det.image_path, det.category
        # AC-055：审计只记录检测实际推理版本，缺失为 None（不回退 current_version）
        detection_model_version = (det.n_tiles or {}).get("model_version")
        # AC-063/064：即时学习同样按 region 消费快照回写 consumption
        _review = (det.n_tiles or {}).get("region_review") or {}
        instant_guard = {"detection_id": det.id,
                         "region_revision": int(_review.get("revision", 0)),
                         "regions": consumable_regions(_review)}
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

    if existing is not None:
        fb_id = existing.id
        pre = region_box_pre(existing.region)[1]
    else:
        with session_scope() as s:
            fb = Feedback(detection_id=detection_id,
                          feedback_type=feedback_type,
                          operator_label=operator_label,
                          defect_type=defect_type, comment=comment,
                          region={"box": region, "pre": pre,
                                  # AC-064：反馈框为人工标注（不与 engine_box 候选混用）
                                  "box_source": "human_box" if region else None},
                          operator=operator,
                          source_event_id=source_event_id,
                          learning_trace={"candidate_only": True, "learning_status": "candidate_queued"}
                          if candidate_only else None)
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

    if candidate_only:
        trace = {"feedback_id": fb_id, "candidate_only": True,
                 "learning_status": "candidate_queued",
                 "model_version_before": model_version_before,
                 "model_version_after": model_version_before,
                 "engine_update": None}
        with session_scope() as s:
            row = s.get(Feedback, fb_id)
            if row is not None:
                row.learning_trace = trace
                row.consumed = False
        log_action("feedback_candidate_queued",
                   f"detection_id={detection_id} type={feedback_type}",
                   user=operator,
                   extra={"feedback_id": fb_id, "detection_id": detection_id,
                          "source_event_id": source_event_id,
                          "category": category,
                          "learning_status": "candidate_queued",
                          "detection_model_version": detection_model_version})
        return {"feedback_id": fb_id, "engine_update": None,
                "note": "已排队候选；未应用、未激活，等待候选任务消费",
                "category": category, "image_path": image_path, "pre": pre,
                "post": None, "learning_trace": trace,
                "feedback_status": "stored", "learning_status": "candidate_queued",
                "consolidation_status": "not_requested",
                "model_activation_status": "unchanged",
                "model_version_before": model_version_before,
                "model_version_after": model_version_before,
                "activated": False, "idempotent": False}

    verdict, label = map_verdict_label(feedback_type, operator_label)
    box = None
    if region and label == 1:
        box = _region_to_norm_box(image_path, region)
    engine_update: Optional[dict] = None
    note: Optional[str] = None
    result_lock = threading.RLock()
    terminal = False
    trace = {"feedback_id": fb_id, "verdict": verdict, "label": label,
             "learning_status": "pending", "model_version_before": model_version_before,
             "model_version_after": model_version_before, "engine_update": None}
    post = None

    def persist_result(update, error=None, *, final=True):
        nonlocal terminal, trace, post
        with result_lock:
            if terminal:
                return
            version = _current_version(category)
            status = ("failed" if error else "applied") if final else "pending"
            trace = {**trace, "learning_status": status,
                     "model_version_after": version,
                     "engine_update": feedback_json(update)}
            if error is not None:
                trace["error"] = str(error)
            if final and error is None:
                try:
                    r = engine.predict_image_path(category, image_path)
                    post = feedback_json({"score": round(float(r.score), 6),
                        "decision": r.decision, "slots": r.slot_scores or {},
                        "model_version": version, "status": "captured"})
                except Exception:
                    post = {"model_version": version, "status": "unavailable"}
                trace["region_consumption"] = _record_region_consumption(
                    fb_id, instant_guard,
                    {"version_num": version, "parent_version": model_version_before},
                    candidate_id=f"instant:feedback:{fb_id}", kind="instant")
            # 回调只捕获标量/引擎，不复用请求线程的 ORM session。
            with session_scope() as s:
                row = s.get(Feedback, fb_id)
                if row is not None:
                    row.consumed = final and error is None
                    row.model_version_before = (f"v{model_version_before}"
                        if isinstance(model_version_before, int) else model_version_before)
                    row.model_version_after = (f"v{version}" if isinstance(version, int) else version)
                    row.post = post
                    row.learning_trace = feedback_json(trace)
            terminal = final

    persist_result(None, final=False)
    try:
        engine_update = get_engine().submit_feedback(
            category, image_path, verdict=verdict, label=label, box=box,
            on_result=persist_result, feedback_id=fb_id)
        engine_update = feedback_json(engine_update)
        persist_result(engine_update, final=not bool(engine_update and engine_update.get("async")))
        notify_event("feedback", {
            "category": category, "detection_id": detection_id,
            "feedback_type": feedback_type, "label": label,
            "image_path": image_path,
            "version": get_engine().current_version(category),
        })
    except Exception as e:  # noqa: BLE001 引擎未就绪不中断反馈落库
        persist_result(None, e)
        note = f"学习失败，反馈已落库（{e}）"

    log_action("feedback",
               f"detection_id={detection_id} type={feedback_type} "
               f"engine_update={'ok' if engine_update else 'skip'}",
               user=operator)
    with result_lock:
        trace = dict(trace)
        engine_update = trace["engine_update"]
        learning_status = trace["learning_status"]
        model_version_after = trace["model_version_after"]
    return {"feedback_id": fb_id, "engine_update": engine_update,
            "note": note, "category": category, "image_path": image_path,
            "pre": pre, "post": post, "learning_trace": trace,
            "feedback_status": "stored",
            "learning_status": learning_status,
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
            # 存档排除（2026-10-09）：全局口径排除存档工单检测的反馈，
            # 保留 workorder_id 为 NULL 的孤儿记录
            arch = archived_workorder_ids(s)
            if arch:
                q = q.filter(Detection.workorder_id.is_(None)
                             | ~Detection.workorder_id.in_(arch))
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
                                Feedback.invalidated == False,
                                Feedback.feedback_type != "open_alert")  # noqa: E712
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
def consolidate(category: str, note: str = "", *,
                activate: Optional[bool] = None) -> Dict:
    """固化在线学习，并记录本次快照包含的反馈集合与版本血缘。

    activate=None 时按设置中心 self_learning.auto_activate（默认 false＝仅生成
    候选版本，人工 A/B 后再激活）；UI-P0.4：此前 None 被当作激活，配置项失效。
    """
    if activate is None:
        from ..core.config import get_settings
        activate = bool(get_settings().get("self_learning", "auto_activate", False))
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
        new_v = engine.consolidate(category, note=note or "consolidate via API",
                                   activate=activate)
    except RuntimeError as e:
        raise ValueError(f"品类 '{category}' 未准备，无法巩固（{e}）")
    with session_scope() as s:
        batch = ConsolidationBatch(
            category=category, from_version=before, to_version=new_v,
            feedback_count=len(feedback_ids), note=note,
            status="activated" if activate else "candidate")
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
            "activated": activate,
            "message": ("巩固落盘完成，已激活为新版本" if activate
                         else "候选版本已生成，等待独立验收和批准激活")}


def build_feedback_candidate(feedback_id: int, parent_version: int) -> Dict:
    """把已落库且未消费的事件显式应用于独立候选，不置生产consumed。"""
    with session_scope() as s:
        fb = s.get(Feedback, feedback_id)
        if (fb is None or fb.invalidated or fb.consumed
                or not (fb.learning_trace or {}).get("candidate_only")):
            raise ValueError("仅显式候选路径的未消费有效反馈可构建候选")
        det = s.get(Detection, fb.detection_id)
        category, path = det.category, det.image_path
        verdict, label = map_verdict_label(fb.feedback_type, fb.operator_label)
        region, _ = region_box_pre(fb.region)
        trace = dict(fb.learning_trace or {})
        review = (det.n_tiles or {}).get("region_review") or {}
        trace.update(
            learning_status="candidate_pending",
            candidate_parent=parent_version,
            candidate_guard={
                "feedback_id": fb.id,
                "detection_id": det.id,
                "invalidated": bool(fb.invalidated),
                "region_revision": int(review.get("revision", 0)),
                # AC-063 region 级消费快照：发布前逐框复核修订/删除/作废（fail-closed）
                "regions": consumable_regions(review),
            })
        fb.learning_trace = trace
    box = _region_to_norm_box(path, region) if region and label == 1 else None
    try:
        out = get_engine().learn_candidate(
            category, path, verdict=verdict, label=label, box=box,
            event_id=f"feedback:{feedback_id}", parent_version=parent_version)
    except Exception as exc:
        from .candidate_release import failure_details
        failure = failure_details(exc, "build")
        with session_scope() as s:
            fb = s.get(Feedback, feedback_id)
            if fb is not None:
                fb.learning_trace = {**trace, "learning_status": "candidate_failed",
                                     "reason": str(exc), **failure}
        raise
    consumption = _record_region_consumption(feedback_id, trace["candidate_guard"], out)
    with session_scope() as s:
        fb = s.get(Feedback, feedback_id)
        fb.learning_trace = {**trace, "learning_status": "candidate", "candidate": out,
                             "region_consumption": consumption}
    return out


def _record_region_consumption(feedback_id: int, guard: Dict, out: Dict,
                               candidate_id: Optional[str] = None,
                               kind: str = "candidate") -> Dict:
    """候选训练/即时学习成功后把 region 级消费记录回写 region_review.consumption（CAS，不改
    revision）。CAS 冲突仅记状态不抛错：fail-closed 以 candidate_guard.regions 为准。"""
    snap = guard.get("regions") or []
    if not snap:
        return {"status": "none", "count": 0}
    out = out or {}
    now = datetime.now().isoformat()
    recs = [{**r, "review_revision": guard.get("region_revision"),
             "candidate_id": candidate_id or f"candidate:feedback:{feedback_id}",
             "feedback_id": feedback_id, "version": out.get("version_num"),
             "parent_version": out.get("parent_version"), "kind": kind,
             "at": now} for r in snap]
    for _ in range(3):
        with session_scope() as s:
            det = s.get(Detection, guard["detection_id"])
            if det is None:
                return {"status": "detection_missing", "count": 0}
            old = det.n_tiles or {}
            rv = old.get("region_review") or {}
            updated = {**old, "region_review": record_consumption(rv, recs)}
            changed = (s.query(Detection).filter(Detection.id == det.id,
                       Detection.n_tiles == old)
                       .update({Detection.n_tiles: updated}, synchronize_session=False))
            if changed == 1:
                return {"status": "recorded", "count": len(recs), "at": now}
    logger.warning("region consumption CAS conflict: feedback=%s", feedback_id)
    return {"status": "cas_conflict", "count": 0}


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


def activate_model(model_id: int, *, allow_stale: bool = False) -> Dict:
    """按 Model 行的 (category, version) 调 engine.activate 热切换，并记录前后版本。"""
    category, version = resolve_model_version(model_id)
    engine = get_engine()
    before = engine.current_version(category)

    # 版本不存在 / 特征表示过期（AC-015）时抛 ValueError
    published = engine.activate(category, version, allow_stale=allow_stale)

    if not isinstance(published, dict) or "model_mirror" not in published:
        with session_scope() as s:
            s.query(Model).filter(Model.category == category,
                                  Model.id != model_id).update(
                {"is_active": False}, synchronize_session=False)
            s.query(Model).filter(Model.id == model_id).update(
                {"is_active": True}, synchronize_session=False)
    else:
        model_id = published["model_mirror"]["model_id"]

    trace = {"model_id": model_id, "category": category,
             "from_version": before, "to_version": version,
             "feedback_ids": [], "reason": "manual_activate"}
    log_action("activate_model", f"model_id={model_id} category={category} "
                                 f"from=v{before} version=v{version}",
               extra=trace)
    return {"ok": True, "activated": True, "model_id": model_id,
            "category": category, "from_version": before,
            "version": f"v{version}", "trace": trace}
