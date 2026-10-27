"""检测接口（2026-08-29 起实时监控由产线流订阅取代，WS 监控会话已移除）。"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image as PILImage
from sqlalchemy import func

from ..core.tasks import task_manager
from ..db.database import session_scope
from ..db.models import Detection, Image, Video
from ..pipeline.service import get_detection_service
from .schemas import DetectImageRequest, DetectVideoRequest, to_dict

router = APIRouter()

# 2026-08-30 并发修复：原全局 _detect_lock 已移除。并发安全改由引擎层
# 品类级锁保证（DetectionEngine._pipe_locks，见 engine/__init__.py）——
# 旧全局锁只护本文件的 REST 同步接口，产线 daemon / RTSP 线程直接调
# svc.detect_image 完全绕过；且全局锁把不同品类的推理也串行化。
# 现在同品类 predict/feedback 在引擎内串行，不同品类并行不互斥。


# ── 同步检测 ─────────────────────────────────────────────────
@router.post("/detect/image")
def api_detect_image(req: DetectImageRequest):
    """同步图片检测：image_id（从 DB 取路径）与 path 二选一。"""
    try:
        kw = {}
        if req.pair_template is not None:
            if not Path(req.pair_template).is_file():
                raise ValueError(f"模板文件不存在: {req.pair_template}")
            kw["pair_template"] = req.pair_template
        svc = get_detection_service()
        # 并发安全由引擎层品类级锁保证（见文件头注释），此处不再加全局锁
        if req.image_id is not None:
            with session_scope() as s:
                img = s.get(Image, req.image_id)
                if img is None:
                    raise ValueError(f"图片不存在: image_id={req.image_id}")
                path = img.path
            out = svc.detect_image(path, req.category, image_id=req.image_id,
                                   target_type="image",
                                   with_heatmap=req.with_heatmap, **kw)
            out["image_id"] = req.image_id
        elif req.path:
            if not Path(req.path).is_file():
                raise ValueError(f"文件不存在: {req.path}")
            out = svc.detect_image(req.path, req.category,
                                   with_heatmap=req.with_heatmap, **kw)
        else:
            raise ValueError("必须提供 image_id 或 path")
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检测失败: {e}")


@router.post("/detect/upload")
async def api_detect_upload(file: UploadFile = File(...),
                            category: str = Form("default"),
                            pair_template: Optional[str] = Form(None)):
    """multipart 上传图片并同步检测（target_type=upload）。"""
    try:
        kw = {}
        if pair_template is not None:
            if not Path(pair_template).is_file():
                raise ValueError(f"模板文件不存在: {pair_template}")
            kw["pair_template"] = pair_template
        data = await file.read()
        pil = PILImage.open(io.BytesIO(data))
        return get_detection_service().detect_pil(pil, category, raw=data, **kw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检测失败: {e}")


@router.post("/detect/frame")
async def api_detect_frame(file: UploadFile = File(...),
                           category: str = Form("default"),
                           with_heatmap: bool = Form(False),
                           persist: bool = Form(True),
                           pair_template: Optional[str] = Form(None)):
    """零拷贝内存帧检测（产线"相机取图直入内存"形态，不足清单 #5 补强）。

    与 /detect/upload（先 PNG 落盘再读回）相对：帧 bytes 直接 imdecode
    进引擎，全程不落盘。相机 SDK/上位机拿到帧缓冲后直接送检用此接口。
    响应带 zero_copy=True 与 latency_e2e_ms（内存直入口径全链路耗时）。
    """
    try:
        kw = {}
        if pair_template is not None:
            if not Path(pair_template).is_file():
                raise ValueError(f"模板文件不存在: {pair_template}")
            kw["pair_template"] = pair_template
        data = await file.read()
        return get_detection_service().detect_frame(
            data, category, persist=persist, with_heatmap=with_heatmap, **kw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"检测失败: {e}")


@router.post("/detect/video")
def api_detect_video(req: DetectVideoRequest):
    """后台任务：视频抽帧逐帧检测并汇总。"""
    with session_scope() as s:
        video = s.get(Video, req.video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="视频不存在")
        video_path = video.path

    def fn(progress_cb):
        svc = get_detection_service()
        n_sampled, n_anomaly, ids = 0, 0, []
        latency_sum = 0.0
        for det in svc.iter_video_detections(video_path, req.category,
                                             interval=req.interval,
                                             video_id=req.video_id):
            n_sampled += 1
            n_anomaly += int(det["is_anomaly"])
            latency_sum += det["latency_ms"]
            if det.get("detection_id"):
                ids.append(det["detection_id"])
            # 总帧数未知（total=-1），仅汇报进度消息
            progress_cb(n_sampled, -1, f"已检测 {n_sampled} 帧")
        return {"n_frames": n_sampled, "n_anomaly": n_anomaly,
                "mean_latency_ms": (latency_sum / n_sampled if n_sampled else None),
                "ids": ids}

    task_id = task_manager.submit("detect_video", req.model_dump(), fn)
    return {"task_id": task_id}


# ── M10b：PLC 触发拍照检测（AOI 上产线标准形态）──────────────
@router.post("/detect/trigger")
def api_detect_trigger(category: str = Form(...)):
    """PLC/光电触发：从 plc.trigger_dir 取最早一张待检图 → 同步检测 →
    移入 processed/ 子目录 → 返回判定结果（target_type=plc）。

    产线对接方式：PLC 光电信号触发相机拍照落图到 trigger_dir，
    上位机/MES 调本接口取判定（OK/NG/gray），节拍内同步返回。
    """
    from ..core.config import get_settings
    from ..db.database import log_action
    settings = get_settings()
    trigger_dir = settings.get("plc", "trigger_dir") or ""
    if not trigger_dir:
        raise HTTPException(status_code=400,
                            detail="plc.trigger_dir 未配置（configs/default.yaml plc 段）")
    tdir = Path(trigger_dir)
    if not tdir.is_dir():
        raise HTTPException(status_code=400, detail=f"触发目录不存在: {trigger_dir}")
    exts = {e.lower() for e in (settings.section("plc").get("exts")
                                or [".png", ".jpg", ".jpeg", ".bmp"])}
    # FIFO：取最早落盘的待检图（跳过 processed 子目录）
    pending = sorted((p for p in tdir.iterdir()
                      if p.is_file() and p.suffix.lower() in exts),
                     key=lambda p: p.stat().st_mtime)
    if not pending:
        raise HTTPException(status_code=404, detail="触发目录无待检图")
    img = pending[0]
    try:
        svc = get_detection_service()
        out = svc.detect_image(img, category, target_type="plc",
                               with_heatmap=True)
        # 移入 processed/（保留证据，防重复检测）
        done_dir = tdir / "processed"
        done_dir.mkdir(exist_ok=True)
        try:
            img.rename(done_dir / img.name)
        except OSError:
            pass
        log_action("plc_trigger",
                   f"PLC 触发检测 {category} {img.name} → {out.get('decision')}",
                   extra={"category": category, "image": img.name,
                          "decision": out.get("decision"),
                          "score": out.get("final_score")})
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"PLC 触发检测失败: {e}")


# ── 检测记录查询 ─────────────────────────────────────────────
@router.get("/detections")
def api_list_detections(category: Optional[str] = None,
                        is_anomaly: Optional[bool] = None,
                        target_type: Optional[str] = None,
                        page: int = 1, page_size: int = 50):
    with session_scope() as s:
        q = s.query(Detection)
        if category:
            q = q.filter(Detection.category == category)
        if is_anomaly is not None:
            q = q.filter(Detection.is_anomaly == is_anomaly)
        if target_type:
            q = q.filter(Detection.target_type == target_type)
        total = q.count()
        rows = (q.order_by(Detection.id.desc())
                 .offset((page - 1) * page_size).limit(page_size).all())
        return {"total": total, "items": [to_dict(r) for r in rows]}


# ── M7a：错检集（须在 /detections/{detection_id} 之前注册，否则
#    "misjudged" 会被路径参数捕获报 422）────────────────────────
_MISJUDGED_TYPES = ("false_positive", "false_negative", "new_defect")


def _misjudged_rows(category: Optional[str]):
    """错检反馈行（按反馈时间倒序）：(Feedback, Detection) 列表。"""
    from ..db.models import Feedback
    with session_scope() as s:
        rows = (s.query(Feedback, Detection)
                .join(Detection, Feedback.detection_id == Detection.id)
                .filter(Feedback.feedback_type.in_(_MISJUDGED_TYPES),
                        Feedback.invalidated == False))  # noqa: E712
        if category:
            rows = rows.filter(Detection.category == category)
        return rows.order_by(Feedback.created_at.desc(),
                             Feedback.id.desc()).all()


def _misjudged_item(fb, det) -> dict:
    from ..self_learning.service import region_box_pre
    _, pre = region_box_pre(fb.region)
    return {
        "detection_id": det.id,
        "image_path": det.image_path,
        "category": det.category,
        "final_score": det.final_score,
        "decision": (det.n_tiles or {}).get("decision"),
        "created_at": det.created_at.isoformat() if det.created_at else None,
        "feedback_type": fb.feedback_type,
        "operator_label": fb.operator_label,
        "pre_score": (pre or {}).get("score"),
        "feedback_time": fb.created_at.isoformat() if fb.created_at else None,
    }


@router.get("/detections/misjudged")
def api_misjudged(category: Optional[str] = None, page: int = 1,
                  page_size: int = 50, format: Optional[str] = None):
    """错检集：有被判错反馈（误检/漏检/新缺陷）的检测记录。

    format=csv 时导出当前页 CSV（列：detection_id,image_path,category,
    score,decision,feedback_type,operator_label,pre_score,feedback_time）。
    post 分数不在此逐行重打分（留给 flip_curve / trace）。
    """
    rows = _misjudged_rows(category)
    items = [_misjudged_item(fb, det) for fb, det in rows]
    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start:start + page_size]

    if format == "csv":
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["detection_id", "image_path", "category", "score",
                    "decision", "feedback_type", "operator_label",
                    "pre_score", "feedback_time"])
        for it in page_items:
            w.writerow([it["detection_id"], it["image_path"], it["category"],
                        it["final_score"], it["decision"], it["feedback_type"],
                        it["operator_label"], it["pre_score"],
                        it["feedback_time"]])
        return Response(content=buf.getvalue(), media_type="text/csv")
    return {"total": total, "page": page, "page_size": page_size,
            "items": page_items}


# ── 产线实时检测流（监控页轮询订阅，须在 /detections/{id} 之前注册）────
_LIVE_TARGET_TYPES = ("pipeline", "stream", "plc")


@router.get("/detections/live")
def api_detections_live(after_id: int = 0, limit: int = 50,
                        category: Optional[str] = None,
                        workorder_id: Optional[int] = None):
    """产线检测流增量拉取：仅产线路径（pipeline/stream/plc）落库的记录。

    after_id=0 返回最新 limit 条（升序），>0 返回 id 大于该值的增量（升序）。
    workorder_id 给定则按工单精确归属过滤（Detection.workorder_id；
    历史空值行回退图像批次 join）。响应附带涉及品类的连续异常滑窗状态
    （AlarmMonitor.status，只读不改窗）。
    """
    limit = max(1, min(int(limit), 200))
    with session_scope() as s:
        q = s.query(Detection).filter(
            Detection.target_type.in_(_LIVE_TARGET_TYPES))
        from ._wo_shared import archived_workorder_ids
        arch = archived_workorder_ids(s)
        if workorder_id is None and arch:
            q = q.filter(Detection.workorder_id.is_(None)
                         | ~Detection.workorder_id.in_(arch))
        if workorder_id is not None:
            from ..db.models import Dataset, Image as ImageRow, WorkOrderSource
            ds_ids = [r[0] for r in s.query(WorkOrderSource.datasource_id)
                      .filter(WorkOrderSource.workorder_id == workorder_id).all()]
            cond = Detection.workorder_id == workorder_id
            if ds_ids:
                legacy = (s.query(Detection.id)
                          .join(ImageRow, Detection.image_id == ImageRow.id)
                          .join(Dataset, ImageRow.dataset_id == Dataset.id)
                          .filter(Dataset.datasource_id.in_(ds_ids)))
                cond = cond | (Detection.workorder_id.is_(None)
                               & Detection.id.in_(legacy))
            q = q.filter(cond)
        if category:
            q = q.filter(Detection.category == category)
        if after_id > 0:
            rows = (q.filter(Detection.id > after_id)
                    .order_by(Detection.id.asc()).limit(limit).all())
        else:
            rows = q.order_by(Detection.id.desc()).limit(limit).all()
            rows.reverse()
        items = [to_dict(r) for r in rows]
        first_id = int(items[0]["id"]) if items else None
        latest_id = int(items[-1]["id"]) if items else None
        # AC-058：ID 洞≠丢帧。区间内被过滤（非产线类型/其他品类/工单）的
        # 记录不算缺口；只有全表也不存在的 ID（删除/未落库）才计 gap。
        filtered_out = dropped_count = 0
        if after_id > 0 and first_id is not None and first_id > after_id + 1:
            exist = (s.query(func.count(Detection.id))
                     .filter(Detection.id > after_id, Detection.id < first_id)
                     .scalar() or 0)
            filtered_out = int(exist)
            dropped_count = max(0, first_id - after_id - 1 - filtered_out)
        gap = dropped_count > 0
        # 游标超过全表最大 ID（库重建/重置）→ 客户端须从 0 重新订阅
        reset_required = False
        if after_id > 0 and not items:
            max_id = s.query(func.max(Detection.id)).scalar()
            reset_required = max_id is None or int(max_id) < after_id
        next_after_id = (latest_id if latest_id is not None
                         else (0 if reset_required else after_id))
        has_more = False
        if latest_id is not None:
            has_more = q.filter(Detection.id > latest_id).first() is not None
    alarm = None
    cats = {str(it.get("category")) for it in items if it.get("category")}
    if cats:
        from ..core.alarm import get_alarm_monitor
        mon = get_alarm_monitor()
        alarm = {c: mon.status(c) for c in cats}
    return {"items": items, "alarm": alarm,
            "next_after_id": next_after_id, "latest_id": latest_id,
            "first_id": first_id, "gap": gap,
            "dropped_count": dropped_count, "filtered_out": filtered_out,
            "reset_required": reset_required,
            "has_more": has_more}


@router.get("/detections/{detection_id}")
def api_get_detection(detection_id: int):
    with session_scope() as s:
        row = s.get(Detection, detection_id)
        if row is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")
        return to_dict(row)


# ── AC-014：单帧解释（实际槽/raw/cal/生效权重/决策路径/逐框证据/trad 测量）────
@router.get("/detections/{detection_id}/explain")
def api_detection_explain(detection_id: int):
    """检测时口径取自记录（不重算融合）；trad 测量解释按需计算（当前激活版本）。
    不含性能增益/融合贡献；分数与 CDF 均非缺陷概率。"""
    from ..engine import get_engine

    with session_scope() as s:
        det = s.get(Detection, detection_id)
        if det is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")
        d = to_dict(det)
    d["score"] = d.get("final_score")
    keys = ("detection_id", "category", "image_path", "decision", "score", "model_version",
            "threshold", "gray_threshold", "raw_scores", "cal_scores", "weights",
            "actual_router", "triggered_slot", "decision_trace", "box_evidence")
    out = {k: d.get(k) for k in keys}
    out["score_semantics"] = "融合异常分（校准后加权），非缺陷概率"
    try:
        ex = get_engine().explain_image(d["category"], d["image_path"])
        out["trad"] = ex["trad"]
        out["trad_model_version"] = ex["model_version"]
        if ex["model_version"] != d.get("model_version"):
            out["trad"]["version_note"] = "trad 解释基于当前激活版本，与检测时版本不同"
    except Exception as e:  # noqa: BLE001 引擎未就绪/图像缺失 → 只返回记录口径
        out["trad"] = {"status": "unavailable", "reason": str(e)[:200]}
    return out


# ── M7a：单帧追溯链 ──────────────────────────────────────────
@router.get("/detections/{detection_id}/trace")
def api_detection_trace(detection_id: int):
    """单帧完整追溯链：检测全字段 + 当时模型版本 + 反馈时间线（含 pre）
    + 当前引擎对该图的重打分 post（仅当最新反馈含 pre 时）。"""
    from ..db.models import Feedback, Model
    from ..engine import get_engine
    from ..self_learning.service import region_box_pre

    with session_scope() as s:
        det = s.get(Detection, detection_id)
        if det is None:
            raise HTTPException(status_code=404, detail="检测记录不存在")
        det_dict = to_dict(det)
        image_path, category = det.image_path, det.category
        model = s.get(Model, det.model_id) if det.model_id else None
        model_dict = to_dict(model) if model is not None else None
        fbs = (s.query(Feedback)
               .filter(Feedback.detection_id == detection_id)
               .order_by(Feedback.created_at.asc(), Feedback.id.asc()).all())
        feedbacks = []
        for fb in fbs:
            box, pre = region_box_pre(fb.region)
            feedbacks.append({
                "id": fb.id, "feedback_type": fb.feedback_type,
                "operator_label": fb.operator_label,
                "defect_type": fb.defect_type, "comment": fb.comment,
                "box": box, "pre": pre,
                "created_at": fb.created_at.isoformat() if fb.created_at else None,
                "operator": fb.operator, "consumed": fb.consumed,
                "invalidated": bool(fb.invalidated),
            })

    post = None
    if feedbacks and feedbacks[-1]["pre"]:
        try:
            r = get_engine().predict_image_path(category, image_path)
            post = {"score": round(float(r.score), 6), "decision": r.decision,
                    "latency_ms": round(float(r.latency_ms), 2)}
        except Exception:  # noqa: BLE001 引擎未就绪 → post=None
            post = None
    return {"detection": det_dict, "model": model_dict,
            "feedbacks": feedbacks, "post": post}
