"""反馈/学习 API：复判反馈即学 + 学习状态透出。"""
from __future__ import annotations

from pathlib import Path
import hashlib

from fastapi import APIRouter, HTTPException

from app.db.database import session_scope
from app.db.models import Detection, Feedback

router = APIRouter()


@router.post("/feedback")
def submit_feedback(payload: dict) -> dict:
    """复判/反馈即学。

    body: {
      detection_id?,          # 有检测记录时填（自动取 category/image_path）
      image_path?,            # 无检测记录时直接给图路径
      category?,
      verdict,                # correct(判对) / wrong(判错) / none(无标签)
      label?,                 # 0=正常 1=缺陷（verdict=wrong 时必填）
      box?,                   # [x,y,w,h] 缺陷框（可选）
      feedback_type?,         # false_positive/false_negative/confirmed/new_defect（落库用）
      operator?, comment?
    }
    """
    from app.engines.feature import get_engine

    verdict = payload.get("verdict")
    if verdict not in ("correct", "wrong", "none"):
        raise HTTPException(status_code=400, detail="verdict 须为 correct/wrong/none")

    detection_id = payload.get("detection_id")
    image_path = payload.get("image_path")
    category = payload.get("category")

    # 从检测记录反查 category/image_path
    if detection_id is not None:
        with session_scope() as s:
            det = s.get(Detection, int(detection_id))
            if det is None:
                raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
            category = category or det.category or ""
            image_path = image_path or det.image_path
    if not image_path:
        raise HTTPException(status_code=400, detail="需要 image_path 或 detection_id")
    if not Path(image_path).is_file():
        raise HTTPException(status_code=400, detail=f"图像不存在: {image_path}")

    label = payload.get("label")
    if verdict == "wrong" and label not in (0, 1):
        raise HTTPException(status_code=400, detail="verdict=wrong 时 label 须为 0 或 1")
    box = payload.get("box")

    try:
        engine = get_engine()
        info = engine.submit_feedback(category or "default", str(image_path),
                                      verdict=verdict, label=label, box=box)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"特征引擎反馈失败: {exc}")

    # 反馈落库（作统计/追溯）
    feedback_type = payload.get("feedback_type")
    if feedback_type is None:
        if verdict == "wrong":
            feedback_type = "false_negative" if label == 1 else "false_positive"
        elif verdict == "correct":
            feedback_type = "confirmed"
        else:
            feedback_type = "none"
    fb_id = None
    source_event_id = hashlib.sha256(
        f"PCB_Dual:{detection_id}:{verdict}:{label}:{box}".encode("utf-8")
    ).hexdigest()
    if detection_id is not None:
        with session_scope() as s:
            if s.get(Detection, int(detection_id)) is None:
                raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
            existing = s.query(Feedback).filter(
                Feedback.source_event_id == source_event_id
            ).first()
            if existing is not None:
                fb_id = existing.id
            else:
                fb = Feedback(
                    detection_id=int(detection_id),
                    feedback_type=feedback_type,
                    operator_label=int(label) if label is not None else 0,
                    defect_type=payload.get("defect_type"),
                    comment=payload.get("comment"),
                    region={"box": box} if box else None,
                    operator=payload.get("operator", "operator"),
                    source_event_id=source_event_id,
                )
                s.add(fb)
                s.flush()
                fb_id = fb.id
    info.update({"feedback_id": fb_id, "feedback_type": feedback_type,
                 "source_event_id": source_event_id})

    from app.config import get_settings
    sync_url = str(get_settings().get("feedback_sync", "url", "") or "").strip()
    if detection_id is not None and sync_url and label in (0, 1) and fb_id is not None:
        with session_scope() as s:
            fb = s.get(Feedback, int(fb_id))
            if fb is not None:
                fb.mainline_sync_status = "pending"
        from app.core.feedback_sync import sync_feedback
        info["mainline_sync"] = sync_feedback(int(fb_id))
    return info


@router.get("/models/{category}/learning")
def learning_status(category: str) -> dict:
    """学习状态透出：双库规模 / 孵育头 / 权重门控统计 / 主动选样队列。"""
    from app.engines.feature import get_engine
    engine = get_engine()
    try:
        pipe = engine.get_pipeline(category)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"品类未准备: {exc}")
    h = getattr(pipe, "handler", None)
    out: dict = {
        "category": category,
        "enabled": h is not None,
        "version": f"v{engine.current_version(category)}",
        "train_auroc": getattr(pipe, "train_auroc", None),
    }
    if h is None:
        return out
    nb = h.normal_bank
    out["normal_bank"] = {
        "core_patches": int(nb.core.shape[0]) if getattr(nb, "core", None) is not None else 0,
        "ext_samples": len(getattr(nb, "ext", []) or []),
        "fuse_blocked": bool(getattr(nb, "fuse_blocked", False)),
    }
    db = h.defect_bank
    out["defect_bank"] = {"samples": len(getattr(db, "samples", []) or [])}
    hm = getattr(pipe, "head_mgr", None)
    out["incubate"] = {
        "trained": bool(hm is not None and hm.head is not None),
        "n_pos": int(hm.head.n_pos) if (hm and hm.head) else 0,
        "log_tail": list(hm.log[-3:]) if hm else [],
    }
    sel = getattr(pipe, "selector", None)
    out["active_queue"] = len(getattr(sel, "queue", []) or []) if sel else 0
    st = getattr(h, "stats", {}) or {}
    out["weight_gate"] = {"apply": int(st.get("weight_apply", 0)),
                          "reject": int(st.get("weight_reject", 0))}
    return out
