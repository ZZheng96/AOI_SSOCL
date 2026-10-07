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
      box?,                   # [x0,y0,x1,y1] 缺陷框（可选，与 algo 一致）
      core_detection_id?,     # 树干 AOI_Core 的 detection_id（缺省从 dual_json 取）
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

    # 从检测记录反查 category/image_path，以及树干 AOI_Core 的 detection_id
    # （本地 detection_id ≠ Core detection_id；后者随特征结果存于 dual_json）
    core_detection_id = payload.get("core_detection_id")
    if detection_id is not None:
        with session_scope() as s:
            det = s.get(Detection, int(detection_id))
            if det is None:
                raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
            category = category or det.category or ""
            image_path = image_path or det.image_path
            if core_detection_id is None and isinstance(det.dual_json, dict):
                dj = det.dual_json
                feat = dj.get("feature") if isinstance(dj.get("feature"), dict) else dj
                core_detection_id = feat.get("detection_id")
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
                                      verdict=verdict, label=label, box=box,
                                      detection_id=core_detection_id,
                                      defect_type=payload.get("defect_type"),
                                      comment=payload.get("comment"))
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", 0)
        raise HTTPException(status_code=status if status in (404, 409) else 500,
                            detail=f"特征引擎反馈失败: {exc}")

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
    return info


@router.get("/models/{category}/learning")
def learning_status(category: str) -> dict:
    """学习状态透出：树干 AOI_Core 掌管学习状态，直接透传其 insight。"""
    from app.engines.feature import get_engine
    engine = get_engine()
    try:
        ins = engine.learning_insight(category)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=getattr(exc, "status", 0) or 502, detail=str(exc))
    return {"category": category, "enabled": True, "backend": "aoi_core",
            "version": f"v{engine.current_version(category)}", "insight": ins}
