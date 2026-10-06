"""异步检测任务 + SSE 流式推送（P1：借鉴 pcbdetect submitFov 快返回 + 流式结果）。

- POST /api/detect/tasks      提交异步检测任务 → task_id（快返回）
- GET  /api/detect/tasks/{id} 查询任务状态/结果
- GET  /api/detect/stream     SSE 长连接订阅检测事件（新订户先收最近快照）
- POST /api/detect/recheck    人工复检（高优先级插队重检）
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.task_queue import PRIO_RECHECK, PRIO_TRIGGER, get_task_queue
from app.db.database import session_scope
from app.db.models import Task

router = APIRouter()


@router.post("/detect/tasks")
def create_detect_task(payload: dict) -> dict:
    """提交异步检测任务（快返回 task_id）。

    body: {template_id, image_path, task_type?=dual, priority?=batch,
           workorder_id?, inspect_id?, board_barcode?, category?}
    priority: recheck=0 / trigger=1 / batch=2
    """
    task_type = payload.get("task_type", "dual")
    image_path = payload.get("image_path")
    if not image_path or not Path(image_path).is_file():
        raise HTTPException(status_code=400, detail="需要有效的 image_path")
    if task_type in ("dual", "traditional") and not payload.get("template_id"):
        raise HTTPException(status_code=400, detail="dual/traditional 需要 template_id")
    if task_type == "feature" and not payload.get("category"):
        raise HTTPException(status_code=400, detail="feature 需要 category")
    prio = {"recheck": PRIO_RECHECK, "trigger": PRIO_TRIGGER,
            "batch": 2}.get(str(payload.get("priority", "batch")), 2)
    q = get_task_queue()
    task_id = q.submit(task_type, {
        "task_type": task_type,
        "image_path": str(image_path),
        "template_id": payload.get("template_id"),
        "workorder_id": payload.get("workorder_id"),
        "inspect_id": payload.get("inspect_id"),
        "board_barcode": payload.get("board_barcode"),
        "category": payload.get("category"),
        "target_type": payload.get("target_type", "task"),
    }, priority=prio)
    return {"task_id": task_id, "status": "pending"}


@router.get("/detect/tasks/{task_id}")
def get_detect_task(task_id: int) -> dict:
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        return {"id": t.id, "task_type": t.task_type, "status": t.status,
                "progress": t.progress, "message": t.message,
                "result": t.result, "created_at": t.created_at.isoformat()
                if t.created_at else None}


@router.get("/detect/tasks")
def list_detect_tasks(status: str | None = None,
                      task_type: str | None = None,
                      limit: int = 50) -> dict:
    with session_scope() as s:
        q = s.query(Task).order_by(Task.id.desc())
        if status:
            q = q.filter(Task.status == status)
        if task_type:
            q = q.filter(Task.task_type == task_type)
        rows = q.limit(limit).all()
        return {"items": [{"id": t.id, "task_type": t.task_type, "status": t.status,
                           "message": t.message} for t in rows]}


@router.post("/detect/recheck")
def recheck(payload: dict) -> dict:
    """人工复检：对指定检测记录对应的图重新双检（高优先级插队）。

    body: {detection_id}
    """
    detection_id = payload.get("detection_id")
    if not detection_id:
        raise HTTPException(status_code=400, detail="需要 detection_id")
    from app.db.models import Detection
    with session_scope() as s:
        det = s.get(Detection, int(detection_id))
        if det is None:
            raise HTTPException(status_code=404, detail=f"检测记录不存在: {detection_id}")
        template_id = det.template_id
        image_path = det.image_path
        workorder_id = det.workorder_id
        inspect_id = det.inspect_id
    if not template_id or not image_path:
        raise HTTPException(status_code=400, detail="该记录无模板/图像，无法复检")
    q = get_task_queue()
    task_id = q.submit("dual", {
        "task_type": "dual", "template_id": template_id,
        "image_path": image_path, "workorder_id": workorder_id,
        "inspect_id": inspect_id, "target_type": "recheck",
    }, priority=PRIO_RECHECK)
    return {"task_id": task_id, "status": "queued", "priority": "recheck"}


@router.get("/detect/stream")
def detect_stream() -> StreamingResponse:
    """SSE 事件流：检测完成/任务状态实时推送。"""
    q = get_task_queue()
    sub = q.subscribe()

    def gen():
        try:
            while True:
                try:
                    ev = sub.get(timeout=30)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except Exception:  # noqa: BLE001 心跳保活
                    yield ": ping\n\n"
        finally:
            q.unsubscribe(sub)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
