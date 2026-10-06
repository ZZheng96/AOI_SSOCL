"""后台任务管理器：线程池执行 + Task 表状态 + 进度回调。

用法：
    task_id = task_manager.submit("pseudo", payload, fn)
    # fn(progress_cb) -> dict 结果
    # progress_cb(done, total, message="")
"""
from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from ..db.database import session_scope
from ..db.models import Task

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, max_workers: int = 4):
        # 4 workers：prepare（模型准备）是长任务，2 线程时一个 prepare 占满
        # 一半池子，两个并发 prepare 就卡死其余全部任务（导入/评估排队无响应）
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._listeners: List[Callable[[Dict], None]] = []
        self._lock = threading.Lock()

    # ── 订阅任务事件（WebSocket 推送用）──────────────────────
    def add_listener(self, cb: Callable[[Dict], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[Dict], None]) -> None:
        """移除订阅（不存在则忽略）。"""
        with self._lock:
            try:
                self._listeners.remove(cb)
            except ValueError:
                pass

    def _emit(self, payload: Dict) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(payload)
            except Exception:
                pass

    # ── 提交与状态 ───────────────────────────────────────────
    def submit(self, task_type: str, payload: Dict,
               fn: Callable[[Callable], Dict]) -> int:
        with session_scope() as s:
            t = Task(task_type=task_type, payload=payload, status="pending")
            s.add(t)
            s.flush()
            task_id = t.id
        self._pool.submit(self._run, task_id, fn)
        return task_id

    def _update(self, task_id: int, **fields) -> None:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t is None:
                return
            for k, v in fields.items():
                setattr(t, k, v)
            snapshot = {"id": t.id, "task_type": t.task_type, "status": t.status,
                        "progress": t.progress, "message": t.message,
                        "payload": t.payload, "result": t.result}
        self._emit(snapshot)

    def _run(self, task_id: int, fn: Callable) -> None:
        self._update(task_id, status="running", progress=0.0)

        def progress_cb(done: int, total: int, message: str = "") -> None:
            prog = (done / total) if total and total > 0 else 0.0
            self._update(task_id, progress=min(prog, 1.0),
                         message=message or f"{done}/{total}")

        try:
            result = fn(progress_cb) or {}
            self._update(task_id, status="done", progress=1.0,
                         message="完成", result=result)
        except Exception as e:  # noqa: BLE001
            logger.error("Task %d failed:\n%s", task_id, traceback.format_exc())
            self._update(task_id, status="failed", message=str(e))

    def get(self, task_id: int) -> Optional[Dict]:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t is None:
                return None
            return {"id": t.id, "task_type": t.task_type, "status": t.status,
                    "progress": t.progress, "message": t.message,
                    "payload": t.payload, "result": t.result}

    def list_recent(self, limit: int = 20) -> List[Dict]:
        with session_scope() as s:
            rows = s.query(Task).order_by(Task.id.desc()).limit(limit).all()
            return [{"id": t.id, "task_type": t.task_type, "status": t.status,
                     "progress": t.progress, "message": t.message,
                     "created_at": t.created_at.isoformat()} for t in rows]


task_manager = TaskManager()
