"""后台任务管理器：线程池执行 + Task 表状态 + 进度回调 + 协作式取消（A15）。

用法：
    task_id = task_manager.submit("pseudo", payload, fn)
    # fn(progress_cb, cancel_event=None) -> dict 结果
    # progress_cb(done, total, message="")
    # cancel_event: threading.Event，长任务循环内检测 .is_set() 后抛
    #   TaskCancelled 或直接返回实现"协作式取消"；不可中断的 C 调用
    #   （sklearn/torch fit）在取消点后自然终止。
"""
from __future__ import annotations

import inspect
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from ..db.database import session_scope
from ..db.models import Task

logger = logging.getLogger(__name__)


class TaskCancelled(Exception):
    """任务被用户取消（A15）。fn 内部检测到 cancel_event 后置位可抛此异常，
    或直接返回——两者都会被标记为 status=cancelled 而非 failed。"""



class TaskManager:
    def __init__(self, max_workers: int = 4):
        # 4 workers：prepare（模型准备）是长任务，2 线程时一个 prepare 占满
        # 一半池子，两个并发 prepare 就卡死其余全部任务（导入/评估排队无响应）
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._listeners: List[Callable[[Dict], None]] = []
        self._lock = threading.Lock()
        self._cancel_flags: Dict[int, threading.Event] = {}  # task_id -> Event

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
        with self._lock:
            self._cancel_flags[task_id] = threading.Event()
        self._pool.submit(self._run, task_id, fn)
        return task_id

    def cancel(self, task_id: int) -> bool:
        """请求取消：置位取消标志并把仍在 pending 的任务直接标记 cancelled。
        正在运行的任务由 fn 协作响应（检测 cancel_event）。返回是否受理。"""
        with self._lock:
            ev = self._cancel_flags.get(task_id)
        if ev is None:
            return False                      # 任务不存在或已结束清理
        ev.set()
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t is None:
                return False
            if t.status == "pending":
                t.status = "cancelled"
                t.message = "已取消（未启动）"
                snapshot = {"id": t.id, "task_type": t.task_type,
                            "status": t.status, "progress": t.progress,
                            "message": t.message, "payload": t.payload,
                            "result": t.result}
                self._emit(snapshot)
        return True

    def is_cancelled(self, task_id: int) -> bool:
        with self._lock:
            ev = self._cancel_flags.get(task_id)
        return bool(ev and ev.is_set())

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
        with self._lock:
            cancel_ev = self._cancel_flags.get(task_id)
        # pending 即被取消：不再启动
        if cancel_ev is not None and cancel_ev.is_set():
            self._update(task_id, status="cancelled", message="已取消（未启动）")
            self._cleanup_flag(task_id)
            return
        self._update(task_id, status="running", progress=0.0)

        def progress_cb(done: int, total: int, message: str = "") -> None:
            prog = (done / total) if total and total > 0 else 0.0
            self._update(task_id, progress=min(prog, 1.0),
                         message=message or f"{done}/{total}")

        # 兼容旧签名 fn(progress_cb)；新签名 fn(progress_cb, cancel_event)
        try:
            n_params = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            n_params = 1
        try:
            if n_params >= 2:
                result = fn(progress_cb, cancel_ev) or {}
            else:
                result = fn(progress_cb) or {}
            if cancel_ev is not None and cancel_ev.is_set():
                self._update(task_id, status="cancelled", message="已取消")
            else:
                self._update(task_id, status="done", progress=1.0,
                             message="完成", result=result)
        except TaskCancelled as e:
            self._update(task_id, status="cancelled",
                         message=str(e) or "已取消")
        except Exception as e:  # noqa: BLE001
            logger.error("Task %d failed:\n%s", task_id, traceback.format_exc())
            self._update(task_id, status="failed", message=str(e))
        finally:
            self._cleanup_flag(task_id)

    def _cleanup_flag(self, task_id: int) -> None:
        with self._lock:
            self._cancel_flags.pop(task_id, None)

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
                     "result": t.result,
                     "created_at": t.created_at.isoformat()} for t in rows]


task_manager = TaskManager()
