"""检测任务队列 + 事件总线（P1：借鉴 pcbdetect 双队列 + subscribeDetectResults）。

- 优先级队列：PRIO_RECHECK(人工复检) > PRIO_TRIGGER(PLC/触发) > PRIO_BATCH(工单/批量)
- 常驻消费 Worker：取任务 → 执行 → 更新 Task 表 + 发布事件
- 事件总线：订阅者队列（SSE 长连接）读到最近事件（环形缓冲 100 条，新订户先发快照）
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)

PRIO_RECHECK = 0    # 人工复检插队
PRIO_TRIGGER = 1    # PLC 触发
PRIO_BATCH = 2      # 工单/批量


class TaskCancelled(Exception):
    pass


@dataclass(order=True)
class _Task:
    priority: int
    seq: int
    payload: dict = field(compare=False)


class DetectTaskQueue:
    _instance: "DetectTaskQueue | None" = None
    _lock = threading.Lock()

    def __init__(self, workers: int = 2) -> None:
        self.workers = workers
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop = False
        self._events: list[dict] = []          # 环形事件缓冲
        self._events_lock = threading.Lock()
        self._subs: list[queue.Queue] = []     # SSE 订阅者

    @classmethod
    def get(cls) -> "DetectTaskQueue":
        with cls._lock:
            if cls._instance is None:
                cls._instance = DetectTaskQueue()
            return cls._instance

    def start(self) -> None:
        if self._threads and all(t.is_alive() for t in self._threads):
            return
        self._stop = False
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, daemon=True,
                                 name=f"task-worker-{i}")
            t.start()
            self._threads.append(t)
        log.info("[taskq] 检测任务队列已启动（%d worker）", self.workers)

    def stop(self) -> None:
        self._stop = True

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    # ── 提交 ──────────────────────────────────────────────
    def submit(self, task_type: str, payload: dict,
               priority: int = PRIO_BATCH) -> int:
        """创建 Task 行并入队。返回 task_id。"""
        timeout_s = payload.get("timeout_s")
        if timeout_s is not None:
            try:
                timeout_s = float(timeout_s)
            except (TypeError, ValueError) as exc:
                raise ValueError("timeout_s 必须是非负数") from exc
            if timeout_s < 0 or not math.isfinite(timeout_s):
                raise ValueError("timeout_s 必须是有限的非负数")
            payload = {**payload, "timeout_s": timeout_s}
        from app.db.database import session_scope
        from app.db.models import Task
        from app.template.store import TemplateStore
        store = TemplateStore() if task_type in ("dual", "traditional") else None
        task_id = None
        try:
            with session_scope() as s:
                t = Task(task_type=task_type, status="pending",
                         payload=payload, progress=0.0)
                s.add(t)
                s.flush()
                task_id = t.id
                if store is not None:
                    tpl = store.load(payload.get("template_id") or "")
                    if tpl is None:
                        raise ValueError(f"模板不存在: {payload.get('template_id')}")
                    version = payload.get("template_version")
                    if version is not None and int(tpl.version) != int(version):
                        raise ValueError("模板版本已变化，请重新提交任务")
                    payload = {**payload, "template_version": tpl.version,
                               "template_snapshot": store.snapshot_for_task(tpl, task_id)}
                    t.payload = payload
        except Exception:
            if store is not None and task_id is not None:
                store.cleanup_task_snapshot(task_id)
            raise
        queued_at = time.monotonic()
        self._q.put(_Task(priority=priority, seq=self._next_seq(), payload={"task_id": task_id, "_queued_at": queued_at, **payload}))
        self.publish({"type": "task_queued", "task_id": task_id,
                      "task_type": task_type, "priority": priority})
        return task_id

    # ── 消费 ──────────────────────────────────────────────
    def _worker(self) -> None:
        while not self._stop:
            try:
                task = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._execute(task.payload)
            except TaskCancelled as exc:
                if self._update_task(task.payload.get("task_id"),
                                     status="cancelled", message=str(exc)):
                    self.publish({"type": "task_cancelled",
                                  "task_id": task.payload.get("task_id"),
                                  "message": str(exc)})
            except TimeoutError as exc:
                log.warning("[taskq] 任务超时: %s", exc)
                if self._update_task(task.payload.get("task_id"),
                                     status="timeout", message=str(exc)):
                    self.publish({"type": "task_timeout", "task_id": task.payload.get("task_id"),
                                  "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                log.warning("[taskq] 任务执行异常: %s", exc)
                if self._update_task(task.payload.get("task_id"),
                                     status="failed", message=str(exc)):
                    self.publish({"type": "task_failed",
                                  "task_id": task.payload.get("task_id"),
                                  "error": str(exc)})
                # 告警（P2）：任务失败上报
                try:
                    from app.db.database import session_scope
                    from app.db.models import AlarmRecord
                    with session_scope() as s:
                        s.add(AlarmRecord(
                            alarm_type="task_failed", level="ERROR",
                            source="task_queue",
                            content=f"任务 {task.payload.get('task_id')} 失败: {exc}",
                            ref_id=task.payload.get("task_id"),
                        ))
                except Exception:  # noqa: BLE001
                    pass
            finally:
                if task.payload.get("template_snapshot"):
                    from app.template.store import TemplateStore
                    TemplateStore().cleanup_task_snapshot(task.payload["task_id"])
                self._q.task_done()

    def _execute(self, payload: dict) -> None:
        task_id = payload.get("task_id")
        task_type = payload.get("task_type") or "dual"
        deadline_s = float(payload.get("timeout_s") or 0)
        queued_at = float(payload.get("_queued_at") or time.monotonic())
        started = time.monotonic()

        def check_cancelled() -> None:
            from app.db.database import session_scope
            from app.db.models import Task
            with session_scope() as s:
                task_row = s.get(Task, task_id)
                if task_row is None or task_row.status == "cancelled":
                    raise TaskCancelled("任务已取消")

        if deadline_s and started - queued_at > deadline_s:
            raise TimeoutError(f"任务排队超过 {deadline_s:g}s")
        def check_deadline() -> None:
            if deadline_s and time.monotonic() - queued_at > deadline_s:
                raise TimeoutError(f"任务超过 {deadline_s:g}s")
        check_cancelled()
        check_deadline()
        if not self._update_task(task_id, status="running"):
            raise TaskCancelled("任务已取消")
        check_cancelled()
        if task_type in ("dual", "traditional"):
            from app.template.model import InspectionTemplate
            from app.template.store import TemplateStore
            snapshot = payload.get("template_snapshot")
            if not snapshot:
                raise RuntimeError("任务缺少模板及标准图快照")
            TemplateStore.verify_task_snapshot(snapshot)
            tpl = InspectionTemplate.from_dict(snapshot)
            expected_version = payload.get("template_version")
            if expected_version is not None and int(tpl.version) != int(expected_version):
                raise RuntimeError(
                    f"模板版本已变化: expected={expected_version}, current={tpl.version}")
            from app.inspect.detect_service import get_detection_service
            svc = get_detection_service()
            if task_type == "dual":
                out = svc.detect_dual(
                    tpl, payload["image_path"],
                    workorder_id=payload.get("workorder_id"),
                    target_type=payload.get("target_type", "task"),
                    inspect_id=payload.get("inspect_id"),
                    board_barcode=payload.get("board_barcode"),
                )
            else:
                # 单传统（强制 traditional 模板模式）
                old_mode = tpl.engine_mode
                tpl.engine_mode = "traditional"
                try:
                    out = svc.detect_dual(tpl, payload["image_path"],
                                          workorder_id=payload.get("workorder_id"))
                finally:
                    tpl.engine_mode = old_mode
        elif task_type == "feature":
            from app.engines.feature import get_engine
            from app.inspect.fusion import feature_to_dict
            r = get_engine().predict_image_path(payload["category"], payload["image_path"])
            out = feature_to_dict(r)
        else:
            raise RuntimeError(f"未知任务类型: {task_type}")
        result = {"overall": out.get("overall"),
                  "final_status": out.get("final_status"),
                  "is_defect": out.get("is_defect"),
                  "requires_review": out.get("requires_review"),
                  "is_system_error": out.get("is_system_error"),
                  "detection_id": out.get("detection_id"),
                  "inspect_id": out.get("inspect_id"),
                  "elapsed_ms": out.get("elapsed_ms"),
                  "align": out.get("align") or {},
                  "summary": out}
        if not self._update_task(task_id, status="done", result=result, progress=1.0):
            return
        self.publish({"type": "detection_done", "task_id": task_id,
                      "overall": out.get("overall"),
                      "final_status": out.get("final_status"),
                      "is_defect": out.get("is_defect"),
                      "requires_review": out.get("requires_review"),
                      "is_system_error": out.get("is_system_error"),
                      "detection_id": out.get("detection_id"),
                      "inspect_id": out.get("inspect_id"),
                      "elapsed_ms": out.get("elapsed_ms")})
        # MES 上报（P2：启用 mes.enabled 后检测完成上报整板判定）
        try:
            from app.core.mes import get_mes_gateway
            gw = get_mes_gateway()
            if gw is not None and out.get("inspect_id"):
                gw.report_detection(out["inspect_id"])
        except Exception:  # noqa: BLE001 上报失败不影响检测
            log.warning("[taskq] MES 上报失败", exc_info=True)

    def cancel(self, task_id: int) -> bool:
        changed = self._update_task(task_id, status="cancelled", message="任务已取消")
        if changed:
            from app.template.store import TemplateStore
            TemplateStore().cleanup_task_snapshot(task_id)
            self.publish({"type": "task_cancelled", "task_id": task_id,
                          "message": "任务已取消"})
        return changed

    def _update_task(self, task_id: Optional[int], **fields) -> bool:
        if task_id is None:
            return False
        from app.db.database import session_scope
        from app.db.models import Task
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t is None:
                return False
            if t.status in ("cancelled", "timeout") and fields.get("status") in (
                    "running", "done", "failed"):
                return False
            if fields.get("status") == "cancelled" and t.status != "pending":
                return False
            for k, v in fields.items():
                setattr(t, k, v)
            t.updated_at = datetime.now()
            return True

    # ── 事件订阅（SSE） ───────────────────────────────────
    def publish(self, event: dict) -> None:
        with self._events_lock:
            self._events.append(event)
            if len(self._events) > 100:
                self._events.pop(0)
            dead = []
            for sub in self._subs:
                try:
                    sub.put_nowait(event)
                except queue.Full:
                    dead.append(sub)
            for sub in dead:
                self._subs.remove(sub)

    def subscribe(self) -> queue.Queue:
        sub: queue.Queue = queue.Queue(maxsize=200)
        with self._events_lock:
            self._subs.append(sub)
            for ev in self._events:
                try:
                    sub.put_nowait(ev)
                except queue.Full:
                    break
        return sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self._events_lock:
            if sub in self._subs:
                self._subs.remove(sub)


get_task_queue = DetectTaskQueue.get
