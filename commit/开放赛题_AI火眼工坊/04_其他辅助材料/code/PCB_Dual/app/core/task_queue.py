"""检测任务队列 + 事件总线（P1：借鉴 pcbdetect 双队列 + subscribeDetectResults）。

- 优先级队列：PRIO_RECHECK(人工复检) > PRIO_TRIGGER(PLC/触发) > PRIO_BATCH(工单/批量)
- 常驻消费 Worker：取任务 → 执行 → 更新 Task 表 + 发布事件
- 事件总线：订阅者队列（SSE 长连接）读到最近事件（环形缓冲 100 条，新订户先发快照）
"""
from __future__ import annotations

import logging
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


@dataclass(order=True)
class _Task:
    priority: int
    seq: int = field(default=0, compare=True)
    payload: dict = field(default_factory=dict, compare=False)


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
        payload = dict(payload)
        payload["task_type"] = task_type
        if "deadline_at" not in payload and payload.get("deadline_s") is not None:
            payload["deadline_at"] = time.monotonic() + max(0.0, float(payload["deadline_s"]))
        from app.db.database import session_scope
        from app.db.models import Task
        with session_scope() as s:
            t = Task(task_type=task_type, status="pending",
                     payload=payload, progress=0.0)
            s.add(t)
            s.flush()
            task_id = t.id
        self._q.put(_Task(priority=priority, seq=self._next_seq(),
                          payload={"task_id": task_id, **payload}))
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
            except Exception as exc:  # noqa: BLE001
                log.warning("[taskq] 任务执行异常: %s", exc)
                self._update_task(task.payload.get("task_id"),
                                  status="failed", message=str(exc))
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
                self._q.task_done()

    def _execute(self, payload: dict) -> None:
        task_id = payload.get("task_id")
        task_type = payload.get("task_type") or "dual"
        deadline_at = payload.get("deadline_at")
        if deadline_at is not None and time.monotonic() >= float(deadline_at):
            message = "任务在开始执行前已超过 deadline；未启动引擎"
            self._update_task(task_id, status="timeout", message=message)
            self.publish({"type": "task_timeout", "task_id": task_id, "message": message})
            return
        self._update_task(task_id, status="running")
        from app.inspect.detect_service import get_detection_service
        svc = get_detection_service()
        if task_type in ("dual", "traditional"):
            from app.template.store import TemplateStore
            tpl = TemplateStore().load(payload["template_id"])
            if tpl is None:
                raise RuntimeError(f"模板不存在: {payload['template_id']}")
            if tpl.status != "published":
                raise RuntimeError(f"模板未发布: {tpl.id}")
            frozen_version = payload.get("template_version")
            if frozen_version is not None and int(frozen_version) != int(tpl.version):
                raise RuntimeError(f"模板版本已变化: 请求 v{frozen_version}，当前 v{tpl.version}")
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
                  "detection_id": out.get("detection_id"),
                  "inspect_id": out.get("inspect_id"),
                  "elapsed_ms": out.get("elapsed_ms"),
                  "align": out.get("align") or {},
                  "summary": out}
        self._update_task(task_id, status="done", result=result, progress=1.0)
        self.publish({"type": "detection_done", "task_id": task_id,
                      "overall": out.get("overall"),
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

    def _update_task(self, task_id: Optional[int], **fields) -> None:
        if task_id is None:
            return
        from app.db.database import session_scope
        from app.db.models import Task
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t is None:
                return
            for k, v in fields.items():
                setattr(t, k, v)
            t.updated_at = datetime.now()

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
