"""WebSocket 推送：后台任务进度。

线程模型：
  - WS /ws/tasks 订阅 task_manager 监听器；监听器在任务工作线程被调，
    经 loop.call_soon_threadsafe 桥接到本连接的 asyncio.Queue

2026-08-29：WS 实时监控会话（/ws/monitor/{id}）已移除——页面私启的检测
会话是脱离工单/队列的影子产线，实时监控改由监控页轮询
/api/detections/live 订阅产线检测流（详见设计文档 §6.29）。
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.security import _resolve_role
from ..core.tasks import task_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ── WebSocket：任务进度 ─────────────────────────────────────
@router.websocket("/ws/tasks")
async def ws_tasks(websocket: WebSocket):
    role = _resolve_role(websocket.headers.get("X-API-Key", ""))
    if role is None:
        await websocket.close(code=4401, reason="未授权")
        return
    await websocket.accept()
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()

    def listener(payload: dict) -> None:
        # listener 由任务工作线程调用，桥接到本连接的事件循环
        try:
            loop.call_soon_threadsafe(aq.put_nowait, payload)
        except Exception:  # noqa: BLE001 连接已关闭则忽略
            pass

    task_manager.add_listener(listener)
    try:
        while True:
            msg = await aq.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        task_manager.remove_listener(listener)
