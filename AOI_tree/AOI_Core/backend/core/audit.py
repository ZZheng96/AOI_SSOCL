"""请求/后台任务共享的轻量审计上下文，不依赖认证 Web 栈。"""
from __future__ import annotations

import contextvars

# anyio 任务树传播上下文；直接 spawn 的 worker 线程使用 operator 默认值。
current_actor: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aoi_current_actor", default="operator")


def log_action(action: str, detail: str = "", user: str | None = None,
               extra: dict | None = None) -> None:
    """使用共享操作者上下文，仍由原数据库函数同步提交审计记录。"""
    from ..db.database import log_action as persist_action
    if user is None:
        user = current_actor.get()
    return persist_action(action, detail, user=user, extra=extra)
