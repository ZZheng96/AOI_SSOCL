"""webhook 异步推送（M5a：MES/PLC 对接）。

notify_event(event, payload)：按 configs/default.yaml 的 notify 段配置
异步 POST JSON（单线程池，不阻塞调用方）；webhook_url 为空即禁用；
超时/连接失败等异常静默捕获并记 OperationLog("webhook_error")。

事件：anomaly / gray / feedback / consolidate。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from .config import get_settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="webhook")


def _post(url: str, body: Dict[str, Any], timeout: float, event: str) -> None:
    import requests
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}")
    except Exception as e:  # noqa: BLE001 推送失败静默，不反噬业务路径
        logger.warning("webhook 推送失败 event=%s url=%s: %s", event, url, e)
        try:
            from ..db.database import log_action
            log_action("webhook_error", f"event={event} url={url} err={e}")
        except Exception:  # noqa: BLE001 记日志失败也不抛出
            pass


def notify_event(event: str, payload: Dict[str, Any]) -> None:
    """异步推送事件。未配置 webhook_url 或事件未订阅时直接返回。"""
    try:
        cfg = get_settings().get_notify()
    except Exception:  # noqa: BLE001 配置异常不阻断业务
        return
    url = cfg["webhook_url"]
    if not url or event not in cfg["events"]:
        return
    body = {"event": event, "ts": time.time(), **payload}
    _executor.submit(_post, url, body, cfg["timeout_s"], event)
