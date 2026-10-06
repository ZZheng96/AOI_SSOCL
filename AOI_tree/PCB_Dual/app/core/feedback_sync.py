"""PCB_Dual 到 AOI_Core 的反馈同步与失败补偿。"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from app.config import get_settings
from app.db.database import session_scope
from app.db.models import Detection, Feedback

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 10.0
_SCAN_INTERVAL_S = 15.0


def _settings() -> tuple[str, float, int, float]:
    settings = get_settings()
    url = str(settings.get("feedback_sync", "url", "") or "").strip()
    timeout = float(settings.get("feedback_sync", "timeout_s", 5) or 5)
    max_attempts = int(settings.get("feedback_sync", "max_attempts", _MAX_ATTEMPTS) or _MAX_ATTEMPTS)
    backoff = float(settings.get("feedback_sync", "base_backoff_s", _BASE_BACKOFF_S) or _BASE_BACKOFF_S)
    return url, timeout, max(1, max_attempts), max(0.1, backoff)


def sync_feedback(feedback_id: int) -> dict[str, Any]:
    """同步一条反馈；网络请求不持有数据库事务。"""
    url, timeout, max_attempts, backoff = _settings()
    if not url:
        return {"status": "disabled"}

    with session_scope() as s:
        fb = s.get(Feedback, int(feedback_id))
        if fb is None:
            return {"status": "missing"}
        det = s.get(Detection, int(fb.detection_id))
        if det is None:
            fb.mainline_sync_status = "failed"
            fb.mainline_sync_error = "检测记录不存在"
            return {"status": "failed", "error": fb.mainline_sync_error}
        if fb.mainline_sync_status == "synced":
            return {"status": "synced"}
        if fb.mainline_sync_attempts >= max_attempts:
            fb.mainline_sync_status = "failed"
            return {"status": "failed", "attempts": fb.mainline_sync_attempts}
        fb.mainline_sync_attempts += 1
        fb.mainline_sync_last_attempt_at = datetime.now()
        attempt = fb.mainline_sync_attempts
        payload = {
            "image_path": det.image_path,
            "category": det.category or "default",
            "label": int(fb.operator_label),
            "defect_type": fb.defect_type,
            "comment": fb.comment,
            "region": (fb.region or {}).get("box") if isinstance(fb.region, dict) else fb.region,
            "source": "PCB_Dual:operator_feedback",
            "source_event_id": fb.source_event_id,
        }

    try:
        resp = requests.post(
            url.rstrip("/") + "/api/feedback/traditional",
            json=payload,
            timeout=timeout,
        )
        status_code = int(resp.status_code)
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:  # noqa: BLE001
        delay = backoff * (2 ** max(0, attempt - 1))
        with session_scope() as s:
            fb = s.get(Feedback, int(feedback_id))
            if fb is not None:
                fb.mainline_sync_http_status = getattr(locals().get("resp"), "status_code", None)
                fb.mainline_sync_error = str(exc)
                if fb.mainline_sync_attempts >= max_attempts:
                    fb.mainline_sync_status = "failed"
                    fb.mainline_sync_next_retry_at = None
                else:
                    fb.mainline_sync_status = "pending"
                    fb.mainline_sync_next_retry_at = datetime.now() + timedelta(seconds=delay)
        return {"status": "failed" if attempt >= max_attempts else "pending", "error": str(exc), "attempts": attempt}

    with session_scope() as s:
        fb = s.get(Feedback, int(feedback_id))
        if fb is not None:
            fb.mainline_sync_status = "synced"
            fb.mainline_sync_error = None
            fb.mainline_sync_http_status = status_code
            fb.mainline_sync_synced_at = datetime.now()
            fb.mainline_sync_next_retry_at = None
    return {"status": "synced", "response": result, "attempts": attempt}


class FeedbackSyncWorker:
    """独立于检测队列运行，避免同步失败阻塞产线检测。"""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="feedback-sync-worker")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(_SCAN_INTERVAL_S):
            try:
                url, _timeout, max_attempts, _backoff = _settings()
                if not url:
                    continue
                now = datetime.now()
                with session_scope() as s:
                    rows = list(s.query(Feedback).filter(
                        Feedback.mainline_sync_status == "pending",
                        Feedback.mainline_sync_attempts < max_attempts,
                        (Feedback.mainline_sync_next_retry_at.is_(None)) |
                        (Feedback.mainline_sync_next_retry_at <= now),
                    ).order_by(Feedback.id).limit(20).all())
                    ids = [int(row.id) for row in rows]
                for feedback_id in ids:
                    sync_feedback(feedback_id)
            except Exception:  # noqa: BLE001
                log.exception("反馈同步补偿扫描失败")


_worker = FeedbackSyncWorker()


def get_feedback_sync_worker() -> FeedbackSyncWorker:
    return _worker
