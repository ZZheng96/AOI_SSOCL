"""任务进度组件。

- TaskProgressBar：QProgressBar + 消息标签，按任务快照更新。
- TaskMonitor：QObject，内部用 websocket-client 在 QThread 中连 /ws/tasks，
  以 task_updated(dict) 信号推送任务快照。
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import TaskWSClient

# 任务状态 → 中文
STATUS_NAMES = {
    "pending": "排队中",
    "running": "运行中",
    "done": "已完成",
    "success": "已完成",
    "failed": "失败",
    "error": "失败",
    "cancelled": "已取消",
}


class TaskProgressBar(QWidget):
    """任务进度条：进度条 + 状态/消息标签 + 可选取消按钮（A15）。

    取消按钮：set_client(api_client) 后，update_task 收到 running 任务会
    显示「取消」；点击请求后端协作式取消（POST /api/tasks/{id}/cancel）。
    """

    cancel_requested = Signal(int)          # task_id

    def __init__(self, title: str = "任务进度",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = None
        self._task_id: int | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        row = QHBoxLayout()
        self._title = QLabel(title)
        self._title.setProperty("subtext", True)
        row.addWidget(self._title)
        row.addStretch(1)
        self._pct = QLabel("")
        self._pct.setProperty("subtext", True)
        row.addWidget(self._pct)
        self._btn_cancel = QPushButton("取消")
        self._btn_cancel.setFixedWidth(56)
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self._btn_cancel)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        lay.addWidget(self.bar)
        self._msg = QLabel("")
        self._msg.setProperty("subtext", True)
        self._msg.setWordWrap(True)
        lay.addWidget(self._msg)
        self.hide()

    def set_client(self, client) -> None:
        """绑定 ApiClient 以启用取消按钮（A15）。"""
        self._client = client

    def _on_cancel(self) -> None:
        if self._task_id is None:
            return
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setText("取消中")
        self.cancel_requested.emit(self._task_id)
        if self._client is not None:
            try:
                self._client.cancel_task(self._task_id)
            except Exception:  # noqa: BLE001 取消失败不阻断 UI
                pass

    def update_task(self, task: dict) -> None:
        """按 /ws/tasks 任务快照刷新显示。"""
        status = str(task.get("status", ""))
        tid = task.get("id")
        if tid is not None:
            try:
                self._task_id = int(tid)
            except (TypeError, ValueError):
                pass
        progress = task.get("progress", 0) or 0
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            progress = 0.0
        if progress <= 1.0:
            progress *= 100
        message = str(task.get("message", "") or "")
        name = STATUS_NAMES.get(status, status or "运行中")

        self.show()
        self.bar.setValue(int(progress))
        self._pct.setText(f"{name} {progress:.0f}%")
        self._msg.setText(message)
        running = status in ("pending", "running")
        self._btn_cancel.setVisible(running and self._client is not None)
        if running:
            self._btn_cancel.setEnabled(True)
            self._btn_cancel.setText("取消")
        if status in ("done", "success"):
            self.bar.setValue(100)
            self._pct.setText("已完成 100%")
        elif status in ("failed", "error", "cancelled"):
            self._pct.setText(name)

    def reset(self) -> None:
        self.bar.setValue(0)
        self._pct.setText("")
        self._msg.setText("")
        self._task_id = None
        self._btn_cancel.setVisible(False)
        self._btn_cancel.setEnabled(True)
        self._btn_cancel.setText("取消")
        self.hide()


class TaskMonitor(QObject):
    """全局任务进度监视器：连接 /ws/tasks，转发 task_updated(dict) 信号。"""

    task_updated = Signal(dict)
    error = Signal(str)

    def __init__(self, ws_url: str, parent: QObject | None = None):
        super().__init__(parent)
        self._ws_url = ws_url
        self._thread: TaskWSClient | None = None

    def start(self) -> None:
        """启动后台 WebSocket 线程（幂等）。"""
        if self._thread is not None and self._thread.isRunning():
            return
        ws_base = self._ws_url
        if ws_base.endswith("/ws/tasks"):
            ws_base = ws_base[: -len("/ws/tasks")]
        self._thread = TaskWSClient(ws_base, self)
        self._thread.task_updated.connect(self.task_updated)
        self._thread.error.connect(self.error)
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程。"""
        if self._thread is not None:
            self._thread.stop()
            self._thread.wait(2000)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()
