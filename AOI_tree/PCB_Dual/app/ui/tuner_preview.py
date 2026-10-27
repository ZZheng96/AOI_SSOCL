"""单任务后台预览：仅保留最新请求，关闭只取消投递，不强杀算法。"""
from __future__ import annotations

from concurrent.futures import Future
import threading

from PySide6.QtCore import QObject, QTimer, Signal


class PreviewController(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, runner, parent=None, debounce_ms=250):
        super().__init__(parent)
        self.runner = runner
        self.generation = 0
        self.closed = False
        self.pending = None
        self.active = None
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(debounce_ms)
        self.debounce.timeout.connect(self._launch)
        self.poll = QTimer(self)
        self.poll.setInterval(25)
        self.poll.timeout.connect(self._collect)

    def submit(self, payload):
        if self.closed:
            return
        self.generation += 1
        self.pending = (self.generation, payload)
        self.debounce.start()

    def cancel(self):
        self.generation += 1
        self.pending = None
        self.debounce.stop()

    def close(self):
        self.cancel()
        self.closed = True
        self.poll.stop()

    def _launch(self):
        if self.closed or self.active is not None or self.pending is None:
            return
        generation, payload = self.pending
        self.pending = None
        future = Future()
        self.active = (generation, future)
        runner = self.runner

        def run():
            try:
                future.set_result(runner(payload))
            except Exception as exc:
                future.set_exception(exc)

        threading.Thread(target=run, daemon=True, name="tuner-preview").start()
        self.poll.start()

    def _collect(self):
        if self.closed or self.active is None:
            return
        generation, future = self.active
        if not future.done():
            return
        self.active = None
        self.poll.stop()
        if generation == self.generation:
            try:
                result = future.result()
            except Exception as exc:
                self.failed.emit(str(exc))
            else:
                self.finished.emit(result)
        if self.pending is not None and not self.debounce.isActive():
            self._launch()
