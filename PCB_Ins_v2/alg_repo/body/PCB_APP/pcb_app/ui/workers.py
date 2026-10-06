"""后台线程 worker：算法服务初始化、检测执行（避免阻塞 UI）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal


class ServiceInitWorker(QObject):
    """在后台构建 DetectorService（含算法池加载，较慢）。"""

    ready = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            from ..core.detector_service import DetectorService
            svc = DetectorService()
            self.ready.emit(svc)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class RoiMatchWorker(QObject):
    """画框/挪框后立即在整图范围内做一次结构匹配预览（不阻塞 UI 线程）。"""

    finished = Signal(object)     # MatchedROI
    failed = Signal(str)

    def __init__(self, service, template, test, roi):
        super().__init__()
        self._svc = service
        self._tpl = template
        self._tst = test
        self._roi = roi

    def run(self):
        try:
            matched = self._svc.compute_roi_match(self._tpl, self._tst, self._roi)
            self.finished.emit(matched)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class DetectWorker(QObject):
    """执行一次检测任务集合。"""

    progress = Signal(int, int, str)
    finished = Signal(object)     # RunSummary
    failed = Signal(str)

    def __init__(self, service, template, test, tasks: List[Dict[str, Any]],
                 sessions_dir=None):
        super().__init__()
        self._svc = service
        self._tpl = template
        self._tst = test
        self._tasks = tasks
        self._sessions_dir = sessions_dir

    def run(self):
        try:
            summary = self._svc.run(
                self._tpl, self._tst, self._tasks,
                auto_save=True, sessions_dir=self._sessions_dir,
                progress_cb=lambda d, t, s: self.progress.emit(d, t, s),
            )
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
