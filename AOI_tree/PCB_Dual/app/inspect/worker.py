"""检测后台线程：UI 只收信号，可取消批量。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.inspect.service import BatchInspectResult, InspectRunResult, InspectService
from app.template.model import InspectionTemplate


class InspectWorker(QThread):
    progress = Signal(int, int, str)
    item_done = Signal(int, object)
    batch_finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: InspectService,
        template: InspectionTemplate,
        paths: list[Path],
        *,
        allow_stub: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.service = service
        self.template = template
        self.paths = list(paths)
        self.allow_stub = allow_stub
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: N802
        try:
            batch = self.service.run_batch_aware(
                self.template,
                self.paths,
                progress_cb=lambda i, n, p: self.progress.emit(i, n, p),
                allow_stub=self.allow_stub,
                write_summary=True,
                archive=True,
                cancel_cb=self._cancelled,
                item_cb=lambda idx, run: self.item_done.emit(idx, run),
            )
            self.batch_finished.emit(batch)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
