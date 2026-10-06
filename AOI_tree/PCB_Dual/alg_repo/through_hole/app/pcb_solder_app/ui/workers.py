"""后台线程 worker：算法服务初始化、锡面预生成、检测执行（避免阻塞 UI）"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..core.models import TemplateConfig


class ServiceInitWorker(QObject):
    """在后台构建 DetectorService（含算法控件初始化）。"""

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


class PrebuildWorker(QObject):
    """后台调用 prebuild_template：标准图 + 锡面获取方式 -> 生成/复用 Mask。"""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, service, template_bgr: np.ndarray, cfg: TemplateConfig):
        super().__init__()
        self._svc = service
        self._tpl = template_bgr
        self._cfg = cfg

    def run(self):
        try:
            result = self._svc.prebuild_template(self._tpl, self._cfg)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class AlignPreviewWorker(QObject):
    """后台调用 preview_test_roi：标准图 ROI 配准映射到测试图坐标的预览。"""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, service, template_bgr: np.ndarray, test_bgr: np.ndarray, cfg: TemplateConfig):
        super().__init__()
        self._svc = service
        self._tpl = template_bgr
        self._tst = test_bgr
        self._cfg = cfg

    def run(self):
        try:
            result = self._svc.preview_test_roi(self._tpl, self._tst, self._cfg)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class DetectWorker(QObject):
    """后台调用 detect：标准图 + 测试图 -> DetectionResult。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, service, template_bgr: np.ndarray, test_bgr: np.ndarray, cfg: TemplateConfig):
        super().__init__()
        self._svc = service
        self._tpl = template_bgr
        self._tst = test_bgr
        self._cfg = cfg

    def run(self):
        try:
            result = self._svc.detect(self._tpl, self._tst, self._cfg)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
