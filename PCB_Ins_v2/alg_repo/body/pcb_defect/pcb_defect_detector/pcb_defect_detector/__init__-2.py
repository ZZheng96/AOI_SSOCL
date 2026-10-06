from __future__ import annotations

from .detector import PCBDefectDetector
from .results import Defect, DetectionResult

__all__ = ["PCBDefectDetector", "DetectionResult", "Defect", "list_algorithms"]

__version__ = "1.0.0"


def list_algorithms():
    """返回 ``{algorithm_code: 中文名}``，等价于 ``PCBDefectDetector.list_algorithms()``。"""
    return PCBDefectDetector.list_algorithms()
