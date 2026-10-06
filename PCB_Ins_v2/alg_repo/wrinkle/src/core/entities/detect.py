"""检测结果类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional



@dataclass
class BoundingBox:
    """
    边界框

    坐标系: (x, y) 为左上角，width/height 为宽高，angle 为旋转角度(度)。
    """
    box_id: int
    box_type: str
    x: int
    y: int
    width: int
    height: int
    detectionConfig = None # 配置列表
    angle: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


class ResultType(Enum):
    COMPONENT = "component"
    PARTS = "parts"
    ERROR = "error"

@dataclass
class DetectBox:
    """检测框基类：公共几何 + metadata。"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    angle: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentDetectBox(DetectBox):
    """器件识别结果。"""
    component_id: Optional[int] = None
    class_id: Optional[int] = None
    component_type: Optional[str] = None
    shm_index: Optional[int] = None
    component_file: Optional[str] = None
    fov_id: Optional[int] = None


@dataclass
class DetectPartBox(DetectBox):
    """器件部件识别结果（焊盘 / 引脚 / OCR 等）。"""
    box_type: Optional[str] = None
    label: Optional[str] = None
    class_id: Optional[int] = None


@dataclass
class AlgorithmResult:
    """算法层统一出口。"""
    code: int
    message: str
    algorithm_code: str
    cost_time: float
    result_type: ResultType
    components: List[ComponentDetectBox] = field(default_factory=list)
    parts: List[DetectPartBox] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
