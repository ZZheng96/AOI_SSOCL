from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class BoundingBox:
    """边界框。坐标系: (x, y) 左上角，width/height 宽高。"""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0
    class_name: str = ""
    confidence: float = 0.0
    detection_config: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class DefectInfo:
    """缺陷信息。"""
    defect_type: str = "UNKNOWN"
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    severity: float = 0.0
    confidence: float = 0.0
    description: str = ""
    mask: Optional[np.ndarray] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    image_path: str = ""
