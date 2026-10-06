from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Defect:
    """单个缺陷框。

    坐标系：相对输入测试图，左上角 (x, y) + 宽高，单位为像素。
    """

    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    description: str = ""
    severity: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class DetectionResult:
    """一次算法检测的结果。

    status 取值：
        - "OK"    未检出缺陷
        - "NG"    检出缺陷
        - "ERROR" 调用过程出错（见 ``error`` 字段）
    """

    algorithm: str
    status: str
    defect_count: int
    defects: List[Defect] = field(default_factory=list)
    cost_time: float = 0.0
    output_image: Optional[np.ndarray] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"

    @property
    def is_ng(self) -> bool:
        return self.status == "NG"

    @property
    def has_error(self) -> bool:
        return self.status == "ERROR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "defect_count": self.defect_count,
            "defects": [d.to_dict() for d in self.defects],
            "cost_time": self.cost_time,
            "error": self.error,
            "extra": dict(self.extra),
        }

    def save_image(self, path: str) -> bool:
        """保存可视化结果图（output_image）到 ``path``。

        无可视化图或写盘失败时返回 False。
        """
        if self.output_image is None:
            return False
        from .image_io import imwrite_unicode

        return bool(imwrite_unicode(path, self.output_image))
