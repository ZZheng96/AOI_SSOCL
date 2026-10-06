from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AlgorithmResult:
    """算法输出统一结构（旧版，detect_roi 用）。"""
    status: str = "OK"                      # OK / NG / WARNING / ERROR
    defects: List[Any] = field(default_factory=list)  # List[DefectInfo]
    statistics: Optional[Any] = None
    mask: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    matching_confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_defects(self) -> bool:
        return len(self.defects) > 0

    def is_ok(self) -> bool:
        return self.status == "OK"
