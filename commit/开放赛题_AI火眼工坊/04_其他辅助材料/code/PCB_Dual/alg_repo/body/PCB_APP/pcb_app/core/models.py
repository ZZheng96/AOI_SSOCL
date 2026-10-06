"""前端数据模型：ROI、检测项、检测结果。

坐标统一使用 **原始图像像素坐标系**（与算法包契约一致），画布负责在显示时做缩放换算。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_roi_counter = itertools.count(1)


@dataclass
class ROI:
    """一个用户绘制的重点检测区域。

    - shape: "rect" 或 "circle"
    - rect:  (x, y, w, h)  左上角 + 宽高（circle 时为其外接矩形，便于统一显示）
    - circle 额外用 cx, cy, r 描述
    - defect_codes: 绑定的瑕疵/算法 code 列表（可绑定多种，空列表表示未绑定）。
      同一个 ROI 绑定的每种瑕疵类型都会在检测时各自生成一条任务，
      即同一区域可以被多种已启用的缺陷检测算法分别检测。
    """

    shape: str
    x: float
    y: float
    w: float
    h: float
    defect_codes: List[str] = field(default_factory=list)
    rid: int = field(default_factory=lambda: next(_roi_counter))
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"ROI {self.rid}"

    # ---- 兼容属性：绑定的第一种瑕疵（用于只需单值的场景，如联动展示参数）----
    @property
    def defect_code(self) -> Optional[str]:
        return self.defect_codes[0] if self.defect_codes else None

    # ---- 圆形便捷属性（以外接矩形推导）----
    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def r(self) -> float:
        return min(self.w, self.h) / 2.0

    def to_engine_roi(self) -> Dict[str, Any]:
        """转换为算法包 ``detect(roi=...)`` 接受的字典。"""
        if self.shape == "circle":
            return {"shape": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}
        return {"shape": "rect", "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def contains(self, px: float, py: float) -> bool:
        if self.shape == "circle":
            return (px - self.cx) ** 2 + (py - self.cy) ** 2 <= self.r ** 2
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rid": self.rid, "name": self.name, "shape": self.shape,
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "defect_codes": list(self.defect_codes),
        }


@dataclass
class DefectBox:
    """一条检测出的缺陷（已换算到整测试图坐标）。"""
    label: str
    defect_code: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    description: str = ""
    roi_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, "defect_code": self.defect_code,
            "x": self.x, "y": self.y, "width": self.width, "height": self.height,
            "confidence": round(self.confidence, 4),
            "description": self.description, "roi_name": self.roi_name,
        }


@dataclass
class MatchedROI:
    """模板 ROI 在待检图中通过结构匹配得到的位置与方向。"""
    rid: int
    name: str
    matched: bool
    fallback: bool
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0
    angle_deg: float = 0.0
    residual_angle_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    reversed_180: bool = False
    normal_score: float = 0.0
    reverse_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rid": self.rid, "name": self.name,
            "matched": self.matched, "fallback": self.fallback,
            "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon],
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "angle_deg": round(self.angle_deg, 3),
            "residual_angle_deg": round(self.residual_angle_deg, 3),
            "dx": round(self.dx, 3), "dy": round(self.dy, 3),
            "reversed_180": self.reversed_180,
            "normal_score": round(self.normal_score, 4),
            "reverse_score": round(self.reverse_score, 4),
            "reason": self.reason,
        }


@dataclass
class RunItemResult:
    """单个"算法×区域"的检测结果。"""
    defect_code: str
    defect_name: str
    roi_name: str          # "整图" 或 ROI 名
    status: str            # OK / NG / ERROR
    defects: List[DefectBox] = field(default_factory=list)
    cost_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class RunSummary:
    """一次完整检测的汇总。"""
    items: List[RunItemResult] = field(default_factory=list)
    total_ms: float = 0.0
    ng_count: int = 0
    ok_count: int = 0
    err_count: int = 0
    session_dir: Optional[str] = None
    matched_rois: List[MatchedROI] = field(default_factory=list)

    @property
    def all_defects(self) -> List[DefectBox]:
        out: List[DefectBox] = []
        for it in self.items:
            out.extend(it.defects)
        return out
