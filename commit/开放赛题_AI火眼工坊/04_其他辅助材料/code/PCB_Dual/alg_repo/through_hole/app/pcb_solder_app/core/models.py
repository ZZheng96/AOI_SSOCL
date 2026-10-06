"""界面内部数据模型：人工框选区域 / 标准图配置 / 检测结果"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..metadata import (
    DEFAULT_ENABLED_DEFECTS,
    PAD_INTERNAL_DEFECTS,
    defects_for_pad_count,
)

_VALID_SOLDER_MODES = ("ellipse", "contour", "roi")
_AUTO_SOLDER_MODES = ("ellipse", "contour")  # solder_mode 字段的合法取值，不包含 "roi"


@dataclass
class ManualRoi:
    """人工框选区域（标准图原始像素坐标系），对应算法 config["solder_roi"]。"""

    shape: str  # "rect" | "circle"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0

    @staticmethod
    def make_rect(x: float, y: float, w: float, h: float) -> "ManualRoi":
        return ManualRoi("rect", x=x, y=y, w=w, h=h)

    @staticmethod
    def make_circle(cx: float, cy: float, r: float) -> "ManualRoi":
        return ManualRoi("circle", cx=cx, cy=cy, r=r)

    def bounds(self) -> Tuple[float, float, float, float]:
        """统一返回外接矩形 (x, y, w, h)，供画布绘制/命中测试使用。"""
        if self.shape == "circle":
            return (self.cx - self.r, self.cy - self.r, self.r * 2, self.r * 2)
        return (self.x, self.y, self.w, self.h)

    def contains(self, px: float, py: float) -> bool:
        if self.shape == "circle":
            return (px - self.cx) ** 2 + (py - self.cy) ** 2 <= self.r ** 2
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def is_valid(self) -> bool:
        if self.shape == "circle":
            return self.r >= 2
        return self.w >= 4 and self.h >= 4

    def to_engine_dict(self) -> Dict[str, Any]:
        if self.shape == "circle":
            return {"shape": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}
        return {"shape": "rect", "x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> Optional["ManualRoi"]:
        if not d:
            return None
        shape = str(d.get("shape", "rect") or "rect").lower()
        if shape == "circle":
            return ManualRoi.make_circle(
                float(d.get("cx", 0)), float(d.get("cy", 0)), float(d.get("r", 0)))
        return ManualRoi.make_rect(
            float(d.get("x", 0)), float(d.get("y", 0)),
            float(d.get("w", 0)), float(d.get("h", 0)))

    def to_bridge_joint(self, jid: int, role: str = "pad") -> Dict[str, Any]:
        """转为算法 bridge_joints 条目。"""
        if self.shape == "circle":
            return {
                "id": int(jid), "type": "circle", "role": role, "label": "",
                "params": {"cx": float(self.cx), "cy": float(self.cy), "r": float(self.r)},
            }
        return {
            "id": int(jid), "type": "rect", "role": role, "label": "",
            "params": {
                "x": float(self.x), "y": float(self.y),
                "w": float(self.w), "h": float(self.h),
            },
        }

    @staticmethod
    def from_bridge_joint(d: Optional[Dict[str, Any]]) -> Optional["ManualRoi"]:
        if not d:
            return None
        jtype = str(d.get("type", d.get("shape", "rect")) or "rect").lower()
        params = d.get("params") if isinstance(d.get("params"), dict) else d
        try:
            if jtype == "circle":
                return ManualRoi.make_circle(
                    float(params["cx"]), float(params["cy"]), float(params["r"]))
            return ManualRoi.make_rect(
                float(params["x"]), float(params["y"]),
                float(params["w"]), float(params["h"]))
        except (KeyError, TypeError, ValueError):
            return None


def rois_from_bridge_joints(joints: Optional[List[Dict[str, Any]]]) -> List[ManualRoi]:
    """仅取 role=pad 的焊点框（排除区暂不在界面编辑）。"""
    out: List[ManualRoi] = []
    for item in joints or []:
        if str(item.get("role", "pad") or "pad").lower() == "exclude":
            continue
        roi = ManualRoi.from_bridge_joint(item)
        if roi is not None and roi.is_valid():
            out.append(roi)
    return out


def bridge_joints_from_rois(rois: List[ManualRoi]) -> List[Dict[str, Any]]:
    return [roi.to_bridge_joint(i + 1, "pad") for i, roi in enumerate(rois)]


@dataclass
class TemplateConfig:
    """一张标准图的完整检测配置，与 ``template_store`` 落盘的 JSON 一一对应。"""

    template_id: str
    template_path: str = ""
    solder_mode: str = "ellipse"           # ellipse | contour | roi
    solder_roi: Optional[ManualRoi] = None  # solder_mode == "roi" 时必填
    enabled_defects: List[int] = field(default_factory=lambda: list(DEFAULT_ENABLED_DEFECTS))
    void_preset: str = "MED"
    insuf_preset: str = "MED"
    enable_review: bool = True
    advanced: Dict[str, Any] = field(default_factory=dict)
    bridge_joints: Optional[List[Dict[str, Any]]] = None

    def manual_pad_rois(self) -> List[ManualRoi]:
        """优先 bridge_joints，否则回退 solder_roi。"""
        pads = rois_from_bridge_joints(self.bridge_joints)
        if pads:
            return pads
        if self.solder_roi is not None and self.solder_roi.is_valid():
            return [self.solder_roi]
        return []

    def effective_solder_mode(self) -> str:
        """有人工框时为 roi；否则为 ellipse/contour。"""
        if self.manual_pad_rois():
            return "roi"
        return self.solder_mode if self.solder_mode in _AUTO_SOLDER_MODES else "ellipse"

    def effective_enabled_defects(self) -> List[int]:
        """多框仅 15；单框/自动为盘内缺陷与勾选的交集。"""
        n = len(self.manual_pad_rois())
        mode_ids = defects_for_pad_count(n)
        if n >= 2:
            return list(mode_ids)
        allowed = set(mode_ids)
        base = list(self.enabled_defects) if self.enabled_defects is not None else list(
            PAD_INTERNAL_DEFECTS)
        return [d for d in base if d in allowed]

    def to_algorithm_config(self) -> Dict[str, Any]:
        """转为算法 config dict。"""
        pads = self.manual_pad_rois()
        mode = "roi" if pads else (
            self.solder_mode if self.solder_mode in _AUTO_SOLDER_MODES else "ellipse")
        joints = bridge_joints_from_rois(pads) if pads else None
        return {
            "template_id": self.template_id,
            "solder_mode": mode,
            "solder_roi": pads[0].to_engine_dict() if pads else None,
            "enabled_defects": self.effective_enabled_defects(),
            "bridge_joints": joints,
            "void_preset": self.void_preset,
            "insuf_preset": self.insuf_preset,
            "enable_review": self.enable_review,
            "advanced": dict(self.advanced),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """落盘用：``solder_mode`` 存"自动方式"原始选择（ellipse/contour），
        而非 ``to_algorithm_config()`` 的有效模式，避免把 "roi" 写入该字段。"""
        return {
            "template_id": self.template_id,
            "template_path": self.template_path,
            "solder_mode": self.solder_mode if self.solder_mode in _AUTO_SOLDER_MODES else "ellipse",
            "solder_roi": self.solder_roi.to_engine_dict() if self.solder_roi else None,
            "enabled_defects": list(self.enabled_defects),
            "bridge_joints": list(self.bridge_joints) if self.bridge_joints else None,
            "void_preset": self.void_preset,
            "insuf_preset": self.insuf_preset,
            "enable_review": self.enable_review,
            "advanced": dict(self.advanced),
        }

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> "TemplateConfig":
        # 兼容 Tin 标注：joints → bridge_joints
        bj = d.get("bridge_joints")
        if bj is None and d.get("joints") is not None:
            bj = d.get("joints")
        return TemplateConfig(
            template_id=str(d.get("template_id") or ""),
            template_path=str(d.get("template_path") or ""),
            solder_mode=str(d.get("solder_mode") or "ellipse"),
            solder_roi=ManualRoi.from_dict(d.get("solder_roi")),
            # 显式判空以区分"字段缺失"与"全部取消勾选 ([])"，避免 `or` 把后者当空值处理
            enabled_defects=[int(x) for x in (
                d.get("enabled_defects")
                if d.get("enabled_defects") is not None else DEFAULT_ENABLED_DEFECTS
            )],
            void_preset=str(d.get("void_preset") or "MED"),
            insuf_preset=str(d.get("insuf_preset") or "MED"),
            enable_review=True,  # 早停复检默认启用，忽略旧配置里可能存的 False
            advanced=dict(d.get("advanced") or {}),
            bridge_joints=list(bj) if isinstance(bj, list) else None,
        )


@dataclass
class DefectBox:
    defect_id: int
    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    reason: str = ""


@dataclass
class DetectionResult:
    ok: bool
    status: str = ""            # OK / NG / REVIEW / ERROR
    defect_ids: List[int] = field(default_factory=list)
    defects: List[DefectBox] = field(default_factory=list)
    coverage: Optional[float] = None
    review_reason: str = ""
    align_method: str = ""
    template_id: str = ""
    cost_ms: float = 0.0
    output_image: Optional[np.ndarray] = None
    error: str = ""
