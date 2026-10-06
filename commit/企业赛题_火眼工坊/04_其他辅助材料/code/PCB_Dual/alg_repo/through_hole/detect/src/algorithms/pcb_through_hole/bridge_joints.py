"""连锡焊点组标注：Joint 数据结构与 JSON 解析（无交互式标注 GUI）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BridgeJoint:
    """一个焊点/排除区标注（标准图像素坐标）。

    type == 'circle': params = {cx, cy, r}
    type == 'rect'  : params = {x, y, w, h}
    role == 'pad' | 'exclude'
    """

    id: int
    type: str
    params: Dict[str, float] = field(default_factory=dict)
    label: str = ""
    role: str = "pad"

    def mask(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        m = np.zeros((h, w), dtype=np.uint8)
        if self.type == "circle":
            cx, cy, r = self.params["cx"], self.params["cy"], self.params["r"]
            cv2.circle(m, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
        elif self.type == "rect":
            x, y, w_, h_ = (
                self.params["x"], self.params["y"],
                self.params["w"], self.params["h"],
            )
            cv2.rectangle(
                m,
                (int(round(x)), int(round(y))),
                (int(round(x + w_)), int(round(y + h_))),
                255, -1,
            )
        else:
            raise ValueError(f"未知的标注类型: {self.type}")
        return m

    def scaled(self, sx: float, sy: float) -> "BridgeJoint":
        """按轴缩放坐标（模板降采样时用）。"""
        p = dict(self.params)
        if self.type == "circle":
            p["cx"] = float(p["cx"]) * sx
            p["cy"] = float(p["cy"]) * sy
            p["r"] = float(p["r"]) * (sx + sy) / 2.0
        else:
            p["x"] = float(p["x"]) * sx
            p["y"] = float(p["y"]) * sy
            p["w"] = float(p["w"]) * sx
            p["h"] = float(p["h"]) * sy
        return BridgeJoint(self.id, self.type, p, self.label, self.role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "type": self.type,
            "params": {k: float(v) for k, v in self.params.items()},
            "label": self.label,
            "role": self.role,
        }

    def to_preview_roi(self) -> Dict[str, Any]:
        if self.type == "circle":
            return {
                "shape": "circle",
                "cx": float(self.params["cx"]),
                "cy": float(self.params["cy"]),
                "r": float(self.params["r"]),
            }
        return {
            "shape": "rect",
            "x": float(self.params["x"]),
            "y": float(self.params["y"]),
            "w": float(self.params["w"]),
            "h": float(self.params["h"]),
        }


def parse_bridge_joints(label: Optional[Dict[str, Any]]) -> List[BridgeJoint]:
    """解析 bridge_joints / joints。"""
    if not label:
        return []
    raw = label.get("bridge_joints")
    if raw is None:
        raw = label.get("joints")
    if not isinstance(raw, list):
        return []
    out: List[BridgeJoint] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        jtype = str(item.get("type", "rect") or "rect").lower()
        if jtype not in ("circle", "rect"):
            continue
        role = str(item.get("role", "pad") or "pad").lower()
        if role not in ("pad", "exclude"):
            role = "pad"
        params_in = item.get("params") or {}
        if not isinstance(params_in, dict):
            # 扁平写法兼容
            params_in = {k: item[k] for k in ("cx", "cy", "r", "x", "y", "w", "h") if k in item}
        try:
            jid = int(item.get("id", i + 1))
            params = {k: float(v) for k, v in params_in.items()}
        except (TypeError, ValueError):
            continue
        required = ("cx", "cy", "r") if jtype == "circle" else ("x", "y", "w", "h")
        if any(k not in params for k in required):
            continue
        out.append(BridgeJoint(
            id=jid, type=jtype, params=params,
            label=str(item.get("label", "") or ""), role=role,
        ))
    return out


def pad_joints(joints: List[BridgeJoint]) -> List[BridgeJoint]:
    return [j for j in joints if j.role != "exclude"]


def exclude_joints(joints: List[BridgeJoint]) -> List[BridgeJoint]:
    return [j for j in joints if j.role == "exclude"]
