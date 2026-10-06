from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.compat import imwrite_unicode

# 落盘根目录：<项目根>/adapter_runs/（向上两级到项目根，不依赖 cwd）
_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / "adapter_runs"

_LOCK = threading.Lock()
# 进程内已分配的最大编号（启动时扫描目录初始化）
_MAX_INDEX = {"value": 0}


def _scan_max_index(root: Path) -> int:
    """扫描已有目录，返回最大数字编号。"""
    if not root.exists():
        return 0
    max_idx = 0
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit():
            max_idx = max(max_idx, int(d.name))
    return max_idx


def _ensure_root() -> Path:
    root = _ADAPTER_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if _MAX_INDEX["value"] == 0:
        _MAX_INDEX["value"] = _scan_max_index(root)
    return root


def _next_index() -> int:
    """分配下一个递增编号（线程安全）。"""
    with _LOCK:
        _ensure_root()
        _MAX_INDEX["value"] += 1
        # 多进程兜底：若目录已存在则继续递增直到空闲
        root = _ADAPTER_ROOT
        while (root / f"{_MAX_INDEX['value']:04d}").exists():
            _MAX_INDEX["value"] += 1
        return _MAX_INDEX["value"]


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """转 BGR 三通道用于写盘。"""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_result(image: np.ndarray, defects: List[Any]) -> np.ndarray:
    """在测试图上画缺陷框 + label + 置信度，返回 BGR 可视化图。

    defects 为 DefectInfo 列表（含 bounding_box / defect_type / severity）。
    """
    vis = _to_bgr(image).copy() if image is not None else None
    if vis is None:
        return None
    for d in defects or []:
        bb = getattr(d, "bounding_box", None)
        if bb is None:
            continue
        x, y = int(bb.x), int(bb.y)
        w, h = int(bb.width), int(bb.height)
        label = getattr(d, "defect_type", "defect")
        sev = getattr(d, "severity", 0.0) or 0.0
        color = (0, 0, 255) if sev > 0 else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        txt = f"{label} {sev:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y - 4)
        cv2.rectangle(vis, (x, ty - th - 2), (x + tw + 4, ty), color, -1)
        cv2.putText(vis, txt, (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


class RunSaver:
    """流式检测落盘单例。"""

    @staticmethod
    def save(
        algorithm_code: str,
        image: np.ndarray,
        template: Optional[np.ndarray],
        result_image: Optional[np.ndarray],
        result_dict: Dict[str, Any],
    ) -> str:
        """保存一次检测的输入输出，返回本次目录路径。

        result_dict 至少包含: code, status, defect_count, defects, cost_time
        """
        # 落盘留档已禁用
        # idx = _next_index()
        # run_dir = _ADAPTER_ROOT / f"{idx:04d}"
        # run_dir.mkdir(parents=True, exist_ok=True)
        #
        # try:
        #     test_bgr = _to_bgr(image)
        #     if test_bgr is not None:
        #         imwrite_unicode(str(run_dir / "test.png"), test_bgr,
        #                         [cv2.IMWRITE_PNG_COMPRESSION, 3])
        #     if template is not None and template.size > 0:
        #         imwrite_unicode(str(run_dir / "template.png"), _to_bgr(template),
        #                         [cv2.IMWRITE_PNG_COMPRESSION, 3])
        #     if result_image is not None:
        #         imwrite_unicode(str(run_dir / "result.png"), _to_bgr(result_image),
        #                         [cv2.IMWRITE_PNG_COMPRESSION, 3])
        #     with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        #         json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)
        # except Exception as exc:
        #     # 落盘失败不得影响主流程，仅记录到 sidecar
        #     try:
        #         with open(run_dir / "_save_error.txt", "w", encoding="utf-8") as f:
        #             f.write(f"落盘失败: {exc}\n")
        #     except Exception:
        #         pass
        #
        # return str(run_dir)
        return ""


def legacy_defects_to_summary(defects: List[Any]) -> List[Dict[str, Any]]:
    """将 DefectInfo 列表转为可 JSON 序列化的摘要。"""
    out = []
    for d in defects or []:
        bb = getattr(d, "bounding_box", None)
        out.append({
            "defect_type": getattr(d, "defect_type", ""),
            "x": float(bb.x) if bb is not None else 0.0,
            "y": float(bb.y) if bb is not None else 0.0,
            "width": float(bb.width) if bb is not None else 0.0,
            "height": float(bb.height) if bb is not None else 0.0,
            "severity": float(getattr(d, "severity", 0.0) or 0.0),
            "confidence": float(getattr(d, "confidence", 0.0) or 0.0),
            "description": getattr(d, "description", ""),
        })
    return out
