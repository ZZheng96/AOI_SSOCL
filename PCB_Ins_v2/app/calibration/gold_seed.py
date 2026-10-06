"""金手指检测的板面取色标定：在标准图上点几个种子点，复用算法仓库自带的
``seed_extract.ranges_from_seeds`` 算出 HSV 范围（非交互、纯函数，不弹独立
的 OpenCV 窗口）。"""
from __future__ import annotations

from typing import Any

from app.calibration.store import load_calib, update_calib


def get_gold_seed(template_name: str | None) -> dict[str, Any]:
    data = load_calib(template_name)
    gs = data.get("gold_seed") or {}
    return {
        "seeds": gs.get("seeds") or [],
        "tol": float(gs.get("tol", 1.0)),
        "ranges": gs.get("ranges") or [],
    }


def set_gold_seed(template_name: str | None, seeds: list, tol: float, ranges: list) -> None:
    update_calib(
        template_name,
        gold_seed={
            "seeds": [[int(x), int(y)] for x, y in seeds],
            "tol": float(tol),
            "ranges": [[list(lo), list(hi)] for lo, hi in ranges],
        },
    )
