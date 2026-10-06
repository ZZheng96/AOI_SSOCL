"""对位服务（P0：借鉴 Java AOI Mark 校准 + 金手指 peizhun 配准）。

对位 = 待测图相对标准图的几何配准。偏移超阈值 → align_warn（demo5 U95 已证
10px 对位偏移是毁灭性弱点）。检测落库时记录 align_offset，供追溯/复判。

复用 alg_repo/gold_finger 的 peizhun.align_image（相位+ORB+候选择优混合配准）。
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

ALIGN_WARN_PX = 10.0  # 对位偏移告警阈值（px）


def _load_peizhun():
    import importlib
    import sys

    from app.config import GOLD_REPO_DIR
    repo = str(GOLD_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return importlib.import_module("peizhun")


_cache = {}
_lock = threading.Lock()


def align(std_bgr: np.ndarray, test_bgr: np.ndarray,
          fast_mode: bool = True) -> dict[str, Any]:
    """对位：返回 {ok, offset_x, offset_y, method, score, warn, affine}。

    偏移 = 仿射平移分量（px），abs 超过 ALIGN_WARN_PX 置 warn。
    """
    with _lock:
        if not _cache:
            _cache["peizhun"] = _load_peizhun()
    peizhun = _cache["peizhun"]
    try:
        ok, _aligned, meta = peizhun.align_image(
            test_bgr, std_bgr, enable_fast_mode=fast_mode)
    except Exception as exc:  # noqa: BLE001 对位失败不阻断检测
        return {"ok": False, "error": str(exc), "method": "failed",
                "offset_x": 0.0, "offset_y": 0.0, "warn": False, "affine": None}
    affine = np.array(meta.get("affine") or [[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    off_x = float(affine[0, 2]) if affine.shape[0] > 0 and affine.shape[1] > 2 else 0.0
    off_y = float(affine[1, 2]) if affine.shape[0] > 1 and affine.shape[1] > 2 else 0.0
    warn = (abs(off_x) > ALIGN_WARN_PX) or (abs(off_y) > ALIGN_WARN_PX)
    return {
        "ok": bool(ok),
        "offset_x": round(off_x, 2),
        "offset_y": round(off_y, 2),
        "method": str(meta.get("method") or ""),
        "score": float(meta.get("score") or 0.0),
        "warn": bool(warn),
        "affine": affine.tolist() if affine is not None else None,
    }
