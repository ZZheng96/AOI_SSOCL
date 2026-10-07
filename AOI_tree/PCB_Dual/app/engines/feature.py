"""特征引擎入口：树枝 PCB_Dual 只通过 HTTP 调用树干 AOI_Core。

- DetectionResult：特征侧结果契约（fuse_dual 融合判定的输入）
- get_engine()：进程单例 FeatureClient（见 feature_client.py）

torch / DINOv2 / 品类模型快照 / 反馈学习全部由 AOI_Core 掌管，树枝不 vendored 算法。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---- 结果契约 ----
@dataclass
class DetectionResult:
    """特征引擎推理结果。"""
    score: float = 0.0                      # fused（含拦截加分）
    decision: str = "normal"                # normal/gray/anomaly
    is_anomaly: bool = False
    threshold: float = 0.0                  # tau_high
    gray_threshold: float = 0.0             # tau_gray
    slot_scores: Dict[str, float] = field(default_factory=dict)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    router_w: Dict[str, float] = field(default_factory=dict)
    defect_boxes: List[Dict[str, Any]] = field(default_factory=list)
    mask: Optional[Any] = None
    heatmap_paths: Dict[str, str] = field(default_factory=dict)
    types: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    n_tiles: int = 0
    triggered_slot: str = ""
    open_alert: bool = False
    align_offset: Optional[List[float]] = None
    align_warn: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


# ---- 进程级单例 ----
_engine: Optional[Any] = None
_engine_lock = threading.Lock()


def get_engine():
    """返回 FeatureClient 单例（HTTP 调树干 AOI_Core）。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from app.config import get_settings
                from app.engines.feature_client import FeatureClient
                settings = get_settings()
                _engine = FeatureClient(settings.aoi_core_url,
                                        timeout=settings.aoi_core_timeout)
    return _engine
