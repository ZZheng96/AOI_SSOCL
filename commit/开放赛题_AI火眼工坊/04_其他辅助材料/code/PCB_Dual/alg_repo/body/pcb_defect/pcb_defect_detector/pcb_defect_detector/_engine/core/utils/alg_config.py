from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from core.entities.detect import (
    AlgorithmResult,
    ResultType,
)


def merge_config(
    defaults: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并默认配置与外部传入的覆盖配置（忽略 None 值）。"""
    merged = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


def safe_index(seq, idx, default=None):
    if seq is None:
        return default
    try:
        return seq[idx]
    except (IndexError, TypeError, KeyError):
        return default


def build_error_result(
    algorithm_code: str,
    exc: BaseException,
    *,
    cost_time: float = 0.0,
) -> AlgorithmResult:
    return AlgorithmResult(
        code=1,
        message=str(exc),
        algorithm_code=algorithm_code,
        cost_time=cost_time,
        result_type=ResultType.ERROR,
        metadata={
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )
