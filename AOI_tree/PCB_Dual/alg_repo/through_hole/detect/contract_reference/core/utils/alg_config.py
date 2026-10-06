"""仅供本仓库单元测试使用；接入检测系统工程时使用工程自带的同名真实模块，不要拷贝本文件"""
from __future__ import annotations

import traceback
from dataclasses import replace
from typing import Any, Dict, List, Optional

from core.entities.detect import (
    AlgorithmResult,
    ComponentDetectBox,
    DetectPartBox,
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


def algorithm_code_of(instance: Any) -> str:
    return str(getattr(instance, "algorithm_code", instance.__class__.__name__))


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


def merge_result_context(
    result: AlgorithmResult,
    **context: Any,
) -> AlgorithmResult:
    """流水线入队前合并 component_id 等上下文字段到 metadata。"""
    extra = {k: v for k, v in context.items() if v is not None}
    if not extra:
        return result
    return replace(result, metadata={**result.metadata, **extra})


def enqueue_with_context(queue: Any, result: AlgorithmResult, **context: Any) -> None:
    queue.put(merge_result_context(result, **context))


def part_box_to_dict(part: DetectPartBox) -> Dict[str, Any]:
    """生产检测路径兼容：将 PartDetectBox 转为 dict。"""
    return {
        "x": part.x,
        "y": part.y,
        "width": part.width,
        "height": part.height,
        "angle": part.angle,
        "confidence": part.confidence,
        "box_type": part.box_type,
        "label": part.label,
        "class_id": part.class_id,
        "metadata": dict(part.metadata),
    }


def component_box_to_dict(box: ComponentDetectBox) -> Dict[str, Any]:
    return {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "angle": box.angle,
        "confidence": box.confidence,
        "component_id": box.component_id,
        "class_id": box.class_id,
        "component_type": box.component_type,
        "component_file": box.component_file,
        "shm_index": box.shm_index,
        "metadata": dict(box.metadata),
    }


def algorithm_result_to_legacy_data(result: AlgorithmResult) -> List[Dict[str, Any]]:
    """生产检测 worker 兼容：导出 parts/components 为 dict 列表。"""
    if result.result_type == ResultType.PARTS:
        return [part_box_to_dict(p) for p in result.parts]
    if result.result_type == ResultType.COMPONENT:
        return [component_box_to_dict(c) for c in result.components]
    if result.metadata:
        return [dict(result.metadata)]
    return []
