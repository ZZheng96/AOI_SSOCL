"""质检门禁：标准图/测试图尺寸差异检查 + 自动缩放对齐。

只做"图像尺寸是否可比"的粗判和整体缩放，不涉及亮度/清晰度检查
（按用户要求已去掉），也不涉及各算法内部自己的像素级精细配准（NCC/ECC/
相位相关等，那些仍由各算法适配器自行处理，本模块不影响）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import get_gate_settings


@dataclass
class GateOutcome:
    blocked: bool
    block_message: str = ""
    aligned_test: np.ndarray | None = None
    note: str = ""


def run_gate(image_std: np.ndarray | None, image_test: np.ndarray | None) -> GateOutcome:
    if image_std is None or image_std.size == 0:
        return GateOutcome(blocked=True, block_message="标准图为空或无法读取，已拦截")
    if image_test is None or image_test.size == 0:
        return GateOutcome(blocked=True, block_message="测试图为空或无法读取，已拦截")

    settings = get_gate_settings()
    if not settings["gate_enabled"]:
        return GateOutcome(blocked=False, aligned_test=image_test)

    h1, w1 = image_std.shape[:2]
    h2, w2 = image_test.shape[:2]
    if h1 == h2 and w1 == w2:
        return GateOutcome(blocked=False, aligned_test=image_test)

    max_diff = max(0, int(settings["size_gate_max_diff_px"]))
    dh, dw = abs(h1 - h2), abs(w1 - w2)

    if dh > max_diff or dw > max_diff:
        msg = (
            f"标准图尺寸 {w1}x{h1} 与测试图尺寸 {w2}x{h2} 差异过大"
            f"（高差{dh}px/宽差{dw}px，超过门禁阈值 {max_diff}px），已拦截，未执行检测"
        )
        return GateOutcome(blocked=True, block_message=msg)

    shrinking = (w2 * h2) > (w1 * h1)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    aligned = cv2.resize(image_test, (w1, h1), interpolation=interp)
    note = (
        f"标准图/测试图尺寸存在 {max(dh, dw)}px 差异（≤ 门禁阈值 {max_diff}px），"
        f"已自动将测试图缩放对齐为 {w1}x{h1}"
    )
    return GateOutcome(blocked=False, aligned_test=aligned, note=note)
