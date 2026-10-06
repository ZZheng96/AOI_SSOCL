"""仅供本仓库单元测试使用；接入检测系统工程时使用工程自带的同名真实模块，不要拷贝本文件"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from core.entities.detect import BoundingBox
from core.entities.detect import AlgorithmResult
from core.utils.alg_config import build_error_result, safe_index


@dataclass
class AlgorithmRegister:
    """算法池注册项。"""
    algorithm_code: str
    algorithm_class: str = None
    description: str = ""
    config_path: str = None
    idle_timeout: int = 5
    timeout: Optional[float] = 5


class IDetectionAlgorithm(ABC):
    """
    检测算法标准接口。

    实现要求:
        - __init__(config_path): 从 JSON 加载默认参数
        - detect: 单图检测（池与 Worker 仅调用此方法）
    batch_detect 提供默认串行实现，支持 GPU 批推理的子类可覆盖。
    """

    algorithm_code: str = "IDetectionAlgorithm"
    version: int = 1

    @abstractmethod
    def __init__(self, config_path: str | None = None):
        self._config_path = config_path

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                param = json.load(f)
                self.algorithm_code = param.get("algorithmCode", self.algorithm_code)
                self.version = param.get("version", self.version)

                if "defaultParam" not in param:
                    self._defaults = self.default_config()
                else:
                    self._defaults = param.get("defaultParam", {})
        else:
            self._defaults = self.default_config()

    @abstractmethod
    def default_config(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox: Optional[BoundingBox] = None,
        original_template_image: Optional[np.ndarray] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> AlgorithmResult:
        raise NotImplementedError

    def batch_run(
        self,
        images: List[np.ndarray],
        configs: List[Dict[str, Any]],
        roi_bbox: Optional[List[BoundingBox]] = None,
        original_template_image: Optional[List[np.ndarray]] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> List[AlgorithmResult]:
        """默认串行批量：逐项调用 detect，单项失败不影响其他。"""
        results: List[AlgorithmResult] = []
        for i, img in enumerate(images):
            cfg = safe_index(configs, i, default={})
            bbox = safe_index(roi_bbox, i, default=None) if roi_bbox else None
            tpl = (
                safe_index(original_template_image, i, default=None)
                if original_template_image
                else None
            )
            multi = (
                safe_index(multiple_roi_images, i, default=None)
                if multiple_roi_images
                else None
            )
            try:
                results.append(
                    self.run(img, cfg, bbox, tpl, multi)
                )
            except Exception as exc:
                results.append(build_error_result(self.algorithm_code, exc))
        return results


def extern_image(roi_bbox: BoundingBox, image, extern_x=40, extern_y=40):
    if not roi_bbox:
        return image
    img_h, img_w = image.shape[:2]
    rw = roi_bbox.width + extern_x * 2
    rh = roi_bbox.height + extern_y * 2
    x = max(0, roi_bbox.x - extern_x)
    y = max(0, roi_bbox.y - extern_y)
    x2 = min(x + rw, img_w)
    y2 = min(y + rh, img_h)
    if x2 - x < rw:
        x = max(0, x2 - rw)
    if y2 - y < rh:
        y = max(0, y2 - rh)
    return image[y:y2, x:x2]
