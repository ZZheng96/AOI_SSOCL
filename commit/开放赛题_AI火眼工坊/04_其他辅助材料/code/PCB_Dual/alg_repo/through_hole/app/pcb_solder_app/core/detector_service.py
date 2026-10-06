"""检测编排服务：封装 ``PcbThroughHoleSolderAlg``，供界面调用"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .. import config as app_config
from .models import DefectBox, DetectionResult, TemplateConfig


class DetectorService:
    def __init__(self):
        app_config.ensure_detect_on_path()
        # 延迟到路径注入后再导入，避免模块加载期 import 失败
        from algorithms.pcb_through_hole.pcb_through_hole_solder_alg import (  # noqa: WPS433
            PcbThroughHoleSolderAlg,
        )

        self._alg = PcbThroughHoleSolderAlg()

    # ---- API ----
    def prebuild_template(self, template_bgr: np.ndarray, cfg: TemplateConfig) -> Dict[str, Any]:
        """标准图 + 当前锡面获取方式 -> 立即生成/复用一次锡面 Mask，不需要测试图。"""
        return self._alg.prebuild_template(template_bgr, cfg.to_algorithm_config())

    def detect(
        self, template_bgr: np.ndarray, test_bgr: np.ndarray, cfg: TemplateConfig,
    ) -> DetectionResult:
        result = self._alg.run(
            test_bgr, cfg.to_algorithm_config(), original_template_image=template_bgr)
        if result.code != 0:
            return DetectionResult(ok=False, status="ERROR", error=result.message)

        meta = result.metadata or {}
        defects = [
            DefectBox(
                defect_id=int(p.class_id) if p.class_id is not None else -1,
                label=str(p.label or ""),
                x=int(p.x), y=int(p.y), width=int(p.width), height=int(p.height),
                confidence=float(p.confidence),
                reason=str((p.metadata or {}).get("reason", "")),
            )
            for p in result.parts
        ]
        return DetectionResult(
            ok=True,
            status=str(meta.get("status", "")),
            defect_ids=[int(x) for x in (meta.get("defect_ids") or [])],
            defects=defects,
            coverage=meta.get("coverage"),
            review_reason=str(meta.get("review_reason", "") or ""),
            align_method=str(meta.get("align_method", "") or ""),
            template_id=str(meta.get("template_id", "") or ""),
            cost_ms=float(result.cost_time) * 1000.0,
            output_image=meta.get("output_image"),
        )

    def preview_test_roi(
        self, template_bgr: np.ndarray, test_bgr: np.ndarray, cfg: TemplateConfig,
    ) -> Dict[str, Any]:
        """标准图 ROI 配准映射到测试图坐标的实时预览（不跑完整检测）。

        返回 dict：`{"ok": True, "roi": {...}}` 或 `{"ok": False, "error": str}`，
        `roi` 是与 `ManualRoi.to_engine_dict()` 同结构的字典 (测试图原始像素坐标)。
        """
        return self._alg.preview_roi(
            test_bgr, cfg.to_algorithm_config(), original_template_image=template_bgr)

    def clear_template_cache(self) -> None:
        """标准图/锡面获取方式发生实质变化（如替换了同名但内容不同的标准图）时，
        用来强制清空缓存，避免误用旧 Mask。"""
        self._alg.clear_template_cache()
