from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (ncc_score, hist_correlation, roi_compare_metrics,
    roi_structural_defect_hint, run_per_component_detect)


class ComponentFlippedTradAlg(ComponentPartyAlg):
    """翻件检测算法。"""

    algorithm_code = "component_flipped"
    version = 1
    SUPPORT_DEFECTS = ['flipped']
    description = "NCC正面+HSV直方图 双特征2/2投票 (移植自 d6_flipped)"
    principle = "NCC(TM_CCOEFF_NORMED) | HSV(32×32) | vote_fail_min=2 | ncc<0.55 | hist<0.75"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "ncc_front_min": 0.55,
            "hist_corr_min": 0.75,
            "vote_fail_min": 2,
        }

    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox=None,
        original_template_image=None,
        multiple_roi_images=None,
    ):
        def detect_core(image_bgr, template_bgr, legacy_cfg):
            ncc_min = float(getattr(legacy_cfg, "ncc_front_min", 0.55))
            hc_min = float(getattr(legacy_cfg, "hist_corr_min", 0.75))
            vote_min = int(getattr(legacy_cfg, "vote_fail_min", 2))

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                metrics = roi_compare_metrics(gold_tpl, test_img)
                hint = roi_structural_defect_hint(metrics)
                if hint:
                    return LegacyResult(status="OK", defects=[], processing_time_ms=0,
                                        metadata={"skip": hint})
                ncc_v = ncc_score(test_img, gold_tpl)
                hist_v = hist_correlation(test_img, gold_tpl)
                fail = (0 if ncc_v >= ncc_min else 1) + (0 if hist_v >= hc_min else 1)
                defects = []
                if fail >= vote_min:
                    score = (1.0 - min(ncc_v / ncc_min, 1.0)) * 0.5 + (1.0 - min(hist_v / hc_min, 1.0)) * 0.5
                    defects.append(DefectInfo("flipped", BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.5,
                        description=f"翻件: ncc={ncc_v:.3f} hist={hist_v:.3f} fail={fail}",
                        extra_data={"ncc_front": ncc_v, "hist_corr": hist_v, "fail_votes": fail}))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
