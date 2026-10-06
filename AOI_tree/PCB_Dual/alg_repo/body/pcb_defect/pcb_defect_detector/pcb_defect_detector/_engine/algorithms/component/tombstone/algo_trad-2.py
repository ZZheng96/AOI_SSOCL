from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (roi_compare_metrics, infer_end_pad_boxes,
    end_pad_change_asymmetry, body_mask, min_area_pose, solder_area, crop,
    run_per_component_detect)


def _aspect_ratio(pose):
    if pose is None:
        return 0.0
    w, h = pose[1]; lo, hi = min(w, h), max(w, h)
    return hi / lo if lo > 1e-6 else 0.0


class ComponentTombstoneTradAlg(ComponentPartyAlg):
    """立碑检测算法。"""

    algorithm_code = "component_tombstone"
    version = 1
    SUPPORT_DEFECTS = ['tombstone']
    description = "aspect_dev+pad不对称+ROI联合三路判定 (移植自 d5_tombstone)"
    principle = "roi_compare_metrics | aspect_dev>0.30 | pad_area_ratio<0.85 | ROI联合"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "aspect_dev_ratio": 0.30,
            "pad_area_ratio_max": 0.85,
            "solder_thresh": 0,
            "roi_diff_min": 0.28,
            "roi_ncc_max": 0.12,
            "roi_body_ratio_min": 0.85,
            "roi_shape_min": 0.01,
            "roi_pad_change_max": 0.70,
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
            p = {
                "aspect_dev_ratio": float(getattr(legacy_cfg, "aspect_dev_ratio", 0.30)),
                "pad_area_ratio_max": float(getattr(legacy_cfg, "pad_area_ratio_max", 0.85)),
                "solder_thresh": int(getattr(legacy_cfg, "solder_thresh", 0)),
                "roi_diff_min": float(getattr(legacy_cfg, "roi_diff_min", 0.28)),
                "roi_ncc_max": float(getattr(legacy_cfg, "roi_ncc_max", 0.12)),
                "roi_body_ratio_min": float(getattr(legacy_cfg, "roi_body_ratio_min", 0.85)),
                "roi_shape_min": float(getattr(legacy_cfg, "roi_shape_min", 0.01)),
                "roi_pad_change_max": float(getattr(legacy_cfg, "roi_pad_change_max", 0.70)),
            }

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                metrics = roi_compare_metrics(gold_tpl, test_img, diff_thresh=40)
                bm_img = body_mask(test_img)
                bm_tpl = body_mask(gold_tpl)
                t_ar = _aspect_ratio(min_area_pose(bm_img))
                g_ar = _aspect_ratio(min_area_pose(bm_tpl))
                ar_dev = metrics["aspect_dev"] if metrics["aspect_dev"] else (
                    abs(t_ar - g_ar) / g_ar if g_ar > 1e-6 else 0.0)
                aspect_abnormal = ar_dev > p["aspect_dev_ratio"]
                pads = infer_end_pad_boxes(gold_tpl)
                pad_ratio = 1.0
                pad_asymmetric = False
                pad_change_asym = 1.0
                if len(pads) >= 2:
                    a1 = solder_area(crop(test_img, pads[0]), p["solder_thresh"])
                    a2 = solder_area(crop(test_img, pads[1]), p["solder_thresh"])
                    hi = max(a1, a2)
                    pad_ratio = (min(a1, a2) / hi) if hi > 0 else 1.0
                    pad_asymmetric = pad_ratio < p["pad_area_ratio_max"]
                    pad_change_asym, _, _ = end_pad_change_asymmetry(
                        gold_tpl, test_img, pads, solder_thresh=p["solder_thresh"])
                is_tombstone = aspect_abnormal and pad_asymmetric
                if not is_tombstone:
                    roi_sig = (
                        metrics["body_ratio"] >= p["roi_body_ratio_min"]
                        and metrics["diff_score"] >= p["roi_diff_min"]
                        and metrics["ncc"] <= p["roi_ncc_max"]
                        and p["roi_shape_min"] <= metrics["shape_dist"] <= 1.0
                        and ar_dev <= p["aspect_dev_ratio"]
                    )
                    roi_pad = pad_asymmetric or pad_change_asym <= p["roi_pad_change_max"]
                    is_tombstone = roi_sig and roi_pad
                defects = []
                if is_tombstone:
                    score = min(1.0, ar_dev * 1.5 + (1.0 - pad_ratio) * 1.5 + metrics["diff_score"] * 0.5)
                    defects.append(DefectInfo("tombstone", BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.55,
                        description=f"立碑: aspect_dev={ar_dev:.3f} pad_ratio={pad_ratio:.3f} diff={metrics['diff_score']:.3f}",
                        extra_data={"aspect_dev": ar_dev, "pad_area_ratio": pad_ratio, "roi_diff": metrics["diff_score"],
                                    "roi_ncc": metrics["ncc"], "shape_dist": metrics["shape_dist"],
                                    "pad_change_asym": pad_change_asym, "body_ratio": metrics["body_ratio"]}))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
