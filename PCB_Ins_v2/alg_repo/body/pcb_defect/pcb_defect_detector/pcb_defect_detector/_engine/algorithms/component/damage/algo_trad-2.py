from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (to_gray, ssim_diff, body_mask, largest_contour,
    align_by_pose, roi_compare_metrics, ncc_score,
    run_per_component_detect)


class ComponentDamageTradAlg(ComponentPartyAlg):
    """破损检测算法。"""

    algorithm_code = "component_damage"
    version = 1
    SUPPORT_DEFECTS = ['damage']
    description = "SSIM差异+轮廓比对, 先精对齐再判破损 (移植自 d7_damage)"
    principle = "align_by_pose | ssim_diff(threshold=0.35) | blob_area≥30 | matchShapes>0.30"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "ssim_diff_thresh": 0.35,
            "blob_area_min": 30,
            "shape_match_max": 0.30,
            "morph_ksize": 3,
            "ncc_fallback_max": 1.0,
            "skip_structural_hint": True,
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
            s_th = float(getattr(legacy_cfg, "ssim_diff_thresh", 0.35))
            blob_min_full = int(getattr(legacy_cfg, "blob_area_min", 30))
            shape_max = float(getattr(legacy_cfg, "shape_match_max", 0.30))
            morph_ks = int(getattr(legacy_cfg, "morph_ksize", 3))
            ncc_fb_max = float(getattr(legacy_cfg, "ncc_fallback_max", 1.0))
            skip_hint = bool(getattr(legacy_cfg, "skip_structural_hint", True))
            # 全局降采样时，面积阈值（px²）按缩放因子平方换算到检测分辨率
            proc_scale = float(getattr(legacy_cfg, "_proc_scale", 1.0)) or 1.0
            blob_min = max(1, int(round(blob_min_full * proc_scale * proc_scale)))

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                aligned = align_by_pose(test_img, gold_tpl)
                metrics = roi_compare_metrics(gold_tpl, aligned)
                skip_by_struct = False
                if skip_hint:
                    from algorithms.component.common import roi_structural_defect_hint
                    skip_by_struct = roi_structural_defect_hint(metrics) is not None
                tpl_ncc = ncc_score(aligned, gold_tpl)
                g_gray = to_gray(gold_tpl); t_gray = to_gray(aligned)
                _, diff_map = ssim_diff(g_gray, t_gray)
                diff_bin = (diff_map > s_th).astype(np.uint8) * 255
                diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, np.ones((morph_ks, morph_ks), np.uint8))
                cnts, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_blob_area = max((cv2.contourArea(c) for c in cnts), default=0.0)
                max_blob = max(cnts, key=cv2.contourArea) if cnts else None
                gm = body_mask(gold_tpl); tm = body_mask(aligned)
                cg = largest_contour(gm); ct = largest_contour(tm)
                shape_dist = float(cv2.matchShapes(cg, ct, cv2.CONTOURS_MATCH_I1, 0.0)) \
                    if cg is not None and ct is not None else 0.0
                roi_area = rh * rw
                if max_blob_area > roi_area * 0.35:
                    is_damage = shape_dist > shape_max
                elif max_blob_area < roi_area * 0.6:
                    is_damage = (max_blob_area > blob_min or shape_dist > shape_max * 1.5) and not skip_by_struct
                else:
                    is_damage = (max_blob_area > blob_min or shape_dist > shape_max) and not skip_by_struct
                if not is_damage and tpl_ncc < ncc_fb_max:
                    is_damage = True
                defects = []
                if is_damage:
                    score = min(1.0, max_blob_area / (rh * rw * 0.01) + shape_dist)
                    if max_blob is not None and max_blob_area > blob_min:
                        x, y, bw, bh = cv2.boundingRect(max_blob)
                        dm = np.zeros((rh, rw), np.uint8); cv2.drawContours(dm, [max_blob], -1, 255, cv2.FILLED)
                        defects.append(DefectInfo("damage", BoundingBox(float(x), float(y), float(bw), float(bh)), score, 0.5,
                            mask=dm, description=f"破损: blob={max_blob_area:.0f}px shape={shape_dist:.3f}",
                            extra_data={"max_blob_area": max_blob_area, "shape_dist": shape_dist}))
                    else:
                        defects.append(DefectInfo("damage", BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.5,
                            description=f"破损: shape_dist={shape_dist:.3f}",
                            extra_data={"max_blob_area": max_blob_area, "shape_dist": shape_dist}))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
