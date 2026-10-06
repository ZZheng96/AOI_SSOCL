from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (body_mask, min_area_pose, hist_correlation,
    largest_contour, roi_compare_metrics, roi_structural_defect_hint,
    run_per_component_detect)


class ComponentWrongTradAlg(ComponentPartyAlg):
    """错件检测算法。

    algorithm_code 与实例配置 JSON 的 algorithmCode 一致。
    """

    algorithm_code = "component_wrong_part"
    version = 1
    SUPPORT_DEFECTS = ['wrong_part']
    description = "尺寸+颜色+轮廓三特征2/3投票 (移植自 d1_wrong_part)"
    principle = "body_mask | min_area_pose | 长边偏差>20% | HSV直方图<0.80 | matchShapes>0.25"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "size_tol_ratio": 0.20,
            "hist_corr_min": 0.80,
            "shape_match_max": 0.25,
            "vote_fail_min": 2,
            # 结构跳过功能已下线：不再暴露给 UI，也不允许通过配置打开。
            "skip_structural_hint": False,
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
            size_tol = float(getattr(legacy_cfg, "size_tol_ratio", 0.20))
            hc_min = float(getattr(legacy_cfg, "hist_corr_min", 0.80))
            shape_max = float(getattr(legacy_cfg, "shape_match_max", 0.25))
            vote_min = int(getattr(legacy_cfg, "vote_fail_min", 2))
            # 结构跳过功能已强制关闭：无论配置/UI 传入什么值都不启用，
            # 始终按错件判据正常投票，不再直接判 OK 跳过。
            skip_hint = False

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                metrics = roi_compare_metrics(gold_tpl, test_img)
                if skip_hint:
                    hint = roi_structural_defect_hint(metrics)
                    if hint:
                        return LegacyResult(status="OK", defects=[], processing_time_ms=0,
                                            metadata={"skip": hint})
                gm = body_mask(gold_tpl); tm = body_mask(test_img)
                gp = min_area_pose(gm); tp = min_area_pose(tm)
                size_dev = 0.0
                if gp and tp:
                    gl, gs = max(gp[1]), min(gp[1]); tl, ts = max(tp[1]), min(tp[1])
                    dl = abs(tl - gl) / gl if gl > 1e-6 else 0.0
                    ds = abs(ts - gs) / gs if gs > 1e-6 else 0.0
                    size_dev = max(dl, ds)
                hist_corr = hist_correlation(test_img, gold_tpl)
                cg = largest_contour(gm); ct = largest_contour(tm)
                shape_dist = float(cv2.matchShapes(cg, ct, cv2.CONTOURS_MATCH_I1, 0.0)) \
                    if cg is not None and ct is not None else 0.0
                fail = (1 if size_dev > size_tol else 0) + \
                       (1 if hist_corr < hc_min else 0) + \
                       (1 if shape_dist > shape_max else 0)
                defects = []
                if fail >= vote_min:
                    score = min(1.0, size_dev * 1.5 + (1.0 - hist_corr) * 0.5 + shape_dist)
                    defects.append(DefectInfo("wrong_part",
                        BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.5,
                        description=f"错件: size={size_dev:.3f} hist={hist_corr:.3f} shape={shape_dist:.3f} fail={fail}",
                        extra_data={"size_dev": size_dev, "hist_corr": hist_corr,
                                    "shape_dist": shape_dist, "fail_features": fail}))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
