from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (estimate_roi_pose, ncc_score,
    run_per_component_detect, is_probable_polarity_reversal)


class ComponentShiftTradAlg(ComponentPartyAlg):
    """移位检测算法。"""

    algorithm_code = "component_shift"
    version = 1
    SUPPORT_DEFECTS = ['shift']
    description = "ROI位姿估计(PCA+边缘角度)检测移位 (移植自 d4_shift)"
    principle = "estimate_roi_pose | dx/dy超公差(0.30mm×20=6px) | theta>10°"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "pos_tol_mm": 0.30,
            "angle_tol_deg": 10.0,
            # 极反预检模式：移位前排除 180° 翻转误判。
            #   "light"=只跑丝印单路（默认，最快）；"full"=四路完整；"off"=不预检（最快但可能把翻转误判为移位）
            "polarity_precheck": "light",
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
            tol_mm = float(getattr(legacy_cfg, "pos_tol_mm", 0.30))
            ang_tol = float(getattr(legacy_cfg, "angle_tol_deg", 10.0))
            pos_tol_full = tol_mm * 20.0  # px_per_mm=20（原图分辨率下的像素公差）
            # 若开启全局降采样，检测分辨率下的位移阈值需按比例缩小，保持等效物理公差
            proc_scale = float(getattr(legacy_cfg, "_proc_scale", 1.0)) or 1.0
            pos_tol = pos_tol_full * proc_scale
            precheck = str(getattr(legacy_cfg, "polarity_precheck", "light")).lower()
            if precheck == "full":
                pol_flags = dict(enable_silkscreen=True, enable_color_band=True,
                                 enable_diode=True, enable_dark_mark=True)
            else:  # light：只跑丝印单路
                pol_flags = dict(enable_silkscreen=True, enable_color_band=False,
                                 enable_diode=False, enable_dark_mark=False)

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                if precheck != "off" and is_probable_polarity_reversal(
                        test_img, gold_tpl, ncc_diff_min=0.06, **pol_flags):
                    return LegacyResult(
                        status="OK", defects=[], processing_time_ms=0,
                        metadata={"skip": "疑似极反，跳过移位判定"},
                    )
                dx, dy, theta = estimate_roi_pose(gold_tpl, test_img)
                is_shift = abs(dx) > pos_tol or abs(dy) > pos_tol or abs(theta) > ang_tol
                defects = []
                if is_shift:
                    score = min(1.0, max(abs(dx), abs(dy)) / max(pos_tol * 3, 1) + abs(theta) / max(ang_tol * 2, 1))
                    defects.append(DefectInfo("shift", BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.6,
                        description=f"移位: dx={dx:.1f}px dy={dy:.1f}px theta={theta:.1f}°",
                        extra_data={"dx_px": dx, "dy_px": dy, "theta_deg": theta, "pos_tol_px": pos_tol,
                                    "ncc": ncc_score(test_img, gold_tpl)}))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
