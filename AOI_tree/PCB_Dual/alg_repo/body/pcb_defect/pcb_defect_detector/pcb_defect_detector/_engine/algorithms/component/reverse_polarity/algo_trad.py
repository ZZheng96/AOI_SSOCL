from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import run_per_component_detect, polarity_detect_all, ncc_score


_TRIGGER_LABEL = {
    "silkscreen": "丝印图案",
    "color_band": "极性色带",
    "diode_die": "灯芯形状",
    "dark_mark": "暗极性标记",
    "ncc_fallback": "模板NCC回退",
    "roi_structure_180": "ROI结构180°匹配",
}


class ComponentReversePolarityTradAlg(ComponentPartyAlg):
    """极反检测算法。"""

    algorithm_code = "component_reverse_polarity"
    version = 1
    SUPPORT_DEFECTS = ['reverse_polarity']
    description = "丝印0°/180°+色带侧别+二极管灯芯 三路极反 (无OCR)"
    principle = "丝印图案NCC | 较宽色带侧别 | 二极管灯芯形状/位置 | 暗标记辅助"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "ncc_diff_min": 0.06,
            "mark_dark_ratio": 0.30,
            "band_side_tol": 0.12,
            "body_ncc_max": 1.0,
            # 四路判据开关：默认只开丝印一路（最快）。色带/灯芯/暗标记按需开启。
            # 按需开启方法：在本算法实例配置 JSON 的 defaultParam 里把对应项置 true（持久生效），
            # 或调用 run(image, config={...}) 时在 config 传入对应项（单次覆盖）。
            "enable_silkscreen": True,
            "enable_color_band": False,
            "enable_diode": False,
            "enable_dark_mark": False,
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
            ncc_min = float(getattr(legacy_cfg, "ncc_diff_min", 0.06))
            mark_ratio = float(getattr(legacy_cfg, "mark_dark_ratio", 0.30))
            band_tol = float(getattr(legacy_cfg, "band_side_tol", 0.12))
            en_silk = bool(getattr(legacy_cfg, "enable_silkscreen", True))
            en_band = bool(getattr(legacy_cfg, "enable_color_band", False))
            en_diode = bool(getattr(legacy_cfg, "enable_diode", False))
            en_dark = bool(getattr(legacy_cfg, "enable_dark_mark", False))
            body_ncc_max = float(getattr(legacy_cfg, "body_ncc_max", 1.0))

            def detect_roi(test_img, gold_tpl):
                rw, rh = test_img.shape[1], test_img.shape[0]
                pol = polarity_detect_all(
                    test_img, gold_tpl,
                    ncc_diff_min=ncc_min,
                    mark_ratio=mark_ratio,
                    band_side_tol=band_tol,
                    enable_silkscreen=en_silk,
                    enable_color_band=en_band,
                    enable_diode=en_diode,
                    enable_dark_mark=en_dark,
                )
                roi_match = getattr(legacy_cfg, "_roi_match", None)
                if (isinstance(roi_match, dict)
                        and roi_match.get("matched")
                        and roi_match.get("orientation_confident")
                        and roi_match.get("reversed_180")):
                    # 结构匹配必须同时满足：180°候选本身够像、且显著优于0°。
                    # 这是独立于丝印/色带等局部判据的一路可靠证据。
                    pol = dict(pol)
                    pol["is_reversed"] = True
                    triggers = list(pol.get("triggers", []))
                    if "roi_structure_180" not in triggers:
                        triggers.append("roi_structure_180")
                    pol["triggers"] = triggers
                    orientation_gap = (
                        float(roi_match.get("reverse_score", 0.0))
                        - float(roi_match.get("normal_score", 0.0)))
                    pol["score_hint"] = max(
                        float(pol.get("score_hint", 0.0)),
                        orientation_gap,
                        float(roi_match.get("confidence", 0.0)) * ncc_min * 2.0)
                if not pol["is_reversed"] and body_ncc_max < 1.0:
                    tpl_ncc = ncc_score(test_img, gold_tpl)
                    if tpl_ncc < body_ncc_max:
                        # 仅"和模板整体比对分数低"不能作为极反证据——欠曝、未对齐、
                        # 破损、脏污等都会让 NCC 变低。必须像丝印判据一样，证明
                        # "把模板转 180° 后确实更像待检图"，才认定是极反，
                        # 避免把这些无关问题误判成极反。
                        tpl_ncc_flip = ncc_score(test_img, cv2.rotate(gold_tpl, cv2.ROTATE_180))
                        if tpl_ncc_flip > tpl_ncc + max(0.10, ncc_min * 1.5) and tpl_ncc_flip > 0.35:
                            pol = dict(pol)
                            pol["is_reversed"] = True
                            pol["triggers"] = list(pol.get("triggers", [])) + ["ncc_fallback"]
                            pol["score_hint"] = max(pol.get("score_hint", 0.0), tpl_ncc_flip - tpl_ncc)
                defects = []
                if pol["is_reversed"]:
                    score = max(min(1.0, pol["score_hint"] / max(ncc_min * 2, 0.01)), 0.3)
                    trigger_txt = "+".join(_TRIGGER_LABEL.get(t, t) for t in pol["triggers"])
                    silk = pol["silkscreen"]
                    band = pol["color_band"]
                    diode = pol["diode_die"]
                    defects.append(DefectInfo(
                        "reverse_polarity", BoundingBox(0.0, 0.0, float(rw), float(rh)), score, 0.5,
                        description=(
                            f"极反[{trigger_txt}]: silk_d={silk.get('silk_diff', 0):.3f} "
                            f"band_g={band.get('gold_band_side', 0):.1f} t={band.get('test_band_side', 0):.1f} "
                            f"die_d={diode.get('die_diff', 0):.3f}"
                        ),
                        extra_data={
                            "triggers": pol["triggers"],
                            "reversed_by_silkscreen": silk.get("reversed_by_silkscreen"),
                            "silk_diff": silk.get("silk_diff"),
                            "has_color_band": band.get("has_band"),
                            "reversed_by_band": band.get("reversed_by_band"),
                            "gold_band_side": band.get("gold_band_side"),
                            "test_band_side": band.get("test_band_side"),
                            "band_strength": band.get("band_strength"),
                            "is_diode": diode.get("is_diode"),
                            "reversed_by_die": diode.get("reversed_by_die"),
                            "die_diff": diode.get("die_diff"),
                            "gold_die_rel_x": diode.get("gold_die_rel_x"),
                            "test_die_rel_x": diode.get("test_die_rel_x"),
                            "reversed_by_dark": pol.get("reversed_by_dark"),
                            "mark_dark_g": pol.get("mark_dark_g"),
                            "mark_dark_t": pol.get("mark_dark_t"),
                        },
                    ))
                return LegacyResult(
                    status="NG" if defects else "OK",
                    defects=defects,
                    processing_time_ms=0,
                    metadata={"polarity_checks": pol},
                )

            return run_per_component_detect(detect_roi, image_bgr, template_bgr)

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
