from typing import Any, Dict, List

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.interfaces import AlgorithmResult as LegacyResult
from core.models import DefectInfo, BoundingBox
from core.utils.alg_config import merge_config
from algorithms.component.adapter_shell import ComponentPartyAlg, run_party_shell
from algorithms.component.common import (to_gray, body_mask, roi_compare_metrics,
    roi_shift_offset, expand_box, clamp_box,
    run_per_component_detect, component_body_bbox,
    ncc_score)


def _dark_body_mask(img):
    """暗色元件本体掩膜（与 common.smt_dark_body_bbox 一致，覆盖黑电阻/黑 IC）"""
    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dark = ((hsv[:, :, 2] < 95) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    else:
        dark = (to_gray(img) < 95).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return dark


def _is_dark_component(golden_roi):
    """在本体区域内判断是否为暗色元件（避免整幅 FOV 绿底干扰）"""
    x, y, bw, bh = component_body_bbox(golden_roi, golden_roi)
    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
    sub = golden_roi[y:y + bh, x:x + bw] if bw > 0 and bh > 0 else golden_roi
    gm = body_mask(sub)
    gd = _dark_body_mask(sub)
    n_bright = cv2.countNonZero(gm)
    n_dark = cv2.countNonZero(gd)
    return n_dark > max(n_bright * 0.45, 60)


def _cfg_float(config, key, default):
    if config is None:
        return float(default)
    try:
        return float(getattr(config, key))
    except AttributeError:
        return float(default)


def _body_presence(body_ratio, body_ratio_dark_raw, is_dark):
    """综合亮/暗通道估计本体保留比例（>1 表示待检图本体不少于模板）"""
    dark_eff = min(float(body_ratio_dark_raw), 1.5)
    bright_eff = min(float(body_ratio), 1.5)
    if is_dark:
        if dark_eff < 0.35:
            return dark_eff
        return max(dark_eff, bright_eff * 0.40)
    return max(bright_eff, dark_eff * 0.55)


class ComponentMissingTradAlg(ComponentPartyAlg):
    """缺件检测算法。"""

    algorithm_code = "component_missing"
    version = 1
    ALG_VERSION = "missing_v5"
    SUPPORT_DEFECTS = ['missing']
    description = "本体像素比+差分+边缘联合判定 (移植自 d2_missing)"
    principle = "暗本体充足/ presence≥0.60→OK；否则 hard: presence<0.30 或 大差分+少边缘"

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            "body_ratio_min": 0.30,
            "presence_ok_min": 0.60,
            "diff_thresh": 40,
            "diff_score_max": 0.50,
            "edge_ratio_min": 0.05,
            "roi_body_ratio_max": 0.55,
            "roi_diff_min": 0.28,
            "tpl_ncc_missing_max": 1.0,
            "diff_score_missing_min": 1.0,
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
            p = {k: _cfg_float(legacy_cfg, k, d) for k, d in [
                ("body_ratio_min", 0.30), ("presence_ok_min", 0.60),
                ("diff_score_max", 0.50), ("edge_ratio_min", 0.05),
                ("roi_body_ratio_max", 0.55), ("roi_diff_min", 0.28),
                ("tpl_ncc_missing_max", 1.0), ("diff_score_missing_min", 1.0),
            ]}
            p["diff_thresh"] = int(_cfg_float(legacy_cfg, "diff_thresh", 40))

            def detect_roi(test_img, gold_tpl):
                h_roi, w_roi = test_img.shape[:2]
                ncc_search_margin = 30
                box = (0, 0, w_roi, h_roi)
                rx, ry, rw, rh = clamp_box(box, gold_tpl.shape)
                golden_roi = gold_tpl[ry:ry + rh, rx:rx + rw]
                sx, sy, sw, sh = expand_box(box, ncc_search_margin, test_img.shape)
                search = test_img[sy:sy + sh, sx:sx + sw]
                dx_ncc, dy_ncc = 0.0, 0.0
                if search.shape[0] >= golden_roi.shape[0] and search.shape[1] >= golden_roi.shape[1]:
                    dx_ncc, dy_ncc = roi_shift_offset(golden_roi, search, margin=ncc_search_margin)
                aligned_x = int(sx + dx_ncc)
                aligned_y = int(sy + dy_ncc)
                aligned_x = max(0, min(aligned_x, w_roi - rw))
                aligned_y = max(0, min(aligned_y, h_roi - rh))
                test_roi = test_img[aligned_y:aligned_y + rh, aligned_x:aligned_x + rw]
                if test_roi.shape[:2] != golden_roi.shape[:2]:
                    test_roi = cv2.resize(test_img, (rw, rh))
                metrics = roi_compare_metrics(golden_roi, test_roi, diff_thresh=p["diff_thresh"])
                body_ratio = metrics["body_ratio"]; diff_score = metrics["diff_score"]
                g_gray = to_gray(golden_roi); t_gray = to_gray(test_roi)
                edges = cv2.Canny(t_gray, 40, 120); edge_ratio = cv2.countNonZero(edges) / max(1, edges.size)
                bm = body_mask(test_roi)
                gm_dark = _dark_body_mask(golden_roi)
                tm_dark = _dark_body_mask(test_roi)
                tmpl_dark = max(1, cv2.countNonZero(gm_dark))
                body_ratio_dark_raw = cv2.countNonZero(tm_dark) / tmpl_dark
                body_ratio_dark = min(body_ratio_dark_raw, 2.0)

                is_dark = _is_dark_component(golden_roi)
                presence = _body_presence(body_ratio, body_ratio_dark_raw, is_dark)
                tpl_ncc = float(metrics.get("ncc", ncc_score(test_roi, golden_roi)))

                reject_reason = None
                is_missing = False

                if tpl_ncc < p["tpl_ncc_missing_max"] or diff_score > p["diff_score_missing_min"]:
                    is_missing = True
                    reject_reason = "fallback_ncc_diff"
                elif is_dark and body_ratio_dark_raw >= 0.65:
                    is_missing = False
                elif body_ratio_dark_raw >= 0.85:
                    is_missing = False
                elif presence >= p["presence_ok_min"]:
                    is_missing = False
                elif body_ratio >= 0.55 and body_ratio_dark_raw >= 0.40:
                    is_missing = False
                elif tpl_ncc >= 0.75 and body_ratio >= 0.50 and diff_score < 0.30:
                    is_missing = False
                elif presence >= 0.50 and diff_score < p["roi_diff_min"]:
                    is_missing = False
                elif tpl_ncc >= 0.82 and presence >= 0.52:
                    is_missing = False
                else:
                    hard_missing = presence < p["body_ratio_min"]
                    hard_missing = hard_missing or (
                        diff_score > p["diff_score_max"] and edge_ratio < p["edge_ratio_min"]
                    )
                    soft_missing = (
                        presence < p["roi_body_ratio_max"] and diff_score > p["roi_diff_min"]
                    )
                    is_missing = hard_missing or soft_missing
                    if hard_missing:
                        reject_reason = "hard"
                    elif soft_missing:
                        reject_reason = "soft_roi"

                defects = []
                if is_missing:
                    score = 1.0 - min(presence, 1.0)
                    dm = cv2.bitwise_not(body_mask(test_roi)) if cv2.countNonZero(body_mask(test_roi)) < 100 else np.zeros((rh, rw), np.uint8)
                    defects.append(DefectInfo("missing",
                        BoundingBox(0.0, 0.0, float(w_roi), float(h_roi)),
                        score, 0.7, mask=dm,
                        description=(
                            f"缺件({reject_reason}): pres={presence:.3f} body_l={body_ratio:.3f} "
                            f"body_d={body_ratio_dark_raw:.3f} diff={diff_score:.3f} "
                            f"edge={edge_ratio:.4f} ncc={tpl_ncc:.3f} dark={is_dark} "
                            f"shift=({dx_ncc:.1f},{dy_ncc:.1f})"
                        ),
                        extra_data={
                            "body_ratio": body_ratio, "body_ratio_dark": body_ratio_dark_raw,
                            "presence": presence, "primary_ratio": presence,
                            "is_dark_component": is_dark, "tpl_ncc": tpl_ncc,
                            "diff_score": diff_score, "edge_ratio": edge_ratio,
                            "reject_reason": reject_reason,
                            "ncc_dx": dx_ncc, "ncc_dy": dy_ncc,
                        }))
                return LegacyResult(status="NG" if defects else "OK",
                                    defects=defects, processing_time_ms=0)

            result = run_per_component_detect(detect_roi, image_bgr, template_bgr)
            meta = dict(result.metadata or {})
            meta["algo_version"] = self.ALG_VERSION
            result.metadata = meta
            return result

        return run_party_shell(
            self.algorithm_code, self._defaults, image, config,
            original_template_image, detect_core, merge_config_fn=merge_config)
