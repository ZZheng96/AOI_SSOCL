from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from .align import AlignCache, build_align_cache
from .bridge_joints import BridgeJoint, pad_joints
from .defect_mode import effective_enabled_defects
from .features import extract, PadFeatures
from .segment import (
    compute_masks,
    compute_masks_from_solder,
    detect_main_roi,
    roi_from_mask,
    Masks,
    PadROI,
)
from .nolead import StandardProfile, build_profile
from .void import VoidStdCache, build_void_std_cache


_MIN_ANNOTATED_MASK_PX = 30


def _binarize_annotated_mask(roi_mask: np.ndarray | None) -> np.ndarray | None:
    """校验并二值化已标注锡面 mask；像素数过少 (损坏/空标注) 时视为不可用。"""
    if roi_mask is None:
        return None
    _, mask_bin = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask_bin) < _MIN_ANNOTATED_MASK_PX:
        return None
    return mask_bin


@dataclass
class TemplateModel:
    templ_id: str = ""
    bgr: np.ndarray | None = None
    masks: Masks | None = None
    roi: PadROI | None = None
    feat: PadFeatures | None = None
    profile: StandardProfile | None = None
    align_cache: AlignCache | None = None
    void_cache: VoidStdCache | None = None
    enabled_defects: list = field(default_factory=list)
    bridge_joints: list = field(default_factory=list)

    @staticmethod
    def build(std_bgr: np.ndarray, roi_mask: np.ndarray | None = None,
              templ_id: str = "",
              bridge_joints: list | None = None) -> "TemplateModel":
        """构建模板模型。

        """
        joints: list[BridgeJoint] = list(bridge_joints or [])
        max_dim = getattr(C, "WORK_MAX_DIM", None)
        if max_dim:
            h, w = std_bgr.shape[:2]
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                nw, nh = int(round(w * s)), int(round(h * s))
                sx, sy = nw / max(1, w), nh / max(1, h)
                std_bgr = cv2.resize(std_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                if roi_mask is not None:
                    roi_mask = cv2.resize(
                        roi_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                joints = [j.scaled(sx, sy) for j in joints]

        annotated = _binarize_annotated_mask(roi_mask)

        if annotated is not None:
            # 复用标注锡面 Mask；ROI 取最大连通域（多框刚性配准在 pipeline 主焊盘路径）
            roi = roi_from_mask(annotated)
            masks = compute_masks_from_solder(std_bgr, annotated)
            profile = build_profile(std_bgr, roi, masks)
        else:
            masks0 = compute_masks(std_bgr)
            roi = detect_main_roi(std_bgr)
            profile0 = build_profile(std_bgr, roi, masks0)
            roi_rect = roi.rect
            masks = compute_masks(
                std_bgr,
                profile0.solder_hsv_low,
                profile0.solder_hsv_high,
                roi_rect=roi_rect,
            )
            profile = build_profile(std_bgr, roi, masks)
            masks = compute_masks(
                std_bgr,
                profile.solder_hsv_low,
                profile.solder_hsv_high,
                roi_rect=roi_rect,
                ref_stats=profile.solder_stats,
            )

        feat = extract(std_bgr, roi, masks)
        ac = build_align_cache(std_bgr, masks.solder, roi.rect)
        vc = build_void_std_cache(roi, masks)
        n_pads = len(pad_joints(joints))
        return TemplateModel(
            templ_id=templ_id, bgr=std_bgr, masks=masks, roi=roi, feat=feat,
            profile=profile, align_cache=ac, void_cache=vc,
            enabled_defects=effective_enabled_defects(
                resolve_enabled_defects(templ_id), n_pads),
            bridge_joints=joints,
        )


def resolve_enabled_defects(templ_id: str) -> list:
    """全局 enabled_defects 按 template_overrides[templ_id] 覆盖。"""
    overrides = getattr(C, "TEMPLATE_OVERRIDES", {}) or {}
    key = str(templ_id)
    if key in overrides:
        return list(overrides[key])
    return list(getattr(C, "ENABLED_DEFECTS", [12, 13, 14, 15, 16]))
