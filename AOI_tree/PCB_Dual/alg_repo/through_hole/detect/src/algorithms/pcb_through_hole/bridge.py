"""连锡(15)：已配准画布上做颜色桥连 + 缝隙收窄判定。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import bridge_color as CM
from . import config as C
from .align import warp_std_layer
from .bridge_joints import BridgeJoint, exclude_joints, pad_joints
from .detection import Detection

DEFECT_ID_BRIDGE = 15


def _bridge_params() -> dict:
    th = C.TH
    return {
        "erode_frac": float(th.get("bridge_erode_frac", 0.18)),
        "erode_min_px": int(th.get("bridge_erode_min_px", 4)),
        "search_margin": int(th.get("bridge_search_margin", 12)),
        "exclude_margin": int(th.get("bridge_exclude_margin", 2)),
        "weights": tuple(th.get("bridge_weights", (0.4, 1.0, 1.0))),
        "std_floor": tuple(th.get("bridge_std_floor", (4.0, 3.0, 3.0))),
        "std_cap": tuple(th.get("bridge_std_cap", (16.0, 12.0, 12.0))),
        "k_sigma": float(th.get("bridge_k_sigma", 2.2)),
        "change_weights": tuple(th.get("bridge_change_weights", (0.5, 1.0, 1.0))),
        "change_tol": tuple(th.get("bridge_change_tol", (10.0, 6.0, 6.0))),
        "change_scale_clip": tuple(th.get("bridge_change_scale_clip", (0.5, 2.0))),
        "change_k": float(th.get("bridge_change_k", 1.0)),
        "change_blur_ksize": int(th.get("bridge_change_blur_ksize", 5)),
        "open_ksize": int(th.get("bridge_open_ksize", 3)),
        "min_bridge_pixels": int(th.get("bridge_min_bridge_pixels", 6)),
        "clearance_bg_percentile": float(th.get("bridge_clearance_bg_percentile", 70.0)),
        "clearance_bg_k": float(th.get("bridge_clearance_bg_k", 3.0)),
        "min_clearance_ratio": float(th.get("bridge_min_clearance_ratio", 0.35)),
        "min_clearance_px": int(th.get("bridge_min_clearance_px", 2)),
    }


def _union_mask(masks, shape_hw) -> np.ndarray:
    u = np.zeros(shape_hw, dtype=np.uint8)
    for m in masks:
        if m is not None:
            u = cv2.bitwise_or(u, m)
    return u


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)


def render_joint_masks(
    joints: List[BridgeJoint], shape_hw: Tuple[int, int],
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """在给定分辨率下渲染 pad masks 与 exclude 并集。"""
    pads = pad_joints(joints)
    excludes = exclude_joints(joints)
    full = {j.id: j.mask(shape_hw) for j in pads}
    excl = _union_mask([j.mask(shape_hw) for j in excludes], shape_hw)
    return full, excl


def warp_joint_masks_to_test(
    joints: List[BridgeJoint],
    std_hw: Tuple[int, int],
    tw: int, th: int,
    warp_dx: float, warp_dy: float,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """标准分辨率 joints → 缩放至测试画布 → 与标准图同一 affine warp。"""
    sh, sw = std_hw
    full_std, excl_std = render_joint_masks(joints, (sh, sw))
    full_t: Dict[int, np.ndarray] = {}
    for jid, m in full_std.items():
        if m.shape[0] != th or m.shape[1] != tw:
            m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
        full_t[jid] = warp_std_layer(m, warp_dx, warp_dy, tw, th, is_mask=True)
    if excl_std.shape[0] != th or excl_std.shape[1] != tw:
        excl_std = cv2.resize(excl_std, (tw, th), interpolation=cv2.INTER_NEAREST)
    excl_t = warp_std_layer(excl_std, warp_dx, warp_dy, tw, th, is_mask=True)
    return full_t, excl_t


def analyze_bridge(
    test_bgr: np.ndarray,
    std_bgr_on_test: np.ndarray,
    pad_masks: Dict[int, np.ndarray],
    exclude_mask: Optional[np.ndarray] = None,
) -> List[Detection]:
    """连锡判定。pad_masks / exclude 须已在测试画布坐标系。"""
    if len(pad_masks) < 2:
        return []
    p = _bridge_params()
    h, w = test_bgr.shape[:2]
    if exclude_mask is None:
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
    elif (
        exclude_mask.shape[0] != h or exclude_mask.shape[1] != w
    ):
        exclude_mask = cv2.resize(
            exclude_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    lab = CM.bgr_to_lab(test_bgr)
    templ_lab = CM.bgr_to_lab(std_bgr_on_test)

    core_masks = {
        jid: CM.erode_mask(m, frac=p["erode_frac"], min_px=p["erode_min_px"])
        for jid, m in pad_masks.items()
    }
    pooled_core = _union_mask(core_masks.values(), (h, w))
    stats = CM.robust_color_stats(lab, pooled_core)
    templ_stats = CM.robust_color_stats(templ_lab, pooled_core)
    if stats is None or templ_stats is None:
        return []
    mean, std, _n = stats
    templ_mean, templ_std, _ = templ_stats

    excl = exclude_mask
    if np.any(excl) and p["exclude_margin"] > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * p["exclude_margin"] + 1, 2 * p["exclude_margin"] + 1),
        )
        excl = cv2.dilate(excl, ek)
    not_excluded = cv2.bitwise_not(excl)

    union_full = _union_mask(pad_masks.values(), (h, w))
    ys, xs = np.where(union_full > 0)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    if len(xs) > 0:
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(hull_mask, hull, 255)
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * p["search_margin"] + 1, 2 * p["search_margin"] + 1),
    )
    search_region = cv2.dilate(cv2.bitwise_or(hull_mask, union_full), dilate_k)
    search_region = cv2.bitwise_and(search_region, not_excluded)

    solder_like, _ = CM.classify_solder(
        lab, mean, std,
        weights=p["weights"], std_floor=p["std_floor"],
        std_cap=p["std_cap"], k=p["k_sigma"],
    )
    changed, _ = CM.classify_change(
        lab, templ_lab, mean, std, templ_mean, templ_std,
        weights=p["change_weights"], tol=p["change_tol"],
        scale_clip=p["change_scale_clip"], k=p["change_k"],
        blur_ksize=p["change_blur_ksize"],
    )
    bridge_candidate = solder_like & changed
    solder_mask = (bridge_candidate & (search_region > 0)).astype(np.uint8) * 255
    solder_mask = cv2.bitwise_or(solder_mask, union_full)
    solder_mask = cv2.bitwise_and(solder_mask, not_excluded)

    if p["open_ksize"] > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p["open_ksize"], p["open_ksize"]))
        opened = cv2.morphologyEx(solder_mask, cv2.MORPH_OPEN, k)
        solder_mask = cv2.bitwise_or(opened, union_full)
        solder_mask = cv2.bitwise_and(solder_mask, not_excluded)

    _num, labels = cv2.connectedComponents(solder_mask, connectivity=8)

    joint_label: Dict[int, int] = {}
    for jid, full in pad_masks.items():
        core = core_masks[jid]
        vals = labels[core > 0]
        vals = vals[vals != 0]
        if len(vals) == 0:
            vals = labels[full > 0]
            vals = vals[vals != 0]
        joint_label[jid] = int(np.bincount(vals).argmax()) if len(vals) else 0

    ids = list(pad_masks.keys())
    dets: List[Detection] = []
    for i in range(len(ids)):
        for k_ in range(i + 1, len(ids)):
            a, b = ids[i], ids[k_]
            same_label = joint_label[a] != 0 and joint_label[a] == joint_label[b]
            bridge_pixels = 0
            bridge_zone = None
            if same_label:
                comp_mask = (labels == joint_label[a]).astype(np.uint8) * 255
                both_full = cv2.bitwise_or(pad_masks[a], pad_masks[b])
                outside = cv2.bitwise_and(comp_mask, cv2.bitwise_not(both_full))
                bridge_pixels = int(np.count_nonzero(outside))
                bridge_zone = outside
            color_bridged = bool(
                same_label and bridge_pixels >= p["min_bridge_pixels"])

            widths_test, axis_test, segs_test = CM.gap_clearance_widths(
                pad_masks[a], pad_masks[b], lab,
                bg_percentile=p["clearance_bg_percentile"],
                bg_k=p["clearance_bg_k"],
            )
            widths_templ, _, _ = CM.gap_clearance_widths(
                pad_masks[a], pad_masks[b], templ_lab,
                bg_percentile=p["clearance_bg_percentile"],
                bg_k=p["clearance_bg_k"],
            )
            clearance_bridged = False
            clearance_ratio = None
            min_gap_test = None
            min_gap_templ = None
            if widths_test and widths_templ:
                min_i = int(np.argmin(widths_test))
                min_gap_test = int(widths_test[min_i])
                min_gap_templ = int(min(widths_templ))
                clearance_ratio = float(min_gap_test / max(min_gap_templ, 1))
                clearance_bridged = (
                    min_gap_test <= p["min_clearance_px"]
                    or clearance_ratio < p["min_clearance_ratio"]
                )

            if not (color_bridged or clearance_bridged):
                continue

            bbox_mask = bridge_zone if (
                bridge_zone is not None and np.any(bridge_zone)
            ) else cv2.bitwise_or(pad_masks[a], pad_masks[b])
            bbox = _bbox_from_mask(bbox_mask)
            if bbox is None:
                bbox = _bbox_from_mask(cv2.bitwise_or(pad_masks[a], pad_masks[b]))
            if bbox is None:
                continue

            parts = []
            if color_bridged:
                parts.append(f"颜色桥连 px={bridge_pixels}")
            if clearance_bridged:
                parts.append(
                    f"缝隙收窄 gap={min_gap_test}/{min_gap_templ}"
                    f" ratio={clearance_ratio:.2f}" if clearance_ratio is not None
                    else f"缝隙收窄 gap={min_gap_test}"
                )
            score = 1.0
            if clearance_ratio is not None:
                score = max(score, 1.0 - float(clearance_ratio))
            if color_bridged:
                score = max(score, min(1.0, bridge_pixels / 50.0))
            dets.append(Detection(
                DEFECT_ID_BRIDGE, float(score), bbox,
                f"连锡[{a}-{b}] " + "；".join(parts),
                pad_id=a * 1000 + b,
            ))
    return dets
