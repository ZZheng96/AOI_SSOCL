"""插件焊点检测流水线: 标准 warp 到测试坐标 -> 测试原图分割 -> 五条规则"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import time

import cv2
import numpy as np

from . import config as C
from .align import (
    AlignMeta,
    align_phase_primary_pad,
    align_phase_roi,
    build_align_cache,
    scale_bgr_to_canvas,
    scale_masks_to_canvas,
    scale_rect,
    std_shift_to_test,
    warp_std_masks,
    warp_std_valid,
    warp_std_layer,
)
from .bridge import DEFECT_ID_BRIDGE, analyze_bridge, warp_joint_masks_to_test
from .bridge_joints import pad_joints
from .features import extract, PadFeatures
from .detection import Detection, bbox_from_mask
from .excess import analyze_excess, compute_wrap_metrics
from .insufficient import (
    analyze_insufficient,
    suppress_insuf_for_nolead,
    suppress_nolead_for_insuf,
)
from .nolead import StandardProfile, evaluate_nolead
from .segment import compute_masks, Masks, PadROI
from .template import TemplateModel
from .void import VoidStdCache, analyze_voids, build_void_std_cache

_K_VALID = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# 进程级线程池（masks / 缺陷规则两阶段各最多 2 任务）
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

STATUS_OK = "OK"
STATUS_NG = "NG"
STATUS_REVIEW = "待复检"


@dataclass
class StdOnTest:
    """标准侧数据 warp 到测试画布后的视图 (仅用于与测试对比)。"""
    bgr: np.ndarray
    masks: Masks
    roi: PadROI
    feat: PadFeatures
    profile: StandardProfile
    void_cache: VoidStdCache | None = None


@dataclass
class InspectResult:
    name: str
    original: np.ndarray
    aligned: np.ndarray
    masks: Masks
    test_roi: PadROI | None = None
    void_mask: np.ndarray | None = None
    align_method: str = ""
    align_meta: AlignMeta = field(default_factory=AlignMeta)
    valid: np.ndarray | None = None
    detections: list[Detection] = field(default_factory=list)
    total_ms: float = 0.0
    feat_t: PadFeatures | None = None
    insuf: tuple | None = None
    excess: tuple | None = None
    nolead: dict | None = None
    status: str = ""
    coverage: float | None = None
    review_reason: str | None = None

    @property
    def is_defect(self) -> bool:
        return len(self.detections) > 0

    @property
    def defect_ids(self) -> list[int]:
        return sorted({d.defect_id for d in self.detections})


def _prepare_test_work(test_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """测试工作画布 (可选降采样); 返回 (work, scale_x=ow/tw, scale_y=oh/th)。"""
    oh, ow = test_bgr.shape[:2]
    max_dim = getattr(C, "WORK_MAX_DIM", None)
    if max_dim and max(oh, ow) > max_dim:
        s = max_dim / max(oh, ow)
        tw, th = int(round(ow * s)), int(round(oh * s))
        work = cv2.resize(test_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        return work, ow / tw, oh / th
    return test_bgr, 1.0, 1.0


def _roi_on_test(std: TemplateModel, tw: int, th: int) -> PadROI:
    sx = tw / std.bgr.shape[1]
    sy = th / std.bgr.shape[0]
    x, y, w, h = scale_rect(std.roi.rect, sx, sy)
    return PadROI(
        std.roi.id, x, y, w, h,
        std.roi.cx * sx, std.roi.cy * sy,
    )


def _apply_preview_shift(roi: PadROI, dx: float, dy: float,
                         tw: int, th: int) -> PadROI:
    """检测窗平移 ``(x-dx,y-dy)`` 并裁入画布。"""
    x = int(round(roi.x - dx))
    y = int(round(roi.y - dy))
    w, h = int(roi.w), int(roi.h)
    if tw <= 0 or th <= 0:
        return PadROI(roi.id, 0, 0, max(1, w), max(1, h), roi.cx - dx, roi.cy - dy)
    x = max(0, min(x, tw - 1))
    y = max(0, min(y, th - 1))
    w = max(1, min(w, tw - x))
    h = max(1, min(h, th - y))
    return PadROI(roi.id, x, y, w, h, roi.cx - dx, roi.cy - dy)


def _profile_on_test(std: TemplateModel, test_roi: PadROI) -> StandardProfile:
    """标准 profile 缩放到测试 ROI 尺寸 (center_gray 保持标准坐标模板)。"""
    w, h = test_roi.w, test_roi.h
    ow, oh = std.profile.center_gray.shape[1], std.profile.center_gray.shape[0]
    if (w, h) == (ow, oh):
        return std.profile
    cg = cv2.resize(std.profile.center_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    cm = cv2.resize(std.profile.center_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return replace(std.profile, center_gray=cg, center_mask=cm)


def compute_coverage(valid_mask: np.ndarray, template_solder_mask: np.ndarray) -> float:
    """配准有效区与模板锡面的重合度。"""
    denom = cv2.countNonZero(template_solder_mask)
    if denom <= 0:
        return 0.0
    inter = cv2.bitwise_and(valid_mask, template_solder_mask)
    return cv2.countNonZero(inter) / denom


def check_reliability(dx: float, dy: float, coverage: float,
                      std_dx: float | None = None, std_dy: float | None = None,
                      align_cfg: dict | None = None,
                      coverage_cfg: dict | None = None) -> tuple[bool, str | None]:
    """配准/覆盖率早停。不可靠时返回 (False, 原因)，否则 (True, None)。"""
    align_cfg = align_cfg if align_cfg is not None else C.ALIGN
    coverage_cfg = coverage_cfg if coverage_cfg is not None else C.COVERAGE

    max_shift = float(align_cfg.get("max_shift_px", 18))
    shift_x = std_dx if std_dx is not None else dx
    shift_y = std_dy if std_dy is not None else dy
    if abs(shift_x) > max_shift or abs(shift_y) > max_shift:
        canvas_note = ""
        if std_dx is not None and std_dy is not None:
            canvas_note = f" 画布dx={dx:.1f} dy={dy:.1f}"
        return False, (
            f"配准超出范围 std_dx={shift_x:.1f} std_dy={shift_y:.1f} "
            f"(>{max_shift:.0f}px 标准分辨率){canvas_note}，无法可靠比较")

    review_min = float(coverage_cfg.get("review_min", 0.85))
    if coverage < review_min:
        return False, f"覆盖率不足 {coverage:.2f} (<{review_min:.2f})，无法可靠比较"

    return True, None


def _apply_valid_std(masks_g: Masks, valid: np.ndarray) -> tuple[Masks, np.ndarray]:
    """标准 warp 越界区不参与掩膜差对比。"""
    valid_e = cv2.erode(valid, _K_VALID)
    masks_g.solder = cv2.bitwise_and(masks_g.solder, valid_e)
    masks_g.red = cv2.bitwise_and(masks_g.red, valid_e)
    masks_g.bright = cv2.bitwise_and(masks_g.bright, valid_e)
    return masks_g, valid_e


def _compute_std_shift(test_work: np.ndarray, std: TemplateModel,
                       work_scale_x: float, work_scale_y: float):
    gh, gw = std.bgr.shape[:2]
    th, tw = test_work.shape[:2]
    sx, sy = tw / gw, th / gh
    test_corr = (
        test_work if (tw, th) == (gw, gh)
        else cv2.resize(test_work, (gw, gh), interpolation=cv2.INTER_AREA)
    )
    pads = pad_joints(std.bridge_joints or [])
    if len(pads) >= 2:
        std_dx, std_dy, meta, method, _, cache = align_phase_primary_pad(
            test_corr, std.bgr, pads, work_scale_x, work_scale_y)
        if cache is None:
            cache = std.align_cache
    else:
        cache = std.align_cache or build_align_cache(
            std.bgr, std.masks.solder, std.roi.rect)
        std_dx, std_dy, meta, method, _ = align_phase_roi(
            test_corr, cache, work_scale_x, work_scale_y)
    dx_t, dy_t = std_shift_to_test(std_dx, std_dy, sx, sy)
    meta.std_dx = std_dx
    meta.std_dy = std_dy
    meta.dx = dx_t
    meta.dy = dy_t
    return dx_t, dy_t, meta, method, cache


def _build_std_on_test(std: TemplateModel, test_work: np.ndarray,
                       dx: float, dy: float) -> tuple[StdOnTest, np.ndarray, np.ndarray]:
    """标准侧叠到检测匹配窗（窗 ``x-dx``，warp ``-dx``）。"""
    th, tw = test_work.shape[:2]
    test_roi = _apply_preview_shift(_roi_on_test(std, tw, th), dx, dy, tw, th)
    warp_dx, warp_dy = -dx, -dy

    std_bgr_s = scale_bgr_to_canvas(std.bgr, tw, th)
    std_masks_s = scale_masks_to_canvas(std.masks, tw, th)
    std_bgr_w = warp_std_layer(std_bgr_s, warp_dx, warp_dy, tw, th)
    std_masks_w = warp_std_masks(std_masks_s, warp_dx, warp_dy, tw, th)
    # 覆盖率早停分母需未扣 valid 边界的模板锡面，先留引用再改写 masks
    tmpl_solder_warped = std_masks_w.solder
    valid = warp_std_valid(tw, th, warp_dx, warp_dy, tw, th)
    std_masks_w, valid_e = _apply_valid_std(std_masks_w, valid)

    profile = _profile_on_test(std, test_roi)
    feat_g = extract(std_bgr_w, test_roi, std_masks_w)
    vc = build_void_std_cache(test_roi, std_masks_w)

    return StdOnTest(std_bgr_w, std_masks_w, test_roi, feat_g, profile, vc), valid_e, tmpl_solder_warped


def _run_void(test_work: np.ndarray, std_ot: StdOnTest, valid_full: np.ndarray):
    x, y, w, h = std_ot.roi.rect
    lo, hi = std_ot.profile.solder_hsv_low, std_ot.profile.solder_hsv_high
    roi_bgr = test_work[y:y + h, x:x + w]
    valid_roi = valid_full[y:y + h, x:x + w]
    prior = std_ot.masks.solder[y:y + h, x:x + w]
    m = compute_masks(
        roi_bgr, lo, hi, prior_mask=prior, roi_rect=(0, 0, w, h),
        ref_stats=std_ot.profile.solder_stats,
    )
    void_mask, dets = analyze_voids(
        std_ot.void_cache, m.hsv, m.solder, m.red, valid_roi=valid_roi)
    return void_mask, dets


def _collect_solder_detections(test_roi: PadROI, insuf, excess, nolead) -> list[Detection]:
    dets: list[Detection] = []
    insuf_hit = insuf[1] and not suppress_insuf_for_nolead(insuf, nolead)
    if insuf_hit:
        bbox = bbox_from_mask(insuf[0], test_roi) if insuf[0] is not None else test_roi.rect
        dets.append(Detection(13, insuf[3], bbox, insuf[4], test_roi.id))
    if excess[1]:
        bbox = bbox_from_mask(excess[0], test_roi) if excess[0] is not None else test_roi.rect
        dets.append(Detection(14, excess[2], bbox, excess[3], test_roi.id))
    suppress_nolead = suppress_nolead_for_insuf(insuf, nolead)
    if nolead["hit"] and not excess[1] and not suppress_nolead:
        dets.append(Detection(
            16, nolead["score"], test_roi.rect,
            f"[{nolead['reason']}] cb={nolead['cb_t']:.2f}/"
            f"{nolead['cb_g']:.2f} diff={nolead['diff_norm']:.2f} "
            f"peak={nolead['peak_ratio']:.2f}",
            test_roi.id,
        ))
    return dets


def align_preview(test_bgr: np.ndarray, std: TemplateModel) -> tuple[AlignMeta, str, tuple[int, int]]:
    """仅做配准 (不分割/不跑四条规则)，供界面"标准图 ROI 映射到测试图"的
    实时预览使用；比完整 ``inspect()`` 轻得多。

    返回 (对齐元数据, 配准方式说明, 测试工作画布尺寸 (tw, th))。
    """
    test_work, work_sx, work_sy = _prepare_test_work(test_bgr)
    th, tw = test_work.shape[:2]
    if C.ENABLE_CALIBRATION:
        _dx, _dy, meta, method, _cache = _compute_std_shift(test_work, std, work_sx, work_sy)
    else:
        meta = AlignMeta(scale_x=work_sx, scale_y=work_sy)
        method = "未校准"
    return meta, method, (tw, th)


def inspect(test_bgr: np.ndarray, std: TemplateModel, name: str = "",
            *, enable_review: bool = True) -> InspectResult:
    """单流水线：配准匹配窗上分割 + 五条规则"""
    t0 = time.perf_counter()
    original = test_bgr
    test_work, work_sx, work_sy = _prepare_test_work(test_bgr)

    if C.ENABLE_CALIBRATION:
        dx, dy, meta, method, _cache = _compute_std_shift(
            test_work, std, work_sx, work_sy)
    else:
        dx, dy = 0.0, 0.0
        meta = AlignMeta(scale_x=work_sx, scale_y=work_sy)
        method = "未校准"
    std_ot, valid_e, tmpl_solder_warped = _build_std_on_test(std, test_work, dx, dy)
    test_roi = std_ot.roi

    coverage_val = None
    if enable_review:
        coverage_val = compute_coverage(valid_e, tmpl_solder_warped)
        reliable, review_reason = check_reliability(
            dx, dy, coverage_val, std_dx=meta.std_dx, std_dy=meta.std_dy)
        if not reliable:
            return InspectResult(
                name=name, original=original, aligned=test_work, masks=std_ot.masks,
                test_roi=test_roi, align_method=method, align_meta=meta, valid=valid_e,
                detections=[], total_ms=(time.perf_counter() - t0) * 1000,
                status=STATUS_REVIEW, coverage=coverage_val, review_reason=review_reason,
            )

    prior = std_ot.masks.solder
    f_void = _EXECUTOR.submit(_run_void, test_work, std_ot, valid_e)
    f_masks = _EXECUTOR.submit(
        compute_masks,
        test_work,
        std.profile.solder_hsv_low,
        std.profile.solder_hsv_high,
        prior,
        test_roi.rect,
        std.profile.solder_stats,
    )
    void_mask, void_dets = f_void.result()
    masks_t = f_masks.result()

    feat_t = extract(test_work, test_roi, masks_t)
    feat_g = std_ot.feat

    wrap = compute_wrap_metrics(
        test_work, masks_t, std_ot.masks, test_roi,
        std_ot.profile.pin_top_core_area)

    f_insuf = _EXECUTOR.submit(
        analyze_insufficient, test_roi, masks_t, std_ot.masks,
        feat_t, feat_g)
    f_nolead = _EXECUTOR.submit(
        evaluate_nolead, test_work, test_roi, masks_t, std_ot.profile,
        wrap=wrap, feat_t=feat_t, feat_g=feat_g)
    insuf = f_insuf.result()
    nolead = f_nolead.result()

    excess = analyze_excess(
        test_roi, masks_t, std_ot.masks, feat_t, feat_g,
        wrap=wrap, nolead_sm=nolead)

    solder_dets = _collect_solder_detections(test_roi, insuf, excess, nolead)

    bridge_dets: list[Detection] = []
    if DEFECT_ID_BRIDGE in std.enabled_defects and len(pad_joints(std.bridge_joints)) >= 2:
        th, tw = test_work.shape[:2]
        sh, sw = std.bgr.shape[:2]
        pad_masks, excl_mask = warp_joint_masks_to_test(
            std.bridge_joints, (sh, sw), tw, th, -dx, -dy)
        bridge_dets = analyze_bridge(
            test_work, std_ot.bgr, pad_masks, excl_mask)
        if bridge_dets and excess[1]:
            solder_dets = [d for d in solder_dets if d.defect_id != 14]

    all_dets = list(void_dets) + solder_dets + bridge_dets
    all_dets = [d for d in all_dets if d.defect_id in std.enabled_defects]
    status = STATUS_NG if all_dets else STATUS_OK

    return InspectResult(
        name=name,
        original=original,
        aligned=test_work,
        masks=masks_t,
        test_roi=test_roi,
        void_mask=void_mask,
        align_method=method,
        align_meta=meta,
        valid=valid_e,
        detections=all_dets,
        total_ms=(time.perf_counter() - t0) * 1000,
        feat_t=feat_t,
        insuf=insuf,
        excess=excess,
        nolead=nolead,
        status=status,
        coverage=coverage_val,
        review_reason=None,
    )
