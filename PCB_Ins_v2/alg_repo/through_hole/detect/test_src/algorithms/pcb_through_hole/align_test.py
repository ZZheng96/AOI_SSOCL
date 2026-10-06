"""algorithms.pcb_through_hole.align / pipeline.check_reliability 单元测试"""
from __future__ import annotations

import numpy as np
import cv2

from algorithms.pcb_through_hole import config as C
from algorithms.pcb_through_hole.align import (
    AlignCache,
    align_phase_primary_pad,
    build_align_cache,
    phase_correlate_shift,
    select_primary_pad,
    solder_mask_to_preview_roi,
)
from algorithms.pcb_through_hole.bridge_joints import BridgeJoint
from algorithms.pcb_through_hole.pipeline import check_reliability, _apply_preview_shift
from algorithms.pcb_through_hole.segment import PadROI


def _textured_gray(size: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size), dtype=np.uint8).astype(np.float32)


def _cache_for(roi_gray: np.ndarray) -> AlignCache:
    h, w = roi_gray.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return AlignCache(
        roi_xy=(0, 0, w, h), roi_gray_f32=roi_gray, roi_hann=hann,
        prior_roi=np.zeros((h, w), np.uint8),
        phase_gray_f32=roi_gray, phase_hann=hann, phase_scale=1.0,
    )


def test_phase_correlate_shift_does_not_clip_large_shift():
    """真实位移超出 max_shift_px 时不应被裁剪，否则 check_reliability 的
    越界判定会失效（见 align.phase_correlate_shift 的说明注释）。"""
    max_px = float(C.ALIGN["max_shift_px"])
    true_shift = int(max_px) + 40  # 明显超出阈值
    base = _textured_gray()
    shifted = np.roll(base, shift=(true_shift, 0), axis=(0, 1))
    cache = _cache_for(base)

    dx, dy, resp = phase_correlate_shift(cache, shifted)

    assert resp > 0.5  # 强纹理平移，相位相关应给出高置信度
    assert abs(dx) > max_px or abs(dy) > max_px


def test_check_reliability_rejects_out_of_range_shift():
    reliable, reason = check_reliability(
        dx=0.0, dy=0.0, coverage=1.0, std_dx=60.0, std_dy=0.0)
    assert reliable is False
    assert "配准超出范围" in reason


def test_check_reliability_accepts_in_range_shift_with_good_coverage():
    reliable, reason = check_reliability(
        dx=0.0, dy=0.0, coverage=0.95, std_dx=5.0, std_dy=-3.0)
    assert reliable is True
    assert reason is None


def _pad_with_offset_pin(size: int = 128, pad_shift: int = 6, pin_extra: int = 10):
    """构造「焊盘平移 + 引脚相对焊盘再偏」的合成图，验证 pad_ring 锁焊盘而非引脚。"""
    rng = np.random.default_rng(1)
    bg = rng.integers(40, 70, (size, size), dtype=np.uint8)
    std = bg.copy()
    test = bg.copy()
    solder = np.zeros((size, size), np.uint8)
    cx = cy = size // 2
    r_out, r_in = 40, 18
    cv2.circle(solder, (cx, cy), r_out, 255, -1)
    # 锡面环：加环形纹理
    yy, xx = np.mgrid[:size, :size]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (rr <= r_out) & (rr >= r_in)
    std[ring] = (90 + (xx[ring] + yy[ring]) % 40).astype(np.uint8)
    # 标准图：引脚居中
    cv2.circle(std, (cx, cy), 8, 240, -1)
    # 测试图：整盘平移 pad_shift，引脚再额外偏 pin_extra（相对焊盘）
    M = np.array([[1, 0, pad_shift], [0, 1, 0]], dtype=np.float32)
    ring_img = np.zeros_like(std)
    ring_img[ring] = std[ring]
    warped_ring = cv2.warpAffine(ring_img, M, (size, size), flags=cv2.INTER_NEAREST)
    test = np.where(warped_ring > 0, warped_ring, test)
    cv2.circle(test, (cx + pad_shift + pin_extra, cy), 8, 240, -1)
    std_bgr = cv2.cvtColor(std, cv2.COLOR_GRAY2BGR)
    test_bgr = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    return std_bgr, test_bgr, solder, pad_shift


def test_pad_ring_locks_pad_not_offset_pin(monkeypatch):
    """引脚相对焊盘偏心时，pad_ring 应给出接近焊盘平移的位移，而不是跟着引脚走。"""
    monkeypatch.setitem(C.ALIGN, "feature", "pad_ring")
    monkeypatch.setitem(C.ALIGN, "channel", "gray")
    monkeypatch.setitem(C.ALIGN, "phase_downsample", 1.0)
    monkeypatch.setitem(C.ALIGN, "pin_suppress_frac", 0.22)
    std_bgr, test_bgr, solder, pad_shift = _pad_with_offset_pin()
    cache = build_align_cache(std_bgr, solder, (0, 0, 128, 128))
    assert cache.feature == "pad_ring"
    assert cache.align_keep is not None
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx, dy, resp = phase_correlate_shift(cache, test_gray)
    # phaseCorrelate: 测试相对标准的位移 ≈ pad_shift（引脚额外偏移不应主导）
    assert abs(dx - pad_shift) <= 2.5, f"expected ~{pad_shift}, got dx={dx:.2f} dy={dy:.2f}"
    assert abs(dy) <= 2.5
    assert resp > 0.1


def test_solder_mask_to_preview_roi_circle():
    mask = np.zeros((100, 100), np.uint8)
    cv2.circle(mask, (50, 48), 30, 255, -1)
    roi = solder_mask_to_preview_roi(mask)
    assert roi["shape"] == "circle"
    assert abs(roi["cx"] - 50) <= 2
    assert abs(roi["cy"] - 48) <= 2
    assert abs(roi["r"] - 30) <= 2


def test_solder_mask_to_preview_roi_ellipse():
    mask = np.zeros((120, 160), np.uint8)
    cv2.ellipse(mask, ((80, 60), (90, 40), 0), 255, -1)
    roi = solder_mask_to_preview_roi(mask)
    assert roi["shape"] == "ellipse"
    assert abs(roi["cx"] - 80) <= 3
    assert abs(roi["cy"] - 60) <= 3
    major = max(roi["axes_w"], roi["axes_h"])
    minor = min(roi["axes_w"], roi["axes_h"])
    assert abs(major - 90) <= 6
    assert abs(minor - 40) <= 6


def test_apply_preview_shift_matches_map_roi_formula():
    """``_apply_preview_shift`` 使用 ``(x-dx, y-dy)``。"""
    roi = PadROI(0, 10, 20, 40, 50, cx=30.0, cy=45.0)
    mapped = _apply_preview_shift(roi, dx=-5.0, dy=3.0, tw=100, th=100)
    assert mapped.x == 15
    assert mapped.y == 17
    assert mapped.w == 40 and mapped.h == 50


def test_select_primary_pad_prefers_circle():
    h, w = 80, 160
    joints = [
        BridgeJoint(1, "rect", {"x": 100, "y": 10, "w": 40, "h": 60}),
        BridgeJoint(2, "circle", {"cx": 40, "cy": 40, "r": 18}),
    ]
    primary, mask = select_primary_pad(joints, (h, w))
    assert primary is not None and primary.id == 2 and primary.type == "circle"
    assert mask is not None and int(cv2.countNonZero(mask)) > 0


def test_multi_pad_primary_pad_ring_recovers_rigid_shift(monkeypatch):
    """连锡多框：主焊盘走 pad_ring（与四种缺陷同款），刚性位移应可找回。"""
    monkeypatch.setitem(C.ALIGN, "feature", "pad_ring")
    monkeypatch.setitem(C.ALIGN, "channel", "gray")
    monkeypatch.setitem(C.ALIGN, "phase_downsample", 1.0)
    monkeypatch.setitem(C.ALIGN, "pin_suppress_frac", 0.22)
    monkeypatch.setitem(C.ALIGN, "phase_min_response", 0.12)

    size = 128
    std_bgr, test_bgr, solder, pad_shift = _pad_with_offset_pin(
        size=size, pad_shift=6, pin_extra=10)
    # 附加一个远离的长条框（不应被选为主焊盘，也不应拖偏）
    joints = [
        BridgeJoint(1, "circle", {"cx": 64, "cy": 64, "r": 40}),
        BridgeJoint(2, "rect", {"x": 2, "y": 2, "w": 12, "h": 40}),
    ]
    # 用真实锡面环图作为标准图内容；主焊盘 mask 与 pad_ring 单测一致
    primary, _ = select_primary_pad(joints, (size, size))
    assert primary is not None and primary.type == "circle"

    dx, dy, _meta, method, resp, cache = align_phase_primary_pad(
        test_bgr, std_bgr, joints)
    assert cache is not None
    assert cache.feature == "pad_ring"
    assert "主焊盘#1/circle" in method
    # align_phase_roi 返回 -est；est≈pad_shift ⇒ dx≈-pad_shift
    assert abs(dx + pad_shift) <= 2.5, f"dx={dx} expected ~{-pad_shift}, resp={resp}"
    assert abs(dy) <= 2.5
