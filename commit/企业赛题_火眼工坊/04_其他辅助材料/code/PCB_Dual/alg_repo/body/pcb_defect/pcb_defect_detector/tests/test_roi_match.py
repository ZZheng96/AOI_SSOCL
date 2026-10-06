from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pcb_defect_detector" / "_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from algorithms.component.common import (  # noqa: E402
    extract_matched_roi,
    match_roi_structure,
    ncc_score,
)


def _scene():
    height, width = 360, 480
    x, y, w, h = 150, 110, 160, 110
    image = np.full((height, width, 3), (40, 100, 40), np.uint8)
    cv2.rectangle(image, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.putText(
        image, "U1", (x + 30, y + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (190, 190, 190), 3)
    cv2.circle(image, (x + 20, y + 20), 8, (240, 240, 240), -1)
    return image, (x, y, w, h)


def _transform(image, roi, angle, dx, dy):
    x, y, w, h = roi
    matrix = cv2.getRotationMatrix2D(
        (x + w / 2.0, y + h / 2.0), angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 100, 40))


def test_small_rotation_and_translation():
    golden, roi = _scene()
    x, y, w, h = roi
    template = golden[y:y + h, x:x + w]
    test = _transform(golden, roi, 12.0, 20.0, -15.0)
    match = match_roi_structure(template, test, roi)
    assert match["matched"]
    assert not match["reversed_180"]
    assert abs(match["residual_angle_deg"] - 12.0) <= 3.0
    assert abs(match["dx"] - 20.0) <= 3.0
    assert abs(match["dy"] + 15.0) <= 3.0
    crop = extract_matched_roi(test, match, (w, h))
    assert crop is not None
    assert ncc_score(crop, template) > 0.90


def test_180_degree_orientation():
    golden, roi = _scene()
    x, y, w, h = roi
    template = golden[y:y + h, x:x + w]
    test = _transform(golden, roi, 180.0, 10.0, 5.0)
    match = match_roi_structure(template, test, roi)
    assert match["matched"]
    assert match["reversed_180"]
    assert match["reverse_score"] > match["normal_score"] + 0.06
    assert abs(match["residual_angle_deg"]) <= 3.0


def test_low_texture_falls_back():
    image = np.full((240, 320, 3), 90, np.uint8)
    roi = (80, 60, 120, 90)
    x, y, w, h = roi
    match = match_roi_structure(image[y:y + h, x:x + w], image, roi)
    assert not match["matched"]
    assert match["reason"] == "low_texture"


def _big_scene():
    """比 ``_scene`` 更大的画布，留出足够空间做"远距离搬移"测试。"""
    height, width = 900, 1200
    x, y, w, h = 120, 100, 160, 110
    image = np.full((height, width, 3), (40, 100, 40), np.uint8)
    cv2.rectangle(image, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.putText(
        image, "U1", (x + 30, y + 65),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (190, 190, 190), 3)
    cv2.circle(image, (x + 20, y + 20), 8, (240, 240, 240), -1)
    return image, (x, y, w, h)


def test_full_image_search_finds_far_shift():
    """元件被挪到远超"原位局部窗口"范围之外时，仍应通过整图搜索定位到。"""
    golden, roi = _big_scene()
    x, y, w, h = roi
    template = golden[y:y + h, x:x + w]
    # 目标搬到画面右下角，远超 search_expand=1.0 的原位局部窗口覆盖范围。
    far_dx, far_dy = 700.0, 600.0
    test = _transform(golden, roi, 5.0, far_dx, far_dy)
    match = match_roi_structure(template, test, roi)
    assert match["matched"]
    assert match["reason"] == "matched_full_image"
    assert abs(match["dx"] - far_dx) <= 4.0
    assert abs(match["dy"] - far_dy) <= 4.0

    # 关闭整图搜索后，同样的远距离位移应无法在原位局部窗口内找到。
    local_only = match_roi_structure(template, test, roi, full_search=False)
    assert not local_only["matched"]


def test_orientation_branch_routing():
    """结构匹配到 0°/180° 分支时，移位与极反应二选一（供检测服务路由使用）。"""
    golden, roi = _scene()
    x, y, w, h = roi
    template = golden[y:y + h, x:x + w]

    normal = _transform(golden, roi, 8.0, 5.0, -4.0)
    m_normal = match_roi_structure(template, normal, roi)
    assert m_normal["matched"] and not m_normal["branch_180"]

    flipped = _transform(golden, roi, 178.0, 5.0, -4.0)
    m_flip = match_roi_structure(template, flipped, roi)
    assert m_flip["matched"] and m_flip["branch_180"]


if __name__ == "__main__":
    tests = [
        test_small_rotation_and_translation,
        test_180_degree_orientation,
        test_low_texture_falls_back,
        test_full_image_search_finds_far_shift,
        test_orientation_branch_routing,
    ]
    for test_fn in tests:
        test_fn()
        print(f"PASS {test_fn.__name__}")
