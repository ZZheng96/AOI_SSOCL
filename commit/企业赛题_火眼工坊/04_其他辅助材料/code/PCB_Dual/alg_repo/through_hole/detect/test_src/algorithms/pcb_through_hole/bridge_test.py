"""连锡规则单元测试：人造两 pad + 桥区 → Detection 15。"""
from __future__ import annotations

import numpy as np

from algorithms.pcb_through_hole.bridge import DEFECT_ID_BRIDGE, analyze_bridge
from algorithms.pcb_through_hole.bridge_joints import BridgeJoint, parse_bridge_joints


def test_parse_bridge_joints_tin_and_native():
    tin = {
        "joints": [
            {"id": 1, "type": "rect", "params": {"x": 0, "y": 0, "w": 10, "h": 10}, "role": "pad"},
            {"id": 2, "type": "circle", "params": {"cx": 20, "cy": 20, "r": 5}, "role": "exclude"},
        ]
    }
    native = {"bridge_joints": tin["joints"]}
    assert len(parse_bridge_joints(tin)) == 2
    assert len(parse_bridge_joints(native)) == 2
    assert parse_bridge_joints({}) == []


def test_analyze_bridge_color_hit():
    h, w = 80, 120
    test = np.full((h, w, 3), (40, 80, 40), dtype=np.uint8)   # 偏绿阻焊
    std = test.copy()
    # 两焊点锡色区域 + 中间桥
    test[20:60, 10:40] = (180, 180, 160)
    test[20:60, 80:110] = (180, 180, 160)
    test[35:50, 35:85] = (180, 180, 160)  # 桥
    # 标准图无桥
    std[20:60, 10:40] = (180, 180, 160)
    std[20:60, 80:110] = (180, 180, 160)

    pad_a = np.zeros((h, w), np.uint8)
    pad_b = np.zeros((h, w), np.uint8)
    pad_a[20:60, 10:40] = 255
    pad_b[20:60, 80:110] = 255

    dets = analyze_bridge(test, std, {1: pad_a, 2: pad_b})
    assert any(d.defect_id == DEFECT_ID_BRIDGE for d in dets)


def test_analyze_bridge_skips_single_pad():
    h, w = 40, 40
    img = np.zeros((h, w, 3), np.uint8)
    m = np.full((h, w), 255, np.uint8)
    assert analyze_bridge(img, img, {1: m}) == []


def test_bridge_joint_scale():
    j = BridgeJoint(1, "rect", {"x": 10, "y": 20, "w": 30, "h": 40})
    s = j.scaled(0.5, 0.5)
    assert s.params["x"] == 5 and s.params["w"] == 15
