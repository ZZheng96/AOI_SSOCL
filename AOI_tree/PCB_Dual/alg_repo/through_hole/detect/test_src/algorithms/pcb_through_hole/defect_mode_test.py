"""框数 → 启用缺陷分流。"""
from __future__ import annotations

from algorithms.pcb_through_hole.defect_mode import (
    DEFECT_ID_BRIDGE,
    PAD_INTERNAL_DEFECTS,
    effective_enabled_defects,
)


def test_multi_pad_forces_bridge_only():
    assert effective_enabled_defects([12, 13, 14, 15, 16], 2) == [DEFECT_ID_BRIDGE]
    assert effective_enabled_defects([12, 13], 3) == [DEFECT_ID_BRIDGE]
    assert effective_enabled_defects(None, 2) == [DEFECT_ID_BRIDGE]


def test_single_or_auto_keeps_pad_internal_only():
    assert effective_enabled_defects([12, 13, 14, 15, 16], 0) == list(PAD_INTERNAL_DEFECTS)
    assert effective_enabled_defects([12, 13, 14, 15, 16], 1) == list(PAD_INTERNAL_DEFECTS)
    assert effective_enabled_defects([12, 14, 15], 1) == [12, 14]
    assert effective_enabled_defects([], 0) == []
    assert effective_enabled_defects(None, 0) == list(PAD_INTERNAL_DEFECTS)
