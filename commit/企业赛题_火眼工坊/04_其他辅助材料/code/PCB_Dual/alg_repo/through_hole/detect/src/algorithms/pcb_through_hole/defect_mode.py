"""按焊点框数分流启用缺陷：≥2 仅连锡(15)；否则仅盘内(12/13/14/16)。"""
from __future__ import annotations

from typing import Iterable, List, Optional

DEFECT_ID_BRIDGE = 15
PAD_INTERNAL_DEFECTS = (12, 13, 14, 16)
_PAD_INTERNAL_SET = frozenset(PAD_INTERNAL_DEFECTS)


def effective_enabled_defects(
    enabled: Optional[Iterable[int]],
    n_pads: int,
) -> List[int]:
    """n_pads 为 role=pad 数量；自动锡面视为 0。"""
    if int(n_pads) >= 2:
        return [DEFECT_ID_BRIDGE]
    if enabled is None:
        return list(PAD_INTERNAL_DEFECTS)
    return [int(d) for d in enabled if int(d) in _PAD_INTERNAL_SET]
