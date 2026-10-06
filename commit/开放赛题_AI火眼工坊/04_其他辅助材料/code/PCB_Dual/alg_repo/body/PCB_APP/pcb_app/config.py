"""全局配置与算法包定位。

负责把 ``pcb_defect_detector`` 包所在目录加入 ``sys.path``，使前端可直接
``from pcb_defect_detector import PCBDefectDetector``。定位优先级：

1. 环境变量 ``PCB_DETECTOR_ROOT``（指向包含 pcb_defect_detector 的目录）；
2. 相对本仓库的默认位置 ``../PCB_defect/pcb_defect_detector``；
3. 逐级向上搜索名为 pcb_defect_detector 的可导入包。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

APP_ROOT = Path(__file__).resolve().parent.parent          # .../PCB_APP
WORKSPACE_ROOT = APP_ROOT.parent                            # .../XDWork

# 结果自动保存根目录（可在设置里改）
DEFAULT_SESSIONS_DIR = APP_ROOT / "sessions"

_DEFAULT_DETECTOR_CANDIDATES = [
    WORKSPACE_ROOT / "PCB_defect" / "pcb_defect_detector",
    WORKSPACE_ROOT / "PCB_defect" / "pcb_defect_detector_cython",
]


def _looks_like_pkg_root(p: Path) -> bool:
    return (p / "pcb_defect_detector" / "__init__.py").is_file()


def locate_detector_root() -> Optional[Path]:
    """返回包含 ``pcb_defect_detector`` 包的目录（找不到返回 None）。"""
    env = os.environ.get("PCB_DETECTOR_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_pkg_root(p):
            return p

    for cand in _DEFAULT_DETECTOR_CANDIDATES:
        if _looks_like_pkg_root(cand):
            return cand.resolve()

    # 向上/向下兜底搜索
    for base in [WORKSPACE_ROOT, APP_ROOT]:
        for sub in base.rglob("pcb_defect_detector"):
            if sub.is_dir() and _looks_like_pkg_root(sub.parent):
                return sub.parent.resolve()
    return None


def ensure_detector_on_path() -> Path:
    """把算法包目录注入 sys.path，返回该目录；找不到则抛出清晰错误。"""
    root = locate_detector_root()
    if root is None:
        raise RuntimeError(
            "未找到 pcb_defect_detector 算法包。请设置环境变量 PCB_DETECTOR_ROOT "
            "指向包含该包的目录，或确认其位于 ../PCB_defect/pcb_defect_detector。"
        )
    sp = str(root)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return root
