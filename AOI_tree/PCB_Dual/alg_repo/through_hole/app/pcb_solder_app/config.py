"""全局配置：定位 ``detect`` 算法包并加入 ``sys.path``"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

APP_ROOT = Path(__file__).resolve().parent.parent            # .../app
REPO_ROOT = APP_ROOT.parent                                   # .../PCB Through-Hole Solder Inspection

# 标准图配置 / 检测结果的默认落盘目录（均可在界面里改）
DEFAULT_TEMPLATES_DIR = APP_ROOT / "templates"
DEFAULT_SESSIONS_DIR = APP_ROOT / "sessions"

_DEFAULT_DETECT_CANDIDATES = [REPO_ROOT / "detect"]

_ALG_REL = Path("algorithms") / "pcb_through_hole" / "pcb_through_hole_solder_alg.py"


def _looks_like_detect_root(p: Path) -> bool:
    """``detect/`` 根：新布局算法在 ``src/algorithms/...``，旧布局在 ``algorithms/...``。"""
    return (p / "src" / _ALG_REL).is_file() or (p / _ALG_REL).is_file()


def locate_detect_root() -> Optional[Path]:
    """返回 ``detect`` 目录（找不到返回 None）。"""
    env = os.environ.get("PCB_SOLDER_DETECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_detect_root(p):
            return p

    for cand in _DEFAULT_DETECT_CANDIDATES:
        if _looks_like_detect_root(cand):
            return cand.resolve()

    for base in (REPO_ROOT, APP_ROOT):
        for sub in base.rglob("detect"):
            if sub.is_dir() and _looks_like_detect_root(sub):
                return sub.resolve()
    return None


def ensure_detect_on_path() -> Path:
    """把算法 ``src/`` 与契约目录注入 ``sys.path``，返回 ``detect`` 目录；
    找不到时抛出 ``RuntimeError``。

    新布局：``detect/src`` + ``detect/contract_reference``（可 import ``algorithms.*`` / ``core.*``）。
    旧布局回退：``detect`` + ``detect/contract_stub``。
    """
    root = locate_detect_root()
    if root is None:
        raise RuntimeError(
            "未找到 detect 算法包目录。请设置环境变量 PCB_SOLDER_DETECT_ROOT "
            "指向该目录，或确认其位于本仓库的 detect/ 下。"
        )

    src = root / "src"
    if (src / _ALG_REL).is_file():
        path_entries = [src]
    else:
        path_entries = [root]

    for name in ("contract_reference", "contract_stub"):
        contract = root / name
        if contract.is_dir():
            path_entries.append(contract)
            break

    for p in path_entries:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return root
