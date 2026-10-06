"""pytest 路径引导：把 src/ 与 contract_reference/ 加入 sys.path（仅本仓库自测用，勿随交付拷入）。"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
_CONTRACT = _REPO_ROOT / "contract_reference"

for _p in (_SRC, _CONTRACT, _REPO_ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
