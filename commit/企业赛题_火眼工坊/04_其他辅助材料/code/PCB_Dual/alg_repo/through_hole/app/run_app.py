#!/usr/bin/env python
"""桌面界面启动入口：``python run_app.py``（在 app/ 目录下或任意目录均可）"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pcb_solder_app.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
