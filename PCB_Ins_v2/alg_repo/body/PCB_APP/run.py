"""启动脚本：``python run.py`` 即可打开 PCB 缺陷检测算法验证台。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
