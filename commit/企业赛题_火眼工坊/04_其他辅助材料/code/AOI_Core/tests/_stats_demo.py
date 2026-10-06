# -*- coding: utf-8 -*-
"""补录脚本：启动一体端并切到「统计报表」页，停留 50 秒后退出（不清理任何数据）。
供项目视频分镜 8（统计/追溯）录制使用。"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.ERROR)

from server import start_server_background  # noqa: E402
start_server_background()

import requests  # noqa: E402
BASE = "http://127.0.0.1:8017"
for _ in range(120):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()
win.goto("统计报表")

# 让图表加载完
deadline = time.time() + 8
while time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)
print("[统计] 统计报表页已就绪", flush=True)

# 停留 50 秒供录制
deadline = time.time() + 50
while time.time() < deadline:
    app.processEvents()
    time.sleep(0.05)
print("[完成] 统计页停留结束", flush=True)
sys.exit(0)
