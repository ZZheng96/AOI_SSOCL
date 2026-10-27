# -*- coding: utf-8 -*-
"""验证产线路径 detect_image(with_heatmap=True) 是否落热力图/叠加图。"""
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

BASE = "http://127.0.0.1:8017"
try:
    if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") != "ok":
        raise RuntimeError("bad")
    print("[启动] 复用已运行后端", flush=True)
except Exception:  # noqa: BLE001
    from server import start_server_background
    start_server_background()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    print("[启动] 后端就绪", flush=True)

from backend.pipeline.service import get_detection_service  # noqa: E402

svc = get_detection_service()
db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "database", "aoi.db")
c = sqlite3.connect(db)
row = c.execute("select path from images where category='02' "
                "and split='test' limit 1").fetchone()
print("img:", row[0], flush=True)
t0 = time.perf_counter()
out = svc.detect_image(row[0], "02", persist=True, with_heatmap=True,
                       target_type="pipeline", workorder_id=1)
ms = (time.perf_counter() - t0) * 1000
print(f"heatmap_path: {out.get('heatmap_path')}", flush=True)
print(f"overlay_path: {out.get('overlay_path')}", flush=True)
print(f"wall: {ms:.0f}ms  e2e: {out.get('latency_e2e_ms')}", flush=True)
