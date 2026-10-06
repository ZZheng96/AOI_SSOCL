# -*- coding: utf-8 -*-
"""验证：工单对话框无数据源提示（含未挂源批次引导）+ 批次挂源接口。"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.CRITICAL)

from server import start_server_background
start_server_background()

import requests  # noqa: E402
BASE = "http://127.0.0.1:8017"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.widgets.workorder_dialogs import WorkOrderCreateDialog  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()
ISSUES = []
_OK = {"n": 0}


def ck(name, cond, detail=""):
    if cond:
        _OK["n"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


def pump(ms=300):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def step():
    # ── 1. 对话框空态提示（当前库无数据源但有未挂源批次）──
    print("\n[1] 工单对话框空态提示", flush=True)
    dlg = WorkOrderCreateDialog(win.client, win)
    dlg.show()
    t0 = time.time()
    txt = ""
    while time.time() - t0 < 10:
        pump(300)
        if dlg.list_sources.count() > 0:
            txt = dlg.list_sources.item(0).text()
            if "未挂数据源" in txt or "新建数据源" in txt:
                break
        time.sleep(0.2)
    ck("空态提示含「未挂数据源的批次」引导", "未挂数据源" in txt, txt.replace("\n", " "))
    ck("空态提示引导去「挂到数据源…」", "挂到数据源" in txt)
    dlg.close()
    pump(200)

    # ── 2. 挂源接口：新建源 → attach 批次 → 工单可选 ──
    print("\n[2] 批次挂到数据源", flush=True)
    dss = requests.get(f"{BASE}/api/datasets", timeout=8).json().get("items", [])
    ds = next((d for d in dss if d.get("datasource_id") is None), None)
    ck("存在未挂源批次", ds is not None, str(ds.get("category") if ds else ""))
    if ds:
        r = requests.post(f"{BASE}/api/datasources",
                          json={"name": "挂源验证", "modality": "image",
                                "label_tier": "L1a"}, timeout=8)
        sid = r.json()["id"]
        r = requests.post(f"{BASE}/api/datasets/{ds['id']}/attach"
                          f"?datasource_id={sid}", timeout=8)
        ck("attach 接口 200", r.status_code == 200, str(r.status_code))
        srcs = requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", [])
        s = next((x for x in srcs if x.get("id") == sid), None)
        cap = (s or {}).get("capability") or {}
        ck("挂源后数据源能力含该批次图片",
           (cap.get("n_images") or 0) > 0, f"n={cap.get('n_images')}")
        # 工单对话框此时应列出数据源
        dlg2 = WorkOrderCreateDialog(win.client, win)
        dlg2.show()
        t0 = time.time()
        n_items = 0
        while time.time() - t0 < 10:
            pump(300)
            n_items = dlg2.list_sources.count()
            if n_items and (dlg2.list_sources.item(0).flags() &
                            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemIsUserCheckable):
                break
            time.sleep(0.2)
        ck("挂源后工单对话框出现可选数据源", n_items >= 1, f"n={n_items}")
        dlg2.close()
        # 清理
        requests.delete(f"{BASE}/api/workorders/{win.current_workorder_id()}" if False else f"{BASE}/api/datasources/{sid}", timeout=8)
    finish()


def finish():
    print(f"\n{'='*46}", flush=True)
    if ISSUES:
        print(f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"验证全部通过 ✅（{_OK['n']}）", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


QTimer.singleShot(400, step)
app.exec()
