# -*- coding: utf-8 -*-
"""UI 冒烟：数据树显示 数据源 → 品类 → 预训练组/检测组；未挂源一键归入。"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.CRITICAL)
from server import start_server_background
start_server_background()
import requests

BASE = "http://127.0.0.1:8017"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

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


def tree_lines(node=None, depth=0, out=None):
    t = win.page_data.tree
    out = [] if out is None else out
    items = [t.topLevelItem(i) for i in range(t.topLevelItemCount())] if node is None \
        else [node.child(i) for i in range(node.childCount())]
    for it in items:
        out.append(("  " * depth) + it.text(0))
        if it.childCount():
            tree_lines(it, depth + 1, out)
    return out


def step():
    # 先清理
    for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)
    # 建源 + 存量归入
    r = requests.post(f"{BASE}/api/datasources",
                      json={"name": "UI分组源", "modality": "image",
                            "label_tier": "L1a",
                            "pretrain_normal": 100, "pretrain_anomaly": 30,
                            "batch_size": 30}, timeout=8)
    sid = r.json()["id"]
    requests.post(f"{BASE}/api/datasources/{sid}/attach-legacy", timeout=8)
    win.goto("数据管理")
    pump(2500)
    lines = tree_lines()
    print("== 树 ==", flush=True)
    for l in lines:
        print("  " + l, flush=True)
    txt = "\n".join(lines)
    ck("树含数据源「UI分组源」", "UI分组源" in txt)
    ck("树显示「预训练组（100 正常 + 30 异常）」", "预训练组（100 正常 + 30 异常）" in txt)
    ck("树显示「检测组（…批）」", any("检测组" in l and "批" in l for l in lines))
    ck("树含品类 zipper", any(l.strip() == "zipper" for l in lines))
    # 清理
    requests.delete(f"{BASE}/api/datasources/{sid}", timeout=8)
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
