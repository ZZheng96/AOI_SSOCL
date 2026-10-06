# -*- coding: utf-8 -*-
"""验证：数据树展示所有数据源（含无数据源"1"，可管理）+ 未挂源数据集组。"""
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


def top_texts():
    t = win.page_data.tree
    return [t.topLevelItem(i).text(0) for i in range(t.topLevelItemCount())]


def wait_tree(cond, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(250)
        if cond(top_texts()):
            return True
        time.sleep(0.2)
    return False


def child_texts():
    t = win.page_data.tree
    out = []
    for i in range(t.topLevelItemCount()):
        root = t.topLevelItem(i)
        for j in range(root.childCount()):
            out.append(root.child(j).text(0))
    return out


def step():
    win.goto("数据管理")
    ok = wait_tree(lambda xs: any("未挂源" in x for x in xs))
    pump(800)
    d = win.page_data._client.images_tree()
    print("client source_groups:", list((d or {}).get("source_groups", {}).keys()), flush=True)
    print("source_mode:", getattr(win.page_data, "_source_mode", None), flush=True)
    xs = top_texts()
    print("树顶层:", xs, flush=True)
    ck("树展示无数据源「1」（可右键管理）", any(x == "1" for x in xs), str(xs))
    ck("树显示「（未挂源）」数据集组", any("未挂源" in x for x in xs),
       str([x for x in xs if "未挂源" in x]))
    # 展开"1"看占位
    t = win.page_data.tree
    for i in range(t.topLevelItemCount()):
        it = t.topLevelItem(i)
        if it.text(0) == "1":
            it.setExpanded(True)
            pump(300)
            ph = it.child(0).text(0) if it.childCount() else ""
            ck("无数据源「1」下显示占位提示", "暂无数据" in ph, ph)
            break
    # 源管理可用：_find_source 能找到"1"（右键编辑/删除入口基于它）
    src = win.page_data._find_source("1")
    ck("「1」源可被源管理定位（右键入口可用）", src is not None,
       str(src.get("id") if src else ""))
    # 工单对话框与数据树都基于同一 DataSource 表 → 对齐
    srcs = requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", [])
    ck("后端数据源列表含「1」", any(d.get("name") == "1" for d in srcs),
       str([d.get("name") for d in srcs]))
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


from PySide6.QtCore import QTimer  # noqa: E402
QTimer.singleShot(400, step)
app.exec()
