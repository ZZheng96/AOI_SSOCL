# -*- coding: utf-8 -*-
"""验证：未挂源数据集组 → 右键批量挂到数据源（挂载后可编辑/删除）。"""
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

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QLineEdit,  # noqa: E402
                               QPushButton)
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


def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


# 清理残留
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    if d.get("name") == "批量源":
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)


def fill_dialog():
    dlg = active_modal()
    if dlg is None:
        print("❌ 挂源对话框未弹出", flush=True)
        return os._exit(1)
    edits = dlg.findChildren(QLineEdit)
    if edits:
        edits[0].setText("批量源")
    pump(200)
    for b in dlg.findChildren(QPushButton):
        if b.text().strip() == "确定":
            b.click()
            break
    # 等待批量挂载完成（不阻塞 exec 返回，用 QTimer 检查）
    QTimer.singleShot(2500, verify)


def verify():
    srcs = requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", [])
    s = next((x for x in srcs if x.get("name") == "批量源"), None)
    cap = (s or {}).get("capability") or {}
    ck("「mvtec（未挂源）」组批量挂源后数据源含 5354 图",
       (cap.get("n_images") or 0) == 5354, f"n={cap.get('n_images')}")
    # mvtec 批次全部挂源
    dss = requests.get(f"{BASE}/api/datasets", timeout=8).json().get("items", [])
    mv = [d for d in dss if str(d.get("dataset_name") or "") == "mvtec"]
    ck("mvtec 组全部批次已挂源",
       len(mv) > 0 and all(d.get("datasource_id") is not None for d in mv),
       f"n={len(mv)}")
    # 清理：删源 → 批次回未挂源
    if s:
        requests.delete(f"{BASE}/api/datasources/{s['id']}", timeout=8)
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


def step():
    win.goto("数据管理")
    pump(800)
    # 探针：观察 _work 内 list_datasets / attach_dataset 实际行为
    cl = win.page_data._client
    _ol = cl.list_datasets

    def _spy_list(category=""):
        r = _ol(category)
        items = (r or {}).get("items", [])
        mv = [d for d in items if str(d.get("dataset_name") or "") == "mvtec"]
        print(f"  [spy] list_datasets n={len(items)} mvtec组={len(mv)}", flush=True)
        return r
    cl.list_datasets = _spy_list
    _oa = cl.attach_dataset

    def _spy_attach(ds, sid):
        r = _oa(ds, sid)
        print(f"  [spy] attach {ds}->{sid}: {bool(r)}", flush=True)
        return r
    cl.attach_dataset = _spy_attach
    print("  [spy] _unlinked_map:", getattr(win.page_data, "_unlinked_map", None),
          flush=True)
    # 触发「mvtec（未挂源）」组挂源（内部弹「挂到数据源」对话框）
    QTimer.singleShot(700, fill_dialog)
    win.page_data._attach_group_to_source("mvtec（未挂源）")


QTimer.singleShot(400, step)
app.exec()
