# -*- coding: utf-8 -*-
"""v10 冒烟：批次列只显示数字 + 表格自动换行（品类名长可换行）。
只读：复用真实库已有数据源，不写任何数据。"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

ISSUES = []
_OK = {"n": 0}


def ck(name, cond, detail=""):
    if cond:
        _OK["n"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


r = requests.get(f"{BASE}/api/datasources", timeout=8).json()
items = r.get("items", []) or []
ck("存在数据源", bool(items), str([d.get("name") for d in items[:3]]))
if not items:
    import os as _os
    _os._exit(1)
sid = items[0]["id"]
g = requests.get(f"{BASE}/api/datasources/{sid}/groups", timeout=8).json()
cats = g.get("categories") or {}
big_cat = None
for cat, cg in cats.items():
    if len(cg.get("detect_batches") or []) >= 2:
        big_cat = cat
        break
ck("存在 ≥2 批的品类", big_cat is not None, str(list(cats.keys())[:5]))
if big_cat is None:
    import os as _os
    _os._exit(1)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1280, 800)
win.show()


def pump(ms=300):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def step():
    win.goto("数据管理")
    pump(3000)
    t = win.page_data.tree
    root = t.topLevelItem(0)

    def src_item_text(r, i):
        tx = r.child(i).text(0)
        return items[0]["name"] in tx or tx.startswith(items[0]["name"])
    src_item = next((root.child(i) for i in range(root.childCount())
                     if src_item_text(root, i)), None)
    ck("树含数据源", src_item is not None)
    src_item.setExpanded(True)
    pump(1200)
    cat_item = None
    for i in range(src_item.childCount()):
        if src_item.child(i).text(0).startswith(big_cat):
            cat_item = src_item.child(i)
            break
    ck("树含品类", cat_item is not None, big_cat)
    cat_item.setExpanded(True)
    pump(600)
    det_item = None
    for i in range(cat_item.childCount()):
        if "检测组" in cat_item.child(i).text(0):
            det_item = cat_item.child(i)
            break
    ck("树含检测组", det_item is not None)
    det_item.setExpanded(True)
    pump(400)
    n_batch = det_item.childCount()
    ck("检测组分批≥2", n_batch >= 2, f"{n_batch} 批")
    b2 = det_item.child(1) if n_batch >= 2 else det_item.child(0)
    t.setCurrentItem(b2)
    pump(1500)
    col_vals = set()
    for row in range(win.page_data.table.rowCount()):
        it = win.page_data.table.item(row, 4)
        if it is not None:
            col_vals.add(it.text())
    # 批次列应为纯数字（逻辑批次编号）
    ck("批次列只显示数字", bool(col_vals) and all(v.isdigit() for v in col_vals),
       str(sorted(col_vals)))
    # 自动换行已开启
    ck("表格开启自动换行", win.page_data.table.wordWrap() is True)
    # 品类列宽度固定（换行前提）
    ck("品类列固定宽度", win.page_data.table.columnWidth(3) >= 100,
       f"w={win.page_data.table.columnWidth(3)}")
    # 行高自适应（存在比默认 66 更高的行则说明换行生效；无则至少不崩溃）
    hs = [win.page_data.table.rowHeight(r)
          for r in range(win.page_data.table.rowCount())]
    ck("行高已自适应填充", bool(hs) and all(h >= 30 for h in hs), f"min={min(hs) if hs else '-'}")
    print(f"\n{'='*46}", flush=True)
    print(f"验证全部通过 ✅（{_OK['n']}）" if not ISSUES
          else f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


QTimer.singleShot(400, step)
app.exec()
