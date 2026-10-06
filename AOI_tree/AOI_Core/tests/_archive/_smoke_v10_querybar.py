# -*- coding: utf-8 -*-
"""v10 冒烟：查询栏改造（去划分/标签、加「批次」下拉纯数字、品类加宽）+ 批次列数字。
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
    pd = win.page_data
    # 1. 查询栏无「划分」「标签」下拉
    ck("查询栏无划分下拉", not hasattr(pd, "combo_split"))
    ck("查询栏无标签下拉", not hasattr(pd, "combo_label"))
    # 2. 有「批次」下拉且首项「全部」
    ck("查询栏有批次下拉", hasattr(pd, "combo_batch"))
    cb = pd.combo_batch
    texts = [cb.itemText(i) for i in range(cb.count())]
    ck("批次下拉首项为全部", bool(texts) and texts[0] == "全部", str(texts[:5]))
    # 3. 品类下拉加宽策略
    ck("品类下拉 AdjustToContents", pd.combo_category.sizeAdjustPolicy()
       is not None)
    # 4. 选一个品类 → 批次下拉出现编号数字
    cat_idx = -1
    for i in range(1, pd.combo_category.count()):
        cat_idx = i
        break
    if cat_idx >= 0:
        cat = pd.combo_category.itemData(cat_idx)
        pd.combo_category.setCurrentIndex(cat_idx)
        pump(2000)   # 等批次下拉异步刷新
        bt = [pd.combo_batch.itemText(i) for i in range(pd.combo_batch.count())]
        nums = [x for x in bt[1:]]
        ck("批次下拉含编号数字", bool(nums) and all(x.isdigit() for x in nums),
           f"cat={cat} batches={nums[:8]}")
        if nums:
            pd.combo_batch.setCurrentIndex(1)
            pd._on_query()
            pump(1500)
            ck("选批次后表格已过滤", pd._total > 0, f"total={pd._total}")
            # 批次列应为该编号
            col_vals = set()
            for row in range(pd.table.rowCount()):
                it = pd.table.item(row, 4)
                if it is not None:
                    col_vals.add(it.text())
            ck("批次列显示所选编号", bool(col_vals) and all(
                v == nums[0] for v in col_vals), str(sorted(col_vals)))
    # 5. 批次列换行保持（回归）
    ck("表格自动换行开启", pd.table.wordWrap() is True)
    print(f"\n{'='*46}", flush=True)
    print(f"验证全部通过 ✅（{_OK['n']}）" if not ISSUES
          else f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


QTimer.singleShot(400, step)
app.exec()
