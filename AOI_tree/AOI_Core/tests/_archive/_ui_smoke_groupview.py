# -*- coding: utf-8 -*-
"""UI 冒烟：①删除图像可用 ②预训练组/检测组/批次可点击查看 ③批次按 30 图划分展示。"""
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


def find_item_by_text(text, node=None):
    t = win.page_data.tree
    items = [t.topLevelItem(i) for i in range(t.topLevelItemCount())] if node is None \
        else [node.child(i) for i in range(node.childCount())]
    for it in items:
        if it.text(0) == text or (text in it.text(0) and len(it.text(0)) < len(text) + 20):
            if it.text(0) == text or text in it.text(0):
                return it
        r = find_item_by_text(text, it)
        if r:
            return r
    return None


def step():
    # 清理 + 建源归入
    for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)
    r = requests.post(f"{BASE}/api/datasources",
                      json={"name": "V源", "modality": "image", "label_tier": "L1a",
                            "pretrain_normal": 100, "pretrain_anomaly": 30,
                            "batch_size": 30}, timeout=8)
    sid = r.json()["id"]
    requests.post(f"{BASE}/api/datasources/{sid}/attach-legacy", timeout=8)
    win.goto("数据管理")
    pump(2500)

    # ① 树里检测组下有具体批次节点
    t = win.page_data.tree
    root = t.topLevelItem(0)
    src_item = None
    for i in range(root.childCount()):
        if "V源" in root.child(i).text(0):
            src_item = root.child(i)
            break
    src_item.setExpanded(True)
    pump(500)
    zipper_item = None
    for i in range(src_item.childCount()):
        if src_item.child(i).text(0) == "zipper":
            zipper_item = src_item.child(i)
            break
    ck("树含品类 zipper", zipper_item is not None)
    det_item = None
    pre_item = None
    for i in range(zipper_item.childCount()):
        tx = zipper_item.child(i).text(0)
        if "检测组" in tx:
            det_item = zipper_item.child(i)
        if "预训练组" in tx:
            pre_item = zipper_item.child(i)
    det_item.setExpanded(True)
    pump(300)
    batch_items = [det_item.child(i).text(0)
                   for i in range(det_item.childCount())]
    ck("③ 检测组下展开为多个批次（每批≤30）",
       len(batch_items) >= 2, str(batch_items))
    ck("③ 批次命名含图数", all("图" in b for b in batch_items))

    # ② 点击预训练组 → 表格 130 图
    t.setCurrentItem(pre_item)
    pump(1200)
    ck("② 点预训练组 → 表格 total=130", win.page_data._total == 130,
       f"total={win.page_data._total}")
    # ② 点击批次 1 → 表格 30 图
    b1 = det_item.child(0)
    t.setCurrentItem(b1)
    pump(1200)
    ck("② 点批次1 → 表格 total=30", win.page_data._total == 30,
       f"total={win.page_data._total}")

    # ① 删除图片：取一张有检测记录的图删（模拟"没反应"场景）
    img_id = 393   # 历史检测过的 zipper 图
    rr = requests.delete(f"{BASE}/api/images/{img_id}", timeout=8)
    ck("① 删除有检测记录的图片成功", rr.status_code == 200,
       f"code={rr.status_code}")
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
