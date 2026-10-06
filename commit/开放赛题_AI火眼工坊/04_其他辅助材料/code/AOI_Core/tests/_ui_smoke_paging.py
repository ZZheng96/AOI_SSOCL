# -*- coding: utf-8 -*-
"""分页冒烟：数据管理页 + 伪异常页 缩略图分页与缓存验证。"""
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

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.pages.common import _THUMB_CACHE  # noqa: E402

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


def wait_cond(name, cond, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(200)
        if cond():
            return True
        time.sleep(0.2)
    return False


def btn(page, text):
    return next((b for b in page.findChildren(QPushButton)
                 if b.text().strip() == text and b.isVisible()), None)


def step1():
    print("\n[数据管理页] 分页", flush=True)
    win.goto("数据管理")
    pump(800)
    ok = wait_cond("页码出现", lambda: win.page_data.lbl_img_page.text() != "第 1 / 1 页"
                   or win.page_data.lbl_img_count.text() != "共 0 条")
    txt = win.page_data.lbl_img_page.text()
    cnt = win.page_data.lbl_img_count.text()
    ck("分页栏显示总条数/页码", ok, f"{cnt} {txt}")
    ck("有「下一页」按钮", btn(win.page_data, "下一页") is not None)
    b_next = btn(win.page_data, "下一页")
    b_prev = btn(win.page_data, "上一页")
    if b_next is not None and b_next.isEnabled():
        b_next.click()
        ok = wait_cond("翻到第 2 页", lambda: win.page_data.lbl_img_page.text().startswith("第 2"))
        ck("点「下一页」翻到第 2 页", ok, win.page_data.lbl_img_page.text())
    if b_prev is not None and b_prev.isEnabled():
        b_prev.click()
        ok = wait_cond("回到第 1 页", lambda: win.page_data.lbl_img_page.text().startswith("第 1"))
        ck("点「上一页」回到第 1 页", ok, win.page_data.lbl_img_page.text())
    # 缓存命中：再次刷新不重新下载（缓存应已累积）
    n0 = len(_THUMB_CACHE)
    win.page_data.reload_images()
    wait_cond("刷新完成", lambda: True, timeout=1)
    pump(1500)
    ck("缩略图缓存已累积", len(_THUMB_CACHE) > 0, f"cache={len(_THUMB_CACHE)}")
    later_step(step2)


def step2():
    print("\n[伪异常页] 分页", flush=True)
    win.goto("伪异常与增强")
    pump(800)
    pg = win.page_augment
    ok = wait_cond("页码出现", lambda: pg.lbl_base_count.text() != "共 0 条")
    ck("基底图列表分页栏显示总条数", ok, f"{pg.lbl_base_count.text()} {pg.lbl_base_page.text()}")
    b_next = btn(pg, "下一页")
    b_prev = btn(pg, "上一页")
    if b_next is not None and b_next.isEnabled():
        b_next.click()
        ok = wait_cond("翻到第 2 页", lambda: pg.lbl_base_page.text().startswith("第 2"))
        ck("伪异常页点「下一页」翻页", ok, pg.lbl_base_page.text())
    if b_prev is not None and b_prev.isEnabled():
        b_prev.click()
        ok = wait_cond("回到第 1 页", lambda: pg.lbl_base_page.text().startswith("第 1"))
        ck("伪异常页点「上一页」回第 1 页", ok, pg.lbl_base_page.text())
    later_step(finish)


def later_step(fn):
    from PySide6.QtCore import QTimer
    QTimer.singleShot(300, fn)


def finish():
    print(f"\n{'='*48}", flush=True)
    pump(500)
    if ISSUES:
        print(f"分页冒烟发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"分页冒烟全部通过 ✅（{_OK['n']}）", flush=True)
    # os._exit：跳过 Qt 关闭时对仍在下载的缩略图线程的析构崩溃
    import os as _os
    _os._exit(1 if ISSUES else 0)


later_step(step1)
app.exec()
