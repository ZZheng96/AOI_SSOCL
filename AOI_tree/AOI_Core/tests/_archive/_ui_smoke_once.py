# -*- coding: utf-8 -*-
"""验证：新建数据源一步式导入=一次弹窗（不再二次弹导入框）+ 路径浏览按钮。"""
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

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QLineEdit, QPushButton, QLabel)
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


def later(ms, fn):
    QTimer.singleShot(int(ms), fn)


def btn(dlg, text):
    for b in dlg.findChildren(QPushButton):
        if b.text().strip() == text:
            return b
    return None


def combo_by(dlg, substr):
    for c in dlg.findChildren(QComboBox):
        if any(substr in str(c.itemText(i)) for i in range(c.count())):
            return c
    return None


def labels_visible(dlg, text):
    for l in dlg.findChildren(QLabel):
        if l.text() == text and l.isVisible():
            return True
    return False


def open_dlg():
    win.goto("数据管理")
    pump(800)
    b = next((x for x in win.page_data.findChildren(QPushButton)
              if x.text().strip() == "新建数据源"), None)
    if b is None:
        print("❌ 无新建数据源按钮", flush=True)
        return os._exit(1)
    later(400, fill1)
    b.click()


def fill1():
    dlg = active_modal()
    if dlg is None:
        print("❌ 建源对话框未弹出", flush=True)
        return os._exit(1)
    ck("初始「数据集根目录」行隐藏", not labels_visible(dlg, "数据集根目录"))
    c = combo_by(dlg, "标准数据集")
    ck("有「导入数据」下拉", c is not None)
    for i in range(c.count()):
        if "标准数据集" in str(c.itemText(i)):
            c.setCurrentIndex(i)
            break
    pump(300)
    ck("选标准数据集后「数据集根目录」行可见", labels_visible(dlg, "数据集根目录"))
    ck("有「数据集根目录 浏览…」按钮", btn(dlg, "浏览…") is not None)
    ck("按钮变「创建并导入」", btn(dlg, "创建并导入") is not None)
    # 填名称 + 根目录
    edits = dlg.findChildren(QLineEdit)
    edits[0].setText("一次弹窗源")
    for e in dlg.findChildren(QLineEdit):
        if e.placeholderText() and "根目录" in e.placeholderText():
            e.setText(r"<PROJECT_ROOT>/data_origin/mvtec")
            break
    pump(200)
    later(200, click_ok)


def click_ok():
    dlg = active_modal()
    b = btn(dlg, "创建并导入")
    if b is None:
        print("❌ 无创建并导入按钮", flush=True)
        return os._exit(1)
    b.click()
    # 点击后不阻塞：等 1.5s 检查是否又弹第二个对话框（应无）
    later(1500, check_once)


def check_once():
    m = active_modal()
    ck("点击后不再弹第二个对话框（一次弹窗）", m is None,
       m.windowTitle() if m else "")
    # 后端应已建源 + 提交 import 任务
    srcs = requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", [])
    s = next((x for x in srcs if x.get("name") == "一次弹窗源"), None)
    ck("数据源已创建", s is not None)
    tasks = requests.get(f"{BASE}/api/tasks/recent", timeout=8).json().get("items", [])
    imp = next((t for t in tasks if t.get("task_type") == "import"), None)
    ck("已直接提交导入任务（不再经过第二个对话框）", imp is not None,
       str(imp.get("status") if imp else ""))
    # 清理
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


later(300, open_dlg)
app.exec()
