# -*- coding: utf-8 -*-
"""实际打开 AOI_sys 桌面 UI 并操作：选 MPDD(bracket_black) → 实时监控 → 从图库选缺陷图 → 检测 → 出结果截图。
复用 main.py 已启动的 8017 后端（不再重复起服务），打开真实 MainWindow 事件驱动操作。
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from ui.main_window import MainWindow
from ui.theme import apply_light_theme

import requests, json
for _ in range(40):
    try:
        if requests.get("http://127.0.0.1:8017/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)
print("[操作] 复用 8017 后端（main.py 已启动）", flush=True)

app = QApplication([])
app.setFont(QFont("Microsoft YaHei UI", 10))
apply_light_theme(app)
win = MainWindow()
win.resize(1440, 900)
win.show()

SHOT_DIR = r"D:\CGAIC\AOI_sys\_operate_shots"
ISSUES = []

def pump(ms=400):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)

def shot(name):
    os.makedirs(SHOT_DIR, exist_ok=True)
    p = os.path.join(SHOT_DIR, name + ".png")
    win.grab().save(p)
    print("  [截图] " + p, flush=True)

def find_btn(widget, text):
    from PySide6.QtWidgets import QPushButton
    for b in widget.findChildren(QPushButton):
        if b.text() == text and not b.isHidden() and b.isEnabled():
            return b
    return None

def click_btn(widget, text, where):
    b = find_btn(widget, text)
    if b is None:
        ISSUES.append("找不到按钮: %s (%s)" % (text, where))
        print("  [FAIL] 找不到按钮「%s」(%s)" % (text, where), flush=True)
        return False
    print("  [点击] %s → %s" % (where, text), flush=True)
    b.click()
    pump(700)
    return True

def later(ms, fn):
    QTimer.singleShot(ms, fn)

# 在事件驱动中操作
def step1_select_category():
    idx = win.combo_category.findText("bracket_black")
    if idx < 0:
        later(400, step1_select_category)
        return
    win.combo_category.setCurrentIndex(idx)
    pump(1500)
    print("[操作] 已选中品类 bracket_black (MPDD), 当前页=%s" % win.currentTitle, flush=True)
    shot("01_工作台_MPDD")
    win.goto("实时监控")
    later(1500, step2_monitor)

def step2_monitor():
    pump(800)
    print("[操作] 实时监控页", flush=True)
    shot("02_实时监控", win.page_monitor)
    later(800, auto_pick)
    click_btn(win.page_monitor, "从图库选择…", "实时监控")

def auto_pick():
    from PySide6.QtWidgets import QTableWidget
    pick = app.activeModalWidget()
    if pick is None:
        later(800, auto_pick)
        return
    pump(2500)
    shot("03_图库选择", pick)
    tbl = pick.findChild(QTableWidget)
    if tbl is None or tbl.rowCount() == 0:
        ISSUES.append("图库为空")
        print("  [FAIL] 图库为空", flush=True)
        pick.close()
        later(800, finish)
        return
    items = getattr(pick, "_items", [])
    row = 0
    for i, it in enumerate(items):
        if str(it.get("label", "")) == "anomaly":
            row = i
            break
    tbl.setCurrentCell(row, 0)
    print("  [点击] 图库第 %d 行(%s) → 选择" % (row, str(items[row].get("label")) if row < len(items) else "?"), flush=True)
    click_btn(pick, "选择", "图库对话框")
    later(3500, step3_result)

def step3_result():
    shot("04_检测结果", win.page_monitor)
    later(400, finish)

def finish():
    print("\n==== 操作结果 ====", flush=True)
    if ISSUES:
        print("[发现 %d 个问题]" % len(ISSUES), flush=True)
        for it in ISSUES:
            print("  - " + it, flush=True)
    else:
        print("[UI 操作流畅，未发现按钮缺失/阻塞问题]", flush=True)
    print("窗口保持打开；3s 后关闭我的操作窗口（main.py 的真实窗口仍在线）", flush=True)
    later(3000, lambda: (win.close(), app.quit()))

if __name__ == "__main__":
    later(600, step1_select_category)
    app.exec()
