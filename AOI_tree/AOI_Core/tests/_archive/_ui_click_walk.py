# -*- coding: utf-8 -*-
"""AOI_sys 真实界面点击走查 v2（事件驱动，不阻塞模态对话框）。
进程内起后端 + 打开主窗口，模拟用户点击按钮走完整操作流，每步截图落盘 _ui_click_shots/。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)         # 压 urllib3 刷屏，聚焦走查输出

from server import start_server_background
start_server_background()

import requests
for _ in range(60):
    try:
        if requests.get("http://127.0.0.1:8017/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton, QTableWidget  # noqa: E402
import PySide6.QtWidgets as QW  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

# 弹窗统计：monkeypatch QMessageBox 静态方法，统计走查全程是否仍有弹窗触发
_QMB_COUNT = {"information": 0, "warning": 0, "question": 0}
for _m in ("information", "warning", "question"):
    _orig = getattr(QW.QMessageBox, _m)

    def _mk(m=_m, o=_orig):
        def _p(*a, **k):
            _QMB_COUNT[m] += 1
            print(f"  [弹窗] QMessageBox.{m} 被调用!", flush=True)
            return o(*a, **k)
        return _p
    setattr(QW.QMessageBox, _m, _mk())

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()
SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ui_click_shots")
ISSUES = []
STATE = {"cat": False, "i": 0}


def pump(ms=400):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def shot(name, page=None):
    os.makedirs(SHOT_DIR, exist_ok=True)
    STATE["i"] += 1
    p = os.path.join(SHOT_DIR, f"{STATE['i']:02d}_{name}.png")
    (page or win).grab().save(p)
    print(f"  [截图] {os.path.basename(p)}", flush=True)


def find_btn(widget, text):
    for b in widget.findChildren(QPushButton):
        if b.text() == text and not b.isHidden() and b.isEnabled():
            return b
    return None


def click_btn(widget, text, where="页面"):
    b = find_btn(widget, text)
    if b is None:
        ISSUES.append(f"找不到按钮: {text} ({where})")
        print(f"  [FAIL] 找不到按钮「{text}」({where})", flush=True)
        return False
    print(f"  [点击] {where} → {text}", flush=True)
    b.click()
    pump(600)
    return True


def later(ms, fn):
    QTimer.singleShot(ms, fn)


def wait_task_done(done_cb, timeout_s=180):
    t0 = time.time()

    def poll():
        if time.time() - t0 > timeout_s:
            ISSUES.append("后台任务超时")
            done_cb(False)
            return
        try:
            tasks = requests.get("http://127.0.0.1:8017/api/tasks/recent",
                                 timeout=3).json()
        except Exception:
            later(1500, poll)
            return
        items = tasks if isinstance(tasks, list) else tasks.get("items", [])
        if not items:
            later(1500, poll)
            return
        latest = items[0]
        if latest.get("status") in ("done", "success"):
            print(f"  [任务完成] {latest.get('kind') or latest.get('type')} "
                  f"{latest.get('progress')}% {latest.get('message', '')}", flush=True)
            done_cb(True)
        elif latest.get("status") in ("failed", "error"):
            ISSUES.append(f"后台任务失败: {latest.get('error')}")
            done_cb(False)
        else:
            later(1500, poll)
    later(800, poll)


# ═════════════════ 各阶段（事件驱动） ═════════════════
def s01_wait_category():
    idx = win.combo_category.findText("bracket_black")
    if idx < 0:
        later(500, s01_wait_category)
        return
    win.combo_category.setCurrentIndex(idx)
    pump(1200)
    print("[操作] 工作台首屏", flush=True)
    shot("01_工作台")
    win.goto("数据管理")
    later(1500, s02_data)


def s02_data():
    pump(800)
    print("[操作] 数据管理页", flush=True)
    click_btn(win.page_data, "查询", "数据管理")
    later(1800, s02_shot)


def s02_shot():
    shot("02_数据管理_查询", win.page_data)
    win.goto("模型管理")
    later(1500, s03_model)


def s03_model():
    pump(800)
    print("[操作] 模型管理页", flush=True)
    shot("03_模型管理", win.page_model)
    # 预注册：准备模型对话框 exec() 期间自动完成 开始准备 → 等任务 → 关闭
    later(800, auto_prepare_dialog)
    click_btn(win.page_model, "准备模型", "模型管理")
    # 对话框 exec 阻塞这里；自动流程跑完后继续


def auto_prepare_dialog():
    dlg = app.activeModalWidget()
    if dlg is None:
        later(800, auto_prepare_dialog)
        return
    pump(1500)                                    # precheck 体检
    shot("04_准备模型对话框", dlg)
    if not click_btn(dlg, "开始准备", "准备对话框"):
        dlg.close()
        later(500, s04_eval)
        return
    wait_task_done(lambda ok: after_prepare(dlg, ok), 200)


def after_prepare(dlg, ok):
    pump(1500)
    shot("05_准备完成", dlg)
    for b in dlg.findChildren(QPushButton):
        if b.text() == "关闭":
            b.click()
            break
    else:
        dlg.close()
    pump(600)
    later(500, s04_eval)


def s04_eval():
    win.goto("评估看板")
    later(1500, s04_run)


def s04_run():
    pump(800)
    print("[操作] 评估看板", flush=True)
    click_btn(win.page_eval, "运行评估", "评估看板")
    wait_task_done(lambda ok: s04_shot(ok), 300)


def s04_shot(ok):
    pump(1500)
    shot("06_评估结果", win.page_eval)
    win.goto("实时监控")
    later(1500, s05_monitor)


def s05_monitor():
    pump(800)
    print("[操作] 实时监控", flush=True)
    later(800, auto_pick_dialog)
    click_btn(win.page_monitor, "从图库选择…", "实时监控")
    # 图库对话框 exec 阻塞；自动流程跑完继续


def auto_pick_dialog():
    pick = app.activeModalWidget()
    if pick is None:
        later(800, auto_pick_dialog)
        return
    pump(2500)                                    # 图片列表加载
    shot("07_图库选择", pick)
    tbl = pick.findChild(QTableWidget)
    if tbl is None or tbl.rowCount() == 0:
        ISSUES.append("图库为空，无法检测")
        print("  [FAIL] 图库为空", flush=True)
        pick.close()
        later(800, s06_feedback)
        return
    items = getattr(pick, "_items", [])
    row = 0
    for i, it in enumerate(items):
        if str(it.get("label", "")) == "anomaly":
            row = i
            break
    tbl.setCurrentCell(row, 0)
    print(f"  [点击] 图库选择第 {row} 行 → 选择", flush=True)
    click_btn(pick, "选择", "图库对话框")
    later(3500, s05_detect_shot)


def s05_detect_shot():
    shot("08_检测结果", win.page_monitor)
    for label, text in (("误检", "标记误检"), ("漏检", "标记漏检"), ("正确", "确认正确")):
        if find_btn(win.page_monitor, text):
            later(800, lambda l=label: auto_feedback_dialog(l))
            click_btn(win.page_monitor, text, "监控反馈")
            return
    later(500, s06_feedback)


def auto_feedback_dialog(label):
    dlg = app.activeModalWidget()
    if dlg is None:
        later(800, lambda: auto_feedback_dialog(label))
        return
    pump(1500)
    shot(f"09_反馈_{label}", dlg)
    # 提交成功后的 QMessageBox.information 在嵌套事件循环里阻塞主线程；
    # 必须点击前注册定时器，由嵌套循环处理 timer 时自动关掉成功提示框。
    QTimer.singleShot(1200, lambda: _close_modal_until_gone(time.time() + 15))
    click_btn(dlg, "提交反馈", "反馈对话框")
    pump(4000)                                   # 期间成功提示框被自动关闭
    later(600, s06_feedback)


def _close_modal_until_gone(deadline_t):
    m = app.activeModalWidget()
    if m is None or time.time() > deadline_t:
        return
    m.close()                                    # 关闭 QMessageBox / 残留模态
    QTimer.singleShot(400, lambda: _close_modal_until_gone(deadline_t))


def s06_feedback():
    win.goto("标注反馈")
    later(1800, s06_shot)


def s06_shot():
    pump(800)
    print("[操作] 标注反馈", flush=True)
    shot("10_标注反馈", win.page_feedback)
    fb = win.page_feedback
    tabs = fb.findChild(QTableWidget, None) or None  # placeholder
    tw = None
    for w in fb.findChildren(QW.QTabWidget):
        tw = w
        break
    if tw is not None:
        tw.setCurrentIndex(1)                        # 反馈记录
        later(1500, lambda: s06_tab_shot(tw))
    else:
        later(300, s07_stats)


def s06_tab_shot(tabs):
    shot("11_反馈记录", win.page_feedback)
    later(300, s07_stats)


def s07_stats():
    win.goto("统计报表")
    later(1500, s07_run)


def s07_run():
    pump(800)
    print("[操作] 统计报表", flush=True)
    click_btn(win.page_stats, "刷新", "统计报表")
    later(1500, s07_shot)


def s07_shot():
    shot("12_统计报表", win.page_stats)
    summary()


def summary():
    print("\n" + "=" * 56, flush=True)
    print(f"[弹窗统计] QMessageBox 触发次数: {_QMB_COUNT}", flush=True)
    if ISSUES:
        print(f"[GUI 走查发现 {len(ISSUES)} 个问题]", flush=True)
        for it in ISSUES:
            print(f"  - {it}", flush=True)
    else:
        print("[GUI 走查未发现问题]", flush=True)
    print(f"[截图目录] {SHOT_DIR}", flush=True)
    win.close()
    app.quit()


if __name__ == "__main__":
    later(800, s01_wait_category)
    app.exec()
