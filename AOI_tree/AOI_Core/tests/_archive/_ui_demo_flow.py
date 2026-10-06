# -*- coding: utf-8 -*-
"""可见窗口全流程演示 v2：建源→工单→预训练→产线→反馈→复核→作废→学习→回队。

基于 _ui_walk_flow.py 扩展：
- 可见模式（不设 offscreen），用户可实时看到窗口自动操作；
- 补充细节步骤：复核判缺陷（feedback_type=review 落库）、反馈作废（invalidated）、
  监控页实时流订阅（detections/live 按工单过滤有帧）。
- 长任务（导入/预训练）有终端进度输出；超时放宽到 900s。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.ERROR)

from server import start_server_background
start_server_background()

import requests  # noqa: E402
BASE = "http://127.0.0.1:8017"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QDialog, QLineEdit, QPushButton, QListWidget)
import PySide6.QtWidgets as QW  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

ISSUES = []
_OK = {"n": 0}
_QMB = {"n": 0}
for _m in ("information", "warning", "question", "critical"):
    _orig = getattr(QW.QMessageBox, _m)

    def _mk(m=_m, o=_orig):
        def _p(*a, **k):
            _QMB["n"] += 1
            print(f"  [弹窗] QMessageBox.{m}（自动应答）", flush=True)
            # 自动化全流程：确认框直接「是」（确认操作的对象均为演示测试数据）；
            # 其余弹窗（info/warning/critical）无返回值
            return QW.QMessageBox.Yes if m == "question" else None
        return _p
    setattr(QW.QMessageBox, _m, _mk())

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()

SRC = "演示源"
WO = "演示工单"
MVTEC_ROOT = r"<PROJECT_ROOT>/data_origin/mvtec"
CATEGORY = "component"   # 演示工单首品类（导入后动态修正）


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


def later(ms, fn):
    def _safe():
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"步骤异常：{type(e).__name__}: {e}")
            print(f"  [异常] {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            finish()
    QTimer.singleShot(int(ms), _safe)


def safe_get(url, timeout=8, retries=4, **kw):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            return r
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            pump(300)
            time.sleep(0.5)
    return None


def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


def buttons(dlg, text):
    return [b for b in dlg.findChildren(QPushButton)
            if b.text().strip() == text and b.isVisible()]


def click_by_text(dlg, text):
    bs = buttons(dlg, text)
    if not bs:
        ISSUES.append(f"找不到按钮「{text}」")
        print(f"  ❌ 找不到按钮「{text}」", flush=True)
        return False
    bs[0].click()
    return True


def wait_task(tid, timeout=900):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        pump(150)
        try:
            last = safe_get(f"{BASE}/api/tasks/{tid}", timeout=8).json()
        except Exception as e:  # noqa: BLE001
            last = {"_err": str(e)}
        if last.get("status") in ("done", "failed"):
            return last
        if int(time.time() - t0) % 15 < 1:
            print(f"  [等待] task={tid} {last.get('status')} "
                  f"{last.get('progress')} {last.get('message', '')}", flush=True)
        time.sleep(2)
    return {"status": "timeout", "id": tid}


def latest_task(task_type, baseline=0):
    rows = safe_get(f"{BASE}/api/tasks/recent", timeout=8).json().get("items", [])
    return next((t for t in rows
                 if t.get("task_type") == task_type and t.get("id", 0) > baseline),
                None)


def det_total():
    r = safe_get(f"{BASE}/api/detections?page=1&page_size=1", timeout=8)
    b = r.json()
    return b.get("total", 0) if isinstance(b, dict) else 0


def live_total():
    """监控页实时流（按当前工单）当前拉到的帧数。"""
    return len(page_monitor._frames)


# ═══════════ 0. 清理残留 ═══════════
for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                       timeout=5).json().get("items", []):
    requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
print("[清理] 完成", flush=True)

page_data = win.page_data
page_monitor = win.page_monitor
page_dashboard = win.page_dashboard
page_model = win.page_model
page_feedback = win.page_feedback
T0 = time.time()


# ═══════════ 1. 数据管理：新建数据源（标准数据集导入）═══════════
def step_datasource():
    print("\n[UI-1] 数据管理：新建数据源（标准数据集导入）", flush=True)
    win.goto("数据管理")
    pump(600)
    b = next((x for x in page_data.findChildren(QPushButton)
              if x.text().strip() == "新建数据源"), None)
    if b is None:
        ck("数据管理页有「新建数据源」按钮", False, "找不到")
        return later(200, step_dashboard)
    ck("数据管理页有「新建数据源」按钮", True)
    later(500, _fill_create_source)
    b.click()


def _fill_create_source():
    dlg = active_modal()
    if dlg is None:
        ck("新建数据源对话框弹出", False)
        return later(200, step_dashboard)
    edits = dlg.findChildren(QLineEdit)
    if edits:
        edits[0].setText(SRC)
    combo_import = next((c for c in dlg.findChildren(QComboBox)
                         if any("标准数据集" in str(c.itemText(i))
                                for i in range(c.count()))), None)
    if combo_import is not None:
        for i in range(combo_import.count()):
            if "标准数据集" in str(combo_import.itemText(i)):
                combo_import.setCurrentIndex(i)
                break
    combo_tier = next((c for c in dlg.findChildren(QComboBox)
                       if any("图像级标注" in str(c.itemText(i))
                              for i in range(c.count()))), None)
    if combo_tier is not None:
        combo_tier.setCurrentIndex(1)   # L1a
    for e in dlg.findChildren(QLineEdit):
        if e.placeholderText() and "根目录" in e.placeholderText():
            e.setText(MVTEC_ROOT)
            break
    pump(200)
    global IMPORT_BASE
    IMPORT_BASE = max((t.get("id") or 0)
                      for t in safe_get(f"{BASE}/api/tasks/recent",
                                        timeout=8).json().get("items", []))
    click_by_text(dlg, "创建并导入")
    later(300, step_import_wait)


def step_import_wait():
    print("  [等待] MVTec 标准数据集导入…（预计 1-3 分钟）", flush=True)
    tid = None
    t0 = time.time()
    while time.time() - t0 < 90:
        pump(200)
        t = latest_task("import", IMPORT_BASE)
        if t:
            tid = t["id"]
            break
        time.sleep(1)
    ck("识别到导入任务", tid is not None)
    if tid is None:
        return later(200, step_dashboard)
    last = wait_task(tid, timeout=900)
    ok = last.get("status") == "done"
    ck("标准数据集导入完成", ok,
       str(last.get("status")) + (f" {last.get('message')}" if last.get("message") else ""))
    r = safe_get(f"{BASE}/api/datasources", timeout=8)
    src = next((x for x in r.json().get("items", []) if x.get("name") == SRC), None)
    cap = (src or {}).get("capability") or {}
    ck("源挂载 ≥391 张", cap.get("n_images", 0) >= 391, f"n_images={cap.get('n_images')}")
    later(200, step_dashboard)


# ═══════════ 2. 工作台：新建工单 ═══════════
def step_dashboard():
    print("\n[UI-2] 工作台：新建工单（挂源 + 人工复判）", flush=True)
    win.goto("工作台")
    pump(600)
    b = next((x for x in page_dashboard.findChildren(QPushButton)
              if "新建工单" in x.text()), None)
    if b is None:
        ck("工作台有「新建工单」按钮", False)
        return later(200, step_model)
    ck("工作台有「新建工单」按钮", True)
    later(700, _fill_create_wo)
    b.click()


def _fill_create_wo(attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _fill_create_wo(attempt + 1))
        ck("新建工单对话框弹出", False)
        return later(200, step_model)
    edits = dlg.findChildren(QLineEdit)
    if edits:
        edits[0].setText(WO)
    lw = dlg.findChild(QListWidget)
    if lw is None or lw.count() == 0 or not (
            lw.item(0).flags() & Qt.ItemIsUserCheckable):
        if attempt < 12:
            return later(500, lambda: _fill_create_wo(attempt + 1))
        ck("建工单对话框数据源列表加载", False)
        return later(200, step_model)
    lw.item(0).setCheckState(Qt.Checked)
    chks = dlg.findChildren(QCheckBox)
    if chks:
        chks[0].setChecked(True)   # 人工复判
    pump(300)
    click_by_text(dlg, "创建工单")
    later(600, step_wo_verify)


def step_wo_verify():
    r = safe_get(f"{BASE}/api/workorders", params={"range": "all"}, timeout=8)
    wo = next((w for w in r.json().get("items", []) if w.get("name") == WO), None)
    ck("UI 建工单成功", wo is not None)
    if wo:
        global CATEGORY
        cats = wo.get("categories") or []
        if cats:
            CATEGORY = str(cats[0])   # 演示工单首品类（动态，非硬编码 zipper）
        ck("工单聚合条件 L1a/L1b", wo.get("conditions", {}).get("label_tier")
           in ("L1a", "L1b"), str(wo.get("conditions")))
        ck("已开启人工复判", bool(wo.get("review_enabled")))
        ck("工单覆盖品类", bool(cats), ",".join(cats))
    pump(500)
    t0 = time.time()
    cur = ""
    while time.time() - t0 < 8:
        pump(300)
        cur = win.combo_workorder.currentText()
        if cur == WO:
            break
        time.sleep(0.3)
    ck("顶栏工单下拉联动", cur == WO, cur)
    later(200, step_model)


# ═══════════ 3. 模型管理：准备模型 ═══════════
def step_model():
    print("\n[UI-3] 模型管理：准备模型（100+30 预训练，预计 3-10 分钟）", flush=True)
    win.goto("模型管理")
    pump(900)
    tree = getattr(page_model, "tree", None)
    if tree is not None:
        for _a in range(10):
            top = tree.topLevelItem(0)
            if top is None:
                pump(800)
                continue
            top.setExpanded(True)
            picked = None
            for i in range(top.childCount()):
                ch = top.child(i)
                d = ch.data(0, Qt.UserRole)
                if isinstance(d, tuple) and d[0] == "category":
                    if CATEGORY in str(ch.text(0)):
                        picked = ch
                        break
                    if picked is None:
                        picked = ch
            if picked is not None:
                tree.setCurrentItem(picked)
                pump(700)
                break
            pump(800)
    b = next((x for x in page_model.findChildren(QPushButton)
              if "准备模型" in x.text()), None)
    if b is None:
        ck("模型页「准备模型」按钮可用", False)
        return later(200, step_monitor)
    ck("模型页「准备模型」按钮可用", True)
    later(900, _fill_prepare)
    b.click()


def _fill_prepare():
    dlg = active_modal()
    if dlg is None:
        ck("准备模型对话框弹出", False)
        return later(200, step_monitor)
    ck("准备模型对话框弹出", True)
    pump(700)
    try:
        # 新版对话框（2026-08-29 重构）：品类/数据条件由模型页树选中
        # 只读带入，只需选运行模式（fast）与重备开关
        profile = getattr(dlg, "cmb_profile", None)
        if profile is not None and profile.count() > 0:
            profile.setCurrentIndex(0)   # fast（单图 <200ms，产线实时）
        chk = getattr(dlg, "chk_force", None)
        if chk is not None:
            chk.setChecked(True)
    except Exception as e:  # noqa: BLE001
        ISSUES.append(f"准备对话框配置异常: {e}")
        print(f"  ❌ 准备对话框配置异常: {e}", flush=True)
    pump(200)
    global PREP_BASE
    PREP_BASE = max((t.get("id") or 0)
                    for t in safe_get(f"{BASE}/api/tasks/recent",
                                      timeout=8).json().get("items", []))
    click_by_text(dlg, "开始准备")

    def _wait_prepare():
        print("  [等待] 100+30 预训练…（终端进度每 15s 刷新）", flush=True)
        tid = None
        t0 = time.time()
        while time.time() - t0 < 90:
            pump(200)
            t = latest_task("model_prepare", PREP_BASE)
            if t:
                tid = t["id"]
                break
            time.sleep(1)
        ck("识别到预训练任务", tid is not None)
        if tid is None:
            _close_dlg(dlg)
            return later(200, step_monitor)
        last = wait_task(tid, timeout=900)
        ok = last.get("status") == "done"
        ck("100+30 预训练完成", ok,
           str(last.get("status")) + (f" {last.get('message')}" if last.get("message") else ""))
        res = last.get("result") or {}
        # 样本量随数据源分组方案而异（演示源预训练组仅 8 张），不硬断言 100+30
        print(f"  [信息] 预训练样本 normal={res.get('n_normal')} "
              f"defect={res.get('n_defect')} val={res.get('n_val')}", flush=True)
        ver = res.get("version")
        ck("预训练生成模型快照",
           (ver not in (None, "")) or _model_versions() >= 1,
           f"version={ver} models={_model_versions()}")
        pump(300)
        _close_dlg(dlg)
        later(300, step_monitor)

    later(300, _wait_prepare)


def _close_dlg(dlg):
    """关闭残留对话框（准备弹窗失败/完成时兜底，避免卡死流程）；找不到关闭
    按钮时静默 reject（对话框可能已自行关闭，非失败）。"""
    try:
        if dlg is None:
            return
        if buttons(dlg, "关闭") or buttons(dlg, "取消"):
            click_by_text(dlg, "关闭") or click_by_text(dlg, "取消")
            pump(400)
            return
        dlg.reject()
        pump(400)
    except RuntimeError:
        pass


# ═══════════ 4. 实时监控：产线控制 + 实时流订阅 ═══════════
def step_monitor():
    print("\n[UI-4] 实时监控：产线控制 + 实时流订阅", flush=True)
    win.goto("实时监控")
    pump(900)
    idx = win.combo_workorder.findData(
        next((w["id"] for w in win._workorders if w.get("name") == WO), -1))
    if idx >= 0:
        win.combo_workorder.setCurrentIndex(idx)
    pump(900)
    txt = page_monitor.lbl_pipeline.text()
    ck("监控页显示产线运行中", "运行中" in txt, txt)
    page_monitor.btn_pp_pause.click()
    pump(1500)
    ck("暂停产线后状态更新", "已暂停" in page_monitor.lbl_pipeline.text(),
       page_monitor.lbl_pipeline.text())
    page_monitor.btn_pp_resume.click()
    pump(1500)
    ck("恢复产线后状态更新", "运行中" in page_monitor.lbl_pipeline.text(),
       page_monitor.lbl_pipeline.text())
    print("  [等待] 产线自动检测落库 + 实时流订阅…", flush=True)
    n0 = det_total()
    ok, last = wait_api("检测", det_total, lambda n: n >= n0 + 3, timeout=300)
    ck("产线自动检测落库 ≥3 条", ok, f"n_det={last}（起={n0}）")
    ok, last = wait_api("实时流", live_total, lambda n: n >= 1, timeout=60)
    ck("监控页实时流订阅到帧（按工单过滤）", ok, f"frames={last}")
    if page_monitor._frames:
        f0 = page_monitor._frames[-1]
        ck("流帧带判定与缺陷框",
           str(f0.get("decision") or "") in ("normal", "gray", "anomaly"),
           f"decision={f0.get('decision')}")
    later(200, step_feedback)


def wait_api(name, fetch, cond, timeout=120):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        pump(80)
        try:
            last = fetch()
            if cond(last):
                return True, last
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.6)
    return False, last


# ═══════════ 5. 错检反馈 ═══════════
def step_feedback():
    print("\n[UI-5] 错检反馈（监控页标记漏检 → 对话框提交）", flush=True)
    r = safe_get(f"{BASE}/api/detections?page=1&page_size=1", timeout=8)
    dets = r.json().get("items", []) if isinstance(r.json(), dict) else []
    if not dets:
        ck("取到检测记录", False)
        return later(200, step_learn)
    det = dets[0]
    page_monitor._current_detection = det
    for _b in (page_monitor.btn_fp, page_monitor.btn_fn, page_monitor.btn_ok):
        _b.setEnabled(True)
    later(900, lambda: _fill_feedback(det))
    page_monitor.btn_fn.click()


def _fill_feedback(det, attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _fill_feedback(det, attempt + 1))
        ck("反馈对话框弹出", False)
        return later(200, step_learn)
    ck("标记漏检弹出反馈对话框", True)
    pump(800)
    for le in dlg.findChildren(QLineEdit):
        if le.placeholderText() and "操作员" in le.placeholderText():
            le.setText("演示")
        elif le.placeholderText() and "划痕" in le.placeholderText():
            le.setText("演示缺陷")
    pump(200)
    click_by_text(dlg, "提交反馈")
    later(2500, step_feedback_verify)


def step_feedback_verify():
    r = safe_get(f"{BASE}/api/feedback", timeout=8)
    items = r.json().get("items", []) if isinstance(r.json(), dict) else []
    fb = next((x for x in items if x.get("feedback_type") == "false_negative"), None)
    ck("漏检反馈已落库", fb is not None)
    later(200, step_review)


# ═══════════ 6. 复核：判缺陷 ═══════════
def step_review():
    print("\n[UI-6] 人工复核：待复核队列 → 判缺陷（即学）", flush=True)
    win.goto("标注反馈")
    pump(1200)
    # 切到「待复核」Tab（索引 1）
    tabs = page_feedback.findChildren(QW.QTabWidget)
    if tabs and tabs[0].count() >= 2:
        tabs[0].setCurrentIndex(1)
    pump(1500)
    tbl = page_feedback.table_review
    ck("复核队列非空（复判工单全量检测进队列）", tbl.rowCount() > 0,
       f"rows={tbl.rowCount()}")
    if tbl.rowCount() == 0:
        return later(200, step_learn)
    tbl.selectRow(0)
    pump(800)
    ck("判缺陷按钮可用", page_feedback.btn_review_ng.isEnabled())
    det_id = page_feedback._review_items[0].get("id")
    n_review = _count_feedback("review")
    page_feedback._on_review_judge(1)
    pump(3000)
    ok, last = wait_api("复核落库", lambda: _count_feedback("review"),
                        lambda n: n > n_review, timeout=30)
    ck("复核判缺陷生成 review 反馈（即学）", ok, f"n_review={last}")
    later(200, step_invalidate)


def _count_feedback(ftype):
    r = safe_get(f"{BASE}/api/feedback", timeout=8)
    return len([x for x in r.json().get("items", [])
                if x.get("feedback_type") == ftype])


# ═══════════ 7. 作废反馈（标错撤回）═══════════
def step_invalidate():
    print("\n[UI-7] 反馈作废：标错撤回", flush=True)
    # 重新加载反馈记录 Tab
    tabs = page_feedback.findChildren(QW.QTabWidget)
    if tabs:
        tabs[0].setCurrentIndex(0)
    pump(1500)
    page_feedback.reload_records()
    pump(1500)
    # 找一条有效（未作废）反馈的作废按钮
    target = None
    for r in range(page_feedback.table.rowCount()):
        w = page_feedback.table.cellWidget(r, 7)
        if isinstance(w, QPushButton) and w.text().strip() == "作废" and w.isVisible():
            target = w
            break
    ck("反馈记录 Tab 有「作废」按钮", target is not None)
    if target is None:
        return later(200, step_learn)
    n_inv = _count_invalidated()
    target.click()
    pump(2500)
    ok, last = wait_api("作废落库", lambda: _count_invalidated(),
                        lambda n: n > n_inv, timeout=30)
    ck("作废反馈落库（invalidated）", ok, f"n_invalidated={last}")
    later(200, step_learn)


def _count_invalidated():
    r = safe_get(f"{BASE}/api/feedback?page=1&page_size=200", timeout=8)
    return len([x for x in r.json().get("items", [])
                if x.get("invalidated")])


# ═══════════ 8. 学习提升 ═══════════
def step_learn():
    print("\n[UI-8] 学习提升（巩固 → 新版本）", flush=True)
    win.goto("实时监控")
    pump(700)
    n_before = _model_versions()
    page_monitor.btn_pp_learn.click()
    pump(4000)
    ok, last = wait_api("learn", lambda: safe_get(f"{BASE}/api/models",
                                                  timeout=8).json(),
                        lambda b: _model_versions() > n_before, timeout=180)
    ck("学习提升后模型版本增加", ok, f"{n_before} → {_model_versions()}")
    later(200, step_queue)


def _model_versions(category=None):
    r = safe_get(f"{BASE}/api/models", timeout=8)
    return len([m for m in r.json().get("items", [])
                if m.get("category") == (category or CATEGORY)])


# ═══════════ 9. 回队重新检测 ═══════════
def step_queue():
    print("\n[UI-9] 数据流队列：回队重新检测", flush=True)
    win.goto("数据管理")
    pump(1000)
    n_before = det_total()
    tbl = page_data.queue_table
    found = False
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 2)
        if it is not None and CATEGORY in it.text():
            tbl.selectRow(r)
            found = True
            break
    ck(f"队列面板含 {CATEGORY} 批次", found)
    if not found:
        return later(200, finish)
    page_data.btn_queue_req.click()
    pump(1500)
    r = safe_get(f"{BASE}/api/workorders/{win.current_workorder_id()}/queue",
                 timeout=8)
    zr = next((x for x in r.json().get("items", [])
               if x.get("category") == CATEGORY), None)
    ck(f"UI 回队成功（{CATEGORY} requeued=true）", bool(zr and zr.get("requeued")))
    ok, last = wait_api("再检", det_total, lambda n: n > n_before, timeout=180)
    ck("回队后重新检测（检测数增加）", ok, f"{n_before} → {last}")
    later(200, finish)


def finish():
    print(f"\n{'='*52}", flush=True)
    pump(600)
    if ISSUES:
        print(f"发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']} 项）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"全流程全部通过 ✅（{_OK['n']} 项，耗时 {time.time()-T0:.0f}s）", flush=True)
    print(f"[弹窗统计] QMessageBox 出现 {_QMB['n']} 次（应 0）", flush=True)
    for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                           timeout=5).json().get("items", []):
        if _w.get("name") == WO:
            requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
    for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
        if _d.get("name") == SRC:
            requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
    print("[清理] 演示数据已移除（检测/反馈记录保留）", flush=True)
    sys.exit(1 if ISSUES else 0)


IMPORT_BASE = 0
PREP_BASE = 0
later(300, step_datasource)
app.exec()
