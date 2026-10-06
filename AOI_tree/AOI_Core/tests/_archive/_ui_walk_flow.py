# -*- coding: utf-8 -*-
"""全流程 UI 走查 v3：建工单→100+30 预训练→产线检测→错检反馈→学习→回队再检。

修复 v2 问题：
- 模态交互先调度后点击（click 同步阻塞 exec）
- 按具体 task id 轮询（/api/tasks/{id} 带 result），import 超时放宽到 600s
- 建工单前等待导入完成（避免 0 张源导致条件 L0）
- MVTec UI 导入为全品类（对话框无品类选择），源能力断言放宽为 ≥391
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QDialog, QLineEdit, QPushButton,
                               QListWidget)
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
            ISSUES.append(f"QMessageBox.{m} 被调用：{a[2] if len(a) > 2 else ''}")
            print(f"  [弹窗] QMessageBox.{m}!", flush=True)
            return o(*a, **k)
        return _p
    setattr(QW.QMessageBox, _m, _mk())

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()

SRC = "UI流源"
WO = "UI流工单"
MVTEC_ROOT = r"<PROJECT_ROOT>/data_origin/mvtec"


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
    """GET 带重试（导入大事务/训练期瞬时读超时/500 不中断走查）。"""
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            return r
        except Exception as e:  # noqa: BLE001
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


def wait_task(tid, timeout=600):
    """泵事件 + 轮询 /api/tasks/{tid} 直到 done/failed。"""
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
        time.sleep(2)
    return {"status": "timeout", "id": tid}


def latest_task(task_type, baseline=0):
    rows = safe_get(f"{BASE}/api/tasks/recent", timeout=8).json().get("items", [])
    return next((t for t in rows
                 if t.get("task_type") == task_type and t.get("id", 0) > baseline),
                None)


def det_total():
    r = safe_get(f"{BASE}/api/detections?category=zipper&page=1&page_size=1",
                 timeout=8)
    b = r.json()
    return b.get("total", 0) if isinstance(b, dict) else 0


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
T0 = time.time()


def step_datasource():
    print("\n[UI-1] 数据管理：新建数据源（一步式导入标准数据集）", flush=True)
    win.goto("数据管理")
    pump(600)
    b = next((x for x in page_data.findChildren(QPushButton)
              if x.text().strip() == "新建数据源"), None)
    if b is None:
        ISSUES.append("数据管理页找不到「新建数据源」按钮")
        print("  ❌ 找不到新建数据源按钮", flush=True)
        return later(200, step_dashboard)
    ck("数据管理页有「新建数据源」按钮", True)
    later(500, _fill_create_source)
    b.click()


def _fill_create_source():
    dlg = active_modal()
    if dlg is None:
        ISSUES.append("新建数据源对话框未弹出")
        print("  ❌ 新建数据源对话框未弹出", flush=True)
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
    else:
        ISSUES.append("建源对话框找不到「导入数据」下拉")
        print("  ❌ 建源对话框无导入下拉", flush=True)
    # 标注档位选 L1a（图像级标注）：默认 L0 会聚合出 L0 条件
    combo_tier = next((c for c in dlg.findChildren(QComboBox)
                       if any("图像级标注" in str(c.itemText(i))
                              for i in range(c.count()))), None)
    if combo_tier is not None:
        combo_tier.setCurrentIndex(1)   # L1a
    # 填标准数据集根目录（内联字段，不再二次弹窗）
    for e in dlg.findChildren(QLineEdit):
        if e.placeholderText() and "根目录" in e.placeholderText():
            e.setText(MVTEC_ROOT)
            break
    pump(200)
    # 记录建源前的最大 task id，用于识别新导入任务
    global IMPORT_BASE
    IMPORT_BASE = max((t.get("id") or 0)
                      for t in safe_get(f"{BASE}/api/tasks/recent",
                                        timeout=8).json().get("items", []))
    # 选「标准数据集」后按钮变为「创建并导入」；点后直接提交导入任务（一次弹窗）
    click_by_text(dlg, "创建并导入")
    later(300, step_import_wait)


def step_import_wait():
    print("  [等待] MVTec 导入任务完成…", flush=True)
    tid = None
    t0 = time.time()
    while time.time() - t0 < 60:
        pump(200)
        t = latest_task("import", IMPORT_BASE)
        if t:
            tid = t["id"]
            break
        time.sleep(1)
    ck("识别到 UI 触发的导入任务", tid is not None)
    if tid is None:
        return later(200, step_dashboard)
    last = wait_task(tid, timeout=600)
    ok = last.get("status") == "done"
    ck("MVTec 导入（UI 触发）完成", ok,
       str(last.get("status")) + (f" {last.get('message')}" if last.get("message") else ""))
    res = last.get("result") or {}
    ck("导入包含 zipper 批次", "zipper" in (res.get("dataset_ids") or {}),
       str(list((res.get("dataset_ids") or {}).keys())[:5]))
    r = safe_get(f"{BASE}/api/datasources", timeout=8)
    src = next((x for x in r.json().get("items", []) if x.get("name") == SRC), None)
    cap = (src or {}).get("capability") or {}
    ck("UI 建源后源挂载 ≥391 张", cap.get("n_images", 0) >= 391,
       f"n_images={cap.get('n_images')}")
    later(200, step_dashboard)


def step_dashboard():
    print("\n[UI-2] 工作台：新建工单", flush=True)
    win.goto("工作台")
    pump(600)
    b = next((x for x in page_dashboard.findChildren(QPushButton)
              if "新建工单" in x.text()), None)
    if b is None:
        ISSUES.append("工作台找不到「新建工单」按钮")
        print("  ❌ 找不到新建工单按钮", flush=True)
        return later(200, step_model)
    ck("工作台有「新建工单」按钮", True)
    later(700, _fill_create_wo)
    b.click()


def _fill_create_wo(attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _fill_create_wo(attempt + 1))
        ISSUES.append("新建工单对话框未弹出")
        print("  ❌ 新建工单对话框未弹出", flush=True)
        return later(200, step_model)
    edits = dlg.findChildren(QLineEdit)
    if edits:
        edits[0].setText(WO)
    lw = dlg.findChild(QListWidget)
    # 数据源列表异步加载：尚未出现可勾选项则重试
    if lw is None or lw.count() == 0 or not (
            lw.item(0).flags() & Qt.ItemIsUserCheckable):
        if attempt < 12:
            return later(500, lambda: _fill_create_wo(attempt + 1))
        ISSUES.append("建工单对话框无数据源列表（加载超时）")
        print("  ❌ 建工单对话框无数据源列表", flush=True)
        return later(200, step_model)
    lw.item(0).setCheckState(Qt.Checked)
    chks = dlg.findChildren(QCheckBox)
    if chks:
        chks[0].setChecked(True)   # 人工复判
    pump(300)
    ok_enabled = bool(getattr(dlg, "btn_ok", None) and dlg.btn_ok.isEnabled())
    ck("「创建工单」按钮可用（数据源必选已勾选）", ok_enabled)
    click_by_text(dlg, "创建工单")
    later(600, step_wo_verify)


def step_wo_verify():
    r = safe_get(f"{BASE}/api/workorders", params={"range": "all"}, timeout=8)
    items = r.json().get("items", [])
    wo = next((w for w in items if w.get("name") == WO), None)
    ck("UI 建工单成功（后端可见）", wo is not None)
    if wo:
        ck("工单聚合条件 L1a/L1b",
           wo.get("conditions", {}).get("label_tier") in ("L1a", "L1b"),
           str(wo.get("conditions")))
        ck("工单已开启人工复判", bool(wo.get("review_enabled")))
        ck("工单覆盖 zipper 品类", "zipper" in (wo.get("categories") or []),
           str((wo.get("categories") or [])[:5]))
    pump(500)
    t0 = time.time()
    cur = ""
    while time.time() - t0 < 8:   # 顶栏工单列表异步刷新，轮询等待联动
        pump(300)
        cur = win.combo_workorder.currentText()
        if cur == WO:
            break
        time.sleep(0.3)
    ck("顶栏工单下拉已联动到新工单", cur == WO, cur)
    later(200, step_model)


def step_model():
    print("\n[UI-3] 模型管理：准备模型（100+30）", flush=True)
    win.goto("模型管理")
    pump(700)
    b = next((x for x in page_model.findChildren(QPushButton)
              if x.text().strip() == "准备模型"), None)
    if b is None:
        ISSUES.append("模型页找不到「准备模型」按钮")
        print("  ❌ 找不到准备模型按钮", flush=True)
        return later(200, step_monitor)
    ck("模型页有「准备模型」按钮", True)
    later(900, _fill_prepare)
    b.click()


def _fill_prepare():
    dlg = active_modal()
    if dlg is None:
        ISSUES.append("准备模型对话框未弹出")
        print("  ❌ 准备模型对话框未弹出", flush=True)
        return later(200, step_monitor)
    ck("准备模型对话框弹出", True)
    pump(700)
    dlg.combo_category.setCurrentText("zipper")
    pump(700)
    dlg.combo_scenario.setCurrentIndex(1)   # L1a
    dlg.combo_profile.setCurrentIndex(0)    # fast
    pump(200)
    global PREP_BASE
    PREP_BASE = max((t.get("id") or 0)
                    for t in safe_get(f"{BASE}/api/tasks/recent",
                                      timeout=8).json().get("items", []))
    click_by_text(dlg, "开始准备")

    def _wait_prepare():
        print("  [等待] 100+30 预训练完成…", flush=True)
        tid = None
        t0 = time.time()
        while time.time() - t0 < 60:
            pump(200)
            t = latest_task("model_prepare", PREP_BASE)
            if t:
                tid = t["id"]
                break
            time.sleep(1)
        ck("识别到 UI 触发的预训练任务", tid is not None)
        if tid is None:
            return later(200, step_monitor)
        last = wait_task(tid, timeout=900)
        ok = last.get("status") == "done"
        ck("100+30 预训练（UI 触发）完成", ok,
           str(last.get("status")) + (f" {last.get('message')}" if last.get("message") else ""))
        res = last.get("result") or {}
        ck("预训练样本量 100 正常 + ≤30 缺陷",
           int(res.get("n_normal") or 0) >= 100 and int(res.get("n_defect") or 0) <= 30,
           f"normal={res.get('n_normal')} defect={res.get('n_defect')}")
        pump(300)
        click_by_text(dlg, "关闭")
        later(300, step_monitor)

    later(300, _wait_prepare)


def step_monitor():
    print("\n[UI-4] 实时监控：产线控制", flush=True)
    win.goto("实时监控")
    pump(800)
    idx = win.combo_workorder.findData(
        next((w["id"] for w in win._workorders if w.get("name") == WO), -1))
    if idx >= 0:
        win.combo_workorder.setCurrentIndex(idx)
    pump(800)
    txt = page_monitor.lbl_pipeline.text()
    ck("监控页显示产线运行中", "运行中" in txt, txt)
    page_monitor.btn_pp_pause.click()
    pump(1200)
    t1 = page_monitor.lbl_pipeline.text()
    ck("暂停产线后状态更新", "已暂停" in t1, t1)
    page_monitor.btn_pp_resume.click()
    pump(1200)
    t2 = page_monitor.lbl_pipeline.text()
    ck("恢复产线后状态更新", "运行中" in t2, t2)
    print("  [等待] 产线自动检测落库（起点 +3）…", flush=True)
    n0 = det_total()
    ok, last = wait_api("检测", det_total,
                        lambda n: n >= n0 + 3, timeout=300)
    ck("产线自动检测落库 ≥3 条新增", ok, f"n_det={last} (起={n0})")
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


def step_feedback():
    print("\n[UI-5] 错检反馈（监控页标记漏检 → 对话框提交）", flush=True)
    r = safe_get(f"{BASE}/api/detections?category=zipper&page=1&page_size=1",
                 timeout=8)
    dets = r.json().get("items", []) if isinstance(r.json(), dict) else []
    if not dets:
        ck("取到检测记录", False, "无")
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
        ISSUES.append("FeedbackDialog 未弹出")
        print("  ❌ FeedbackDialog 未弹出", flush=True)
        return later(200, step_learn)
    ck("监控页「标记漏检」弹出反馈对话框", True)
    pump(800)
    for le in dlg.findChildren(QLineEdit):
        if le.placeholderText() and "操作员" in le.placeholderText():
            le.setText("UI走查")
        elif le.placeholderText() and "划痕" in le.placeholderText():
            le.setText("走查缺陷")
    pump(200)
    click_by_text(dlg, "提交反馈")
    later(2500, step_feedback_verify)


def step_feedback_verify():
    r = safe_get(f"{BASE}/api/feedback", timeout=8)
    items = r.json().get("items", []) if isinstance(r.json(), dict) else []
    ck("反馈已落库（≥1 条）", len(items) >= 1, f"n={len(items)}")
    later(200, step_learn)


def step_learn():
    print("\n[UI-6] 学习提升（监控页按钮）", flush=True)
    win.goto("实时监控")
    pump(600)
    n_before = _model_versions()
    page_monitor.btn_pp_learn.click()
    pump(4000)
    ok, last = wait_api(
        "learn", lambda: safe_get(f"{BASE}/api/models", timeout=8).json(),
        lambda b: _model_versions() > n_before, timeout=150)
    ck("学习提升后模型版本增加", ok, f"{n_before} → {_model_versions()}")
    later(200, step_queue)


def _model_versions():
    r = safe_get(f"{BASE}/api/models", timeout=8)
    return len([m for m in r.json().get("items", [])
                if m.get("category") == "zipper"])


def step_queue():
    print("\n[UI-7] 数据流队列：回队（重新检测）", flush=True)
    win.goto("数据管理")
    pump(1000)
    n_before = det_total()
    tbl = page_data.queue_table
    found = False
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 2)
        if it is not None and "zipper" in it.text():
            tbl.selectRow(r)
            found = True
            break
    ck("队列面板含 zipper 批次", found)
    if not found:
        return later(200, finish)
    page_data.btn_queue_req.click()
    pump(1500)
    r = safe_get(f"{BASE}/api/workorders/{win.current_workorder_id()}/queue",
                 timeout=8)
    items = r.json().get("items", [])
    zr = next((x for x in items if x.get("category") == "zipper"), None)
    ck("UI 回队成功（requeued=true）", bool(zr and zr.get("requeued")))
    ok, last = wait_api("再检", det_total, lambda n: n > n_before, timeout=180)
    ck("回队后重新检测（检测数增加）", ok, f"{n_before} → {last}")
    later(200, finish)


def finish():
    print(f"\n{'='*52}", flush=True)
    pump(600)
    if ISSUES:
        print(f"UI 走查发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']} 项）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"UI 走查全部通过 ✅（{_OK['n']} 项，耗时 {time.time()-T0:.0f}s）", flush=True)
    print(f"[弹窗统计] QMessageBox 出现 {_QMB['n']} 次（要求 0）", flush=True)
    for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                           timeout=5).json().get("items", []):
        if _w.get("name") == WO:
            requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
    for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
        if _d.get("name") == SRC:
            requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
    sys.exit(1 if ISSUES else 0)


IMPORT_BASE = 0
PREP_BASE = 0
later(300, step_datasource)
app.exec()
