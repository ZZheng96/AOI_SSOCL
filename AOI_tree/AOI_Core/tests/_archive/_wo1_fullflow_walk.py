# -*- coding: utf-8 -*-
"""工单「1」(id=1) 全流程 UI 走查：准备模型→评估→产线检测→反馈→学习→回队。

针对库内已存在的工单「1」（挂数据源「1」，MPDD 6 品类 1386 图）：
- 不新建/不删除工单与数据源，全程在真实 UI 上点击操作（事件驱动）。
- 选未准备过的品类 metal_plate 走「准备模型」，其余步骤用工单现有数据。
- 每步 ck() 断言 + 截图落盘 _wo1_shots/；统计 QMessageBox 模态弹窗（应 0）。
"""
import os
import sys
import time
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
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
from PySide6.QtWidgets import (QApplication, QDialog, QLabel,  # noqa: E402
                               QLineEdit, QPushButton)
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

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_wo1_shots")
WO_NAME, WO_ID = "1", 1
CAT = "metal_plate"   # 工单1数据源内未准备过的品类
DS_ID = 1
T0 = time.time()
_state = {"i": 0}


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


def shot(name, widget=None):
    os.makedirs(SHOT_DIR, exist_ok=True)
    _state["i"] += 1
    p = os.path.join(SHOT_DIR, f"{_state['i']:02d}_{name}.png")
    (widget or win).grab().save(p)
    print(f"  [截图] {os.path.basename(p)}", flush=True)


def later(ms, fn):
    def _safe():
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ISSUES.append(f"步骤异常：{type(e).__name__}: {e}")
            print(f"  [异常] {type(e).__name__}: {e}", flush=True)
            finish()
    QTimer.singleShot(int(ms), _safe)


def safe_get(url, timeout=8, retries=4, **kw):
    for i in range(retries):
        try:
            return requests.get(url, timeout=timeout, **kw)
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            pump(300)
            time.sleep(0.5)
    return None


def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


def wait_api(name, fetch, cond, timeout=120, poll=0.6):
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
        time.sleep(poll)
    return False, last


def det_total():
    r = safe_get(f"{BASE}/api/detections?category={CAT}&page=1&page_size=1",
                 timeout=8)
    b = r.json()
    return b.get("total", 0) if isinstance(b, dict) else 0


def model_versions():
    r = safe_get(f"{BASE}/api/models", params={"category": CAT}, timeout=8)
    return len([m for m in r.json().get("items", [])
                if m.get("category") == CAT])


# ═══════════ 0. 顶栏选工单「1」 ═══════════
def step_select_wo():
    print("\n[0] 顶栏选择工单「1」", flush=True)
    ok, _ = wait_api("工单加载", lambda: win.combo_workorder.count(),
                     lambda n: n >= 1, timeout=20)
    ck("顶栏工单下拉已加载", ok, f"n={win.combo_workorder.count()}")
    idx = win.combo_workorder.findData(WO_ID)
    ck("工单下拉包含工单「1」", idx >= 0)
    if idx >= 0:
        win.combo_workorder.setCurrentIndex(idx)
        pump(1000)
    ck("当前工单=「1」", win.combo_workorder.currentText() == WO_NAME,
       win.combo_workorder.currentText())
    shot("顶栏_工单1")
    later(200, step_dashboard)


# ═══════════ 1. 工作台 ═══════════
def step_dashboard():
    print("\n[1] 工作台：工单「1」行与统计", flush=True)
    win.goto("工作台")
    pump(1500)
    row_txt = []
    tbl = win.page_dashboard.table
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 0)
        if it:
            row_txt.append(it.text())
    ck("工作台工单表含工单「1」", WO_NAME in row_txt, str(row_txt))
    shot("工作台_工单表")
    later(200, step_prepare)


# ═══════════ 2. 模型管理：准备 metal_plate ═══════════
def step_prepare():
    print(f"\n[2] 模型管理：树选「{CAT}」→ 准备模型", flush=True)
    win.goto("模型管理")
    pump(1500)
    tree = win.page_model.tree
    target = None
    t0 = time.time()
    while time.time() - t0 < 40:   # 树异步加载（datasource_groups）
        pump(500)
        for i in range(tree.topLevelItemCount()):
            src = tree.topLevelItem(i)
            for j in range(src.childCount()):
                it = src.child(j)
                d = it.data(0, Qt.UserRole)
                if isinstance(d, tuple) and len(d) == 4 \
                        and d[0] == "category" and d[3] == CAT:
                    target = it
                    break
        if target is not None:
            break
    ck("模型树出现 metal_plate 品类节点", target is not None)
    if target is None:
        return later(200, step_eval)
    tree.setCurrentItem(target)
    pump(800)
    ck("选中品类后面板进入品类视图", win.page_model._cur is not None
       and win.page_model._cur.get("category") == CAT,
       str(win.page_model._cur))
    ck("「准备模型」按钮可见", win.page_model.btn_prepare.isVisible())
    later(600, _prepare_dlg)
    win.page_model.btn_prepare.click()


def _prepare_dlg(attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _prepare_dlg(attempt + 1))
        ck("准备模型对话框弹出", False, "未弹出")
        return later(200, step_eval)
    ck("准备模型对话框弹出", True)
    # 等数据体检回填
    ok, _ = wait_api("体检", lambda: dlg.lbl_counts.text(),
                     lambda t: t and "检查中" not in t, timeout=30)
    ck("数据体检回填（预训练组统计）", ok, dlg.lbl_counts.text()[:90])
    ck("数据条件自动判定", "数据条件" in dlg.lbl_scenario.text(),
       dlg.lbl_scenario.text()[:60])
    shot("准备模型_体检", dlg)
    dlg.btn_ok.click()   # 开始准备
    print("  [等待] metal_plate 准备任务完成…", flush=True)
    ok, st = wait_api("准备", lambda: dlg.lbl_status.text(),
                      lambda t: ("准备完成" in t) or ("准备失败" in t),
                      timeout=900, poll=2.0)
    ck("metal_plate 准备完成", ok and "准备完成" in (st or ""), (st or "")[:120])
    shot("准备模型_结果", dlg)
    ok2, _ = wait_api("版本落库", model_versions, lambda n: n >= 1, timeout=30)
    ck("metal_plate 模型版本已落库", ok2, f"versions={model_versions()}")
    # 关闭对话框（成功后 btn_ok 文本变为「完成」）
    dlg.btn_ok.click()
    pump(500)
    later(200, step_eval)


# ═══════════ 3. 评估看板 ═══════════
def step_eval():
    print("\n[3] 评估看板：运行评估", flush=True)
    win.goto("评估看板")
    pump(1200)
    runs0 = len(safe_get(f"{BASE}/api/eval/runs", params={"page_size": 100},
                         timeout=8).json().get("items", []))
    # 多品类工单：评估页自带品类下拉，选本次准备的 metal_plate
    ci = win.page_eval.combo_category.findData(CAT)
    ck("评估页品类下拉已填充（多品类可选）", ci >= 0)
    if ci >= 0:
        win.page_eval.combo_category.setCurrentIndex(ci)
        pump(300)
    win.page_eval.btn_run.click()
    pump(2000)
    # UI 提交成功标志：TaskProgressBar 显示（任务已创建并推送进度）
    submitted = win.page_eval.progress.isVisible()
    ck("评估任务已从 UI 提交（进度条出现）", submitted)
    shot("评估看板_点击运行后")
    if submitted:
        # eval run 是任务完成后才落库，等待记录出现
        ok, n = wait_api("评估记录落库", lambda: len(safe_get(
            f"{BASE}/api/eval/runs", params={"page_size": 100},
            timeout=8).json().get("items", [])),
            lambda x: x > runs0, timeout=600, poll=2.0)
        ck("评估记录落库（任务完成）", ok, f"runs {runs0}→{n}")
    else:
        ck("多品类工单下评估页可运行", False,
           "点击运行后未出现进度条（按钮被禁用或提交失败）")
    later(200, step_pipeline)


# ═══════════ 4. 实时监控：产线控制 + 自动检测 ═══════════
def step_pipeline():
    print("\n[4] 实时监控：暂停/恢复产线 + 自动检测落库", flush=True)
    win.goto("实时监控")
    pump(1200)
    txt = win.page_monitor.lbl_pipeline.text()
    ck("监控页显示产线运行中", "运行中" in txt, txt)
    win.page_monitor.btn_pp_pause.click()
    pump(1500)
    t1 = win.page_monitor.lbl_pipeline.text()
    ck("暂停产线后状态更新", "已暂停" in t1, t1)
    shot("监控_产线已暂停")
    n_pause = det_total()
    pump(3000)
    ck("暂停期间无新检测落库", det_total() == n_pause,
       f"{n_pause}→{det_total()}")
    win.page_monitor.btn_pp_resume.click()
    pump(1500)
    t2 = win.page_monitor.lbl_pipeline.text()
    ck("恢复产线后状态更新", "运行中" in t2, t2)
    # 2026-08-29 修复验证：按 pipeline_summary 分支断言
    # - 有积压 -> 等自动检测落库；- 无积压 -> 监控页应给出空转提示
    sm = (safe_get(f"{BASE}/api/workorders/{WO_ID}", timeout=8)
          .json().get("pipeline_summary") or {})
    n_pending = int(sm.get("n_pending") or 0)
    if n_pending > 0:
        print("  [等待] metal_plate 自动检测落库（+2）…", flush=True)
        n0 = det_total()
        ok, last = wait_api("检测", det_total, lambda n: n >= n0 + 2,
                            timeout=300)
        ck("恢复后自动检测落库 ≥2 条新增", ok, f"n_det={last}（起={n0}）")
    else:
        ok, txt = wait_api(
            "空转提示", lambda: win.page_monitor.lbl_pipeline_hint.text(),
            lambda t: bool(t) and ("待检测 0" in t
                                   or "暂无可检数据" in t), timeout=30)
        ck("无可检数据时监控页给出空转提示", ok, txt)
        ck("空转时显示「去准备未就绪品类」引导按钮",
           win.page_monitor.btn_go_prepare.isVisible()
           and "准备" in win.page_monitor.btn_go_prepare.text(),
           win.page_monitor.btn_go_prepare.text())
    shot("监控_自动检测后")
    later(200, step_feedback)


# ═══════════ 5. 错检反馈 ═══════════
def step_feedback():
    print("\n[5] 监控页：标记漏检 → 反馈对话框提交", flush=True)
    r = safe_get(f"{BASE}/api/detections?category={CAT}&page=1&page_size=1",
                 timeout=8)
    dets = r.json().get("items", []) if isinstance(r.json(), dict) else []
    ck("取到 metal_plate 检测记录", bool(dets))
    if not dets:
        return later(200, step_learn)
    det = dets[0]
    win.page_monitor._current_detection = det
    for b in (win.page_monitor.btn_fp, win.page_monitor.btn_fn,
              win.page_monitor.btn_ok):
        b.setEnabled(True)
    later(900, lambda: _fill_feedback(det))
    win.page_monitor.btn_fn.click()


def _fill_feedback(det, attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _fill_feedback(det, attempt + 1))
        ck("反馈对话框弹出", False, "未弹出")
        return later(200, step_learn)
    ck("「标记漏检」弹出反馈对话框", True)
    pump(800)
    for le in dlg.findChildren(QLineEdit):
        ph = le.placeholderText() or ""
        if "操作员" in ph:
            le.setText("走查员")
        elif "划痕" in ph or "缺陷" in ph:
            le.setText("走查漏检")
    shot("反馈对话框", dlg)
    btn = next((b for b in dlg.findChildren(QPushButton)
                if "提交" in b.text()), None)
    ck("反馈对话框有提交按钮", btn is not None)
    if btn:
        btn.click()
    pump(2500)
    ok, items = wait_api("反馈", lambda: safe_get(
        f"{BASE}/api/feedback", params={"page_size": 100},
        timeout=8).json().get("items", []), lambda it: len(it) >= 1,
        timeout=30)
    ck("反馈已落库（≥1 条）", ok,
       f"n={len(items) if isinstance(items, list) else '?'}")
    later(200, step_learn)


# ═══════════ 6. 学习提升 ═══════════
def step_learn():
    print("\n[6] 监控页：学习提升", flush=True)
    win.goto("实时监控")
    pump(800)
    n_before = model_versions()
    win.page_monitor.btn_pp_learn.click()
    print(f"  [等待] 学习固化生成新版本（当前 {n_before} 个）…", flush=True)
    ok, last = wait_api("learn", model_versions, lambda n: n > n_before,
                        timeout=300, poll=2.0)
    ck("学习提升后模型版本增加", ok, f"{n_before} → {last}")
    shot("监控_学习提升后")
    later(200, step_queue)


# ═══════════ 7. 数据流队列：回队再检 ═══════════
def step_queue():
    print("\n[7] 数据管理：队列回队（重新检测）", flush=True)
    win.goto("数据管理")
    pump(1500)
    tbl = win.page_data.queue_table
    found = -1
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 2)
        if it is not None and it.text() == CAT:
            found = r
            break
    ck("队列面板含 metal_plate 批次", found >= 0)
    if found < 0:
        return later(200, finish)
    n_before = det_total()
    tbl.selectRow(found)
    pump(300)
    win.page_data.btn_queue_req.click()
    pump(2000)
    items = safe_get(f"{BASE}/api/workorders/{WO_ID}/queue",
                     timeout=8).json().get("items", [])
    zr = next((x for x in items if x.get("category") == CAT), None)
    ck("UI 回队成功（requeued=true）", bool(zr and zr.get("requeued")))
    shot("数据管理_队列回队")
    print("  [等待] 回队批次重新检测（检测数增加）…", flush=True)
    ok, last = wait_api("再检", det_total, lambda n: n > n_before,
                        timeout=300)
    ck("回队后重新检测（检测数增加）", ok, f"{n_before} → {last}")
    later(200, finish)


def finish():
    print(f"\n{'='*52}", flush=True)
    pump(500)
    if ISSUES:
        print(f"工单1 全流程 UI 走查发现 {len(ISSUES)} 个问题 ❌"
              f"（通过 {_OK['n']} 项，耗时 {time.time()-T0:.0f}s）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"工单1 全流程 UI 走查全部通过 ✅"
              f"（{_OK['n']} 项，耗时 {time.time()-T0:.0f}s）", flush=True)
    print(f"[弹窗统计] QMessageBox 出现 {_QMB['n']} 次（要求 0）", flush=True)
    _state["exit"] = 1 if ISSUES else 0
    win.close()
    app.quit()   # QTimer 回调内 sys.exit 不生效，改为退出事件循环后统一退出


later(300, step_select_wo)
app.exec()
sys.exit(_state.get("exit", 1))
