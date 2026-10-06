# -*- coding: utf-8 -*-
"""UI 全按钮遍历测试：不只要主干，所有页面所有按钮/对话框/行内控件全点一遍。

设计（2026-08-29）：
- 启动后端，API/DB 造最小测试数据（工单+数据源+图像+检测+反馈），
  让每个按钮都有数据可交互。
- patch ApiClient 的重任务/破坏性方法（记录调用、返回假结果）：
  按钮点击→对话框→提交链路全部真实走到 API 调用层，但引擎重任务
  （预训练/导入/评估/回放/巩固）不真正执行；删除类依赖确认框自动点「否」。
- 每个按钮点击后泵事件捕获异常；QMessageBox 自动应答（默认拒绝）；
  模态对话框自动进入遍历内部按钮后关闭。
- 覆盖：顶层按钮、对话框内按钮、表格 cellWidget 按钮、下拉框全项切换、
  复选框切换。输出逐页报告 + ISSUES 汇总。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.ERROR)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from server import start_server_background  # noqa: E402
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
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QMessageBox, QPushButton, QTableWidget)
import PySide6.QtWidgets as QW  # noqa: E402
from shiboken6 import isValid  # noqa: E402
from ui.api_client import ApiClient  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

ISSUES: list = []
_OK = {"n": 0}
_QMB = {"n": 0}
EXC = {"n": 0}
PATCHED = []          # 记录被调用的重任务方法
CLICK_LOG = []        # 按钮点击日志

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ui_tmp")
os.makedirs(TMP, exist_ok=True)
CAT = "uitest_cat"
SRC = "全按钮源"
WO = "全按钮工单"


_last_progress = [time.time()]   # 看门狗：60s 无新 ck 输出即视为卡死


def _heartbeat():
    _last_progress[0] = time.time()


# 看门狗：主循环停滞 90s → 打印现场并强制关闭所有模态，继续扫描
def _watchdog():
    idle = time.time() - _last_progress[0]
    if idle > 90:
        import traceback
        ISSUES.append(f"看门狗触发：主循环停滞 {int(idle)}s，强制清模态继续")
        print(f"  ⚠ 看门狗：停滞 {int(idle)}s，清模态继续（{CLICK_LOG[-1] if CLICK_LOG else '-'}）",
              flush=True)
        traceback.print_stack()
        for tl in QApplication.topLevelWidgets():
            try:
                if isinstance(tl, QDialog) and tl.isVisible():
                    tl.reject()
            except RuntimeError:
                pass
        _last_progress[0] = time.time()
    QTimer.singleShot(10000, _watchdog)


QTimer.singleShot(10000, _watchdog)


def ck(name, cond, detail=""):
    _heartbeat()
    if cond:
        _OK["n"] += 1
        print(f"    ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"    ❌ {name}（{detail}）", flush=True)


def pump(ms=250):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.015)


def later(ms, fn):
    QTimer.singleShot(int(ms), fn)


# ── 异常捕获：按钮点击后 3s 内无新异常 ──
_exc_before = 0


def _excepthook(et, ev, tb):
    import traceback
    global EXC
    EXC["n"] += 1
    traceback.print_exception(et, ev, tb)
    ISSUES.append(f"未捕获异常 {et.__name__}: {ev}")


sys.excepthook = _excepthook


def no_new_exc():
    return EXC["n"] <= _exc_before


# ── patch：重任务/破坏性 client 方法 ──
def fake_det(**kw):
    d = {"detection_id": 990001, "image_path": os.path.join(TMP, "d0.png"),
         "heatmap_path": None, "overlay_path": None, "final_score": 0.42,
         "is_anomaly": False, "decision": "gray", "threshold": 0.6,
         "gray_threshold": 0.3, "slots": {"sem": 0.4, "disc": 0.5, "shead": 0.3},
         "weights": {"sem": 0.5, "disc": 0.3, "shead": 0.2}, "triggered_slot": "",
         "defect_boxes": [], "defect_types": [], "open_alert": False,
         "align_warn": False, "align_offset": None, "latency_ms": 15.0,
         "total_latency_ms": 18.0}
    d.update(kw)
    return d


def _patch(name, result_fn):
    orig = getattr(ApiClient, name)

    def _fake(self, *a, **k):
        PATCHED.append(name)
        print(f"      [patched→] {name}(args={a[1:] if a else ''}, kw={k})", flush=True)
        return result_fn(*a, **k)
    setattr(ApiClient, name, _fake)
    return orig


_patch("prepare_model", lambda *a, **k: {"task_id": "fake_prepare"})
_patch("self_learning_update", lambda *a, **k: {"task_id": "fake_update"})
_patch("workorder_learn", lambda *a, **k: {"message": "学习完成（fake）"})
_patch("queue_requeue", lambda *a, **k: {"requeued": []})
_patch("import_folder", lambda *a, **k: {"task_id": "fake_import"})
_patch("import_mvtec", lambda *a, **k: {"task_id": "fake_import"})
_patch("import_video", lambda *a, **k: {"task_id": "fake_import"})
_patch("upload_image", lambda *a, **k: {"id": 1, "path": "tmp.png"})
_patch("detect_image", lambda *a, **k: fake_det())
_patch("detect_upload", lambda *a, **k: fake_det())
_patch("detect_video", lambda *a, **k: fake_det())
_patch("eval_accuracy", lambda *a, **k: {"accuracy": 0.9, "n": 1})
_patch("eval_benchmark", lambda *a, **k: {"items": [], "total": 0})
_patch("eval_robustness", lambda *a, **k: {"ok": True})
_patch("learning_curve", lambda *a, **k: {"task_id": "fake_curve"})
_patch("learning_contribution", lambda *a, **k: {"task_id": "fake_contrib"})
_patch("rollback_model", lambda *a, **k: {"ok": True, "message": "已回滚（fake）"})
_patch("activate_model", lambda *a, **k: {"ok": True, "gate_report": {}})
_patch("save_config", lambda *a, **k: {"ok": True})
_patch("save_augment_config", lambda *a, **k: {"ok": True})
_patch("save_plan", lambda *a, **k: {"ok": True})
_patch("pseudo_preview", lambda *a, **k: {"path": os.path.join(TMP, "d0.png")})
_patch("clear_datasource_data", lambda *a, **k: {"cleared": 0})
_patch("orphan_clean", lambda *a, **k: {"cleaned": 0})
# 破坏性删除：记录调用（UI 真删有确认框保护，这里双重保险）
for _dn in ("delete_image", "delete_dataset", "delete_datasource",
            "delete_eval_run", "delete_plan", "delete_workorder",
            "delete_workorder_template"):
    _patch(_dn, lambda *a, **k: {"ok": True})
# 对话框内「创建」类提交：假响应避免反复污染测试库（链路仍真实走到 API 调用层）
_patch("create_workorder", lambda *a, **k: {"id": 999001, "name": "fake_wo"})
_patch("create_datasource", lambda *a, **k: {"id": 999002, "name": "fake_ds"})
_patch("create_workorder_template", lambda *a, **k: {"id": 999003})
# 文件对话框：取消
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
# 静态 QMessageBox（question/warning 等）在 Windows 走原生对话框，
# activeModalWidget() 看不到会卡死事件循环 → 统一返回固定「否」并计数
for _m in ("information", "warning", "question", "critical"):
    def _mk(m=_m):
        def _p(*a, **k):
            _QMB["n"] += 1
            return QW.QMessageBox.No
        return _p
    setattr(QW.QMessageBox, _m, _mk())


# ── 全局模态守护：每 150ms 检查未处理的 QMessageBox/QDialog ──
# 处理异步弹窗（如点击后先 run_async 请求、返回后才 exec 的对话框），
# 以及 _visit_dialog 遍历按钮时连锁弹出的新对话框
_answered = set()


def _modal_guard():
    m = QApplication.activeModalWidget()
    if isinstance(m, QMessageBox):
        if id(m) not in _answered:
            _answered.add(id(m))
            _QMB["n"] += 1
            print(f"      [弹窗自动应答] {m.windowTitle()}", flush=True)
            for b in m.findChildren(QPushButton):
                t = b.text().strip()
                if t in ("否", "No", "取消", "Cancel"):
                    b.click()
                    return
            for b in m.findChildren(QPushButton):
                t = b.text().strip()
                if t in ("是", "Yes", "确定", "OK"):
                    b.click()
                    return
    elif isinstance(m, QDialog) and id(m) not in _VISITED:
        _visit_dialog(m, f"守护::{type(m).__name__}")
    QTimer.singleShot(150, _modal_guard)


QTimer.singleShot(150, _modal_guard)


# ── 工具：模态对话框查找 / 按钮查找 ──
def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


def buttons(wdg, text):
    return [b for b in wdg.findChildren(QPushButton)
            if b.text().strip() == text and b.isVisible()]


def click_by_text(wdg, text):
    bs = buttons(wdg, text)
    if bs:
        bs[0].click()
        return True
    return False


# ── 按钮分类 ──
DANGER_KW = ("删除", "作废", "清空", "回滚", "取消回队", "移除", "取消回队")
TASK_KW = ("准备", "导入", "学习", "巩固", "回队", "运行", "生成", "回放",
           "保存", "上传", "试检", "激活", "重新读取", "预览合成", "更新")
NAV_KW = ("去复核", "去巩固", "去设置中心", "去准备未就绪")
SKIP_KW = ("取消", "关闭")


def classify(text: str) -> str:
    if any(k in text for k in SKIP_KW):
        return "skip"
    if any(k in text for k in NAV_KW):
        return "nav"
    if any(k in text for k in DANGER_KW):
        return "danger"
    if any(k in text for k in TASK_KW):
        return "task"
    return "safe"


# ── 对话框遍历：进入后遍历内部按钮，最后关闭 ──
_VISITED = set()   # 已处理过的 dialog（防 _after_modal_click 重复进入死循环）


def _visit_dialog(dlg: QDialog, label: str):
    if id(dlg) in _VISITED:
        return
    _VISITED.add(id(dlg))
    ck(f"对话框弹出：{label}（{type(dlg).__name__}）", True)
    pump(500)
    # 表格类对话框（如图库选图）：先选中首行，让「选择/确定」可生效
    for tbl in dlg.findChildren(QTableWidget):
        try:
            if tbl.rowCount() > 0:
                tbl.selectRow(0)
        except Exception:  # noqa: BLE001
            pass
    # 遍历内部下拉/复选框
    for combo in dlg.findChildren(QComboBox):
        try:
            if combo.count() > 1:
                for i in range(min(combo.count(), 4)):
                    combo.setCurrentIndex(i)
                    pump(120)
                combo.setCurrentIndex(0)
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{label}] 下拉切换异常 {combo.objectName()}: {e}")
    for chk in dlg.findChildren(QCheckBox):
        try:
            chk.toggle()
            pump(100)
            chk.toggle()
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{label}] 复选框异常: {e}")
    # 遍历内部按钮（排除导航类）
    for b in dlg.findChildren(QPushButton):
        try:
            t = b.text().strip()
            if not t or not isValid(b) or not b.isVisible():
                continue
            cat = classify(t)
            if cat == "skip":
                continue
            _try_click(b, f"{label}>>{t}", cat)
        except RuntimeError:
            continue
    # 关闭对话框
    closed = click_by_text(dlg, "关闭") or click_by_text(dlg, "取消")
    if not closed:
        dlg.reject()
    pump(400)


def _try_click(btn: QPushButton, label: str, cat: str):
    global _exc_before
    try:
        if not isValid(btn):
            return
        if not btn.isVisible() or not btn.isEnabled():
            CLICK_LOG.append(f"skip(disabled) {label}")
            return
    except RuntimeError:
        return
    CLICK_LOG.append(f"click {label}")
    _exc_before = EXC["n"]
    _before_modal = None
    try:
        # 所有按钮统一调度「弹窗处理」：模态 exec 阻塞时 QTimer 仍触发
        #（dialog 自带事件循环），350ms 后进入处理；无弹窗则直接返回
        QTimer.singleShot(350, lambda: _after_modal_click(label))
        btn.click()
        pump(600)
        # 非模态点击：检查异常
        if not no_new_exc():
            return
        ck(f"点击无异常：{label}", True)
    except Exception as e:  # noqa: BLE001
        ISSUES.append(f"[{label}] 点击异常 {type(e).__name__}: {e}")
        print(f"    ❌ 点击异常 {label}: {type(e).__name__}: {e}", flush=True)


def _after_modal_click(label: str):
    """模态按钮点击后（exec 阻塞解除）处理弹窗；同一 dialog 只处理一次。"""
    dlg = active_modal()
    if dlg is None:
        return
    if id(dlg) in _VISITED:
        return
    _visit_dialog(dlg, label)


# ── 表格 cellWidget 按钮遍历 ──
def _scan_table_cells(page):
    for tbl in page.findChildren(QTableWidget):
        try:
            for r in range(tbl.rowCount()):
                for c in range(tbl.columnCount()):
                    w = tbl.cellWidget(r, c)
                    if isinstance(w, QPushButton) and isValid(w) and w.isVisible():
                        _try_click(w, f"{type(page).__name__}[tbl r{r}c{c}]::{w.text()}",
                                   classify(w.text()))
        except RuntimeError:
            continue
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{type(page).__name__}] 表格遍历异常: {e}")


# ── 页面扫描 ──
def _scan_page(page, page_name: str):
    print(f"\n  [页] {page_name}", flush=True)
    pump(900)  # 等异步加载
    # 特殊页面预处理：先选择条目，使条件可见/可用的按钮出现
    if page_name == "ModelPage":
        tree = getattr(page, "tree", None)
        if tree is not None:
            for _attempt in range(10):
                top = tree.topLevelItem(0)
                if top is None:
                    pump(800)
                    continue
                top.setExpanded(True)
                for i in range(top.childCount()):
                    ch = top.child(i)
                    d = ch.data(0, Qt.UserRole)
                    if isinstance(d, tuple) and d[0] == "category":
                        tree.setCurrentItem(ch)
                        pump(700)
                        break
                else:
                    pump(800)
                    continue
                break
    elif page_name == "FeedbackPage":
        # 待复核 Tab：选中首行使「判正常/判缺陷」可用
        tbl = getattr(page, "table_review", None)
        if tbl is not None and tbl.rowCount() > 0:
            tbl.selectRow(0)
            pump(500)
    pump(600)
    # 顶层按钮
    handled = set()
    for b in list(page.findChildren(QPushButton)):
        try:
            if not isValid(b):
                continue
            t = b.text().strip()
            if not t:
                continue
            key = (id(b), t)
            if key in handled:
                continue
            handled.add(key)
            if not b.isVisible():
                continue
            _try_click(b, f"{page_name}::{t}", classify(t))
            pump(250)
            # 若点击导致模态弹窗残留，统一走防重入处理
            _after_modal_click(f"{page_name}::{t}")
        except RuntimeError:
            continue
    # 表格行内按钮
    _scan_table_cells(page)
    # 下拉切换（页面级）
    for combo in page.findChildren(QComboBox):
        try:
            if combo.count() > 1 and combo.isVisible():
                for i in range(min(combo.count(), 4)):
                    combo.setCurrentIndex(i)
                    pump(120)
                combo.setCurrentIndex(0)
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{page_name}] 下拉切换异常: {e}")
    # 复选框
    for chk in page.findChildren(QCheckBox):
        try:
            if chk.isVisible():
                chk.toggle()
                pump(100)
                chk.toggle()
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{page_name}] 复选框异常: {e}")


# ═══════════ 0. 造最小测试数据 ═══════════
def _seed_data():
    print("\n[造数] 最小测试数据（工单+数据源+图像+检测+反馈）", flush=True)
    try:
        return _seed_data_impl()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ISSUES.append(f"造数异常 {type(e).__name__}: {e}")
        print(f"  ❌ 造数异常 {type(e).__name__}: {e}", flush=True)
        return None, None


def _seed_data_impl():
    # 清理上次残留
    try:
        for w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                              timeout=5).json().get("items", []):
            if w.get("name") == WO:
                requests.delete(f"{BASE}/api/workorders/{w['id']}", timeout=5)
        for d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
            if d.get("name") == SRC:
                requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=5)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] 清理残留失败: {e}", flush=True)
    from backend.db.database import session_scope
    from backend.db.models import (Dataset, Detection, Feedback,
                                   Image as ImageRow, WorkOrder, WorkOrderSource)
    # 图像文件（真实小 PNG，缩略图加载可正常渲染）
    from PIL import Image as PILImage
    files = []
    for i in range(6):
        p = os.path.join(TMP, f"d{i}.png")
        PILImage.new("RGB", (64, 64), (100 + i * 20, 80, 60)).save(p)
        files.append(p)
    # 数据源 + 批次
    r = requests.post(f"{BASE}/api/datasources",
                      json={"name": SRC, "modality": "image"}, timeout=8)
    ds = r.json()
    ds_id = ds["id"]
    with session_scope() as s:
        dset = Dataset(name=f"{CAT}_b1", category=CAT, datasource_id=ds_id)
        s.add(dset)
        s.flush()
        img_ids = []
        for i, p in enumerate(files):
            img = ImageRow(path=p, category=CAT, split="train",
                           label="normal" if i < 4 else "anomaly",
                           source="upload", dataset_id=dset.id)
            s.add(img)
            s.flush()
            img_ids.append(img.id)
        # 工单（复判开启）
        wo = WorkOrder(name=WO, review_enabled=True,
                       pipeline_status="running",
                       label_tier="L1a", per_category=True,
                       has_template=False, queue_json={})
        s.add(wo)
        s.flush()
        s.add(WorkOrderSource(workorder_id=wo.id, datasource_id=ds_id))
        # 检测：2 正常 / 2 灰区 / 2 异常（pipeline + workorder 归属）
        decs = ["normal", "gray", "anomaly"] * 2
        scores = [0.15, 0.45, 0.85] * 2
        det_ids = []
        for i, (dec, sc) in enumerate(zip(decs, scores)):
            d = Detection(target_type="pipeline", workorder_id=wo.id,
                          image_id=img_ids[i], image_path=files[i],
                          category=CAT, final_score=sc,
                          is_anomaly=(dec == "anomaly"), latency_ms=12.0,
                          n_tiles={"slots": {"sem": sc, "disc": sc, "shead": sc},
                                   "decision": dec})
            s.add(d)
            s.flush()
            det_ids.append(d.id)
        # 反馈：误检/漏检/复核各 1（1 条已作废）
        fbs = [
            (det_ids[0], "false_positive", 0, True, False),
            (det_ids[1], "false_negative", 1, True, False),
            (det_ids[2], "review", 1, True, False),
            (det_ids[3], "false_positive", 0, True, True),   # 已作废
        ]
        for det_id, ftype, label, consumed, inv in fbs:
            fb = Feedback(detection_id=det_id, feedback_type=ftype,
                          operator_label=label, operator="uitest",
                          consumed=consumed, invalidated=inv)
            s.add(fb)
        wo_id = wo.id
    ck("造数完成（工单+数据源+6检测+4反馈）", wo_id is not None)
    return ds_id, wo_id


# ═══════════ 1. 逐页全按钮遍历 ═══════════
def run_all():
    ds_id, wo_id = _seed_data()
    if wo_id is None:
        print("造数失败，中止", flush=True)
        sys.exit(1)
    stack = win.stack
    # 当前工单联动到测试工单
    idx = win.combo_workorder.findData(wo_id)
    if idx >= 0:
        win.combo_workorder.setCurrentIndex(idx)
        pump(800)

    n_pages = stack.count()
    for i in range(n_pages):
        pg = stack.widget(i)
        name = type(pg).__name__
        try:
            stack.setCurrentIndex(i)
            pump(600)
            _scan_page(pg, name)
        except Exception as e:  # noqa: BLE001
            ISSUES.append(f"[{name}] 页面扫描异常 {type(e).__name__}: {e}")
            print(f"  ❌ 页面扫描异常 {name}: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 52, flush=True)
    if ISSUES:
        print(f"发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']} 项）", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print(f"全部通过 ✅（{_OK['n']} 项）", flush=True)
    print(f"[点击日志] {len(CLICK_LOG)} 次按钮点击；[弹窗] QMessageBox {_QMB['n']} 次；"
          f"[patched] 重任务调用 {len(PATCHED)} 次：{sorted(set(PATCHED))}", flush=True)
    # 清理测试数据
    try:
        requests.delete(f"{BASE}/api/workorders/{wo_id}", timeout=5)
        requests.delete(f"{BASE}/api/datasources/{ds_id}", timeout=5)
    except Exception:  # noqa: BLE001
        pass
    sys.exit(1 if ISSUES else 0)


later(400, run_all)
app.exec()
