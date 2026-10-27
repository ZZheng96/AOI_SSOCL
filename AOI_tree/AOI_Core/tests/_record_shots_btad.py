# -*- coding: utf-8 -*-
"""AOI_Core 实操全流程逐分镜录制驱动（默认 BTAD 02，可切换 MPDD tubes）。

方案与 AOI_sys tests/_record_shots_v2.py 相同：逐分镜独立录 mp4
（2s 引导 + 正片 + 3s 余量，剪辑裁 ss=2），逐 clip 烧字幕后拼接。
分镜表/字幕见 practice/BTAD/录制脚本_BTAD.md。

与 v2 的差异：
- 主窗口最大化；仅抓取 QWidget 窗口信号并合成所属顶层弹窗，不采集屏幕。
- 真实实操：分镜内真实执行建源/导入/建工单/准备/评估/反馈/固化，
  长任务（导入/准备/评估）拆「开始+结果」两分镜，分镜间轮询
  /api/tasks/{本次提交ID} 等待完成（不录制等待过程）。

用法：
    python tests\\_record_shots_btad.py                 # BTAD 02 全部 22 分镜
    python tests\\_record_shots_btad.py 05_prepare      # 只录指定分镜（可多个）
    $env:AOI_DEMO_DATASET='mpdd'; $env:AOI_DEMO_CATEGORY='tubes'; python tests\\_record_shots_btad.py

前置：关闭已运行的 AOI 实例（窗口标题唯一）；演示数据目录必须存在；
MPDD tubes 使用标准目录 MPDD\\tubes；干净演示环境（database/ 不存在则自动建库）。
素材输出：d:\\CGAIC\\AOI_SSOCL\\practice\\BTAD\\raw\\clips\\<时间戳>\\<shot>.mp4（+ .log）
"""
import os
import subprocess
import sys
import time
import json
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FF = (r"C:\Users\imia\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(tempfile.gettempdir(), "AOI_window_clips")
os.makedirs(CLIPS, exist_ok=True)

WIN_TITLE = "AOI 实时在线 AI 质检系统"
DEMO_DATASET = os.environ.get("AOI_DEMO_DATASET", "btad").strip().lower()
if DEMO_DATASET == "mpdd":
    _RUN_TAG = time.strftime("%m%d_%H%M%S")
    CATEGORY = os.environ.get("AOI_DEMO_CATEGORY", "tubes").strip()
    WO_NAME = f"MPDD-{CATEGORY} 质检演示 {_RUN_TAG}"
    DS_NAME = f"MPDD {CATEGORY} {_RUN_TAG}"
    DATA_ROOT = r"D:\CGAIC\data_origin\MPDD"
    DATASET_LABEL = "MPDD"
else:
    WO_NAME = "BTAD-02 质检演示"
    DS_NAME = "BTAD 金属件"
    DATA_ROOT = r"D:\CGAIC\data_origin\BTAD"
    CATEGORY = os.environ.get("AOI_DEMO_CATEGORY", "02").strip()
    DATASET_LABEL = "BTAD"
BTAD_ROOT = DATA_ROOT
LEAD = 2.0    # 引导段（剪辑裁掉）
TAIL = 3.0    # 录制总长 = 正片 + TAIL（含引导 2s + 余量 1s）
BASE = os.environ.get("AOI_DEMO_BASE", "http://127.0.0.1:8017")
# 每次实录使用独立目录，保留已有素材和成片。
CLIPS = os.path.join(CLIPS, time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(CLIPS, exist_ok=False)

import logging
logging.basicConfig(level=logging.ERROR)

import requests  # noqa: E402

# 后端：已被占用则复用（同进程 UI 直连），否则拉起
try:
    if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") != "ok":
        raise RuntimeError("bad")
    print("[启动] 复用已运行后端", flush=True)
except Exception:  # noqa: BLE001
    from server import start_server_background
    start_server_background()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import Qt, QTimer, QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,  # noqa: E402
                               QLineEdit, QListWidget, QPushButton,
                               QTableWidget)
import PySide6.QtWidgets as QW  # noqa: E402
from ui.theme import apply_light_theme  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

ISSUES = []
for _m in ("information", "warning", "question", "critical"):
    _orig = getattr(QW.QMessageBox, _m)

    def _mk(m=_m, o=_orig):
        def _p(*a, **k):
            print(f"  [弹窗] QMessageBox.{m}（自动应答）", flush=True)
            return QW.QMessageBox.Yes if m == "question" else None
        return _p
    setattr(QW.QMessageBox, _m, _mk())

app = QApplication([])
apply_light_theme(app)
win = MainWindow(BASE)
win.showMaximized()   # 用户规则：录制窗口最大化
win.raise_()
win.activateWindow()
_pump_end = time.time() + 1.0
while time.time() < _pump_end:
    app.processEvents()
    time.sleep(0.02)
_fg = win.frameGeometry()
WIN_X, WIN_Y, WIN_W, WIN_H = _fg.x(), _fg.y(), _fg.width(), _fg.height()
print(f"[窗口] 最大化窗口 {WIN_W}x{WIN_H}；素材目录 {CLIPS}", flush=True)

page_dashboard = win.page_dashboard
page_data = win.page_data
page_model = win.page_model
page_eval = win.page_eval
page_monitor = win.page_monitor
page_feedback = win.page_feedback
page_learning = win.page_learning
page_stats = win.page_stats
page_feature = win.page_feature
page_augment = win.page_augment
page_settings = win.page_settings
TASK_IDS = {}
DS_ID = None
WO_ID = None
if os.environ.get("AOI_DEMO_EXISTING_WO"):
    only_requested = set(sys.argv[1:])
    if not only_requested or not only_requested <= {"13_learning", "14_stats"}:
        raise RuntimeError("复用既有工单仅允许局部重录学习/统计，不执行创建或候选操作")
    WO_ID = int(os.environ["AOI_DEMO_EXISTING_WO"])
    existing = requests.get(f"{BASE}/api/workorders/{WO_ID}", timeout=15)
    existing.raise_for_status()
    existing = existing.json()
    sources = existing.get("datasources") or []
    if len(sources) != 1 or set(existing.get("categories") or []) != {CATEGORY}:
        raise RuntimeError("既有工单数据源/品类不唯一")
    DS_ID, DS_NAME = sources[0]["id"], sources[0]["name"]
    WO_NAME = existing["name"]
    print(f"[复用绑定] 工单 {WO_ID} / 数据源 {DS_ID} / {CATEGORY}；仅局部实录", flush=True)
EVAL_NAME = f"{WO_NAME} 精度评估 {time.time_ns()}"
FEEDBACK_ID = None
SPACE_DIALOG = None
LEARN_RESULT = None
CANDIDATE_DIALOG = None
CANDIDATE_RESULT = None
CANDIDATE_BINDING = None
ONLINE_EVIDENCE = None
client = win.client
_original_post = client.post


def capture_post(path, **kw):
    global FEEDBACK_ID, LEARN_RESULT, DS_ID, WO_ID
    payload = kw.get("json") or {}
    if path == "/api/datasets/adapt_import":
        if payload.get("categories") != [CATEGORY] or payload.get("datasource_id") != DS_ID:
            raise RuntimeError("导入必须仅绑定本次数据源与目标品类")
    if path == "/api/workorders":
        if payload.get("name") != WO_NAME or payload.get("datasource_ids") != [DS_ID]:
            raise RuntimeError("创建工单的数据源不是本次唯一数据源")
    if path in ("/api/models/prepare", "/api/eval/accuracy"):
        if payload.get("category") != CATEGORY:
            raise RuntimeError(f"拒绝提交其他品类：{payload}")
        if path == "/api/models/prepare" and payload.get("datasource_id") != DS_ID:
            raise RuntimeError("准备模型未绑定本次数据源")
    # 评估只用少量真实 test 图片，不运行全量评估。
    if path == "/api/eval/accuracy":
        kw["json"] = dict(payload, limit=12, name=EVAL_NAME)
    result = _original_post(path, **kw)
    if path == "/api/datasources" and payload.get("name") == DS_NAME:
        DS_ID = (result or {}).get("id")
    if path == "/api/workorders":
        WO_ID = (result or {}).get("id")
    label = None
    if path.startswith("/api/feedback/") and path.endswith("/candidate"):
        label = "构建候选反馈"
    elif path.startswith("/api/candidates/") and path.endswith("/acceptance"):
        label = "独立验收"
    elif path == "/api/datasets/adapt_import":
        label = f"导入 {DATASET_LABEL}"
    elif path == "/api/models/prepare":
        label = "准备模型"
    elif path == f"/api/models/feature-set/{CATEGORY}/report":
        label = "生成特征贡献档案"
    elif path == "/api/eval/accuracy":
        label = "精度评估"
    elif path.startswith("/api/workorders/") and path.endswith("/learn"):
        # 当前接口同步返回 applied_categories，并不创建后台 task。
        LEARN_RESULT = result or {}
    if label:
        TASK_IDS[label] = (result or {}).get("task_id")
        print(f"  [提交] {label} task_id={TASK_IDS[label]}", flush=True)
    if path == "/api/feedback":
        FEEDBACK_ID = (result or {}).get("id")
    return result


client.post = capture_post


def pump(ms=200):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


def wait_modal(title, timeout=300):
    """已废弃：主线程 pump 会被弹窗 exec() 嵌套事件循环捕获（首跑实测卡死）。
    改用 main() 里的 poll_modal（QTimer 轮询）。"""
    raise NotImplementedError


def wait_tasks_idle(label, timeout=1800):
    """只查询本次真实提交返回的 ID，绝不猜测最近任务。"""
    if label == "线上证据":
        return wait_online_evidence()
    if label == "固化学习快照":
        t0 = time.monotonic()
        while LEARN_RESULT is None and time.monotonic() - t0 < timeout:
            print(f"  [等待] 同步固化响应 {time.monotonic()-t0:.0f}s", flush=True)
            time.sleep(1)
        if LEARN_RESULT and LEARN_RESULT.get("applied_categories"):
            return True
        ISSUES.append("固化没有返回真实 applied_categories，停止学习结果分镜")
        return False
    task_id = TASK_IDS.get(label)
    if not task_id:
        ISSUES.append(f"缺少本次提交 task_id：{label}；停止依赖分镜")
        return False
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            task = client.get_task(task_id)
            if not task:
                raise RuntimeError("任务查询无响应")
            status = task.get("status")
            print(f"  [等待] {label} #{task_id} {status} "
                  f"{task.get('progress', '')} {time.monotonic()-t0:.0f}s", flush=True)
            if status == "done":
                return True
            if status in ("failed", "cancelled", "error"):
                ISSUES.append(f"任务失败：{label} #{task_id} {task.get('message')}")
                return False
        except Exception as exc:
            print(f"  [等待] #{task_id} 查询异常：{exc}", flush=True)
        time.sleep(3)
    ISSUES.append(f"等任务完成超时：{label} #{task_id}")
    return False


def close_feedback_evidence():
    from ui.widgets.feedback_dialog import FeedbackEvidenceDialog
    for widget in list(QApplication.topLevelWidgets()):
        if isinstance(widget, FeedbackEvidenceDialog) and widget.isVisible():
            widget.close()
            print("  [清理] 关闭反馈证据窗（仅应用自有窗）", flush=True)
    pump(300)


def close_modal():
    close_feedback_evidence()
    dlg = active_modal()
    if dlg is None:
        return
    if "数据集识别与确认" in dlg.windowTitle():
        return  # 识别确认窗留给 02 分镜处理，不能误关
    for txt in ("完成", "取消", "关闭"):
        b = [x for x in dlg.findChildren(QPushButton) if x.text().strip() == txt]
        if b:
            b[0].click()
            pump(300)
            return
    dlg.reject()
    pump(300)


def select_wo():
    """创建后按本次返回ID绑定，绝不回退旧工单。"""
    if WO_ID is None:
        return False
    for _ in range(12):
        idx = win.combo_workorder.findData(WO_ID)
        if idx >= 0:
            win.combo_workorder.setCurrentIndex(idx)
            pump(500)
            wo = win.current_workorder() or {}
            sources = {s.get("id") for s in wo.get("datasources") or []}
            if (win.combo_workorder.currentData() != WO_ID
                    or win.combo_workorder.currentText() != WO_NAME
                    or sources != {DS_ID}
                    or set(wo.get("categories") or []) != {CATEGORY}):
                raise RuntimeError(f"活动工单绑定不一致：{wo}")
            return True
        pump(500)
    raise RuntimeError(f"活动工单未切换到本次 {WO_NAME} #{WO_ID}")


def goto(page, settle=1000):
    if win.combo_workorder.count():
        select_wo()
    win.goto(page)
    pump(settle)


def _modal_btns(dlg):
    return dlg.findChildren(QPushButton) if dlg is not None else []


def _click_modal_btn(dlg, texts, tag):
    for b in _modal_btns(dlg):
        if b.text().strip() in texts:
            b.click()
            print(f"  [op] {tag}（{b.text().strip()}）", flush=True)
            return True
    ISSUES.append(f"找不到按钮：{tag}")
    print(f"  ❌ 找不到按钮：{tag}", flush=True)
    return False


# ═══════════ 01 新建数据源 ═══════════
def op_open_new_source():
    b = next((x for x in page_data.findChildren(QPushButton)
              if x.text().strip() == "新建数据源"), None)
    if b:
        b.click()
        print("  [op] 打开新建数据源弹窗", flush=True)
    else:
        ISSUES.append("数据管理页找不到「新建数据源」")


def op_fill_source():
    dlg = active_modal()
    if dlg is None:
        ISSUES.append("新建数据源弹窗未弹出")
        return
    for le in dlg.findChildren(QLineEdit):
        ph = le.placeholderText() or ""
        if "贴片机" in ph:
            le.setText(DS_NAME)
        elif "D:/datasets" in ph:
            le.setText(BTAD_ROOT)
        elif "树顶层分组名" in ph:
            le.setText(DATASET_LABEL)
    for cb in dlg.findChildren(QComboBox):
        i = cb.findData("L1b")
        if i >= 0:
            cb.setCurrentIndex(i)   # 标注档位
        j = cb.findData("adapt")
        if j >= 0:
            cb.setCurrentIndex(j)   # 导入数据=智能识别适配
    for ck in dlg.findChildren(QCheckBox):
        if "分品类" in ck.text():
            ck.setChecked(True)
    print("  [op] 数据源表单已填（名称/档位 L1b/分品类/智能识别/根目录）", flush=True)


def op_click_create_import():
    _click_modal_btn(active_modal(), ("创建并导入", "创建"), "创建并导入")


# ═══════════ 02 识别确认 ═══════════
def op_confirm_pick_02():
    dlg = active_modal()
    if dlg is None:
        ISSUES.append("识别确认窗不在")
        return
    checks = getattr(dlg, "_cat_checks", {})
    if CATEGORY not in checks:
        raise RuntimeError(f"确认窗找不到品类 {CATEGORY}")
    for name, checkbox in checks.items():
        checkbox.setChecked(name == CATEGORY)
    if [name for name, ck in checks.items() if ck.isChecked()] != [CATEGORY]:
        raise RuntimeError("导入确认未限制为唯一目标品类")
    print(f"  [op] 仅勾选导入品类 {CATEGORY}，取消其他 {len(checks)-1} 项", flush=True)


def op_click_confirm_import():
    _click_modal_btn(active_modal(), ("确认导入",), "确认导入")


# ═══════════ 03 导入完成展示 ═══════════
def _cat_of(item):
    # 数据树节点文本形如 "02 (629)"；模型树形如 "02" 或 "02 · v1 · 已投产"
    return item.text(0).split("·")[0].split("（")[0].split(" (")[0].strip()


def op_tree_pick_02():
    tree = page_data.tree
    def _walk(item, depth):
        ancestor = item
        own_source = False
        while ancestor is not None:
            if (ancestor.data(0, Qt.UserRole + 1) == DS_NAME
                    or ancestor.text(0).split("（")[0] == DS_NAME):
                own_source = True
                break
            ancestor = ancestor.parent()
        if _cat_of(item) == CATEGORY and own_source:
            tree.setCurrentItem(item)
            print(f"  [op] 数据树选中品类 {CATEGORY}（{item.text(0)}）", flush=True)
            return True
        if depth >= 3:
            return False
        item.setExpanded(True)
        for i in range(item.childCount()):
            if _walk(item.child(i), depth + 1):
                return True
        return False

    for _ in range(12):
        for t in range(tree.topLevelItemCount()):
            if _walk(tree.topLevelItem(t), 0):
                return
        pump(500)
    raise RuntimeError(f"数据树找不到本次源 {DS_NAME} 品类 {CATEGORY}")


def op_queue_pick_02():
    tbl = page_data.queue_table
    tbl.clearSelection()
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 2)  # 真实品类列，不匹配批次名或数量。
        key = tbl.item(r, 0)
        if (it is not None and it.text().strip() == CATEGORY
                and key is not None and key.data(Qt.UserRole)):
            tbl.selectRow(r)
            print(f"  [op] 队列选中品类 {CATEGORY} 批次（行 {r}）", flush=True)
            return
    tbl.clearSelection()
    raise RuntimeError(f"队列没有品类 {CATEGORY} 批次，禁止回队其他品类")


# ═══════════ 04 新建工单 ═══════════
def op_open_new_wo():
    b = next((x for x in page_dashboard.findChildren(QPushButton)
              if x.text().strip() == "新建工单"), None)
    if b:
        b.click()
        print("  [op] 打开新建工单弹窗", flush=True)
    else:
        ISSUES.append("工作台找不到「新建工单」")


def op_fill_wo():
    for _ in range(10):   # 数据源列表异步加载
        dlg = active_modal()
        if dlg is not None:
            lists = dlg.findChildren(QListWidget)
            if lists and lists[0].count():
                break
        pump(400)
    dlg = active_modal()
    if dlg is None:
        ISSUES.append("新建工单弹窗未弹出")
        return
    for le in dlg.findChildren(QLineEdit):
        if "SMT" in (le.placeholderText() or ""):
            le.setText(WO_NAME)
    if DS_ID is None:
        raise RuntimeError("缺少本次创建数据源ID")
    lw = dlg.list_sources
    matched = []
    for i in range(lw.count()):
        it = lw.item(i)
        if it.flags() & Qt.ItemIsUserCheckable:
            chosen = it.data(Qt.UserRole) == DS_ID
            it.setCheckState(Qt.Checked if chosen else Qt.Unchecked)
            if chosen:
                matched.append(it)
    if len(matched) != 1 or DS_NAME not in matched[0].text():
        raise RuntimeError(f"工单弹窗未唯一匹配本次数据源 {DS_NAME} #{DS_ID}")
    print(f"  [op] 工单仅勾选数据源「{matched[0].text()}」", flush=True)
    print("  [op] 工单表单已填", flush=True)


def op_click_create_wo():
    _click_modal_btn(active_modal(), ("创建工单",), "创建工单")


# ═══════════ 05/06 模型管理 ═══════════
def op_model_tree_02():
    tree = getattr(page_model, "tree", None)
    if tree is None:
        ISSUES.append("模型页无 tree")
        return
    if not select_wo():
        raise RuntimeError("准备模型前未创建本次工单")
    for _ in range(12):
        for t in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(t)
            top.setExpanded(True)
            for i in range(top.childCount()):
                ch = top.child(i)
                if ch.data(0, Qt.UserRole) == ("category", DS_ID, DS_NAME, CATEGORY):
                    tree.setCurrentItem(ch)
                    print(f"  [op] 模型树选中本次源 {DS_NAME} / {CATEGORY}", flush=True)
                    return
        pump(500)
    raise RuntimeError(f"模型树找不到本次源 {DS_NAME} 品类 {CATEGORY}")


def op_open_prepare():
    page_model.btn_prepare.click()
    print("  [op] 打开准备模型弹窗", flush=True)


def op_start_prepare():
    _click_modal_btn(active_modal(), ("开始准备",), "开始准备")


# ═══════════ 07/08 评估看板 ═══════════
def op_eval_pick_02():
    cb = page_eval.combo_category
    idx = cb.findData(CATEGORY)
    if idx < 0:
        idx = next((i for i in range(cb.count())
                    if cb.itemText(i).split(" ")[0].strip() == CATEGORY), -1)
    if idx >= 0:
        cb.setCurrentIndex(idx)
        print(f"  [op] 评估品类选 {CATEGORY}", flush=True)
    else:
        ISSUES.append(f"评估看板无品类 {CATEGORY}")


def op_eval_run():
    b = next((x for x in page_eval.findChildren(QPushButton)
              if x.text().strip() == "运行精度评估"), None)
    if b:
        b.click()
        print("  [op] 运行精度评估", flush=True)
    else:
        ISSUES.append("评估看板找不到「运行精度评估」")


def op_eval_refresh():
    b = next((x for x in page_eval.findChildren(QPushButton)
              if x.text().strip() == "刷新"), None)
    if b:
        b.click()
        print("  [op] 评估记录刷新", flush=True)
    for _ in range(12):
        for row, run in enumerate(getattr(page_eval, "_runs", [])):
            if run.get("name") == EVAL_NAME and run.get("category") == CATEGORY:
                page_eval.table.selectRow(row)
                page_eval.table.scrollToItem(page_eval.table.item(row, 2))
                print(f"  [op] 选中本次真实评估 #{run.get('id')} {EVAL_NAME}", flush=True)
                return
        pump(500)
    raise RuntimeError("没有本次提交的精度评估记录，禁止展示旧记录")


# ═══════════ 09/10/12 实时监控 ═══════════
def op_ensure_pipeline():
    txt = page_monitor.lbl_pipeline.text()
    if "运行中" not in txt:
        page_monitor.btn_pp_resume.click()
        pump(1200)
        print(f"  [op] 产线恢复运行（原状态 {txt}）", flush=True)
    else:
        print(f"  [op] 产线运行中（{txt}）", flush=True)


def op_layer_heatmap():
    """实时流切热力图图层：产线帧已落热力图，画面随检测流更新。"""
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        records = client.get("/api/detections/live", params={
            "category": CATEGORY, "workorder_id": WO_ID, "limit": 20}) or {}
        ready = False
        for det in records.get("items", []):
            if not det.get("is_anomaly") or not det.get("image_id"):
                continue
            images = client.list_images(ids=str(det["image_id"]), category=CATEGORY) or {}
            if any(i.get("label") == "normal" for i in images.get("items", [])):
                ready = True
                break
        if ready:
            break
        print("  [真实产线等待] 等正常样本经过推理，不制造误检", flush=True)
        pump(1000)
    b = (page_monitor._layer_btns or {}).get("heatmap")
    if b:
        b.click()
        print("  [op] 实时流切热力图图层", flush=True)


def op_layer_overlay():
    """切回叠加图图层（默认视图）。"""
    b = (page_monitor._layer_btns or {}).get("overlay")
    if b:
        b.click()
        print("  [op] 实时流切回叠加图图层", flush=True)


def op_pause_pipeline():
    page_monitor.btn_pp_pause.click()
    pump(800)
    print(f"  [op] 暂停产线（{page_monitor.lbl_pipeline.text()}）", flush=True)


def op_mark_fp():
    try:
        params = {
            "category": CATEGORY,
            "workorder_id": WO_ID,
            "after_id": 0,
            "limit": 20,
        }
        if not select_wo():
            raise RuntimeError("反馈前缺少本次工单")
        r = requests.get(f"{BASE}/api/detections/live", params=params, timeout=8)
        r.raise_for_status()
        dets = r.json().get("items", [])
    except Exception as e:  # noqa: BLE001
        dets = []
        print(f"  [op] 取品类 02 产线检测失败 {e}", flush=True)
    if not dets:
        raise RuntimeError("真实误检缺口：品类02无可核验的产线检测记录")
    det = None
    for candidate in dets:
        decision = candidate.get("decision") or (candidate.get("n_tiles") or {}).get("decision")
        if not candidate.get("is_anomaly") or decision not in (None, "anomaly"):
            continue
        image_id = candidate.get("image_id")
        if image_id is None:
            continue
        images = client.list_images(ids=str(image_id), category=CATEGORY) or {}
        image = next((x for x in images.get("items", [])
                      if x.get("id") == image_id and x.get("label") == "normal"), None)
        if image is not None:
            det = candidate
            break
    if det is None:
        raise RuntimeError("真实误检缺口：最近20条品类02产线记录未发现预测异常且真实标签normal的样本")
    page_monitor.chk_follow.setChecked(False)
    page_monitor._show_detection(det)
    # 列表返回的持久检测记录已由真实 image_id 和标签核验。
    page_monitor._open_feedback("false_positive")
    print(f"  [op] 标记品类 02 产线误检（检测 #{det.get('id')}）", flush=True)


def _feedback_dlg():
    for _ in range(10):
        dlg = getattr(page_monitor, "_feedback_dialog", None)
        if dlg is not None:
            return dlg
        pump(300)
    return None


def op_fill_feedback():
    dlg = _feedback_dlg()
    if dlg is None:
        ISSUES.append("反馈对话框未弹出")
        return
    for le in dlg.findChildren(QLineEdit):
        ph = le.placeholderText() or ""
        if "操作员" in ph:
            le.setText("演示")
        elif "划痕" in ph:
            le.clear()  # 真值为正常，不虚构缺陷类型。
    print("  [op] 反馈对话框已填写", flush=True)


def op_submit_feedback():
    dlg = _feedback_dlg()
    if dlg is None:
        ISSUES.append("提交时反馈对话框不在")
        return
    _click_modal_btn(dlg, ("提交反馈",), "提交反馈")


def op_learn():
    global LEARN_RESULT
    LEARN_RESULT = None
    page_monitor.btn_pp_learn.click()
    print("  [op] 固化学习快照", flush=True)


# ═══════════ 11 标注反馈 ═══════════
def op_records_tab():
    tabs = page_feedback.findChildren(QW.QTabWidget)
    if tabs:
        tabs[0].setCurrentIndex(0)
    for combo, value in ((page_feedback.combo_fb_type, "false_positive"),
                         (page_feedback.combo_fb_cat, CATEGORY)):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
    page_feedback.reload_records()
    print("  [op] 切「反馈记录」页签", flush=True)


# ═══════════ 候选学习：只消费真实队列，绝不批准/激活 ═══════════
def op_candidate_open():
    global CANDIDATE_DIALOG
    for row in range(page_feedback.table_pending.rowCount()):
        item = page_feedback.table_pending.item(row, 0)
        if item is not None and item.text() == CATEGORY:
            cell = page_feedback.table_pending.cellWidget(row, 3)
            button = next((b for b in cell.findChildren(QPushButton)
                           if b.text() == "候选学习…"), None) if cell else None
            if button:
                def remember():
                    global CANDIDATE_DIALOG
                    CANDIDATE_DIALOG = active_modal()
                QTimer.singleShot(200, remember)
                button.click()
                return
    raise RuntimeError("反馈状态表尚无品类02候选学习入口")


def op_candidate_bind():
    global CANDIDATE_BINDING
    dlg = CANDIDATE_DIALOG
    if dlg is None or not dlg._feedback_items:
        raise RuntimeError("真实候选反馈尚未加载")
    models = client.list_models(CATEGORY) or []
    active = [m for m in models if m.get("is_active") and m.get("category") == CATEGORY]
    if len(active) != 1:
        raise RuntimeError("当前激活父版本不唯一")
    parent = int(str(active[0]["version"]).lstrip("v"))
    if int(dlg.edit_parent.text()) != parent:
        raise RuntimeError("UI父版本与当前真实激活版本不一致")
    index = next((i for i, item in enumerate(dlg._feedback_items)
                  if item.get("id") == FEEDBACK_ID), -1)
    if index < 0:
        raise RuntimeError("候选列表缺少本次反馈，禁止消费旧反馈")
    feedback = dlg._feedback_items[index]
    # 原有人工结论只核验不改写，禁止将真实缺陷标正常。
    images = client.list_images(category=CATEGORY, page_size=2,
                                ids=str(client.get(f"/api/detections/{feedback['detection_id']}")["image_id"])) or {}
    expected = "anomaly" if feedback.get("operator_label") == 1 else "normal"
    if not any(i.get("label") == expected for i in images.get("items", [])):
        raise RuntimeError("候选反馈人工标签与登记图片真值不一致")
    dlg.combo_feedback.setCurrentIndex(index)
    CANDIDATE_BINDING = {"feedback_id": feedback["id"], "parent_version": parent}
    print(f"  [候选绑定] {CANDIDATE_BINDING}，保留原人工标签", flush=True)


def op_candidate_build():
    if not CANDIDATE_BINDING:
        raise RuntimeError("未核验真实候选绑定")
    CANDIDATE_DIALOG.btn_build.click()


def op_candidate_status():
    global CANDIDATE_RESULT
    task = client.get_task(TASK_IDS["构建候选反馈"])
    CANDIDATE_RESULT = task.get("result") or {}
    version = CANDIDATE_RESULT.get("version_num")
    if not version or CANDIDATE_RESULT.get("parent_version") != CANDIDATE_BINDING["parent_version"]:
        raise RuntimeError(f"候选任务结果缺少绑定版本：{CANDIDATE_RESULT}")
    print(f"  [候选真实结果] {CANDIDATE_RESULT}", flush=True)
    CANDIDATE_DIALOG.edit_version.setText(str(version))
    CANDIDATE_DIALOG.btn_status.click()


def op_acceptance_boundary():
    from backend.self_learning.candidate_release import business_policy
    policy = business_policy()
    # 后端仅冻结四项，延迟预算和报告有效期没有可读取的已批准值。
    reason = (f"独立验收未提交：后端冻结政策 {policy}；max_predict_ms 与 valid_seconds "
              "需业务确认，不能自行编造。test表还须两侧正常/缺陷且排除候选/WAL训练资产。"
              "本镜仅展示真实候选状态和验收入口；未批准、未激活。")
    CANDIDATE_DIALOG.lbl_result.setText(reason)
    print(f"  [验收阻碍] {reason}", flush=True)
    if CANDIDATE_DIALOG.lbl_status.text().startswith("候选状态读取失败"):
        raise RuntimeError(CANDIDATE_DIALOG.lbl_status.text())
    print(f"  [候选状态] {CANDIDATE_DIALOG.lbl_status.text()}", flush=True)
    import tempfile
    evidence = os.path.join(tempfile.gettempdir(), f"btad_candidate_{time.time_ns()}.png")
    CANDIDATE_DIALOG.grab().save(evidence)
    print(f"  [候选窗口证据] {evidence}", flush=True)


def wait_online_evidence():
    global ONLINE_EVIDENCE
    import json
    import tempfile
    import threading
    result, errors = {}, []
    def fetch():
        try:
            check = requests.get(f"{BASE}/api/feedback", params={"category": CATEGORY, "page_size": 1}, timeout=15)
            check.raise_for_status()
            feedback_total = int(check.json().get("total", 0))
            if feedback_total > 20:
                raise RuntimeError("反馈超过20条，停止录前重打分，禁止全量扫描")
            response = requests.get(f"{BASE}/api/learning/flip_curve/{CATEGORY}", timeout=600)
            response.raise_for_status()
            flip = response.json()
            if flip.get("task_id"):
                TASK_IDS["翻案曲线"] = flip["task_id"]
                if not wait_tasks_idle("翻案曲线", 900):
                    raise RuntimeError("翻案任务未成功")
                flip = client.get_task(flip["task_id"]).get("result")
            if not isinstance(flip, dict) or not isinstance(flip.get("points"), list):
                raise RuntimeError(f"翻案接口缺少真实points：{flip}")
            # 权重图优先使用真实候选快照；这是独立候选证据，不冒充生产更新。
            candidate_response = requests.get(
                f"{BASE}/api/learning/candidate_weight_history/{CATEGORY}", timeout=60)
            candidate_response.raise_for_status()
            candidate = candidate_response.json()
            candidate_points = candidate.get("points") if isinstance(candidate, dict) else None
            if not isinstance(candidate_points, list) or not candidate_points:
                raise RuntimeError(f"候选权重历史为空，禁止用生产快照冒充候选：{candidate}")
            weight = candidate
            misjudged = requests.get(
                f"{BASE}/api/detections/misjudged",
                params={"category": CATEGORY, "page": 1, "page_size": 20}, timeout=30)
            misjudged.raise_for_status()
            misjudged = misjudged.json()
            trend = requests.get(
                f"{BASE}/api/stats/mistake_trend",
                params={"category": CATEGORY, "range_name": "30d"}, timeout=30)
            trend.raise_for_status()
            trend = trend.json()
            online = requests.get(
                f"{BASE}/api/learning/online_curve/{CATEGORY}", timeout=30)
            online.raise_for_status()
            online = online.json()
            result.update(flip=flip, weight=weight, misjudged=misjudged,
                          trend=trend, online=online, feedback_total=feedback_total)
        except Exception as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    start = time.monotonic()
    while worker.is_alive():
        print(f"  [录前等待] 现有真实反馈翻案重打分 {time.monotonic()-start:.0f}s", flush=True)
        worker.join(3)
    if errors:
        ISSUES.append(f"学习证据就绪失败：{errors[0]}")
        return False
    ONLINE_EVIDENCE = result
    # 同次预取的真实响应交给原UI渲染，避免切页重复启动慢重打分。
    client.flip_curve = lambda category: result["flip"] if str(category) == CATEGORY else None
    client.weight_history = lambda category: result["weight"] if str(category) == CATEGORY else None
    path = os.path.join(tempfile.gettempdir(), f"btad_learning_{time.time_ns()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [真实证据] flip points={len(result['flip']['points'])}；candidate weight points="
          f"{len(result['weight'].get('points', []))}；misjudged records="
          f"{result['misjudged'].get('total', 0)}；trend reviewed="
          f"{sum(int(x.get('reviewed', 0) or 0) for x in result['trend'].get('items', []))}；online points="
          f"{len(result['online'].get('points', []))}；n_labeled_fb="
          f"{result['weight'].get('n_labeled_fb', 'candidate-snapshot')}；TEMP={path}", flush=True)
    return True


# ═══════════ 13 学习效果 ═══════════
def op_learning_online_refresh():
    tabs = getattr(page_learning, "tabs", None)
    if tabs:
        tabs.setCurrentIndex(0)
    cb = page_learning.combo_online_cat
    for _ in range(10):   # 品类列表异步加载，等它填充
        idx = cb.findData(CATEGORY)
        if idx < 0:
            idx = next((i for i in range(cb.count())
                        if cb.itemText(i).split(" ")[0].strip() == CATEGORY), -1)
        if idx >= 0:
            cb.setCurrentIndex(idx)
            if ONLINE_EVIDENCE is None:
                raise RuntimeError("学习证据未在录制前就绪")
            candidate_idx = page_learning.combo_weight_source.findData("candidate")
            if candidate_idx < 0:
                raise RuntimeError("学习效果页缺少候选权重来源")
            page_learning.combo_weight_source.setCurrentIndex(candidate_idx)
            evidence = ONLINE_EVIDENCE
            if not evidence["flip"].get("points"):
                raise RuntimeError("翻案图没有真实points，拒绝录空图")
            if not evidence["weight"].get("points"):
                raise RuntimeError("候选权重图没有真实points，拒绝录空图")
            if not evidence["online"].get("points"):
                raise RuntimeError("线上曲线没有真实points，拒绝录空图")
            reviewed = sum(int(x.get("reviewed", 0) or 0)
                           for x in evidence["trend"].get("items", []))
            if reviewed <= 0:
                raise RuntimeError("趋势图没有真实复核记录，拒绝录空图")
            page_learning._fill_flip_curve(evidence["flip"])
            page_learning._fill_weight_history(evidence["weight"])
            page_learning._fill_misjudged(evidence["misjudged"])
            page_learning._fill_trend(evidence["trend"])
            page_learning._fill_online_curve(evidence["online"])
            print(f"  [op] 学习效果进入候选/线上真实证据：flip={len(evidence['flip']['points'])}，"
                  f"weight={len(evidence['weight']['points'])}，online={len(evidence['online']['points'])}", flush=True)
            return
        pump(600)
    ISSUES.append("学习效果页无品类 02")


# ═══════════ 14 存档统计 ═══════════
def op_stats_refresh():
    b = next((x for x in page_stats.findChildren(QPushButton)
              if x.text().strip() == "刷新"), None)
    if b:
        b.click()
        print("  [op] 存档统计刷新", flush=True)
    # 树异步填充后选中本工单，驱动右侧统计面板
    for _ in range(16):
        tree = page_stats.tree
        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            for j in range(g.childCount()):
                it = g.child(j)
                if it.data(0, Qt.UserRole) == win.combo_workorder.currentData():
                    name = it.text(0)   # 先取文本：选中后刷新会 clear 树
                    tree.setCurrentItem(it)
                    pump(1500)
                    print(f"  [op] 存档统计选中工单（{name}）", flush=True)
                    return
        pump(500)
    ISSUES.append("存档统计树未找到工单")


# ═══════════ 03b 数据增强：真实预览 + 三维特征空间 ═══════════
def op_augment_select_normal():
    tree = page_augment.tree_base

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            img = child.data(0, Qt.UserRole)
            if isinstance(img, dict) and str(img.get("category")) == CATEGORY:
                dataset = page_augment._feature_datasets.get(img.get("dataset_id")) or {}
                if dataset.get("datasource_id") == DS_ID and img.get("label") == "normal":
                    return child
            found = walk(child)
            if found:
                return found
        return None

    for _ in range(12):
        target = None
        for i in range(tree.topLevelItemCount()):
            target = walk(tree.topLevelItem(i))
            if target is not None:
                break
        if target:
            tree.setCurrentItem(target)
            print(f"  [op] 数据增强选择品类 {CATEGORY} 正常基底图", flush=True)
            pump(800)
            return
        if tree.topLevelItemCount():
            tree.expandAll()
        pump(500)
    ISSUES.append("数据增强页找不到品类 02 正常基底图")


def op_augment_preview():
    b = next((x for x in page_augment.findChildren(QPushButton)
              if x.text().strip() == "预览合成"), None)
    if b:
        b.click()
        print("  [op] 数据增强预览合成", flush=True)
        pump(1200)
    else:
        ISSUES.append("数据增强页找不到预览合成")


def op_augment_preprocess():
    b = getattr(page_augment, "btn_pre_preview", None)
    if b:
        b.click()
        print("  [op] 数据增强预览当前预处理", flush=True)
        pump(1200)


def op_augment_space():
    b = next((x for x in page_augment.findChildren(QPushButton)
              if x.text().strip() == "真实特征空间分析"), None)
    if not b:
        ISSUES.append("数据增强页找不到真实特征空间分析")
        return
    # click 进入 exec()，开始定时器必须先安排。
    QTimer.singleShot(1000, lambda: _safe_op("03b_augment", 0, op_feature_space_start))
    b.click()
    print("  [op] 三维弹窗已关闭", flush=True)


def op_feature_space_start():
    global SPACE_DIALOG
    dlg = active_modal()
    SPACE_DIALOG = dlg
    if dlg is not None and hasattr(dlg, "_task_id"):
        original_poll = dlg._poll

        def capture_dialog_task():
            # _poll 首次调用前 _task_id 已赋值，done 回调随后会清空它。
            if dlg._task_id:
                TASK_IDS["三维分析"] = dlg._task_id
            return original_poll()

        dlg._poll = capture_dialog_task
    if dlg is None:
        ISSUES.append("三维特征空间分析弹窗未打开")
        return
    b = next((x for x in dlg.findChildren(QPushButton)
              if x.text().strip() == "开始分析"), None)
    if b:
        b.click()
        print("  [op] 三维特征空间开始分析", flush=True)
    else:
        ISSUES.append("三维弹窗找不到开始分析")


# ═══════════ 06b 特征工程 ═══════════
def op_feature_cat02():
    cb = page_feature.combo_category
    for _ in range(10):
        idx = cb.findText(CATEGORY)
        if idx >= 0:
            cb.setCurrentIndex(idx)
            pump(1500)
            print(f"  [op] 特征工程切品类 {CATEGORY}", flush=True)
            return
        pump(600)
    ISSUES.append("特征工程页无品类 02")


def op_feature_report():
    b = getattr(page_feature, "btn_report", None)
    if b:
        b.click()
        print("  [op] 生成特征贡献档案", flush=True)
    else:
        ISSUES.append("特征工程页找不到分析当前特征影响")


def op_feature_refresh():
    b = next((x for x in page_feature.findChildren(QPushButton)
              if x.text().strip() == "刷新分析结果"), None)
    if b:
        b.click()
        print("  [op] 刷新特征分析结果", flush=True)
        pump(1800)
    else:
        ISSUES.append("特征工程页找不到刷新分析结果")


def require_space_result():
    dlg = SPACE_DIALOG
    if dlg is None or dlg._closed or not dlg._samples or not dlg._artists:
        raise RuntimeError("三维任务未生成真实样本/散点，拒绝录制空结果")
    return dlg


def op_space_rotate():
    dlg = require_space_result()
    dlg.axes.view_init(elev=32, azim=125)
    dlg.canvas.draw()


def op_space_pick():
    from matplotlib.backend_bases import MouseEvent, PickEvent
    dlg = require_space_result()
    artist = next((a for a, points in dlg._artists.items()
                   if points and points[0]["kind"] in ("normal", "defect")), None)
    if artist is None:
        raise RuntimeError("三维图没有可点选的真实样本")
    event = PickEvent("pick_event", dlg.canvas,
                      MouseEvent("button_press_event", dlg.canvas, 0, 0, button=1),
                      artist, ind=[0])
    dlg.canvas.callbacks.process("pick_event", event)
    if dlg.image.pixmap().isNull():
        raise RuntimeError("真实样本点选后未显示缩略图")


def op_feature_evidence():
    report = page_feature._fs.get("report") or {}
    if not report.get("trad_groups"):
        task = client.get_task(TASK_IDS["生成特征贡献档案"])
        profile = (task.get("result") or {}).get("profile") or {}
        ablation = profile.get("ablation") or {}
        single = ablation.get("single") or {}
        loo = ablation.get("leave_one_out") or {}
        if task.get("status") != "done" or not single or not loo:
            raise RuntimeError("本次真实贡献任务没有可展示的槽位消融数值")
        if (report.get("ablation") or {}) != ablation:
            raise RuntimeError("应用展示的槽位档案与本次真实任务不一致")
        if not page_feature.table.rowCount() or not page_feature._fs.get("production_slots"):
            raise RuntimeError("特征左侧未展示真实生产槽位")
        page_feature._record_slot_profile = profile
        print(f"  [真实槽位贡献] {ablation}", flush=True)
        show_widget(page_feature.group_chart)
        return
    groups = (report.get("trad_groups") or {}).get("groups") or {}
    if not isinstance(groups, dict) or not any(
            isinstance(g, dict) and any(isinstance(g.get(k), (int, float))
                                       for k in ("single_auroc", "rebuild_loo_drop"))
            for g in groups.values()):
        raise RuntimeError("右侧特征贡献 groups 缺少真实数值证据")
    show_widget(page_feature.group_chart)
    if not page_feature.combo_image.count() or not page_feature._image_rows:
        raise RuntimeError("特征档案缺少真实逐图贡献")


def op_feature_image(index):
    if getattr(page_feature, "_record_slot_profile", None):
        show_widget(page_feature.group_chart)
        return
    if index >= page_feature.combo_image.count():
        raise RuntimeError(f"逐图贡献缺少第 {index+1} 个真实样本")
    page_feature.combo_image.setCurrentIndex(index)
    show_widget(page_feature.image_table)
    if not page_feature.image_table.rowCount():
        raise RuntimeError("所选真实样本没有逐图贡献行")


def show_widget(widget):
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QW.QScrollArea):
            parent.ensureWidgetVisible(widget, 20, 20)
            break
        parent = parent.parentWidget()
    widget.setFocus()


def op_select_feedback_record():
    if FEEDBACK_ID is None:
        raise RuntimeError("没有本次提交反馈ID，不能展示其他记录冒充")
    for row, record in enumerate(page_feedback._record_items):
        if record.get("id") == FEEDBACK_ID:
            page_feedback.table.selectRow(row)
            page_feedback.table.scrollToItem(page_feedback.table.item(row, 0))
            return
    raise RuntimeError(f"反馈记录未加载本次记录 #{FEEDBACK_ID}")


SETTINGS_ORIGINAL = None


def op_settings_edit():
    global SETTINGS_ORIGINAL
    for key, editor in page_settings._editors.items():
        if isinstance(editor, QCheckBox):
            SETTINGS_ORIGINAL = (key, editor.isChecked())
            show_widget(editor)
            editor.setChecked(not editor.isChecked())
            return
    raise RuntimeError("设置页没有可安全临时切换的复选框")


def op_settings_restore():
    if SETTINGS_ORIGINAL is None:
        raise RuntimeError("没有临时设置可恢复")
    key, value = SETTINGS_ORIGINAL
    page_settings._editors[key].setChecked(value)
    # 不调用保存、恢复默认或清理接口。


def op_settings_logs():
    idx = page_settings.combo_log_action.findData("feedback")
    if idx >= 0:
        page_settings.combo_log_action.setCurrentIndex(idx)
    show_widget(page_settings.table_logs)


# ═══════════ 15 回队 ═══════════
def op_requeue():
    op_queue_pick_02()
    page_data.btn_queue_req.click()
    print("  [op] 批次回队（重新检测）", flush=True)


# ═══════════ 分镜表：(name, 页面, 正片时长s, [(offset, op)], prep) ═══════════
# prep：None=直接录；("modal",)=等识别确认窗出现（QTimer 轮询）；
#       ("task", 标签, 超时, refresh)=worker 线程轮询任务完成后再录
def _refresh_data():
    # 切走再切回（goto 同页可能是 no-op，不触发刷新）
    win.goto("工作台")
    pump(600)
    if hasattr(page_data, "reload"):
        page_data.reload()
    pump(500)


def _refresh_model():
    if hasattr(page_model, "reload"):
        page_model.reload()
    pump(500)


SHOTS = [
    ("01_new_source", "工作台", 33, [
        (3, lambda: goto("数据管理")),
        (5, op_open_new_source), (8, op_fill_source), (21, op_click_create_import)],
     None),
    ("02_probe_confirm", "数据管理", 20, [
        (4, op_confirm_pick_02), (12, op_click_confirm_import)],
     ("modal",)),
    ("03_imported", "数据管理", 12, [
        (2, op_tree_pick_02)],
     ("task", f"导入 {DATASET_LABEL}", 1800, _refresh_data)),
    ("03b_augment", "数据增强", 24, [
        (2, op_augment_select_normal),
        (6, op_augment_preview),
        (10, op_augment_preprocess),
        (14, op_augment_space)],
     None),
    ("03c_space_done", "数据增强", 20, [
        (3, op_space_rotate), (9, op_space_pick)],
     ("task", "三维分析", 600, require_space_result)),
    ("04_workorder", "工作台", 18, [
        (2, op_open_new_wo), (5, op_fill_wo), (13, op_click_create_wo)],
     None),
    ("05_prepare", "模型管理", 35, [
        (2, op_model_tree_02), (6, op_open_prepare), (14, op_start_prepare)],
     None),
    ("06_prepared", "模型管理", 10, [(2, op_model_tree_02)],
     ("task", "准备模型", 1800, _refresh_model)),
    ("06b_feature", "特征工程", 14, [
        (2, op_feature_cat02),
        (7, op_feature_report)],
     None),
    ("06c_feature_done", "特征工程", 26, [
        (2, op_feature_cat02), (5, op_feature_refresh),
        (9, op_feature_evidence),
        (14, lambda: op_feature_image(0)),
        (20, lambda: op_feature_image(1))],
     ("task", "生成特征贡献档案", 900, None)),
    ("07_eval", "评估看板", 25, [
        (2, op_eval_pick_02), (5, op_eval_run)],
     None),
    ("08_eval_done", "评估看板", 12, [(2, op_eval_refresh)],
     ("task", "精度评估", 1800, None)),
    ("09_monitor", "实时监控", 25, [(2, op_ensure_pipeline),
                                    (10, op_layer_heatmap),
                                    (18, op_layer_overlay)],
     None),
    ("10_feedback", "实时监控", 15, [
        (1, op_mark_fp), (4, op_fill_feedback), (8, op_submit_feedback)],
     None),
    ("11_review", "标注反馈", 12, [
        (2, op_records_tab), (6, op_select_feedback_record)],
     None),
    ("12_learn", "实时监控", 25, [
        (2, op_pause_pipeline), (6, op_learn)],
     None),
    ("12b_candidate", "标注反馈", 18, [
        (2, op_candidate_open), (6, op_candidate_bind), (10, op_candidate_build)],
     None),
    ("12c_acceptance_status", "标注反馈", 18, [
        (2, op_candidate_status), (8, op_acceptance_boundary)],
     ("task", "构建候选反馈", 900, None)),
    ("13_learning", "学习效果", 20, [(2, op_learning_online_refresh)],
     ("task", "线上证据", 900, None)),
    ("14_stats", "存档统计", 24, [
        (2, op_stats_refresh), (7, lambda: show_widget(page_stats.table_batch)),
        (11, lambda: show_widget(page_stats.trend_chart)),
        (15, lambda: show_widget(page_stats.pareto_chart)),
        (19, lambda: show_widget(page_stats.latency_chart))],
     None),
    ("14b_settings", "系统设置", 16, [
        (2, op_settings_edit), (6, op_settings_restore), (10, op_settings_logs)],
     None),
    ("15_requeue", "数据管理", 10, [
        (2, op_queue_pick_02), (4, op_requeue)],
     None),
]


def start_rec(name, dur):
    import threading
    out = os.path.join(CLIPS, f"{name}.mp4")
    lf = open(os.path.join(CLIPS, f"{name}.log"), "xb")
    win.showMaximized()
    width, height = win.width() // 2 * 2, win.height() // 2 * 2
    proc = subprocess.Popen(
        [FF, "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE, stdout=lf, stderr=lf)
    latest = [None]
    proc.stop_encoding = threading.Event()
    proc.started_at = time.monotonic()
    proc.frame_count = 0
    capture = QTimer(win)
    capture.setInterval(33)

    def frame():
        image = QImage(width, height, QImage.Format_RGB888)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.drawPixmap(0, 0, win.grab())
        owned = []
        for widget in QApplication.topLevelWidgets():
            if widget is win or not widget.isVisible():
                continue
            parent = widget.parentWidget()
            while parent is not None and parent is not win:
                parent = parent.parentWidget()
            if parent is win:
                owned.append(widget)
        modal = QApplication.activeModalWidget()
        owned.sort(key=lambda w: w is modal or w is QApplication.activeWindow())
        origin = win.mapToGlobal(QPoint(0, 0))
        for widget in owned:
            point = widget.mapToGlobal(QPoint(0, 0)) - origin
            painter.drawPixmap(point, widget.grab())
        painter.end()
        # RGB888 的行可能有对齐填充；仅传实际像素。
        raw = bytes(image.constBits())
        stride = image.bytesPerLine()
        latest[0] = b"".join(raw[y*stride:y*stride+width*3] for y in range(height))

    frame()
    capture.timeout.connect(frame)
    capture.start()

    def encode():
        started = time.monotonic()
        try:
            index = 0
            while not proc.stop_encoding.is_set():
                time.sleep(max(0, started + index / 30 - time.monotonic()))
                proc.stdin.write(latest[0])
                index += 1
                proc.frame_count = index
        except (BrokenPipeError, OSError) as exc:
            ISSUES.append(f"{name} 编码失败：{exc}")
        finally:
            proc.stdin.close()

    threading.Thread(target=encode, daemon=True).start()
    proc.capture_timer = capture
    return proc, out, lf


def verify(name, out, dur):
    try:
        r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", out],
                           capture_output=True, text=True, timeout=30)
        d = float(r.stdout.strip())
    except Exception as e:  # noqa: BLE001
        d = -1
        print(f"  [校验] ffprobe 异常 {e}", flush=True)
    ok = d >= dur + TAIL - 1.5
    print(f"  [校验] {name}.mp4 时长 {d:.1f}s（期望 ≥{dur + TAIL - 1.5:.1f}）"
          f" {'✅' if ok else '❌'}", flush=True)
    if not ok:
        ISSUES.append(f"{name} 录制时长异常 {d}")


CUR_SHOT = ""  # 当前分镜；ffmpeg 秒退时防 QTimer 串分镜执行


def _safe_op(name, off, fn):
    if CUR_SHOT != name or ISSUES:
        return  # 录制已失败进入下一分镜，丢弃迟到的定时 op
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ISSUES.append(f"{name} op@{off}s 异常 {type(e).__name__}: {e}")
        print(f"  [op 异常] {type(e).__name__}: {e}", flush=True)


def poll_modal(title, on_found, timeout=300):
    """QTimer 轮询等模态弹窗出现（不能用主线程 pump 等：弹窗 exec() 会
    把主线程捕获进嵌套事件循环，首跑实测卡死）。"""
    t0 = time.time()

    def _poll():
        m = active_modal()
        if m is not None and title in m.windowTitle():
            print(f"  [等待] 弹窗「{title}」已出现（{time.time()-t0:.0f}s）",
                  flush=True)
            on_found()
            return
        if time.time() - t0 >= timeout:
            ISSUES.append(f"等弹窗超时：{title}")
            print(f"  ❌ 等弹窗超时：{title}", flush=True)
            app.exit(1)
            return
        if int(time.time() - t0) % 15 == 0:
            print(f"  [等待] 等弹窗「{title}」… {time.time()-t0:.0f}s", flush=True)
        QTimer.singleShot(1000, _poll)
    _poll()


MANIFEST = {"dataset": DEMO_DATASET, "category": CATEGORY, "shots": [], "complete": False}


def ui_ready(fn):
    name = fn.__name__
    dlg = active_modal()
    if name in ("op_fill_source", "op_click_create_import", "op_confirm_pick_02",
                "op_click_confirm_import", "op_start_prepare"):
        return dlg is not None and any(b.isEnabled() for b in _modal_btns(dlg))
    if name == "op_fill_wo":
        return dlg is not None and hasattr(dlg, "list_sources") and any(
            dlg.list_sources.item(i).data(Qt.UserRole) == DS_ID for i in range(dlg.list_sources.count()))
    if name == "op_candidate_bind":
        return CANDIDATE_DIALOG is not None and any(
            x.get("id") == FEEDBACK_ID for x in CANDIDATE_DIALOG._feedback_items)
    if name == "op_select_feedback_record":
        return any(x.get("id") == FEEDBACK_ID for x in getattr(page_feedback, "_record_items", []))
    if name == "op_feature_evidence":
        return bool(TASK_IDS.get("生成特征贡献档案"))
    return True


def main():
    """QTimer 驱动的分镜状态机（不能主线程 pump 等待：弹窗 exec() 会把
    主线程捕获进嵌套事件循环，首跑实测卡死在等确认窗）。"""
    import threading
    only = set(sys.argv[1:])
    shots = [s for s in SHOTS if not only or s[0] in only]
    state = {"i": 0}

    def step():
        global CUR_SHOT
        if ISSUES or state["i"] >= len(shots):
            MANIFEST.update(complete=not ISSUES and len(shots) == 22,
                            recording_scope="full" if len(shots) == 22 else "partial",
                            requested_shots=[s[0] for s in shots],
                            requested_complete=not ISSUES and len(MANIFEST["shots"]) == len(shots),
                            capture_method="maximized_qwidget_window_signal", issues=ISSUES,
                            datasource_id=DS_ID, workorder_id=WO_ID, feedback_id=FEEDBACK_ID,
                            tasks=TASK_IDS, datasource_name=DS_NAME, workorder_name=WO_NAME)
            with open(os.path.join(CLIPS, "manifest.json"), "w", encoding="utf-8") as stream:
                json.dump(MANIFEST, stream, ensure_ascii=False, indent=2)
            print(f"\n{'='*50}", flush=True)
            if ISSUES:
                print(f"完成，{len(ISSUES)} 个问题：", flush=True)
                for it in ISSUES:
                    print(f"  ❌ {it}", flush=True)
            else:
                print("全部分镜录制完成 ✅", flush=True)
            app.exit(1 if ISSUES else 0)
            return
        name, page, dur, ops, prep = shots[state["i"]]
        if name == "13_learning":
            print("  [录前] 等待当前真实翻案points和权重历史，不把快照当轨迹", flush=True)
        state["i"] += 1
        CUR_SHOT = name
        print(f"\n[shot] {name} → {page}（正片 {dur}s）", flush=True)

        def begin():
            if active_modal() is None:
                goto(page)
            else:
                print("  [提示] 弹窗开着，跳过切页", flush=True)
            if active_modal() is None:
                win.raise_()
                win.activateWindow()
            QTimer.singleShot(400, lambda: record(name, dur, ops))

        if prep is None:
            begin()
        elif prep[0] == "modal":
            poll_modal("数据集识别与确认", begin)
        else:  # ("task", 标签, 超时, refresh)：worker 线程轮询，完成回 UI 线程
            _tag, label, timeout, refresh = prep
            ev = threading.Event()
            outcome = [False]

            def _w():
                outcome[0] = wait_tasks_idle(label, timeout)
                ev.set()
            threading.Thread(target=_w, daemon=True).start()

            def _poll_task():
                if ev.is_set():
                    if not outcome[0]:
                        step()
                        return
                    try:
                        if refresh is not None:
                            # 让弹窗自己的 done 回调先绘制真实结果。
                            pump(1000)
                            refresh()
                    except Exception as exc:
                        ISSUES.append(f"{name} 展示前校验失败：{exc}")
                        step()
                        return
                    begin()
                else:
                    QTimer.singleShot(1000, _poll_task)
            _poll_task()

    def record(name, dur, ops):
        proc, out, lf = start_rec(name, dur)
        entry = {"name": name, "page": next(s[1] for s in SHOTS if s[0] == name), "operations": []}
        MANIFEST["shots"].append(entry)
        cursor = {"i": 0, "wait": time.monotonic()}

        def advance():
            if ISSUES:
                proc.stop_encoding.set()
                end_shot(name, proc, out, lf, max(0, proc.frame_count / 30 - TAIL))
                return
            i = cursor["i"]
            if i == len(ops):
                # 提交成功回调/缩略图绘制留出系统响应时间，不盲目进入依赖镜头。
                QTimer.singleShot(2000, finish)
                return
            fn = ops[i][1]
            if not ui_ready(fn):
                elapsed = time.monotonic() - cursor["wait"]
                if elapsed > 60:
                    ISSUES.append(f"{name} UI未就绪：{fn.__name__}")
                elif int(elapsed) % 5 == 0:
                    print(f"  [UI等待] {fn.__name__} {elapsed:.0f}s", flush=True)
                QTimer.singleShot(250, advance)
                return
            cursor["i"] += 1
            cursor["wait"] = time.monotonic()
            entry["operations"].append({"op": fn.__name__, "at": proc.frame_count / 30})
            print(f"  [序列] {i+1}/{len(ops)} {fn.__name__} @ {proc.frame_count/30:.2f}s", flush=True)
            # 在点击 exec() 前安排后续操作，模态事件循环内仍可检查真实控件就绪。
            nested = fn.__name__ in ("op_open_new_source", "op_open_new_wo", "op_open_prepare",
                                     "op_augment_space", "op_candidate_open")
            if nested:
                QTimer.singleShot(1000, advance)
            _safe_op(name, i, fn)
            if not nested:
                QTimer.singleShot(1000, advance)

        def finish():
            if name == "03b_augment" and not TASK_IDS.get("三维分析"):
                QTimer.singleShot(250, finish)
                return
            proc.stop_encoding.set()
            entry["duration"] = proc.frame_count / 30
            end_shot(name, proc, out, lf, max(0, entry["duration"] - TAIL))

        QTimer.singleShot(int((LEAD + (3 if name == "01_new_source" else 1)) * 1000), advance)

    def end_shot(name, proc, out, lf, dur):
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < 10:
            app.processEvents()
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        proc.capture_timer.stop()
        proc.capture_timer.deleteLater()
        lf.close()
        if proc.returncode != 0:
            ISSUES.append(f"{name} ffmpeg 退出码 {proc.returncode}")
        verify(name, out, dur)
        if ISSUES:
            import shutil
            import tempfile
            failed = os.path.join(tempfile.gettempdir(), f"btad_failed_{time.time_ns()}_{name}.mp4")
            shutil.move(out, failed)
            print(f"  [失败素材隔离] {failed}", flush=True)
        if name not in ("03b_augment", "12b_candidate") or ISSUES:
            close_modal()  # 三维任务跨分镜保活，结果镜头结束再关。
        dlg = getattr(page_monitor, "_feedback_dialog", None)
        if dlg is not None:
            dlg.close()
        QTimer.singleShot(400, step)

    QTimer.singleShot(1500, step)
    sys.exit(app.exec_())


main()
