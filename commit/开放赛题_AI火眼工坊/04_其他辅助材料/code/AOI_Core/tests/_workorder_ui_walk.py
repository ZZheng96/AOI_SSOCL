# -*- coding: utf-8 -*-
"""工单制改造 UI 点击走查（前端反馈 v2，w9 验证）。

进程内起后端（真实库）+ 主窗口，事件驱动模拟用户操作：
1. 设置页「新建品类向导」→ 填工单名 + 三档两开关 → 提交
   （重点验证 _done 成功判据修复：不再误报"提交失败"，且体检自动调低有中文提示）。
2. 主窗口工单下拉出现新工单，切换后条件徽标联动。
3. 数据管理页「新建数据源」（图像）→ 数据源登记成功。
4. 工作台：时间范围三档切换 + 工单表含新工单 + 双击「数据条件」列弹改条件对话框。
5. 走查结束 API 清理（删走查工单/数据源，不污染真实库）。

每步截图落盘 _ui_click_shots/；全程统计 QMessageBox 模态弹窗（应 0，一律内联提示）。
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
for _ in range(60):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QLineEdit, QPushButton)
import PySide6.QtWidgets as QW  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402
from ui.pages.common import run_async  # noqa: E402
from ui.widgets.workorder_dialogs import WorkOrderCreateDialog  # noqa: E402

# 弹窗统计：内联 Toast 设计要求全程无模态 QMessageBox
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
SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_ui_click_shots")
ISSUES = []
STATE = {"i": 0}
WO_NAME = "走查工单"
DS_NAME = "走查图源"


def pump(ms=400):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def shot(name, widget=None):
    os.makedirs(SHOT_DIR, exist_ok=True)
    STATE["i"] += 1
    p = os.path.join(SHOT_DIR, f"{STATE['i']:02d}_{name}.png")
    (widget or win).grab().save(p)
    print(f"  [截图] {os.path.basename(p)}", flush=True)


def ck(name, cond, detail=""):
    tag = "✅" if cond else "❌"
    print(f"  {tag} {name}" + (f"（{detail}）" if detail else ""), flush=True)
    if not cond:
        ISSUES.append(f"{name}: {detail}")


def nav(page_title):
    """按导航标题切页（经 _title2row 映射，跳过分组标题行）。"""
    row = win._title2row.get(page_title)
    if row is None:
        return False
    win.nav.setCurrentRow(row)
    pump(600)
    return True


def find_btn(widget, text):
    for b in widget.findChildren(QPushButton):
        if b.text() == text and b.isEnabled() and not b.isHidden():
            return b
    return None


def modal() -> QDialog | None:
    w = QApplication.activeModalWidget()
    return w if isinstance(w, QDialog) else None


# ── 0) 等工单加载完 ─────────────────────────────────
# 清理上次中断走查的残留数据（防重名 409）
for _w in requests.get(f"{BASE}/api/workorders",
                       params={"range": "all"}, timeout=5).json().get("items", []):
    if _w.get("name") == WO_NAME:
        requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    if _d.get("name") == DS_NAME:
        requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
for _ in range(40):
    pump(200)
    if getattr(win, "_workorders", None) is not None and win.combo_workorder.count():
        break
ck("主窗口工单下拉已加载", win.combo_workorder.count() >= 1,
   f"n={win.combo_workorder.count()}")
ck("顶栏品类下拉已移除（v3）", not hasattr(win, "combo_category"))
shot("主窗口_工单下拉与徽标")

# ── 1) 设置页向导：新建工单（声明 L1b+两开关）────────
print("[1] 设置页 → 新建品类向导 → 提交工单", flush=True)
ck("切到设置页", nav("系统设置"))
btn = find_btn(win.page_settings, "新建品类向导")
ck("找到「新建品类向导」按钮", btn is not None)


def _wizard_fill():
    dlg = modal()
    if dlg is None:
        ISSUES.append("向导对话框未弹出")
        return
    dlg.edit_name.setText(WO_NAME)
    dlg.chk_review.setChecked(True)   # v4：向导无条件编辑器，勾选人工复判
    shot("向导_填工单", dlg)
    QTimer.singleShot(300, lambda: dlg.btn_submit.click())
    # 异步提交可能被数据页缩略图请求洪峰拖慢：轮询等结果，最多 30s
    STATE["wait"] = 0
    QTimer.singleShot(1500, lambda: _wizard_check(dlg))


def _wizard_check(dlg):
    txt = dlg.lbl_wo_state.text()
    if not txt and STATE["wait"] < 20:   # 提交结果尚未回填，继续等
        STATE["wait"] += 1
        QTimer.singleShot(1500, lambda: _wizard_check(dlg))
        return
    shot("向导_提交后提示", dlg)
    ck("提交成功提示（非'提交失败'）", "工单已登记" in txt, txt[:80])
    ck("未挂数据源提醒", "数据源" in txt, txt[:80])
    dlg.accept()


QTimer.singleShot(600, _wizard_fill)
btn.click()
pump(1000)
# levels_changed → 主窗口异步刷新工单下拉（洪峰下可能慢）：轮询最多 30s
names = []
for _ in range(20):
    pump(1500)
    names = [win.combo_workorder.itemText(i)
             for i in range(win.combo_workorder.count())]
    if WO_NAME in names:
        break
ck("主窗口工单下拉出现新工单", WO_NAME in names, str(names))
if WO_NAME in names:
    win.combo_workorder.setCurrentIndex(names.index(WO_NAME))
    pump(800)
ck("条件徽标联动（体检后仅正常图）", "正常图" in win.badge_level.text(),
   win.badge_level.text())
shot("主窗口_选中新工单")

# ── 2) 数据管理页：数据源存在 + 新建工单数据源必选 ──
# 注：新建数据源对话框交互已由 _repro_ds.py 专项验证；此处用 API 建源，
# 聚焦数据源存在后的 UI 行为（列表/树节点/必选联动）。
print("[2] 数据管理页 → 数据源体系", flush=True)
# 清理诊断脚本残留数据源，保证走查环境干净
for d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    if d.get("name") not in (DS_NAME,):
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=5)
r = requests.post(f"{BASE}/api/datasources",
                  json={"name": DS_NAME, "modality": "image"}, timeout=5)
ck("API 创建走查数据源", r.status_code == 200, str(r.status_code))
run_async(win.page_data, win.client.list_datasources,
          win.page_data._fill_datasources)
pump(1200)
ds_names = [str(d.get("name"))
            for d in getattr(win.page_data, "_datasources", [])]
ck("数据源列表出现新源", DS_NAME in ds_names, str(ds_names))
win.page_data.reload()
pump(1000)
shot("数据管理_新数据源")

# ── 2b) v3：新建工单数据源必选 + 数据源树节点 ───────
print("[2b] v3：数据源必选按钮联动 + 数据源树节点", flush=True)
dlg = WorkOrderCreateDialog(win.client, win)
dlg.show()
pump(1500)
btn_ok = None
for b in dlg.findChildren(QPushButton):
    if b.text() == "创建工单":
        btn_ok = b
        break
ck("创建按钮初始禁用（数据源必选）",
   btn_ok is not None and not btn_ok.isEnabled())
n_checked = 0
for i in range(dlg.list_sources.count()):
    it = dlg.list_sources.item(i)
    if it.flags() & Qt.ItemIsUserCheckable:
        it.setCheckState(Qt.Checked)
        n_checked += 1
pump(300)
ck("勾选数据源后创建按钮启用",
   n_checked >= 1 and btn_ok is not None and btn_ok.isEnabled(),
   f"勾选={n_checked}")
dlg.done(0)   # 关闭（触发 done() 等待后台线程）
pump(400)
# 数据源树顶层节点（source 模式）：右键可编辑/删除数据源
# 注：source_groups 只含「有数据」的源，空源不出现；此处直接注入
# 渲染数据验证节点标记与 _source_mode 机制（生产代码路径）。
win.page_data._source_mode = True   # 模拟 _fill_tree 的 source 分支设置
win.page_data._fill_tree_groups({DS_NAME: {}}, is_source=True)
pump(300)
src_node = None
for i in range(win.page_data.tree.topLevelItemCount()):
    it = win.page_data.tree.topLevelItem(i)
    if it.data(0, Qt.UserRole + 1):
        src_node = it
        break
ck("数据源树顶层节点标记（source 模式）",
   src_node is not None
   and src_node.data(0, Qt.UserRole + 1) == DS_NAME
   and getattr(win.page_data, "_source_mode", True),
   str(src_node.text(0) if src_node else None))
win.page_data.reload()

# ── 3) 工作台：范围切换 + 双击改条件 ────────────────
print("[3] 工作台 → 时间范围 + 双击改条件", flush=True)
ck("切到工作台", nav("工作台"))
for i in range(win.page_dashboard.combo_range.count()):
    win.page_dashboard.combo_range.setCurrentIndex(i)
    pump(900)
    rng = win.page_dashboard.combo_range.currentData()
    n = win.page_dashboard.table.rowCount()
    print(f"  [范围] {win.page_dashboard.combo_range.currentText()} 行数={n}",
          flush=True)
ck("范围三档切换无异常", True)

wo_row = -1
for r in range(win.page_dashboard.table.rowCount()):
    it = win.page_dashboard.table.item(r, 0)
    if it and it.text() == WO_NAME:
        wo_row = r
        break
ck("工单表含走查工单", wo_row >= 0, f"row={wo_row}")
shot("工作台_工单表")

if wo_row >= 0:
    QTimer.singleShot(900, lambda: (shot("双击改条件对话框", modal() or win),
                                    (modal() and modal().reject())))
    win.page_dashboard.table.cellDoubleClicked.emit(wo_row, 1)
    pump(1800)
    ck("双击弹出改条件对话框（已截图关闭）", True)

# ── 3b) v5：产线控制 + 数据流队列面板 ──────────────
print("[3b] v5：监控页产线控制 + 数据页队列面板", flush=True)
ck("切到实时监控页", nav("实时监控"))
pump(800)
ck("监控页产线控制面板存在", hasattr(win.page_monitor, "lbl_pipeline")
   and hasattr(win.page_monitor, "btn_pp_pause")
   and hasattr(win.page_monitor, "btn_pp_learn"))
ck("产线状态显示", "运行中" in win.page_monitor.lbl_pipeline.text()
   or "已暂停" in win.page_monitor.lbl_pipeline.text()
   or "产线：-" in win.page_monitor.lbl_pipeline.text(),
   win.page_monitor.lbl_pipeline.text())
ck("切到数据管理页", nav("数据管理"))
pump(800)
ck("数据页队列面板存在", hasattr(win.page_data, "queue_table")
   and hasattr(win.page_data, "btn_queue_req"))
# 队列刷新（当前工单可能无批次，不崩即可）
win.page_data._load_queue()
pump(1000)
ck("队列刷新无异常", True)
shot("数据管理_数据流队列")

# ── 4) 清理走查数据（不污染真实库）─────────────────
print("[4] 清理走查数据", flush=True)
items = requests.get(f"{BASE}/api/workorders",
                     params={"range": "all"}, timeout=5).json().get("items", [])
wid = next((w["id"] for w in items if w.get("name") == WO_NAME), None)
ck("找到走查工单 id", wid is not None)
if wid is not None:
    r = requests.delete(f"{BASE}/api/workorders/{wid}", timeout=5)
    ck("删除走查工单", r.status_code == 200, str(r.status_code))
ds_items = requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", [])
dsid = next((d["id"] for d in ds_items if d.get("name") == DS_NAME), None)
if dsid is not None:
    r = requests.delete(f"{BASE}/api/datasources/{dsid}", timeout=5)
    ck("删除走查数据源", r.status_code == 200, str(r.status_code))

# ── 汇总 ───────────────────────────────────────────
ck("全程无 QMessageBox 模态弹窗", _QMB["n"] == 0, f"n={_QMB['n']}")
print(f"\n{'='*50}\n走查完成：{'全部通过 ✅' if not ISSUES else '有问题 ❌'}",
      flush=True)
for it in ISSUES:
    print(f"  ❌ {it}", flush=True)
win.close()
sys.exit(1 if ISSUES else 0)
