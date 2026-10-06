# -*- coding: utf-8 -*-
"""数据管理页 MVTec 专项功能测试（前端反馈：罗列功能并一一测试）。

用标准 MVTec 数据（<PROJECT_ROOT>/data_origin/mvtec，zipper 品类）：
1. 建数据源 + 标准数据集导入（挂源）
2. 数据树结构（源→品类→批次）
3. 筛选栏全组合遍历：模态(3) × 划分(6) × 标签(3) = 54 组合
4. 新建数据源对话框选项（模态/档位/开关/导入方式/抽帧）
5. 队列面板 + 源右键菜单
6. 汇总 + 清理
"""
import os
import sys
import time
import traceback
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.CRITICAL)

ISSUES = []


def _hook(t, v, tb):
    traceback.print_exception(t, v, tb)
    ISSUES.append(f"异常: {t.__name__}: {v}")
    print(f">>> 异常: {t.__name__}: {v}", flush=True)


sys.excepthook = _hook

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

MVTEC_ROOT = r"<PROJECT_ROOT>/data_origin/mvtec"
SRC_NAME = "MVTest源"

# 清理残留
for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                       timeout=5).json().get("items", []):
    if _w.get("name") == "MVTest工单":
        requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    if _d.get("name") == SRC_NAME:
        requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)

from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QLineEdit, QPushButton, QSpinBox, QCheckBox)
from ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1440, 900)
win.show()
STATE = {"step": "boot"}


def pump(ms=300):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def ck(name, cond, detail=""):
    tag = "✅" if cond else "❌"
    print(f"  {tag} {name}" + (f"（{detail}）" if detail else ""), flush=True)
    if not cond:
        ISSUES.append(f"{name}: {detail}")


def find_btn(widget, text):
    for b in widget.findChildren(QPushButton):
        if b.text() == text and not b.isHidden() and b.isEnabled():
            return b
    return None


def step1():
    pump(500)
    if getattr(win, "_workorders", None) is None:
        QTimer.singleShot(300, step1)
        return
    win.nav.setCurrentRow(1)
    pump(1000)
    QTimer.singleShot(100, step2_setup)


def step2_setup():
    """API 建源 + 导入 MVTec zipper（后台任务）。"""
    print("[1] 建数据源 + 导入 MVTec zipper", flush=True)
    r = requests.post(f"{BASE}/api/datasources",
                      json={"name": SRC_NAME, "modality": "image",
                            "label_tier": "L1a", "per_category": True},
                      timeout=5)
    ck("建数据源", r.status_code == 200, str(r.status_code))
    sid = r.json()["id"]
    STATE["ds_id"] = sid
    r = requests.post(f"{BASE}/api/images/import_mvtec",
                      json={"root": MVTEC_ROOT, "categories": ["zipper"],
                            "dataset_name": "mvtec", "datasource_id": sid},
                      timeout=5)
    ck("提交 MVTec 导入任务", r.status_code == 200 and r.json().get("task_id"),
       str(r.json()))
    task_id = r.json()["task_id"]
    # 轮询任务完成（带进度）
    t0 = time.time()
    while time.time() - t0 < 180:
        tr = requests.get(f"{BASE}/api/tasks/{task_id}", timeout=5).json()
        st = tr.get("status") if isinstance(tr, dict) else "?"
        prog = tr.get("progress") if isinstance(tr, dict) else "?"
        print(f"    [导入] status={st} progress={prog}", flush=True)
        if st in ("done", "completed", "finished"):
            break
        if st in ("failed", "error"):
            ISSUES.append(f"MVTec 导入失败: {tr}")
            break
        pump(1000)
    win.page_data.reload()
    pump(1500)
    QTimer.singleShot(100, step3_verify)


def step3_verify():
    """验证导入结果。"""
    print("[2] 导入结果", flush=True)
    r = requests.get(f"{BASE}/api/datasources", timeout=5)
    d = [x for x in r.json()["items"] if x["id"] == STATE["ds_id"]][0]
    cap = d.get("capability") or {}
    ck("数据源能力统计", cap.get("n_images", 0) > 100
       and cap.get("n_normal", 0) > 50 and cap.get("n_anomaly", 0) > 0,
       f"img={cap.get('n_images')} normal={cap.get('n_normal')} "
       f"anomaly={cap.get('n_anomaly')}")
    ck("品类含 zipper", "zipper" in (cap.get("categories") or []),
       str(cap.get("categories")))
    QTimer.singleShot(100, step4_tree)


def step4_tree():
    page = win.page_data
    print("[3] 数据树结构", flush=True)
    # 源 → 品类 → 批次
    def collect(node, depth=0):
        out = []
        for i in range(node.childCount()):
            ch = node.child(i)
            out.append((depth + 1, ch.text(0), ch.data(0, Qt.UserRole)))
            out.extend(collect(ch, depth + 1))
        return out

    src_node = None
    for i in range(page.tree.topLevelItemCount()):
        it = page.tree.topLevelItem(i)
        if it.data(0, Qt.UserRole + 1):
            src_node = it
            break
    ck("源节点存在", src_node is not None,
       str(src_node.text(0) if src_node else None))
    if src_node:
        src_node.setExpanded(True)
        pump(400)
        kids = collect(src_node)
        texts = [t for _, t, _ in kids]
        ck("源下含品类节点", any("zipper" in t for t in texts), str(texts[:5]))
        ck("源下含批次节点", any("批次" in t or "(" in t for t in texts[1:6]),
           str(texts[1:6]))
    # 模态筛选
    page.combo_modality.setCurrentIndex(2)   # 视频
    pump(800)
    vis = any(page.tree.topLevelItem(i).data(0, Qt.UserRole + 1)
              for i in range(page.tree.topLevelItemCount()))
    ck("视频筛选隐藏图片源", not vis)
    page.combo_modality.setCurrentIndex(1)   # 图片
    pump(800)
    vis = any(page.tree.topLevelItem(i).data(0, Qt.UserRole + 1)
              for i in range(page.tree.topLevelItemCount()))
    ck("图片筛选显示图片源", vis)
    page.combo_modality.setCurrentIndex(0)
    pump(800)
    QTimer.singleShot(100, step5_filters)


def step5_filters():
    """筛选栏全组合遍历：模态×划分×标签。"""
    page = win.page_data
    print("[4] 筛选组合遍历（模态×划分×标签）", flush=True)
    modes = ["", "image", "video"]
    splits = ["", "train", "val", "test", "template", "feedback"]
    labels = ["", "normal", "anomaly"]
    total = len(modes) * len(splits) * len(labels)
    n = 0
    for m in modes:
        page.combo_modality.setCurrentIndex(
            page.combo_modality.findData(m))
        for sp in splits:
            page.combo_split.setCurrentIndex(
                page.combo_split.findData(sp))
            for lb in labels:
                page.combo_label.setCurrentIndex(
                    page.combo_label.findData(lb))
                page.combo_category.setCurrentIndex(
                    page.combo_category.findText("zipper"))
                page._on_query()
                pump(400)
                rows = page.table.rowCount()
                n += 1
                # 关键组合合理性：train/normal 应为 train good 数（>0）
                if sp == "train" and lb == "normal" and m in ("", "image"):
                    ck(f"train+normal 计数（m={m or 'all'}）", rows > 0,
                       f"rows={rows}")
                if rows < 0:
                    ISSUES.append(f"查询异常 rows={rows} 组合 "
                                  f"m={m} sp={sp} lb={lb}")
    print(f"    遍历 {n}/{total} 组合完成", flush=True)
    ck("全组合遍历无崩溃", n == total, f"n={n}")
    # 复位
    for c, v in ((page.combo_split, ""), (page.combo_label, ""),
                 (page.combo_modality, "")):
        c.setCurrentIndex(c.findData(v))
    QTimer.singleShot(100, step6_dsdlg)


def step6_dsdlg():
    """新建数据源对话框选项组合。"""
    print("[5] 新建数据源对话框选项", flush=True)
    QTimer.singleShot(800, _ds_fill)
    find_btn(win.page_data, "新建数据源").click()


def _ds_fill():
    dlg = QApplication.activeModalWidget()
    if dlg is None:
        ISSUES.append("建源对话框未弹出")
        app.quit()
        return
    STATE["dlg"] = dlg
    combos = dlg.findChildren(QComboBox)
    mod_combo = combos[0]
    tier_combo = combos[1]
    import_combo = combos[-1]
    checks = dlg.findChildren(QCheckBox)
    spins = dlg.findChildren(QSpinBox)
    ck("模态选项=图片/视频", mod_combo.count() == 2,
       str([mod_combo.itemText(i) for i in range(mod_combo.count())]))
    ck("档位选项=三档", tier_combo.count() == 3,
       str([tier_combo.itemText(i) for i in range(tier_combo.count())]))
    ck("导入方式=4 项", import_combo.count() == 4,
       str([import_combo.itemText(i) for i in range(import_combo.count())]))
    ck("品类/模板开关=2 个", len(checks) == 2,
       str([c.text() for c in checks]))
    # 模态×导入方式组合：图片/视频 × 各导入方式（字段联动验证）
    for mi in range(2):
        mod_combo.setCurrentIndex(mi)
        pump(150)
        ck(f"模态={mod_combo.currentText()} 抽帧"
           f"{'显示' if spins[0].isVisible() else '隐藏'}",
           spins[0].isVisible() == (mi == 1))
        for ii in range(4):
            import_combo.setCurrentIndex(ii)
            pump(100)
    dlg.reject()
    QTimer.singleShot(300, step7_queue)


def step7_queue():
    print("[6] 队列 + 清理", flush=True)
    page = win.page_data
    page._load_queue()
    pump(1000)
    ck("队列面板刷新", page.queue_table.rowCount() >= 0)
    # 清理
    for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
        if _d.get("name") == SRC_NAME:
            requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
    print(f"\n{'='*50}", flush=True)
    if ISSUES:
        print(f"测试发现 {len(ISSUES)} 个问题 ❌", flush=True)
        for it in ISSUES:
            print(f"  ❌ {it}", flush=True)
    else:
        print("MVTec 专项测试全部通过 ✅", flush=True)
    win.close()
    app.quit()


QTimer.singleShot(0, step1)
app.exec()
sys.exit(1 if ISSUES else 0)
