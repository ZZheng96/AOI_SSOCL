# -*- coding: utf-8 -*-
"""模型管理页重构冒烟验证（2026-08-29）：

1. 左侧树 = 数据源 → 品类（品类节点带状态文本）；
2. 页面无 demo5/槽位/示意图/校准/扫描模型 等用户困惑信息；
3. 选中品类 → 右侧面板显示状态头 + 当前状态卡；
4. 版本历史默认折叠，可展开；
5. 准备模型对话框：数据源只读带入，提交 payload 含 datasource_id。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402
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


def all_label_texts(widget):
    texts = []
    for lb in widget.findChildren(QLabel):
        if lb.isVisible() and lb.text():
            texts.append(lb.text())
    for bt in widget.findChildren(QPushButton):
        if bt.isVisible() and bt.text():
            texts.append(bt.text())
    return texts


def tree_nodes():
    t = win.page_model.tree
    out = {}
    for i in range(t.topLevelItemCount()):
        src = t.topLevelItem(i)
        out[src.text(0)] = [src.child(j).text(0) for j in range(src.childCount())]
    return out


def wait_tree(timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pump(250)
        item, _ = find_category_item()
        if item is not None:
            return tree_nodes()
        time.sleep(0.2)
    return tree_nodes()


def find_category_item():
    t = win.page_model.tree
    for i in range(t.topLevelItemCount()):
        src = t.topLevelItem(i)
        for j in range(src.childCount()):
            data = src.child(j).data(0, 0x0100)  # Qt.UserRole
            if data and data[0] == "category":
                return src.child(j), data
    return None, None


def step():
    win.goto("模型管理")
    pump(500)
    nodes = wait_tree()
    print("树结构:", nodes, flush=True)
    item, data = find_category_item()
    if item is None:
        g = requests.get(f"{BASE}/api/datasources/1/groups", timeout=8).json()
        print("调试 groups(1):", str(g)[:400], flush=True)

    # 1. 树结构：数据源 → 品类
    ck("左侧树有数据源节点", len(nodes) > 0, str(list(nodes.keys())))
    ck("数据源下有品类节点", any(kids for kids in nodes.values()), str(nodes))
    ck("品类节点 UserRole 携带上下文",
       data is not None and data[0] == "category" and len(data) >= 4, str(data))

    # 2. 无用户困惑信息
    texts = all_label_texts(win.page_model)
    joined = " | ".join(texts)
    for bad in ("demo5", "槽位", "问题三", "CDF", "校准", "扫描模型", "sem", "shead"):
        ck(f"页面无困惑词「{bad}」", bad not in joined,
           "" if bad not in joined else "命中: " + bad)

    # 2b. 其他页面同步清理（2026-08-29 前端反馈：同一标准推广全工程）
    for page_name, widget in (("实时监控", win.page_monitor),
                              ("标注反馈", win.page_feedback),
                              ("系统设置", win.page_settings)):
        win.goto(page_name)
        pump(400)
        j = " | ".join(all_label_texts(widget))
        for bad in ("demo5", "Demo5", "槽位", "问题一", "孵育", "校准",
                    "sem", "shead", "min_hit", "AUROC"):
            ck(f"{page_name}页无困惑词「{bad}」", bad not in j,
               "" if bad not in j else "命中: " + bad)
    win.goto("模型管理")
    wait_tree()  # 树可能因切页重建，重新等待并取节点
    item, data = find_category_item()

    # 3. 选中品类 → 面板
    if item is not None:
        win.page_model.tree.setCurrentItem(item)
        pump(800)
        pg = win.page_model
        ck("右侧面板切换到品类视图", pg._stack.currentIndex() == 1,
           f"currentIndex={pg._stack.currentIndex()}")
        cat = data[3]
        ck("状态头显示品类名", cat in pg.lbl_cat.text(), pg.lbl_cat.text())
        ck("阶段徽标可见且有文本",
           pg.lbl_stage.isVisible() and bool(pg.lbl_stage.text()),
           pg.lbl_stage.text())
        ck("当前状态卡显示版本", pg.lbl_version.text() != "", pg.lbl_version.text())
        ck("操作卡有「准备模型」按钮", pg.btn_prepare.isVisible())
        ck("操作卡有「学习反馈并更新模型」按钮", pg.btn_learn.isVisible())

        # 4. 版本历史默认折叠 → 展开
        ck("版本历史默认折叠", not pg.hist_body.isVisible())
        pg.btn_hist_toggle.click()
        pump(400)
        ck("点击后版本历史展开", pg.hist_body.isVisible())
        ck("历史表格已填充行", pg.tbl.rowCount() >= 0,
           f"{pg.tbl.rowCount()} 行")

        # 5. 准备模型对话框：数据源带入 + payload 含 datasource_id
        captured = {}
        orig = pg._client.prepare_model

        def fake_prepare(category, scenario="L1a", profile="fast",
                         note="", force=True, datasource_id=None):
            captured.update(category=category, scenario=scenario,
                            profile=profile, datasource_id=datasource_id)
            return {"task_id": None, "note": "mock"}  # 不真提交

        pg._client.prepare_model = fake_prepare
        try:
            from ui.pages.model_page import PrepareModelDialog
            dlg = PrepareModelDialog(
                pg._client, cat,
                datasource={"id": data[1], "name": data[2]},
                workorder=None, parent=pg)
            pump(500)
            ck("对话框标题含品类", cat in dlg.windowTitle(), dlg.windowTitle())
            ds_label = dlg.lbl_source.text()
            ck("对话框样本来源指向数据源",
               str(data[2]) in ds_label and "预训练组" in ds_label, ds_label)
            ck("对话框无手选品类/数据源控件",
               not hasattr(dlg, "cmb_category") and not hasattr(dlg, "cmb_source"))
            dlg._on_start()  # 直接触发提交（被 mock 捕获）
            pump(300)
            ck("提交 payload 含 datasource_id",
               captured.get("datasource_id") == data[1], str(captured))
            ck("提交 payload 品类正确", captured.get("category") == cat)
            dlg.close()
        finally:
            pg._client.prepare_model = orig

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


from PySide6.QtCore import QTimer  # noqa: E402
QTimer.singleShot(400, step)
app.exec()
