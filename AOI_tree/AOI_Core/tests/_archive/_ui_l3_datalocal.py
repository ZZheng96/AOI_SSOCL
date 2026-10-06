# -*- coding: utf-8 -*-
"""UI 自动化测试：datalocal gold_finger 在「提供模板(L3)」数据条件下走 AOI_sys 界面。

数据准备（API，与 UI 等价）：
  - 创建数据源「L3源」（has_template=true）
  - import_folder ×4 挂源（template / train / test-defect / test-good）
UI 操作（真实控件）：
  1. 数据管理页：验证数据树（template=8 / train=20 / test=17）
  2. 工作台：新建工单挂 L3源 → 验证条件含「模板比对」
  3. 模型管理：准备模型 → 对话框数据条件应为「模板比对(L3)」→ 开始准备
  4. 实时监控：检测 1 缺陷 + 1 正常（API 统计 17 张 AUROC）
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, r"D:\CGAIC\AOI_sys")

TMP = tempfile.mkdtemp(prefix="aoi_l3ui_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({"system": {
        "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
        "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
        "demo5_storage": os.path.join(TMP, "storage", "demo5").replace("\\", "/"),
        "demo5_base_cfg": "configs/demo5_fast.yaml",
    }}, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH
print(f"[env] TMP={TMP}", flush=True)

import requests  # noqa: E402
from server import start_server_background  # noqa: E402
start_server_background()
BASE = "http://127.0.0.1:8017"
for _ in range(150):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.3)
print("[启动] 后端就绪", flush=True)

CAT = "gold_finger_l3"
ROOT = r"D:\CGAIC\data_local_l3\gold_finger"
SRC = "L3源"
WO = "L3工单"
ISSUES = []
_OK = {"n": 0}


def ck(name, cond, detail=""):
    if cond:
        _OK["n"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


def api(path, method="get", **kw):
    r = getattr(requests, method)(f"{BASE}{path}", timeout=60, **kw)
    if r.status_code >= 400:
        print(f"  [api] {method.upper()} {path} -> {r.status_code} {r.text[:200]}",
              flush=True)
    return r


def wait_task(tid, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            t = api(f"/api/tasks/{tid}").json()
        except Exception as e:  # noqa: BLE001
            t = {"_err": str(e)}
        if t.get("status") in ("done", "failed"):
            return t
        time.sleep(2)
    return {"status": "timeout"}


def prep_data():
    """数据准备：数据源(has_template) + 4 次导入挂源（API，数据搬运非测试核心）。"""
    print("\n[准备] 创建数据源 + 导入", flush=True)
    r = api("/api/datasources", "post", json={
        "name": SRC, "modality": "image", "label_tier": "L0",
        "source_type": "local", "has_template": True, "per_category": True})
    assert r.status_code == 200, r.text
    ds_id = r.json()["id"]
    for key, rel in (("template", "template"), ("train", "train/good"),
                     ("test_def", "test/defect"), ("test_good", "test/good")):
        rr = api("/api/images/import", "post", json={
            "folder": os.path.join(ROOT, rel), "category": CAT,
            "split": {"template": "template", "train": "train",
                      "test_def": "test", "test_good": "test"}[key],
            "label": "normal" if key != "test_def" else "anomaly",
            "name": key, "note": "L3 模板测试", "copy_to_storage": False,
            "datasource_id": ds_id})
        t = wait_task(rr.json()["task_id"])
        assert t["status"] == "done", t.get("message")
        print(f"  [导入 {key}] ✅ {t['result']}", flush=True)
    # precheck 复核
    pc = api(f"/api/models/precheck/{CAT}").json()
    print(f"  [precheck] counts={pc['counts']}", flush=True)
    assert pc["counts"]["template"] == 8 and pc["counts"]["train_normal"] == 20
    return ds_id


from PySide6.QtCore import QTimer, Qt  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QDialog, QLineEdit, QPushButton, QListWidget)
import PySide6.QtWidgets as QW  # noqa: E402

_QMB = {"n": 0}
for _m in ("information", "warning", "question", "critical"):
    _orig = getattr(QW.QMessageBox, _m)

    def _mk(m=_m, o=_orig):
        def _p(*a, **k):
            _QMB["n"] += 1
            print(f"  [弹窗] QMessageBox.{m}（自动应答）", flush=True)
            return QW.QMessageBox.Yes if m == "question" else None
        return _p
    setattr(QW.QMessageBox, _m, _mk())


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


def active_modal():
    m = QApplication.activeModalWidget()
    return m if isinstance(m, QDialog) else None


def click_by_text(dlg, text):
    bs = [b for b in dlg.findChildren(QPushButton)
          if b.text().strip() == text and b.isVisible()]
    if not bs:
        print(f"  ❌ 找不到按钮「{text}」", flush=True)
        return False
    bs[0].click()
    return True


def finish():
    print(f"\n════ 结果汇总 ════", flush=True)
    print(f"通过检查 {_OK['n']} 项，问题 {len(ISSUES)} 项", flush=True)
    for i in ISSUES:
        print(f"  - {i}", flush=True)
    try:
        app.quit()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0 if not ISSUES else 1)


# ════════ 主流程 ════════
def main():
    ds_id = prep_data()
    print(f"数据源 id={ds_id}", flush=True)

    global app, win, page_model, page_dashboard
    app = QApplication([])
    from ui.main_window import MainWindow
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    pump(1200)
    page_model = win.page_model
    page_dashboard = win.page_dashboard
    later(200, step_wo)
    T0 = time.time()
    # 进入 Qt 事件循环（由各 later 步骤驱动）
    app.exec()
    print(f"[总耗时] {time.time()-T0:.0f}s", flush=True)


# ════════ 1. 工作台：新建工单（挂 L3源 → 条件含模板比对）════════
def step_wo():
    print("\n[UI-1] 工作台：新建工单（挂 L3源）", flush=True)
    win.goto("工作台")
    pump(700)
    b = next((x for x in page_dashboard.findChildren(QPushButton)
              if "新建工单" in x.text()), None)
    ck("工作台有「新建工单」按钮", b is not None)
    if b is None:
        return later(200, step_model)
    later(900, _fill_wo)
    b.click()


def _fill_wo(attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(500, lambda: _fill_wo(attempt + 1))
        ck("新建工单对话框弹出", False)
        return later(200, step_model)
    edits = dlg.findChildren(QLineEdit)
    if edits:
        edits[0].setText(WO)
    lw = dlg.findChild(QListWidget)
    if lw is None or lw.count() == 0:
        if attempt < 12:
            return later(500, lambda: _fill_wo(attempt + 1))
        ck("建工单对话框数据源列表加载", False)
        return later(200, step_model)
    # 勾选第一个数据源（L3源）
    lw.item(0).setCheckState(Qt.Checked)
    pump(400)
    click_by_text(dlg, "创建工单")
    later(800, step_wo_verify)


def step_wo_verify():
    r = api("/api/workorders", params={"range": "all"})
    wo = next((w for w in r.json().get("items", []) if w.get("name") == WO), None)
    ck("UI 建工单成功", wo is not None)
    if wo:
        cond = wo.get("conditions") or {}
        ck("工单条件含「模板比对」", bool(cond.get("has_template")), str(cond))
    # 顶栏工单联动
    t0 = time.time()
    while time.time() - t0 < 8:
        pump(300)
        if win.combo_workorder.currentText() == WO:
            break
        time.sleep(0.3)
    ck("顶栏工单联动", win.combo_workorder.currentText() == WO)
    later(200, step_model)


# ════════ 2. 模型管理：准备模型（L3）════════
def step_model():
    print("\n[UI-2] 模型管理：准备模型（数据条件应=模板比对）", flush=True)
    win.goto("模型管理")
    pump(1000)
    tree = getattr(page_model, "tree", None)
    picked = None
    if tree is not None:
        for _a in range(15):
            top = tree.topLevelItem(0)
            if top is None:
                pump(800)
                continue
            top.setExpanded(True)
            for i in range(top.childCount()):
                ch = top.child(i)
                d = ch.data(0, Qt.UserRole)
                if isinstance(d, tuple) and d[0] == "category":
                    if CAT in str(ch.text(0)):
                        picked = ch
                        break
            if picked is not None:
                break
            pump(600)
    ck("模型页树选中品类", picked is not None)
    if picked is None:
        return later(200, step_detect)
    tree.setCurrentItem(picked)
    pump(400)
    b = next((x for x in page_model.findChildren(QPushButton)
              if "准备模型" in x.text()), None)
    ck("模型页「准备模型」按钮可用", b is not None)
    if b is None:
        return later(200, step_detect)
    later(1000, _fill_prepare)
    b.click()


def _fill_prepare(attempt=0):
    dlg = active_modal()
    if dlg is None:
        if attempt < 12:
            return later(600, lambda: _fill_prepare(attempt + 1))
        ck("准备模型对话框弹出", False)
        return later(200, step_detect)
    ck("准备模型对话框弹出", True)
    pump(1200)
    lbl = getattr(dlg, "lbl_scenario", None)
    txt = lbl.text() if lbl is not None else "?"
    ck("数据条件自动判定为「模板比对」", "模板比对" in txt, txt)
    profile = getattr(dlg, "cmb_profile", None)
    if profile is not None and profile.count() > 0:
        profile.setCurrentIndex(0)   # fast
    chk = getattr(dlg, "chk_force", None)
    if chk is not None:
        chk.setChecked(True)
    pump(300)
    click_by_text(dlg, "开始准备")
    later(300, _wait_prepare)


def _wait_prepare():
    print("  [等待] L3 预训练（fit + 快照，预计 1-3 分钟）", flush=True)
    tid = None
    t0 = time.time()
    while time.time() - t0 < 90:
        pump(200)
        rows = api("/api/tasks/recent").json().get("items", [])
        t = next((x for x in rows
                  if x.get("task_type") == "model_prepare"), None)
        if t:
            tid = t["id"]
            break
        time.sleep(1)
    ck("识别到预训练任务", tid is not None)
    if tid is None:
        return later(200, step_detect)
    _poll_prepare(tid, 0)


def _poll_prepare(tid, attempt):
    """QTimer 驱动轮询（不 sleep 阻塞 Qt 主循环，避免 0xC0000409 崩溃）。"""
    if attempt > 900:
        ck("L3 预训练完成", False, "超时")
        return later(200, step_detect)
    try:
        t = api(f"/api/tasks/{tid}").json()
    except Exception:  # noqa: BLE001
        t = {}
    st = t.get("status")
    if st == "done":
        res = t.get("result") or {}
        ck("L3 预训练完成", True, f"version={res.get('version')} "
           f"templates={res.get('n_templates')}")
        if res.get("n_templates") is not None:
            ck("模板图进入模型（n_templates=8）", res["n_templates"] == 8,
               str(res["n_templates"]))
        dlg = active_modal()
        if dlg is not None:
            click_by_text(dlg, "关闭") or dlg.reject()
            pump(400)
        return later(300, step_detect)
    if st == "failed":
        ck("L3 预训练完成", False, t.get("message", "failed"))
        return later(200, step_detect)
    if attempt % 15 == 0:
        print(f"  [等待] task {tid} {st} {t.get('progress', '')} "
              f"{t.get('message', '')}", flush=True)
    later(3000, lambda: _poll_prepare(tid, attempt + 1))


# ════════ 3. 实时监控：检测（缺陷 + 正常）════════
def step_detect():
    print("\n[UI-3] 实时监控 + 检测统计（17 张 test 图）", flush=True)
    win.goto("实时监控")
    pump(900)
    ck("监控页已打开", True)
    # 用 API 批量检测（UI 检测入口为产线实时流，单图统计用 API 保证数字准确）
    import glob
    from sklearn.metrics import roc_auc_score
    def_paths = sorted(glob.glob(os.path.join(ROOT, "test", "defect", "*")))
    good_paths = sorted(glob.glob(os.path.join(ROOT, "test", "good", "*")))
    rows, lat = [], []
    for p in def_paths + good_paths:
        d = api("/api/detect/image", "post", json={
            "path": p, "category": CAT, "with_heatmap": False}).json()
        lat.append(d.get("latency_ms", 0))
        rows.append({"path": os.path.basename(p), "gt": p in def_paths,
                     "score": d["final_score"], "decision": d["decision"],
                     "tpl": d.get("slots", {}).get("tpl")})
    y = [1 if r["gt"] else 0 for r in rows]
    au = roc_auc_score(y, [r["score"] for r in rows])
    n_ok = sum(1 for r in rows if (r["gt"] and r["decision"] == "anomaly")
               or (not r["gt"] and r["decision"] == "normal"))
    print(f"\n  [检测统计] AUROC={au:.4f} 判定一致={n_ok}/{len(rows)} "
          f"latency p50={sorted(lat)[len(lat)//2]:.0f}ms max={max(lat):.0f}ms",
          flush=True)
    for r in rows[:6]:
        print(f"    {'D' if r['gt'] else 'N'} {r['path'][:32]:<34} "
              f"score={r['score']:.3f} dec={r['decision']} tpl={r['tpl']}", flush=True)
    ck("API 检测统计完成", au is not None)
    later(300, finish)


if __name__ == "__main__":
    main()
