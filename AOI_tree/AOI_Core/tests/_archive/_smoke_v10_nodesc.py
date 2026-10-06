# -*- coding: utf-8 -*-
"""v10 冒烟：无「未挂源」概念。

验证：
1. 删除数据源级联删除其下批次/图片（不再归为未挂源）；
2. 导入对话框数据源下拉无「不挂数据源」选项；
3. 数据树顶层只有「全部图像」→ 数据源（无「未挂源批次」节点）。
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

from backend.db.database import session_scope  # noqa: E402
from backend.db.models import Dataset, Image as ImageRow  # noqa: E402

BASE = "http://127.0.0.1:8017"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/api/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:
        pass
    time.sleep(0.3)

ISSUES = []
_OK = {"n": 0}


def ck(name, cond, detail=""):
    if cond:
        _OK["n"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


# ── 准备：清理同名残留 → 建数据源 + 直接写 1 个批次、3 张图 ──
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    if d.get("name") in ("v10删除级联", "v10分组"):
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)

r = requests.post(f"{BASE}/api/datasources",
                  json={"name": "v10删除级联", "modality": "image",
                        "label_tier": "L1a", "pretrain_normal": 2,
                        "pretrain_anomaly": 1, "batch_size": 2}, timeout=8)
sid = r.json()["id"]
ck("建数据源", r.status_code == 200 and sid, f"id={sid}")

import tempfile
_tmp = tempfile.mkdtemp()
paths = [os.path.join(_tmp, f"img{i}.png") for i in range(3)]
for p in paths:
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + os.urandom(64))

with session_scope() as s:
    ds = Dataset(name="v10批次", category="bottle", source_type="folder",
                 source_path=_tmp, datasource_id=sid)
    s.add(ds)
    s.flush()
    for p in paths:
        s.add(ImageRow(path=p, category="bottle", split="train",
                       label="normal" if p.endswith("0.png") else "anomaly",
                       dataset_id=ds.id))
    s.commit()
    ds_id = ds.id

# 1. 删除数据源 → 批次/图片级联删除
r = requests.delete(f"{BASE}/api/datasources/{sid}", timeout=8)
b = r.json()
ck("删除数据源级联返回", r.status_code == 200 and b.get("deleted"),
   f"datasets={b.get('datasets')} images={b.get('images')}")
with session_scope() as s:
    n_ds = s.query(Dataset).filter(Dataset.id == ds_id).count()
    n_img = s.query(ImageRow).filter(ImageRow.path.in_(paths)).count()
ck("批次已级联删除", n_ds == 0, f"datasets={n_ds}")
ck("图片已级联删除", n_img == 0, f"images={n_img}")

# 2. 再建源 → 验证分组树（预训练/检测组），再删除
r = requests.post(f"{BASE}/api/datasources",
                  json={"name": "v10分组", "modality": "image",
                        "pretrain_normal": 1, "pretrain_anomaly": 1,
                        "batch_size": 2}, timeout=8)
sid2 = r.json()["id"]
r = requests.get(f"{BASE}/api/datasources/{sid2}/groups", timeout=8)
ck("分组接口 200", r.status_code == 200)
requests.delete(f"{BASE}/api/datasources/{sid2}", timeout=8)

# ── 3. 前端：树构建无「未挂源」节点；下拉无「不挂数据源」 ──
from PySide6.QtWidgets import QApplication  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
win = MainWindow()
win.resize(1280, 800)
win.show()


def pump(ms=300):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def step2():
    win.goto("数据管理")
    pump(2500)
    # 树顶层：全部图像 → 数据源（无未挂源）
    t = win.page_data.tree
    texts = []
    for i in range(t.topLevelItemCount()):
        root = t.topLevelItem(i)
        texts.append(root.text(0))
        for j in range(root.childCount()):
            texts.append(f"  {root.child(j).text(0)}")
    ck("树顶层为「全部图像」", texts and "全部图像" in texts[0], texts[0] if texts else "-")
    ck("树无「未挂源」节点", not any("未挂源" in x for x in texts),
       "; ".join(texts[:8]))
    # 导入对话框下拉：无「不挂数据源」
    combo = win.page_data._on_import_data.__globals__  # 仅验证函数存在
    ck("_on_import_data 存在", callable(win.page_data._on_import_data))
    # 数据源下拉填充（空库 → 占位提示）
    import contextlib
    from PySide6.QtWidgets import QComboBox, QDialog  # noqa: E402
    cb = QComboBox()
    win.page_data._fill_source_combo(cb, "")
    items = [cb.itemText(i) for i in range(cb.count())]
    ck("下拉无「不挂数据源」", not any("不挂数据源" in x for x in items), "; ".join(items))
    ck("下拉有占位提示或数据源", bool(items), "; ".join(items))
    print(f"\n{'='*46}", flush=True)
    print(f"验证全部通过 ✅（{_OK['n']}）" if not ISSUES
          else f"验证发现 {len(ISSUES)} 个问题 ❌（通过 {_OK['n']}）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
    import os as _os
    _os._exit(1 if ISSUES else 0)


from PySide6.QtCore import QTimer  # noqa: E402
QTimer.singleShot(400, step2)
app.exec()
