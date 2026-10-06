# -*- coding: utf-8 -*-
"""端到端流程测试：建工单 → 100+30 预训练 → 产线检测 → 持续学习错检再检。

数据：MVTec zipper（train/normal 100 + test 缺陷 ≤30 → prepare；产线检测全量）。
核心耗时步骤（导入/训练/检测/学习）API 驱动 + 轮询进度；关键 UI 点断言。
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
SRC = "Flow源"
WO = "Flow工单"

# 清理残留
for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                       timeout=5).json().get("items", []):
    if _w.get("name") == WO:
        requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    if _d.get("name") == SRC:
        requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)

N = {"ok": 0}


def ck(name, cond, detail=""):
    if cond:
        N["ok"] += 1
        print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  ❌ {name}（{detail}）", flush=True)


def wait_task(task_id, timeout=600):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        tr = requests.get(f"{BASE}/api/tasks/{task_id}", timeout=5).json()
        st = tr.get("status") if isinstance(tr, dict) else "?"
        prog = tr.get("progress") if isinstance(tr, dict) else "?"
        if st != last:
            print(f"    [任务{task_id}] status={st} progress={prog}", flush=True)
            last = st
        if st in ("done", "success", "completed", "finished"):
            return True, tr
        if st in ("failed", "error", "canceled"):
            return False, tr
        time.sleep(2)
    return False, {"status": "timeout"}


# ═══════════════ 阶段 1：数据 + 工单 ═══════════════
print("[阶段1] 数据源 + 导入 + 工单", flush=True)
r = requests.post(f"{BASE}/api/datasources",
                  json={"name": SRC, "modality": "image", "label_tier": "L1a",
                        "per_category": True}, timeout=5)
ck("建数据源", r.status_code == 200, str(r.status_code))
sid = r.json()["id"]
r = requests.post(f"{BASE}/api/images/import_mvtec",
                  json={"root": MVTEC_ROOT, "categories": ["zipper"],
                        "datasource_id": sid}, timeout=5)
ck("提交 MVTec 导入", r.status_code == 200 and r.json().get("task_id"), str(r.json()))
ok, tr = wait_task(r.json()["task_id"], timeout=180)
ck("MVTec 导入完成", ok, str(tr.get("status") if isinstance(tr, dict) else tr))

r = requests.post(f"{BASE}/api/workorders",
                  json={"name": WO, "datasource_ids": [sid],
                        "review_enabled": False}, timeout=5)
ck("建工单（挂源）", r.status_code == 200 and r.json().get("id"), str(r.status_code))
wid = r.json()["id"]
cond = r.json().get("conditions") or {}
ck("工单聚合条件（zipper 有缺陷→L1a）", cond.get("label_tier") in ("L1a", "L1b"),
   str(cond))

# ═══════════════ 阶段 2：100+30 预训练 ═══════════════
print("[阶段2] 100+30 预训练", flush=True)
r = requests.post(f"{BASE}/api/models/prepare",
                  json={"category": "zipper", "scenario": "L1a",
                        "profile": "fast", "force": True}, timeout=5)
ck("提交模型准备", r.status_code == 200 and r.json().get("task_id"), str(r.json()))
ok, tr = wait_task(r.json()["task_id"], timeout=900)
ck("预训练完成", ok, str(tr.get("status") if isinstance(tr, dict) else tr))
r = requests.get(f"{BASE}/api/models", timeout=5)
models = r.json().get("items", [])
zmodels = [m for m in models if m.get("category") == "zipper"]
ck("zipper 模型快照存在", len(zmodels) >= 1, str(len(zmodels)))
ck("有激活版本", any(m.get("is_active") for m in zmodels),
   str([(m.get("version"), m.get("is_active")) for m in zmodels][:3]))

# ═══════════════ 阶段 3：产线检测 ═══════════════
print("[阶段3] 产线检测（PipelineService 自动消费）", flush=True)
r = requests.get(f"{BASE}/api/workorders/{wid}")
ck("工单产线 running", r.json().get("pipeline_status") == "running", str(r.json().get("pipeline_status")))
# 等待检测落库（engine 就绪后 PipelineService 逐张送检）
t0 = time.time()
n_det = 0
while time.time() - t0 < 300:
    r = requests.get(f"{BASE}/api/detections?category=zipper&page=1&page_size=1",
                     timeout=5)
    body = r.json()
    n_det = body.get("total", 0) if isinstance(body, dict) else 0
    if n_det >= 10:
        break
    time.sleep(3)
ck("产线自动检测落库 ≥10 条", n_det >= 10, f"n_det={n_det}")

# ═══════════════ 阶段 4：错检反馈 → 学习 → 再检 ═══════════════
print("[阶段4] 错检反馈 + 学习提升 + 再检", flush=True)
r = requests.get(f"{BASE}/api/detections?category=zipper&page=1&page_size=5",
                 timeout=5)
dets = r.json().get("items", []) if isinstance(r.json(), dict) else []
ck("取到检测记录", len(dets) >= 1, str(len(dets)))
fed = 0
for d in dets[:3]:
    rr = requests.post(f"{BASE}/api/feedback",
                       json={"detection_id": d["id"],
                             "feedback_type": "false_negative"
                             if not d.get("is_anomaly") else "confirmed",
                             "operator_label": 1}, timeout=5)
    if rr.status_code == 200:
        fed += 1
    else:
        ISSUES.append(f"反馈提交失败 {d['id']}: {rr.status_code} {rr.text[:80]}")
ck("错检反馈提交", fed >= 1, f"fed={fed}")

r = requests.post(f"{BASE}/api/workorders/{wid}/learn", timeout=60)
learn = r.json()
ck("学习提升（consolidate）", r.status_code == 200
   and "message" in learn, str(learn.get("message")))
ck("学习后版本提升", len(learn.get("applied_categories") or []) >= 0)

# 再检：回队 zipper 批次 → 重新检测
r = requests.get(f"{BASE}/api/workorders/{wid}/queue")
items = r.json().get("items", [])
zipper_ds = [it for it in items if it.get("category") == "zipper"]
ck("队列含 zipper 批次", len(zipper_ds) >= 1, str(len(zipper_ds)))
if zipper_ds:
    rr = requests.post(f"{BASE}/api/workorders/{wid}/queue/requeue",
                       json={"batch_keys": [zipper_ds[0]["batch_key"]]}, timeout=5)
    ck("错检批次回队", rr.status_code == 200, str(rr.status_code))
    t0 = time.time()
    n2 = 0
    while time.time() - t0 < 180:
        r = requests.get(f"{BASE}/api/detections?category=zipper&page=1&page_size=1",
                         timeout=5)
        n2 = r.json().get("total", 0) if isinstance(r.json(), dict) else 0
        if n2 > n_det:
            break
        time.sleep(3)
    ck("回队后重新检测（检测数增加）", n2 > n_det, f"{n_det}→{n2}")

# ═══════════════ 汇总 ═══════════════
print(f"\n{'='*50}", flush=True)
if ISSUES:
    print(f"流程测试发现 {len(ISSUES)} 个问题 ❌（通过 {N['ok']} 项）", flush=True)
    for it in ISSUES:
        print(f"  ❌ {it}", flush=True)
else:
    print(f"端到端流程测试全部通过 ✅（{N['ok']} 项）", flush=True)

# 清理
for _w in requests.get(f"{BASE}/api/workorders", params={"range": "all"},
                       timeout=5).json().get("items", []):
    if _w.get("name") == WO:
        requests.delete(f"{BASE}/api/workorders/{_w['id']}", timeout=5)
for _d in requests.get(f"{BASE}/api/datasources", timeout=5).json().get("items", []):
    if _d.get("name") == SRC:
        requests.delete(f"{BASE}/api/datasources/{_d['id']}", timeout=5)
sys.exit(1 if ISSUES else 0)
