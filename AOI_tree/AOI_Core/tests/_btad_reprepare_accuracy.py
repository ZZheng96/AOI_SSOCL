# -*- coding: utf-8 -*-
"""品类 02 用 accuracy 工作点重新准备模型：补回 fast 模式裁掉的
trad/blob/layout/inp 内置特征槽位（特征工程页依此展示生产特征组）。

用法：python tests\_btad_reprepare_accuracy.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

BASE = "http://127.0.0.1:8017"

# 后端：复用或拉起
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

r = requests.post(f"{BASE}/api/models/prepare", json={
    "category": "02", "scenario": "L1b", "profile": "accuracy",
    "force": True, "note": "补回内置特征槽位（accuracy 全槽位）"}, timeout=15)
print(f"[准备] 提交 → {r.status_code} {r.text[:200]}", flush=True)
task_id = r.json().get("task_id")

t0 = time.time()
while True:
    time.sleep(5)
    try:
        tasks = requests.get(f"{BASE}/api/tasks/recent", timeout=8).json()
    except Exception:  # noqa: BLE001
        continue
    items = tasks.get("items") if isinstance(tasks, dict) else tasks
    t = next((x for x in (items or []) if x.get("id") == task_id
              or x.get("task_id") == task_id), None)
    if t is None:
        continue
    st = t.get("status")
    prog = t.get("progress")
    msg = (prog.get("message") if isinstance(prog, dict)
           else (f"{prog:.0%}" if isinstance(prog, (int, float)) else ""))
    msg = msg or str(t.get("message") or "")
    print(f"[准备] {time.time()-t0:.0f}s {st} {msg}", flush=True)
    if st in ("done", "failed", "error"):
        print(f"[准备] 结束：{st} {str(t.get('result') or t.get('error') or '')[:300]}",
              flush=True)
        break
    if time.time() - t0 > 1800:
        print("[准备] 超时退出", flush=True)
        break

# 校验特征槽位
try:
    fs = requests.get(f"{BASE}/api/models/feature-set/02", timeout=15).json()
    groups = fs.get("groups") or []
    print(f"[校验] feature-set/02 status={fs.get('status')} "
          f"启用组={fs.get('enabled_groups')}", flush=True)
    for g in groups:
        print(f"  - {g.get('key')} dim={g.get('dim')} enabled={g.get('enabled')}",
              flush=True)
except Exception as e:  # noqa: BLE001
    print(f"[校验] 失败 {e}", flush=True)
