# -*- coding: utf-8 -*-
"""验证：prepare 带 datasource_id 时样本取自数据源预训练组（100+30）。"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.CRITICAL)
from server import start_server_background
start_server_background()
import requests

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


# 清理
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)

r = requests.post(f"{BASE}/api/datasources",
                  json={"name": "PT源", "modality": "image", "label_tier": "L1a",
                        "pretrain_normal": 100, "pretrain_anomaly": 30,
                        "batch_size": 30}, timeout=8)
sid = r.json()["id"]
requests.post(f"{BASE}/api/datasources/{sid}/attach-legacy", timeout=8)

r = requests.post(f"{BASE}/api/models/prepare",
                  json={"category": "zipper", "scenario": "L1a",
                        "profile": "fast", "force": True,
                        "datasource_id": sid}, timeout=8)
tid = r.json().get("task_id")
ck("提交 prepare（带 datasource_id）", tid is not None, f"task={tid}")
last = {}
t0 = time.time()
while time.time() - t0 < 300:
    last = requests.get(f"{BASE}/api/tasks/{tid}", timeout=8).json()
    if last.get("status") in ("done", "failed"):
        break
    time.sleep(3)
ck("prepare 完成", last.get("status") == "done", str(last.get("status")))
res = last.get("result") or {}
ck("预训练组样本 =100 正常+30 异常",
   int(res.get("n_normal") or 0) >= 100 and int(res.get("n_defect") or 0) >= 30,
   f"normal={res.get('n_normal')} defect={res.get('n_defect')}")

# 清理
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)

print(f"\n{'='*46}")
print(f"验证{'全部通过 ✅' if not ISSUES else '发现 ' + str(len(ISSUES)) + ' 个问题 ❌'}（{_OK['n']}）")
for it in ISSUES:
    print(f"  ❌ {it}")
sys.exit(1 if ISSUES else 0)
