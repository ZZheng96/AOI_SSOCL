# -*- coding: utf-8 -*-
"""API 冒烟：数据源分组（预训练组/检测组·批次）、方案另存/复用、存量归入。"""
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


# 清理残留
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)
for p in requests.get(f"{BASE}/api/datasource-plans", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasource-plans/{p['id']}", timeout=8)

# 1. 建数据源（配置 100+30 / 批 30）
r = requests.post(f"{BASE}/api/datasources",
                  json={"name": "分组源", "modality": "image", "label_tier": "L1a",
                        "pretrain_normal": 100, "pretrain_anomaly": 30,
                        "batch_size": 30}, timeout=8)
sid = r.json()["id"]
ck("建数据源（含分组配置）", r.status_code == 200, f"id={sid}")

# 2. 存量一键归入
r = requests.post(f"{BASE}/api/datasources/{sid}/attach-legacy", timeout=8)
ck("存量一键归入", r.status_code == 200 and r.json().get("attached", 0) >= 20,
   f"batches={r.json().get('attached')}")

# 3. 分组
r = requests.get(f"{BASE}/api/datasources/{sid}/groups", timeout=8)
b = r.json()
cats = b.get("categories") or {}
zipper = cats.get("zipper") or {}
ck("分组含 zipper", bool(zipper))
pn = zipper.get("pretrain_counts") or {}
ck("zipper 预训练组 100 正常+30 异常",
   pn.get("normal") >= 100 and pn.get("anomaly") >= 30,
   f"normal={pn.get('normal')} anomaly={pn.get('anomaly')}")
n_batches = len(zipper.get("detect_batches") or [])
ck("zipper 检测组分批（每批≤30）",
   n_batches >= 1 and all(len(x) <= 30 for x in zipper.get("detect_batches", [])),
   f"batches={n_batches} 首批={len((zipper.get('detect_batches') or [[]])[0])}")
det = zipper.get("detect_count")
total = zipper.get("total")
ck("检测组 = 其余全部",
   det == total - (pn.get("normal") + pn.get("anomaly")),
   f"detect={det} total={total}")

# 4. 另存方案
r = requests.post(f"{BASE}/api/datasources/{sid}/plan",
                  json={"name": "zipper方案"}, timeout=8)
pid = r.json().get("id") if r.status_code == 200 else None
ck("另存方案", pid is not None, f"plan={pid}")
plans = requests.get(f"{BASE}/api/datasource-plans", timeout=8).json().get("items", [])
ck("方案列表可见", any(p.get("id") == pid for p in plans))

# 5. 方案复用：先删分组源（批次回未挂源），再用方案建新源
requests.delete(f"{BASE}/api/datasources/{sid}", timeout=8)
r = requests.post(f"{BASE}/api/datasources",
                  json={"name": "复用源", "modality": "image",
                        "plan_id": pid}, timeout=8)
sid2 = r.json().get("id")
requests.post(f"{BASE}/api/datasources/{sid2}/attach-legacy", timeout=8)
b2 = requests.get(f"{BASE}/api/datasources/{sid2}/groups", timeout=8).json()
z2 = (b2.get("categories") or {}).get("zipper") or {}
ck("复用方案：预训练组固定同清单",
   (z2.get("pretrain_counts") or {}).get("normal")
   == (zipper.get("pretrain_counts") or {}).get("normal"),
   f"normal={ (z2.get('pretrain_counts') or {}).get('normal')}")
ck("复用源 using_plan", bool(b2.get("using_plan")))

# 6. 导出方案 JSON
r = requests.get(f"{BASE}/api/datasource-plans/{pid}/export", timeout=8)
ck("方案导出 JSON", r.status_code == 200 and '"config"' in r.text,
   f"len={len(r.text)}")

# 清理
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=8)
for p in requests.get(f"{BASE}/api/datasource-plans", timeout=8).json().get("items", []):
    requests.delete(f"{BASE}/api/datasource-plans/{p['id']}", timeout=8)

print(f"\n{'='*46}")
print(f"验证{'全部通过 ✅' if not ISSUES else '发现 ' + str(len(ISSUES)) + ' 个问题 ❌'}（{_OK['n']}）")
for it in ISSUES:
    print(f"  ❌ {it}")
sys.exit(1 if ISSUES else 0)
