# -*- coding: utf-8 -*-
"""PCB_Ins v2 冒烟：数据集适配模块（识别 probe + 确认导入 adapt_import）。

与 AOI_sys dataset_adapter 同口径（2026-08-31 同步）：
1. probe 识别各数据集根（mvtec/BTAD/GYU-DET/data_local/test_images_solder）；
2. adapt_import 导入小数据集到新数据源 -> 验证入库 -> 删除数据源级联清理。

用法：python tests/_smoke_dataset_adapter.py（先无 server 运行亦可，脚本自启）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from server import start_server_background  # noqa: E402

start_server_background()

BASE = "http://127.0.0.1:8021/api"
for _ in range(100):
    try:
        if requests.get(f"{BASE}/health", timeout=1).json().get("status") == "ok":
            break
    except Exception:  # noqa: BLE001
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


def probe(root):
    r = requests.post(f"{BASE}/datasources/probe", json={"root": root},
                      timeout=120)
    return r.json() if r.status_code == 200 else {"_err": r.text}


print("── 1. probe 识别各数据集根 ─────────────────", flush=True)

rep = probe(r"D:\CGAIC\data_origin\mvtec")
ck("mvtec 识别为 mvtec_like", rep.get("format") == "mvtec_like",
   f"cats={len(rep.get('categories') or [])}")

rep = probe(r"D:\CGAIC\data_origin\BTAD")
ck("BTAD 识别为 mvtec_like（包装目录下钻）", rep.get("format") == "mvtec_like",
   f"cats={[c['name'] for c in rep.get('categories') or []]}")
b01 = {c["name"]: c for c in rep.get("categories") or []}.get("01", {})
ck("BTAD 01 ok/ko 正确映射",
   b01.get("counts", {}).get("train_normal", 0) == 400
   and b01.get("counts", {}).get("test_anomaly", 0) == 49,
   f"{b01.get('counts')} masks={b01.get('n_masks')}")

rep = probe(r"D:\CGAIC\data_origin\GYU-DET")
cnt = ((rep.get("categories") or [{}])[0].get("counts") or {})
ck("GYU-DET 识别为 yolo_split", rep.get("format") == "yolo_split")
ck("GYU-DET txt 标签生效（缺陷占多数）",
   cnt.get("train_anomaly", 0) > cnt.get("train_normal", 0),
   f"train N/A={cnt.get('train_normal')}/{cnt.get('train_anomaly')}")

rep = probe(r"D:\CGAIC\data_local")
cats = {c["name"]: c for c in rep.get("categories") or []}
ck("data_local 识别为 mvtec_like",
   rep.get("format") == "mvtec_like" and "component" in cats)
ck("data_local component 模板命名识别",
   cats.get("component", {}).get("counts", {}).get("template", 0) == 8,
   f"{cats.get('component', {}).get('counts')}")

rep = probe(r"D:\CGAIC\test_images\test_images_solder")
ck("solder 识别为 dir_rules", rep.get("format") == "dir_rules",
   f"images={rep.get('n_images')}")
groups = {g["name"]: g for g in
          ((rep.get("categories") or [{}])[0].get("groups") or [])}
ck("solder _OK 成对正常图识别",
   groups.get("bridge_pair", {}).get("normal", 0) > 0)

r = requests.post(f"{BASE}/datasources/probe",
                  json={"root": r"D:\CGAIC\not_exist_dir"}, timeout=8)
ck("不存在目录返回 400", r.status_code == 400)

print("── 2. adapt_import 导入 solder -> 验证 -> 级联清理 ──", flush=True)

r = requests.post(f"{BASE}/datasources", json={
    "name": "适配源-solder", "modality": "image", "label_tier": "L1a",
    "pretrain_normal": 100, "pretrain_anomaly": 30, "batch_size": 30},
    timeout=8)
sid = r.json()["id"]
ck("建数据源", bool(sid), f"id={sid}")

r = requests.post(f"{BASE}/datasources/adapt_import", json={
    "root": r"D:\CGAIC\test_images\test_images_solder",
    "format": "dir_rules", "category": "solder", "ok_role": "train",
    "group_overrides": {"insufficient": "anomaly", "excess": "anomaly"},
    "dataset_name": "test_images_solder", "datasource_id": sid}, timeout=300)
body = r.json()
ck("adapt_import 提交", r.status_code == 200 and body.get("imported", 0) > 0,
   f"imported={body.get('imported')} total={body.get('total')}")

# 验证入库
data = requests.get(f"{BASE}/datasources/{sid}", timeout=8).json()
ck("数据源详情带批次", len(data.get("batches") or []) == 1,
   f"batches={len(data.get('batches') or [])}")
n_images = data.get("n_images", 0)
ck("图片入库", n_images > 0, f"n_images={n_images}")

# 删除数据源级联清理（不影响其他数据）
r = requests.delete(f"{BASE}/datasources/{sid}", timeout=120)
ck("删除数据源级联清理", r.status_code == 200 and r.json().get("deleted"),
   f"images={r.json().get('images')}")

print(f"\n{'='*46}")
print(f"验证{'全部通过 ✅' if not ISSUES else '发现 ' + str(len(ISSUES)) + ' 个问题 ❌'}（{_OK['n']}）")
for it in ISSUES:
    print(f"  ❌ {it}")
sys.exit(1 if ISSUES else 0)
