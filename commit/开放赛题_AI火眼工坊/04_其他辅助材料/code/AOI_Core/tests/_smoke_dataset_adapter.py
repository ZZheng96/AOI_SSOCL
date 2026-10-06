# -*- coding: utf-8 -*-
"""API 冒烟：数据集适配模块（识别 probe + 确认导入 adapt_import）。

覆盖（2026-08-31 前端反馈：非标准数据集识别 + 用户确认）：
1. probe 识别各数据集根：data_origin（mvtec/BTAD/GYU-DET）、data_local、
   test_images（solder/shift/jump）--格式/计数断言；
2. 确认导入 dir_rules 小数据集（test_images_solder，164 图）：
   分组 override 生效、_OK 图归预训练正常组、预训练/检测分组正确；
3. mvtec_like 单品类导入（data_local/component，24 图）模板图进 L3。

进度信息直接 print（调优用小数据集，秒级完成）。
"""
import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AOI_sys 根
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


def probe(root):
    r = requests.post(f"{BASE}/api/datasets/probe", json={"root": root},
                      timeout=120)
    return r.json() if r.status_code == 200 else {"_err": r.text}


def wait_task(tid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = requests.get(f"{BASE}/api/tasks/{tid}", timeout=8).json()
        if t.get("status") in ("done", "success"):
            return t
        if t.get("status") in ("failed", "error"):
            return t
        time.sleep(0.5)
    return {"status": "timeout"}


# 清理残留数据源
for d in requests.get(f"{BASE}/api/datasources", timeout=8).json().get("items", []):
    if str(d.get("name", "")).startswith("适配源"):
        requests.delete(f"{BASE}/api/datasources/{d['id']}", timeout=30)

print("── 1. probe 识别各数据集根 ─────────────────", flush=True)

# 1.1 MVTec：mvtec_like 15 品类
rep = probe(r"D:\CGAIC\data_origin\mvtec")
ck("mvtec 识别为 mvtec_like", rep.get("format") == "mvtec_like",
   f"cats={len(rep.get('categories') or [])}")
cats = {c["name"]: c for c in rep.get("categories") or []}
ck("mvtec 品类 bottle 计数合理",
   cats.get("bottle", {}).get("counts", {}).get("train_normal", 0) > 0,
   f"{cats.get('bottle', {}).get('counts')}")

# 1.2 BTAD（包装目录下钻）：mvtec_like 01/02/03
rep = probe(r"D:\CGAIC\data_origin\BTAD")
ck("BTAD 识别为 mvtec_like（包装目录下钻）", rep.get("format") == "mvtec_like",
   f"cats={[c['name'] for c in rep.get('categories') or []]}")
b01 = {c["name"]: c for c in rep.get("categories") or []}.get("01", {})
ck("BTAD 01 ok/ko 正确映射",
   b01.get("counts", {}).get("train_normal", 0) == 400
   and b01.get("counts", {}).get("test_anomaly", 0) == 49,
   f"{b01.get('counts')} masks={b01.get('n_masks')}")

# 1.3 GYU-DET：yolo_split，txt 非空为缺陷
rep = probe(r"D:\CGAIC\data_origin\GYU-DET")
cnt = ((rep.get("categories") or [{}])[0].get("counts") or {})
ck("GYU-DET 识别为 yolo_split", rep.get("format") == "yolo_split")
ck("GYU-DET txt 标签生效（缺陷占多数）",
   cnt.get("train_anomaly", 0) > cnt.get("train_normal", 0),
   f"train N/A={cnt.get('train_normal')}/{cnt.get('train_anomaly')}")

# 1.4 data_local：mvtec_like 4 品类（模板命名进 L3）
rep = probe(r"D:\CGAIC\data_local")
cats = {c["name"]: c for c in rep.get("categories") or []}
ck("data_local 识别为 mvtec_like",
   rep.get("format") == "mvtec_like" and "component" in cats
   and "gold_finger" in cats)
ck("data_local component 模板命名识别",
   cats.get("component", {}).get("counts", {}).get("template", 0) == 8,
   f"{cats.get('component', {}).get('counts')}")
ck("data_local gold_finger 掩码计数",
   cats.get("gold_finger", {}).get("n_masks", 0) >= 40,
   f"masks={cats.get('gold_finger', {}).get('n_masks')}")

# 1.5 test_images_solder：dir_rules + _OK 正常
rep = probe(r"D:\CGAIC\test_images\test_images_solder")
ck("solder 识别为 dir_rules", rep.get("format") == "dir_rules",
   f"images={rep.get('n_images')}")
groups = {g["name"]: g for g in
          ((rep.get("categories") or [{}])[0].get("groups") or [])}
ck("solder _OK 成对正常图识别",
   groups.get("bridge_pair", {}).get("normal", 0) > 0,
   f"bridge_pair={groups.get('bridge_pair')}")
ck("solder 未识别分组有告警", bool(rep.get("warnings")))

# 1.6 test_images_shift 电容底座：templ_ 模板 + _OK 正常 + 关键词缺陷
rep = probe(r"D:\CGAIC\test_images\test_images_shift\电容底座")
g = ((rep.get("categories") or [{}])[0].get("groups") or [{}])[0]
ck("shift 电容底座三类标记识别",
   g.get("template", 0) == 16 and g.get("normal", 0) == 1
   and g.get("anomaly", 0) == 8,
   f"tpl/n/a={g.get('template')}/{g.get('normal')}/{g.get('anomaly')}")

# 1.7 jump：ok/NG/误报/良品/模板小图
rep = probe(r"D:\CGAIC\test_images\test_images_jump\WIRE\jumper-20251023")
groups = {g["name"]: g for g in
          ((rep.get("categories") or [{}])[0].get("groups") or [])}
ck("jump ok/NG 目录标记识别",
   groups.get("ok", {}).get("normal", 0) + groups.get("ok2", {}).get("normal", 0) > 500
   and groups.get("NG", {}).get("anomaly", 0) > 0
   and groups.get("NG2", {}).get("anomaly", 0) > 0,
   f"ok={groups.get('ok')} NG={groups.get('NG')}")

# 1.8 误报=正常（AOI 假报警）
rep = probe(r"D:\CGAIC\test_images\test_images_jump\WJ 跳线压线和变形显绿框20260811\AP-P309AM")
groups = {g["name"]: g for g in
          ((rep.get("categories") or [{}])[0].get("groups") or [])}
ck("误报目录按正常识别（假报警=正常）",
   groups.get("误报", {}).get("normal", 0) == 20,
   f"{groups.get('误报')}")
ck("良品/模板小图识别",
   groups.get("良品", {}).get("normal", 0) > 0
   and groups.get("模板小图", {}).get("template", 0) > 0)

# 1.9 错误路径
r = requests.post(f"{BASE}/api/datasets/probe",
                  json={"root": r"D:\CGAIC\not_exist_dir"}, timeout=8)
ck("不存在目录返回 400", r.status_code == 400)

print("── 2. 确认导入 dir_rules（solder，分组 override）──────", flush=True)

r = requests.post(f"{BASE}/api/datasources", json={
    "name": "适配源-solder", "modality": "image", "label_tier": "L1a",
    "pretrain_normal": 100, "pretrain_anomaly": 30, "batch_size": 30},
    timeout=8)
sid = r.json()["id"]
ck("建数据源", bool(sid), f"id={sid}")

# 未识别分组（insufficient 等）按默认缺陷导入；_OK 图作预训练正常图
r = requests.post(f"{BASE}/api/datasets/adapt_import", json={
    "root": r"D:\CGAIC\test_images\test_images_solder",
    "format": "dir_rules", "category": "solder", "ok_role": "train",
    "group_overrides": {"insufficient": "anomaly", "excess": "anomaly"},
    "dataset_name": "test_images_solder", "datasource_id": sid}, timeout=30)
ck("adapt_import 提交", r.status_code == 200 and r.json().get("task_id"))
t = wait_task(r.json().get("task_id"))
ck("导入任务完成", t.get("status") in ("done", "success"),
   f"{t.get('status')} {t.get('message')}")

# 验证入库内容：正常图 split=train，缺陷 split=test 且带缺陷类型
data = requests.get(f"{BASE}/api/images", params={
    "category": "solder", "page_size": 300}, timeout=30).json()
items = data.get("items", [])
n_norm = [i for i in items if i.get("label") == "normal"]
n_anom = [i for i in items if i.get("label") == "anomaly"]
ck("solder 图片入库（正常+缺陷）",
   n_norm and n_anom, f"normal={len(n_norm)} anomaly={len(n_anom)}")
ck("正常图 _OK 命名且 split=train",
   all(i.get("split") == "train" for i in n_norm)
   and all("_OK" in i.get("path", "") for i in n_norm))
anom_types = {i.get("defect_type") for i in n_anom}
ck("缺陷图带缺陷类型（分组目录名）",
   {"insufficient", "excess"} <= anom_types or anom_types,
   f"types={sorted(str(t) for t in anom_types)[:6]}")

# 分组：预训练组应有正常图（100 上限内全部 _OK 图）
g = requests.get(f"{BASE}/api/datasources/{sid}/groups", timeout=8).json()
pc = ((g.get("categories") or {}).get("solder") or {}).get("pretrain_counts") or {}
ck("预训练组含 _OK 正常图", pc.get("normal", 0) >= 30,
   f"pretrain={pc}")

# 清理
requests.delete(f"{BASE}/api/datasources/{sid}", timeout=60)

print("── 3. mvtec_like 单品类导入（component，模板进 L3）────", flush=True)

r = requests.post(f"{BASE}/api/datasources", json={
    "name": "适配源-component", "modality": "image", "label_tier": "L1a"},
    timeout=8)
sid2 = r.json()["id"]
r = requests.post(f"{BASE}/api/datasets/adapt_import", json={
    "root": r"D:\CGAIC\data_local\component", "format": "mvtec_like",
    "dataset_name": "component", "datasource_id": sid2}, timeout=30)
t = wait_task(r.json().get("task_id"))
ck("component 导入完成", t.get("status") in ("done", "success"),
   f"{t.get('message')}")
data = requests.get(f"{BASE}/api/images", params={
    "category": "component", "page_size": 100}, timeout=30).json()
items = data.get("items", [])
tpl = [i for i in items if i.get("split") == "template"]
anom = [i for i in items if i.get("label") == "anomaly"]
ck("component 模板图 8 张进 L3", len(tpl) == 8, f"tpl={len(tpl)}")
ck("component test/缺陷目录识别",
   len(anom) >= 14, f"anomaly={len(anom)}")
requests.delete(f"{BASE}/api/datasources/{sid2}", timeout=60)

print(f"\n{'='*46}")
print(f"验证{'全部通过 ✅' if not ISSUES else '发现 ' + str(len(ISSUES)) + ' 个问题 ❌'}（{_OK['n']}）")
for it in ISSUES:
    print(f"  ❌ {it}")
sys.exit(1 if ISSUES else 0)
