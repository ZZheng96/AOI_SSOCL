# -*- coding: utf-8 -*-
"""给 BTAD-02 演示工单补一批反馈（误检/漏检/确认正确混合），
让「学习效果」页有可见的翻案/趋势数据，供 12/13 分镜重录。

用法：python tests\_btad_feedback.py
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

r = requests.get(f"{BASE}/api/detections?page=1&page_size=40", timeout=10)
items = r.json().get("items", [])
print(f"检测记录 {len(items)} 条", flush=True)
if items:
    print("字段样例：", {k: items[0].get(k) for k in list(items[0])[:12]}, flush=True)

# 按判定分组：is_anomaly=True 判异常（可标误检/确认正确），False 判正常（可标漏检）
abn = [d for d in items if d.get("is_anomaly")]
nor = [d for d in items if not d.get("is_anomaly")
       and d.get("id") not in (107, 108, 109, 110)]  # 已提交过漏检反馈
print(f"判异常 {len(abn)} / 判正常 {len(nor)}", flush=True)

plan = []
for d in abn[:4]:    # 2 条误检 + 2 条确认正确
    plan.append((d, "false_positive", 0, "划痕"))
for d in abn[4:6]:
    plan.append((d, "confirmed", 1, ""))
for d in nor[:4]:    # 漏检
    plan.append((d, "false_negative", 1, "划伤"))

ok = 0
for d, ft, label, defect in plan:
    did = d.get("id") or d.get("detection_id")
    try:
        resp = requests.post(f"{BASE}/api/feedback", json={
            "detection_id": did, "feedback_type": ft,
            "operator_label": label, "defect_type": defect,
            "comment": "", "operator": "演示"}, timeout=10)
        print(f"  反馈 #{did} {ft} → {resp.status_code}", flush=True)
        ok += resp.status_code == 200
    except Exception as e:  # noqa: BLE001
        print(f"  反馈 #{did} 异常 {e}", flush=True)
print(f"提交完成 {ok}/{len(plan)}", flush=True)
