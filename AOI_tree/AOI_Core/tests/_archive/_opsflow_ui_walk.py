# -*- coding: utf-8 -*-
"""AOI_sys 界面操作流走查：模拟用户按 demo5用户操作流 §2-§7 在界面逐步点按钮，
每个按钮背后就是这里的 API 调用。MPDD bracket_black 走通 导入→体检→准备→评估→检测→反馈→追溯→导出。
"""
import sys
import time
import json
import requests

sys.stdout.reconfigure(line_buffering=True)

BASE = "http://127.0.0.1:8017"
CAT = "bracket_black"
MPDD_ROOT = r"D:\CGAIC\data_origin\MPDD"
issues = []


def call(method, path, **kw):
    r = requests.request(method, BASE + path, timeout=120, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def wait_task(task_id, poll=2.0, timeout_s=600, label=""):
    t0 = time.time()
    last_print = -10
    while time.time() - t0 < timeout_s:
        st = call("GET", f"/api/tasks/{task_id}")
        s = st.get("status")
        prog = st.get("progress", {})
        done = prog.get("done", 0) if isinstance(prog, dict) else 0
        total = prog.get("total", 0) if isinstance(prog, dict) else 0
        msg = st.get("message", "")
        if s in ("completed", "done", "finished", "success"):
            print(f"  [task:{label}] 完成 ({time.time()-t0:.0f}s) result={st.get('result')}")
            return st.get("result")
        if s in ("failed", "error", "cancelled"):
            err = st.get("error", "?")
            issues.append(f"任务[{label}]失败: {err}")
            print(f"  [task:{label}] 失败: {err}")
            return None
        if int(time.time() - t0) - last_print >= 10:
            last_print = int(time.time() - t0)
            print(f"  [task:{label}] {s} {done}/{total} {msg}")
        time.sleep(poll)
    issues.append(f"任务[{label}]超时 {timeout_s}s")
    return None


def main():
    # ---- 顶栏/系统状态（对应顶栏指示灯）----
    info = call("GET", "/api/system/info")
    print(f"[1 系统状态] device={info.get('device')} role={info.get('role')} "
          f"latency_budget={info.get('latency_budget_ms')}ms")
    cats = call("GET", "/api/categories")
    print(f"[2 品类] 现有 {cats}")

    # ---- 数据管理 → 导入MVTec（对应「导入MVTec」按钮）----
    print("\n[3 数据管理] 导入MVTec: MPDD bracket_black ...")
    r = call("POST", "/api/images/import_mvtec",
             json={"root": MPDD_ROOT, "categories": [CAT]})
    tid = r["task_id"]
    imp = wait_task(tid, label="import_mvtec")
    if imp is None:
        return
    q = call("GET", "/api/images", params={"category": CAT, "page_size": 500})
    total = q.get("total", 0)
    print(f"  导入完成: 库内 {total} 张")

    # ---- 模型管理 → 准备模型（先「数据体检」再「开始准备」）----
    print("\n[4 模型管理] 数据体检 precheck ...")
    pc = call("GET", f"/api/models/precheck/{CAT}")
    print(f"  precheck: {json.dumps(pc, ensure_ascii=False)[:400]}")
    print("  [4] 点击「开始准备」(scenario=L1a profile=fast) ...")
    r = call("POST", "/api/models/prepare",
             json={"category": CAT, "scenario": "L1a", "profile": "fast", "note": "opsflow-walk"})
    prep = wait_task(r["task_id"], label="prepare", timeout_s=900)
    if prep is None:
        return
    models = call("GET", "/api/models", params={"category": CAT})
    for m in models.get("items", [])[:3]:
        print(f"  模型快照 v{m.get('version')} active={m.get('active')} "
              f"n_normal={m.get('n_normal')} n_defect={m.get('n_defect')} "
              f"auroc={m.get('meta', {}).get('train_auroc')}")

    # ---- 评估看板 → 运行评估 ----
    print("\n[5 评估看板] 运行评估 (split=test) ...")
    r = call("POST", "/api/eval/accuracy", json={"category": CAT, "split": "test", "name": "opsflow-walk"})
    ev = wait_task(r["task_id"], label="eval_accuracy", timeout_s=900)
    runs = call("GET", "/api/eval/runs", params={"page_size": 3})
    for run in runs.get("items", [])[:2]:
        m = run.get("metrics", {}) or {}
        print(f"  EvalRun#{run.get('id')} name={run.get('name')} "
              f"auroc={m.get('auroc')} f1={m.get('f1')} ms/img={run.get('ms_per_image')}")

    # ---- 实时监控 → 选择图片并检测（异常图 + 正常图各一张）----
    print("\n[6 实时监控] 选择图片并检测 ...")
    imgs = call("GET", "/api/images", params={"category": CAT, "page_size": 500})
    def_ids = [i for i in imgs.get("items", []) if i.get("label") == "anomaly"]
    good_ids = [i for i in imgs.get("items", []) if i.get("label") == "normal" and i.get("split") == "test"]
    dets = []
    for name, pool in (("缺陷图", def_ids), ("正常图", good_ids)):
        if not pool:
            print(f"  [skip] {name} 无样本")
            continue
        rid = pool[0]["id"]
        r = call("POST", "/api/detect/image",
                 json={"image_id": rid, "category": CAT, "with_heatmap": True})
        d = r.get("detection", r)
        dets.append(r)
        print(f"  {name} image#{rid}: decision={d.get('decision')} fused={d.get('fused')} "
              f"boxes={len(d.get('boxes') or [])} types={d.get('types')} "
              f"ms={d.get('ms_per_image')}")
    if not dets:
        issues.append("检测未产出记录")
        return

    # ---- 标注反馈 → 判正常/判缺陷 ----
    print("\n[7 标注反馈] 对检测记录打反馈 ...")
    for i, det in enumerate(dets[:2]):
        det_id = det.get("detection_id") or det.get("id") or (det.get("detection") or {}).get("id")
        if det_id is None:
            issues.append("检测返回无 detection_id，无法打反馈")
            continue
        fb_type = "confirmed" if det.get("detection", det).get("label", 1) == 1 else "false_positive"
        op_label = 1 if det.get("detection", det).get("label", 1) == 1 else 0
        r = call("POST", "/api/feedback",
                 json={"detection_id": det_id, "feedback_type": fb_type,
                       "operator_label": op_label, "operator": "admin",
                       "comment": "opsflow-walk"})
        print(f"  feedback#{r.get('id')} detection={det_id} type={fb_type} -> {r.get('status', 'ok')}")

    # ---- 追溯举证 ----
    print("\n[8 追溯] 查询检测轨迹 ...")
    first = dets[0]
    det_id = first.get("detection_id") or first.get("id") or (first.get("detection") or {}).get("id")
    if det_id:
        tr = call("GET", f"/api/detections/{det_id}/trace")
        keys = list(tr.keys()) if isinstance(tr, dict) else []
        print(f"  trace 字段: {keys[:12]}")
        slot = tr.get("slots") or tr.get("slot_scores")
        if isinstance(slot, dict):
            print(f"  槽位分: { {k: round(v,3) for k,v in slot.items()} }")
        else:
            print(f"  trace 内容: {json.dumps(tr, ensure_ascii=False)[:300]}")
    else:
        issues.append("无 detection_id 可追溯")

    # ---- 统计报表 → 审计日志 CSV 导出 ----
    print("\n[9 统计报表] 导出审计日志 CSV ...")
    try:
        r = requests.get(BASE + "/api/logs/export", timeout=30)
        print(f"  /api/logs/export -> {r.status_code} len={len(r.content)}B")
    except Exception as e:
        issues.append(f"logs/export 异常: {e}")
    try:
        r = requests.get(BASE + "/api/stats/export", timeout=30)
        print(f"  /api/stats/export -> {r.status_code} len={len(r.content)}B")
    except Exception as e:
        issues.append(f"stats/export 异常: {e}")

    # ---- 汇总 ----
    print("\n" + "=" * 56)
    if issues:
        print(f"[界面走查发现 {len(issues)} 个问题]")
        for it in issues:
            print(f"  - {it}")
    else:
        print("[界面走查未发现问题]")


if __name__ == "__main__":
    main()
