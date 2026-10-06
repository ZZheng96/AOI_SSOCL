"""M14 冒烟：demo5 设计 §0/§2 前端对应补全。

覆盖：
1. M14a L3 模板链路：annotate 标记模板 → precheck counts.template →
   prepare L3 携带 templates → detect 响应含 tpl 槽位分。
2. M14b 在线学习内部状态：/self_learning/insight 返回双库/孵育/权重门控。
3. M14c 不完备报告：/review/incomplete_report 结构 + open_alert 落库。
4. M14d 数据集分层徽章：/datasets items[].layers 推断正确。

隔离环境：独立临时 sqlite/storage/cfg（同 M10/M11 模式）。
数据：mvtec bottle 10 正常 + 3 缺陷 + 2 张模板标记（fast profile）。
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AOI_sys 根

TMP = tempfile.mkdtemp(prefix="aoi_m14_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "system": {
            "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
            "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
            "demo5_storage": os.path.join(TMP, "storage", "demo5").replace("\\", "/"),
            "demo5_base_cfg": "configs/demo5_fast.yaml",
        },
        "archive": {"enabled": False},
    }, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

CAT = "bottle_m14"
MV = r"D:\CGAIC\data_origin\mvtec\bottle"

ok_count = 0


def ck(name, cond, detail=""):
    global ok_count
    assert cond, f"❌ {name}: {detail}"
    ok_count += 1
    print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)


def _ls(d, n):
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]


def wait_task(task_manager, task_id, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = task_manager.get(task_id)
        if t["status"] in ("done", "failed"):
            return t
        time.sleep(3)
    raise TimeoutError(f"task {task_id} 超时")


def main():
    from fastapi.testclient import TestClient
    from backend.api.app import app
    from backend.core.tasks import task_manager
    from backend.db.database import session_scope
    from backend.db.models import Image as ImageRow

    with TestClient(app) as client:
        # ── 数据登记 ─────────────────────────────────────
        print("[M14-0] 数据登记", flush=True)
        good = _ls(os.path.join(MV, "train", "good"), 10)
        defects = []
        for sub in sorted(os.listdir(os.path.join(MV, "test"))):
            dd = os.path.join(MV, "test", sub)
            if os.path.isdir(dd) and sub != "good":
                defects += _ls(dd, 1)
            if len(defects) >= 3:
                break
        ids = {}
        with session_scope() as s:
            for p in good:
                row = ImageRow(path=p, category=CAT, split="train",
                               label="normal", source="mvtec")
                s.add(row)
                s.flush()
                ids[p] = row.id
            for p in defects:
                s.add(ImageRow(path=p, category=CAT, split="test",
                               label="anomaly", source="mvtec"))

        # ── 1) M14a L3 模板链路 ──────────────────────────
        print("[M14-1] L3 模板链路", flush=True)
        tpl_ids = [ids[good[0]], ids[good[1]]]
        for iid in tpl_ids:
            r = client.post(f"/api/images/{iid}/annotate",
                            json={"split": "template"})
            ck(f"标记模板图 #{iid}", r.status_code == 200, f"{r.status_code}")
        r = client.get(f"/api/models/precheck/{CAT}")
        ck("precheck 模板计数", r.json()["counts"].get("template") == 2,
           f"counts={r.json()['counts']}")
        ck("precheck 模板存在性", r.json().get("has_template") is True)
        ck("precheck L3 提示", any("L3" in w for w in r.json()["warnings"]))

        r = client.post("/api/models/prepare",
                        json={"category": CAT, "scenario": "L3",
                              "profile": "fast", "force": True})
        assert r.status_code == 200, f"prepare 提交失败: {r.text}"
        t = wait_task(task_manager, r.json()["task_id"])
        ck("L3 prepare 完成", t["status"] == "done", t.get("message", ""))
        res = t.get("result") or {}
        ck("prepare 携带模板数", res.get("n_templates") == 2,
           f"n_templates={res.get('n_templates')}")

        r = client.post("/api/detect/image",
                        json={"path": defects[0], "category": CAT,
                              "with_heatmap": False})
        assert r.status_code == 200, f"detect 失败: {r.text}"
        det = r.json()
        ck("L3 激活 tpl 槽位", "tpl" in (det.get("slots") or {}),
           f"slots={list((det.get('slots') or {}).keys())}")

        # ── 2) M14b 在线学习内部状态 ─────────────────────
        print("[M14-2] 引擎内部状态透出", flush=True)
        r = client.get("/api/self_learning/insight", params={"category": CAT})
        ins = r.json()
        ck("insight enabled", ins.get("enabled") is True, f"keys={list(ins.keys())}")
        ck("insight 双库字段", "normal_bank" in ins and "defect_bank" in ins,
           f"core={ins.get('normal_bank', {}).get('core_patches')}")
        ck("insight 孵育/门控字段", "incubate" in ins and "weight_gate" in ins)
        ck("insight train_auroc", ins.get("train_auroc") is not None,
           f"train_auroc={ins.get('train_auroc')}")
        r = client.get("/api/self_learning/insight",
                       params={"category": "no_such_cat"})
        ck("未准备品类空态", r.json().get("enabled") is False)

        # ── 3) M14c 不完备报告 ───────────────────────────
        print("[M14-3] 不完备报告", flush=True)
        r = client.get("/api/review/incomplete_report")
        rep = r.json()
        ck("incomplete_report 结构",
           "reports" in rep and "any_alert" in rep,
           f"reports={len(rep.get('reports', []))}")
        cat_rep = [x for x in rep["reports"] if x["category"] == CAT]
        ck("报告含本品类窗口统计", bool(cat_rep)
           and cat_rep[0]["window_n"] >= 1, f"{cat_rep}")
        # open_alert 落库字段存在（M14c 原料）
        r = client.get("/api/detections", params={"category": CAT, "page_size": 5})
        items = r.json().get("items", [])
        ck("检测记录含 open_alert 落库",
           bool(items) and "open_alert" in (items[0].get("n_tiles") or {}))

        # ── 4) M14d 数据集分层徽章 ───────────────────────
        print("[M14-4] 数据集分层徽章", flush=True)
        from backend.core import datasets as ds_helper
        with session_scope() as s:
            rows = s.query(ImageRow).filter(ImageRow.category == CAT).all()
            ds_id = ds_helper.create_dataset(CAT, "test", name="m14_分层验证批次")
            for row in rows:
                row.dataset_id = ds_id
        r = client.get("/api/datasets", params={"category": CAT})
        items = r.json()["items"]
        mine = [x for x in items if x["id"] == ds_id]
        ck("批次含 layers 字段", bool(mine) and "layers" in mine[0])
        layers = mine[0]["layers"]
        ck("层级推断 L0/L1a/L2/L3",
           all(x in layers for x in ("L0", "L1a", "L2", "L3")),
           f"layers={layers}")

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n✅ M14 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
