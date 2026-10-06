"""M11 冒烟：评审 §11 复评改进项。

覆盖：
1. M11c 多角色权限（security.keys）：
   operator 仅读/检测（scan/prepare/删图/删批次 403）；
   engineer 可 scan/prepare/删图（删批次 403）；admin 全通（删批次 200）。
2. M11b 归因透出：detect 响应含 defect_types；非正常判定下 types 非空。
3. M11d 对位预警：快照含 align_ref.npy；平移图触发 align_warn + offset。
4. M11e 日志轮转：server._setup_logging 幂等挂 RotatingFileHandler 落盘。

隔离环境：独立临时 sqlite/storage/cfg（security.keys 三角色）。
数据：mvtec bottle 10 正常 + 3 缺陷（fast profile，同 M10 口径）。
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AOI_sys 根

# ---- 隔离环境（必须在 import backend 之前设置 AOI_CONFIG）----
TMP = tempfile.mkdtemp(prefix="aoi_m11_")
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
        "security": {"keys": {"op-key": "operator",
                              "eng-key": "engineer",
                              "adm-key": "admin"}},
        "archive": {"enabled": False},
    }, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

CAT = "bottle_m11"
MV = r"D:\CGAIC\data_origin\mvtec\bottle"
OP = {"X-API-Key": "op-key"}
ENG = {"X-API-Key": "eng-key"}
ADM = {"X-API-Key": "adm-key"}

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

    with TestClient(app) as client:
        # ── 1) M11c 多角色权限 ───────────────────────────
        print("[M11-1] 多角色权限", flush=True)
        r = client.get("/api/categories", headers=OP)
        ck("operator 只读放行", r.status_code == 200, f"status={r.status_code}")
        r = client.post("/api/models/scan", headers=OP)
        ck("operator scan 被拒 403", r.status_code == 403, f"status={r.status_code}")
        r = client.post("/api/models/prepare",
                        json={"category": CAT, "scenario": "L1a",
                              "profile": "fast"}, headers=OP)
        ck("operator prepare 被拒 403", r.status_code == 403)
        r = client.post("/api/models/1/activate", headers=OP)
        ck("operator activate 被拒 403", r.status_code == 403)
        r = client.post("/api/self_learning/update",
                        json={"category": CAT}, headers=OP)
        ck("operator 巩固落盘被拒 403", r.status_code == 403)
        r = client.delete("/api/images/1", headers=OP)
        ck("operator 删图被拒 403", r.status_code == 403)
        r = client.delete("/api/datasets/1", headers=ENG)
        ck("engineer 删批次被拒 403", r.status_code == 403)
        r = client.post("/api/models/scan", headers=ENG)
        ck("engineer scan 放行", r.status_code == 200, f"status={r.status_code}")

        from backend.db.database import session_scope
        from backend.db.models import Image as ImageRow
        from backend.core import datasets as ds_helper
        with session_scope() as s:
            s.add(ImageRow(path=os.path.join(TMP, "dummy.png"), category=CAT,
                           split="unlabeled", label="unknown", source="test"))
            s.flush()
            dummy_img_id = s.query(ImageRow).order_by(ImageRow.id.desc()).first().id
        r = client.delete(f"/api/images/{dummy_img_id}", headers=ENG)
        ck("engineer 删图放行", r.status_code == 200, f"status={r.status_code}")
        ds_id = ds_helper.create_dataset(CAT, "test", name="m11_权限验证批次")
        r = client.delete(f"/api/datasets/{ds_id}", headers=ADM)
        ck("admin 删批次放行", r.status_code == 200, f"status={r.status_code}")

        # ── 2) 数据登记 + prepare（engineer 权限）────────
        print("[M11-2] 数据登记 + prepare", flush=True)
        good = _ls(os.path.join(MV, "train", "good"), 10)
        defects = []
        for sub in sorted(os.listdir(os.path.join(MV, "test"))):
            dd = os.path.join(MV, "test", sub)
            if os.path.isdir(dd) and sub != "good":
                defects += _ls(dd, 1)
            if len(defects) >= 3:
                break
        with session_scope() as s:
            for p in good:
                s.add(ImageRow(path=p, category=CAT, split="train",
                               label="normal", source="mvtec"))
            for p in defects:
                s.add(ImageRow(path=p, category=CAT, split="train_anomaly",
                               label="anomaly", source="mvtec"))
        r = client.post("/api/models/prepare",
                        json={"category": CAT, "scenario": "L1a",
                              "profile": "fast", "force": True}, headers=ENG)
        assert r.status_code == 200, f"prepare 提交失败: {r.text}"
        t = wait_task(task_manager, r.json()["task_id"])
        ck("prepare 完成", t["status"] == "done", t.get("message", ""))

        # 快照产物：align_ref.npy（M11d 持久化）
        snap_root = os.path.join(TMP, "storage", "demo5", "snapshots", CAT)
        versions = [d for d in os.listdir(snap_root) if d.startswith("v")]
        ck("快照版本存在", bool(versions), f"versions={versions}")
        vdir = os.path.join(snap_root, sorted(versions)[-1])
        ck("align_ref.npy 持久化", os.path.exists(os.path.join(vdir, "align_ref.npy")))

        # ── 3) M11b 归因透出 + M11d 对位预警 ─────────────
        print("[M11-3] 归因透出 + 对位预警", flush=True)
        r = client.post("/api/detect/image",
                        json={"path": defects[0], "category": CAT,
                              "with_heatmap": False}, headers=OP)
        assert r.status_code == 200, f"detect 失败: {r.text}"
        det = r.json()
        ck("响应含 defect_types", isinstance(det.get("defect_types"), list),
           f"types={det.get('defect_types')}")
        ck("响应含对位字段", "align_warn" in det and "align_offset" in det,
           f"align_offset={det.get('align_offset')}")
        if det.get("decision") != "normal":
            ck("非正常判定归因非空", len(det["defect_types"]) >= 1,
               f"decision={det['decision']} top={det['defect_types'][0].get('type')}")

        # 平移图触发对位预警（roll 40px @原图 → 256 参考系约 10px > warn_px 4）
        import numpy as np
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(good[0]).convert("RGB"))
        shifted = np.roll(arr, shift=40, axis=1)
        sh_path = os.path.join(TMP, "storage", "shifted.png")
        PILImage.fromarray(shifted).save(sh_path)
        r = client.post("/api/detect/image",
                        json={"path": sh_path, "category": CAT,
                              "with_heatmap": False}, headers=OP)
        assert r.status_code == 200, f"平移图 detect 失败: {r.text}"
        det2 = r.json()
        ck("平移图触发 align_warn", det2.get("align_warn") is True,
           f"offset={det2.get('align_offset')} warn={det2.get('align_warn')}")
        ck("偏移量量级正确", det2.get("align_offset")
           and abs(det2["align_offset"][0]) > 4.0,
           f"offset={det2.get('align_offset')}")
        # 原图不预警（对照）
        r = client.post("/api/detect/image",
                        json={"path": good[1], "category": CAT,
                              "with_heatmap": False}, headers=OP)
        det3 = r.json()
        ck("正常对位图不预警", det3.get("align_warn") is False,
           f"offset={det3.get('align_offset')}")

        # ── 4) M11e 日志轮转 ─────────────────────────────
        print("[M11-4] 日志轮转", flush=True)
        import logging
        import server
        server._setup_logging()
        server._setup_logging()   # 幂等：二次调用不重复挂
        rotating = [h for h in logging.getLogger().handlers
                    if getattr(h, "_aoi_rotating", False)]
        ck("RotatingFileHandler 挂载且幂等", len(rotating) == 1,
           f"n={len(rotating)}")
        logging.getLogger("m11_smoke").info("日志轮转写入验证")
        for h in rotating:
            h.flush()
        log_file = os.path.join(TMP, "storage", "logs", "server.log")
        ck("server.log 落盘", os.path.exists(log_file)
           and os.path.getsize(log_file) > 0)

    print(f"\n✅ M11 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
