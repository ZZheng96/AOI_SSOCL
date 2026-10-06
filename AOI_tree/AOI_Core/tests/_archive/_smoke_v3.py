"""前端反馈 v3 后端冒烟：数据源增删改查（编辑 PUT / 删除解挂）。

隔离环境：独立临时 sqlite/storage/cfg（同 _smoke_workorder 模式）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="aoi_v3_")
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

ok = 0


def ck(name, cond, detail=""):
    global ok
    assert cond, f"❌ {name}: {detail}"
    ok += 1
    print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)


def main() -> None:
    from fastapi.testclient import TestClient
    from backend.api.app import app

    with TestClient(app) as client:
        # ── 1) 数据源编辑（PUT） ─────────────────────
        print("[V1] 数据源编辑：改名/备注/采样/重名/404", flush=True)
        r = client.post("/api/datasources",
                        json={"name": "v3_src", "modality": "image",
                              "note": "旧备注"})
        ck("新建数据源", r.status_code == 200, str(r.status_code))
        sid = r.json()["id"]
        r = client.post("/api/datasources",
                        json={"name": "v3_src2", "modality": "video",
                              "sampling": {"frame_interval": 3}})
        sid2 = r.json()["id"]

        r = client.put(f"/api/datasources/{sid}",
                       json={"name": "v3_src_renamed", "note": "新备注",
                             "sampling": {"frame_interval": 7}})
        ck("编辑：改名+备注+采样",
           r.status_code == 200 and r.json().get("name") == "v3_src_renamed"
           and r.json().get("note") == "新备注"
           and (r.json().get("sampling") or {}).get("frame_interval") == 7,
           str(r.json()))
        ck("编辑后模态未变（锁定）", r.json().get("modality") == "image",
           str(r.json().get("modality")))
        r = client.put(f"/api/datasources/{sid}",
                       json={"name": "v3_src2"})
        ck("编辑重名 409", r.status_code == 409, str(r.status_code))
        r = client.put("/api/datasources/99999", json={"name": "x"})
        ck("编辑不存在 404", r.status_code == 404, str(r.status_code))
        r = client.get("/api/datasources")
        names = [d.get("name") for d in r.json().get("items", [])]
        ck("列表含新名/旧名不残留",
           "v3_src_renamed" in names and "v3_src" not in names, str(names))

        # ── 2) 删除解挂 ──────────────────────────────
        print("[V2] 删除数据源：解挂工单", flush=True)
        r = client.post("/api/workorders",
                        json={"name": "v3_wo", "label_tier": "L1a",
                              "datasource_ids": [sid, sid2]})
        ck("建工单挂双源", r.status_code == 200, str(r.status_code))
        r = client.delete(f"/api/datasources/{sid}")
        ck("删除数据源", r.status_code == 200 and r.json().get("deleted"),
           str(r.status_code))
        r = client.get("/api/workorders")
        wo = [w for w in r.json().get("items", []) if w.get("name") == "v3_wo"]
        ck("删源后工单剩余 1 源", wo and len(wo[0].get("datasources", [])) == 1
           and wo[0]["datasources"][0]["id"] == sid2,
           str(wo[0].get("datasources") if wo else None))
        r = client.delete(f"/api/datasources/{sid}")
        ck("删除不存在 404", r.status_code == 404, str(r.status_code))

        # ── 3) 新建工单数据源必选（后端契约） ─────────
        print("[V3] 工单创建：空数据源仍可建（UI 强制必选）", flush=True)
        r = client.post("/api/workorders", json={"name": "v3_wo2"})
        ck("无数据源建单（后端允许，UI 层拦截）",
           r.status_code == 200, str(r.status_code))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok} 项）", flush=True)


if __name__ == "__main__":
    main()
