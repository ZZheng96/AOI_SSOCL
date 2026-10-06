"""前端反馈 v6 后端冒烟：数据源声明（分品类/提供模板）+ 聚合 + 体检警告。

隔离环境：独立临时 sqlite/storage/cfg。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="aoi_v6_")
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
        # ── 1) 建源声明：分品类 + 提供模板（无模板图数据） ──
        print("[V1] 数据源声明：per_category / has_template", flush=True)
        r = client.post("/api/datasources",
                        json={"name": "v6_src", "modality": "image",
                              "label_tier": "L0",
                              "per_category": True, "has_template": True})
        ck("建源声明分品类+模板", r.status_code == 200
           and r.json().get("per_category") is True
           and r.json().get("has_template") is True, str(r.status_code))
        sid = r.json()["id"]
        r = client.get("/api/datasources")
        d = [x for x in r.json()["items"] if x["name"] == "v6_src"][0]
        ck("列表附声明与 check", d.get("per_category") is True
           and "check" in d and d["check"]["declared"].get("has_template") is True,
           str(d.get("check", {}).get("declared")))
        ck("体检警告：声明模板但无模板图",
           any("模板图" in w for w in d["check"]["warnings"]),
           str(d["check"]["warnings"]))
        ck("体检警告：声明分品类但无品类说明",
           any("品类" in w for w in d["check"]["warnings"]),
           str(d["check"]["warnings"]))

        # ── 2) 工单聚合：声明参与聚合 ──────────────────
        print("[V2] 工单聚合：声明（无数据）参与 per_category/has_template", flush=True)
        r = client.post("/api/workorders",
                        json={"name": "v6_wo", "datasource_ids": [sid]})
        cond = r.json().get("conditions") or {}
        ck("聚合品类独立=声明", cond.get("per_category") is True, str(cond))
        ck("聚合模板比对=声明", cond.get("has_template") is True, str(cond))

        # ── 3) 编辑源声明 ──────────────────────────────
        print("[V3] 编辑源声明", flush=True)
        r = client.put(f"/api/datasources/{sid}",
                       json={"has_template": False, "per_category": False})
        ck("编辑关闭声明", r.json().get("has_template") is False
           and r.json().get("per_category") is False, str(r.json()))
        r = client.get("/api/workorders")
        wo = [w for w in r.json()["items"] if w["name"] == "v6_wo"][0]
        cond = wo.get("conditions") or {}
        ck("编辑后聚合回退（无数据）", cond.get("per_category") is False
           and cond.get("has_template") is False, str(cond))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok} 项）", flush=True)


if __name__ == "__main__":
    main()
