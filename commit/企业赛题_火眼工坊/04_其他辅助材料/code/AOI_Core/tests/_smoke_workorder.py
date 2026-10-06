"""W-workorder 冒烟：工单制改造（前端反馈 v2）端到端验证。

覆盖：
1. 数据源：新建图片/视频源（视频含采样配置）、重名 409、列表附 capability。
2. 工单 CRUD + 体检：新建即体检（声明超数据支撑自动调低并记 applied）、
   range=today/7d/all 列表、单工单详情、改条件、只读体检、换挂数据源、重名 409。
3. 任务模板：存模板 → 从模板建工单（配置导入）→ 删模板（工单 template_id 置空）。
4. 工单口径统计：/workorders/{id}/stats?range= 三档 + /stats/overview?workorder_id=。
5. 数据树：/images/tree 含 source_groups（数据源→品类→批次四维）。

隔离环境：独立临时 sqlite/storage/cfg（同 _smoke_m16 模式），不碰真实库。

附带清库：--real-db-drop-legacy 对真实库 database/aoi.db 删旧
category_work_orders 表（旧品类工单脏数据；默认不执行）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # AOI_sys 根

TMP = tempfile.mkdtemp(prefix="aoi_wo_")
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

ok_count = 0


def ck(name, cond, detail=""):
    global ok_count
    assert cond, f"❌ {name}: {detail}"
    ok_count += 1
    print(f"  ✅ {name}" + (f"（{detail}）" if detail else ""), flush=True)


def drop_legacy_real_db() -> None:
    """清库：真实库删旧 category_work_orders 表（新模型已无该表）。"""
    import sqlite3
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "database", "aoi.db")
    if not os.path.exists(path):
        print(f"[清库] 真实库不存在，跳过：{path}")
        return
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    print(f"[清库] 真实库表（{len(tables)}）：{sorted(tables)}", flush=True)
    if "category_work_orders" in tables:
        n = cur.execute("SELECT COUNT(*) FROM category_work_orders").fetchone()[0]
        cur.execute("DROP TABLE category_work_orders")
        con.commit()
        print(f"[清库] 已删除旧表 category_work_orders（{n} 行脏数据）", flush=True)
    else:
        print("[清库] 无 category_work_orders 表，无需清理", flush=True)
    con.close()


def main() -> None:
    from fastapi.testclient import TestClient
    from backend.api.app import app

    with TestClient(app) as client:
        # ── 1) 数据源 ─────────────────────────────────
        print("[W-1] 数据源：新建/重名/列表 capability", flush=True)
        r = client.post("/api/datasources",
                        json={"name": "smoke_img_src", "modality": "image"})
        ck("新建图片数据源", r.status_code == 200, str(r.status_code))
        ds1 = r.json()["id"]
        r = client.post("/api/datasources",
                        json={"name": "smoke_vid_src", "modality": "video",
                              "sampling": {"frame_interval": 5}})
        ck("新建视频数据源（抽帧间隔）",
           r.status_code == 200
           and r.json().get("sampling", {}).get("frame_interval") == 5,
           str(r.json()))
        ds2 = r.json()["id"]
        r = client.post("/api/datasources", json={"name": "smoke_img_src"})
        ck("数据源重名 409", r.status_code == 409, str(r.status_code))
        r = client.get("/api/datasources")
        items = r.json().get("items", [])
        ck("数据源列表 2 条且附 capability",
           len(items) == 2 and all("capability" in d for d in items),
           f"n={len(items)}")

        # ── 2) 工单 CRUD + 体检 ───────────────────────
        print("[W-2] 工单：新建即体检/列表 range/改条件/换挂", flush=True)
        r = client.post("/api/workorders",
                        json={"name": "smoke_wo", "label_tier": "L1b",
                              "per_category": True, "has_template": True,
                              "datasource_ids": [ds1]})
        ck("新建工单（声明 L1b+两开关）", r.status_code == 200, str(r.status_code))
        body = r.json()
        chk = body.get("check") or {}
        ck("新建即体检：返回 check.declared/applied",
           "declared" in chk and "applied" in chk, str(list(body)))
        applied = chk.get("applied") or {}
        ck("空数据支撑 → 自动调低档位/关开关",
           applied.get("label_tier") == "L0"
           and applied.get("per_category") is False
           and applied.get("has_template") is False, str(applied))

        wid = None
        for rng in ("all", "today", "7d"):
            r = client.get("/api/workorders", params={"range": rng})
            items = r.json().get("items", [])
            ck(f"工单列表 range={rng}", r.status_code == 200 and len(items) == 1,
               f"n={len(items)}")
            if rng == "all":
                wid = items[0]["id"]
                ck("列表项含条件/数据源/统计/体检字段",
                   all(k in items[0] for k in
                       ("conditions", "conditions_cn", "datasources",
                        "stats", "check", "n_images")),
                   str(sorted(items[0])))
                ck("体检后条件已调低（L0/关/关）",
                   items[0]["conditions"] == {"label_tier": "L0",
                                              "per_category": False,
                                              "has_template": False},
                   str(items[0]["conditions"]))
        r = client.get(f"/api/workorders/{wid}")
        ck("单工单详情", r.status_code == 200 and r.json().get("id") == wid)
        r = client.post("/api/workorders", json={"name": "smoke_wo"})
        ck("工单重名 409", r.status_code == 409, str(r.status_code))

        r = client.put(f"/api/workorders/{wid}",
                       json={"label_tier": "L1a", "per_category": True})
        ck("改条件（L1a+品类独立）", r.status_code == 200, str(r.status_code))
        r = client.get(f"/api/workorders/{wid}")
        cond = r.json().get("conditions", {})
        ck("改条件生效", cond.get("label_tier") == "L1a"
           and cond.get("per_category") is True, str(cond))

        r = client.post(f"/api/workorders/{wid}/check",
                        params={"apply": "false"})
        body = r.json()
        chk = body.get("check") or {}
        ck("只读体检：capability/declared/suggestion/matched",
           all(k in chk for k in
               ("capability", "declared", "suggestion", "matched"))
           and not chk.get("applied"), str(list(chk)))
        ck("只读体检不改库（仍 L1a）", chk["declared"]["label_tier"] == "L1a",
           str(chk["declared"]))

        r = client.post(f"/api/workorders/{wid}/datasources",
                        json={"datasource_ids": [ds1, ds2]})
        ck("换挂数据源（图+视频双源）", r.status_code == 200, str(r.status_code))
        r = client.get(f"/api/workorders/{wid}")
        ck("挂接生效 2 源", len(r.json().get("datasources", [])) == 2,
           str(r.json().get("datasources")))

        # ── 3) 任务模板 ───────────────────────────────
        print("[W-3] 任务模板：存/导入/删", flush=True)
        r = client.post("/api/workorder-templates",
                        json={"name": "smoke_tpl", "label_tier": "L1a",
                              "per_category": True, "has_template": False})
        ck("存模板", r.status_code == 200
           and r.json().get("config", {}).get("label_tier") == "L1a",
           str(r.json()))
        tid = r.json()["id"]
        r = client.get("/api/workorder-templates")
        ck("模板列表", len(r.json().get("items", [])) == 1)
        r = client.post("/api/workorders",
                        json={"name": "smoke_wo2", "from_template": tid,
                              "label_tier": "L1b"})
        ck("从模板建工单（配置导入并关联 template_id）",
           r.status_code == 200 and r.json().get("template_id") == tid,
           str(r.json().get("template_id")))
        r = client.delete(f"/api/workorder-templates/{tid}")
        ck("删模板", r.status_code == 200 and r.json().get("deleted"))
        r = client.get("/api/workorders", params={"range": "all"})
        wo2 = [w for w in r.json()["items"] if w["name"] == "smoke_wo2"]
        ck("删模板后工单 template_id 置空",
           wo2 and wo2[0].get("template_id") is None, str(wo2))

        # ── 4) 工单口径统计 ───────────────────────────
        print("[W-4] 统计：工单 stats + overview workorder_id", flush=True)
        for rng in ("all", "today", "7d"):
            r = client.get(f"/api/workorders/{wid}/stats",
                           params={"range": rng})
            body = r.json()
            ck(f"工单统计 range={rng}（含 timeseries）",
               r.status_code == 200 and "stats" in body
               and "timeseries" in body, str(r.status_code))
        r = client.get("/api/stats/overview",
                       params={"workorder_id": wid, "range": "all"})
        body = r.json()
        ck("overview scope_stats（工单口径）",
           r.status_code == 200 and "scope_stats" in body
           and "categories" in body["scope_stats"], str(body.get("scope_stats")))

        # ── 5) 数据树四维 ─────────────────────────────
        print("[W-5] 数据树：source_groups", flush=True)
        r = client.get("/api/images/tree")
        ck("images/tree 含 source_groups",
           r.status_code == 200 and "source_groups" in r.json(),
           str(list(r.json())))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok_count} 项）", flush=True)


if __name__ == "__main__":
    if "--real-db-drop-legacy" in sys.argv:
        drop_legacy_real_db()
    main()
