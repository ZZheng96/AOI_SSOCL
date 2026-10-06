"""前端反馈 v4 后端冒烟：条件下沉数据源 + 工单聚合 + 人工复判 + 模板。

隔离环境：独立临时 sqlite/storage/cfg；数据用 ORM 直插（能力统计只看 DB 记录）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="aoi_v4_")
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


def seed() -> dict:
    """直插数据源下的数据集/图片/检测记录，返回 id 映射。"""
    from backend.db.database import session_scope
    from backend.db.models import DataSource, Dataset, Detection, Image as ImageRow

    with session_scope() as s:
        # 数据源 A：声明 L1b（实际支撑 L1a + 模板图）
        a = DataSource(name="v4_srcA", modality="image", label_tier="L1b")
        s.add(a); s.flush()
        da = Dataset(name="批次A", category="catA", datasource_id=a.id)
        s.add(da); s.flush()
        for i, (lbl, split) in enumerate([
                ("normal", "train"), ("normal", "train"), ("normal", "train"),
                ("anomaly", "train"), ("anomaly", "train"),
                ("normal", "template")]):   # 模板图
            s.add(ImageRow(path=f"/tmp/v4/a_{i}.png", category="catA",
                           dataset_id=da.id, split=split, label=lbl))
        # 数据源 B：声明 L0（实际支撑 L0）
        b = DataSource(name="v4_srcB", modality="image", label_tier="L0")
        s.add(b); s.flush()
        db_ = Dataset(name="批次B", category="catB", datasource_id=b.id)
        s.add(db_); s.flush()
        for i in range(2):
            s.add(ImageRow(path=f"/tmp/v4/b_{i}.png", category="catB",
                           dataset_id=db_.id, split="train", label="normal"))
        s.flush()
        return {"A": a.id, "B": b.id, "da": da.id}


def seed_detections(cats: list) -> list:
    """直插检测记录（catA: 正常+异常各1；catB: 正常1），返回 id。

    复核提交会把检测图像登记到反馈库（register_image 需要真实文件），
    故 image_path 指向真实存在的临时文件。
    """
    from backend.db.database import session_scope
    from backend.db.models import Detection

    files = []
    for i in range(3):
        p = os.path.join(TMP, f"d{i}.png")
        with open(p, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        files.append(p)
    dets = [
        {"category": "catA", "is_anomaly": True, "decision": "anomaly"},
        {"category": "catA", "is_anomaly": False, "decision": "normal"},
        {"category": "catB", "is_anomaly": False, "decision": "normal"},
    ]
    with session_scope() as s:
        ids = []
        for i, d in enumerate(dets):
            row = Detection(
                target_type="pipeline", image_path=files[i],
                category=d["category"], final_score=0.9 if d["is_anomaly"] else 0.2,
                is_anomaly=d["is_anomaly"], latency_ms=10.0,
                n_tiles={"decision": d["decision"]})
            s.add(row); s.flush()
            ids.append(row.id)
        return ids


def main() -> None:
    from fastapi.testclient import TestClient
    from backend.api.app import app

    with TestClient(app) as client:
        ids = seed()

        # ── 1) 数据源体检：声明档位 vs 实际支撑 ──────
        print("[V1] 数据源：声明档位 + 体检校正", flush=True)
        r = client.get("/api/datasources")
        items = {d["name"]: d for d in r.json()["items"]}
        ck("列表含 label_tier 与 check", "label_tier" in items["v4_srcA"]
           and "check" in items["v4_srcA"], str(list(items["v4_srcA"])))
        ck("源A 支撑 L1a（有异常图）",
           items["v4_srcA"]["capability"]["tier"] == "L1a",
           str(items["v4_srcA"]["capability"]["tier"]))
        ck("源A 有模板图", items["v4_srcA"]["capability"]["n_templates"] >= 1,
           str(items["v4_srcA"]["capability"]["n_templates"]))
        ck("源A 声明 L1b 未匹配（高于支撑）",
           items["v4_srcA"]["check"]["matched"] is False,
           str(items["v4_srcA"]["check"]["declared"]))
        r = client.post(f"/api/datasources/{ids['A']}/check", params={"apply": "true"})
        ck("源A 体检校正为 L1a", r.json().get("label_tier") == "L1a"
           and (r.json().get("check") or {}).get("applied") == {"label_tier": "L1a"},
           str(r.json().get("label_tier")))

        # ── 2) 工单：条件自动聚合 ────────────────────
        print("[V2] 工单：数据源属性聚合", flush=True)
        r = client.post("/api/workorders",
                        json={"name": "v4_wo", "datasource_ids": [ids["A"], ids["B"]],
                              "review_enabled": True})
        ck("建工单挂双源+复判", r.status_code == 200, str(r.status_code))
        body = r.json()
        cond = body.get("conditions") or {}
        ck("聚合条件：档位 L1a（取源最高）", cond.get("label_tier") == "L1a",
           str(cond))
        ck("聚合条件：品类独立（源有品类）", cond.get("per_category") is True,
           str(cond))
        ck("聚合条件：模板比对（源A有模板图）", cond.get("has_template") is True,
           str(cond))
        ck("工单 review_enabled 落库", body.get("review_enabled") is True,
           str(body.get("review_enabled")))
        ck("聚合品类并集", sorted(body.get("categories")) == ["catA", "catB"],
           str(body.get("categories")))
        wid = body["id"]

        # ── 3) 人工复判：全部进队列 + 复核后才算 ──────
        print("[V3] 人工复判：队列 + 统计只认复核后", flush=True)
        dets = seed_detections(["catA", "catB"])
        r = client.get("/api/review/queue")
        items = r.json()
        ck("复判工单品类的全部检测进队列（3条）", items["total"] == 3,
           str(items["total"]))
        r = client.get("/api/workorders")
        wo = [w for w in r.json()["items"] if w["name"] == "v4_wo"][0]
        ck("未复核统计=0（复核后才算）", wo["stats"]["inspected"] == 0
           and wo["stats"]["anomaly"] == 0, str(wo["stats"]))
        # 复核 catA 异常检测
        r = client.post(f"/api/review/{dets[0]}", json={"label": 1})
        ck("复核提交生成反馈", r.status_code == 200
           and r.json().get("feedback_id"), str(r.status_code))
        r = client.get("/api/workorders")
        wo = [w for w in r.json()["items"] if w["name"] == "v4_wo"][0]
        ck("复核1条后统计=1", wo["stats"]["inspected"] == 1
           and wo["stats"]["anomaly"] == 1, str(wo["stats"]))
        r = client.get("/api/review/queue")
        ck("队列剩 2 条（1条已复核）", r.json()["total"] == 2, str(r.json()["total"]))

        # ── 4) 模板：存挂接+复判 → 从模板建单 ─────────
        print("[V4] 任务模板：复制配置", flush=True)
        r = client.post("/api/workorder-templates",
                        json={"name": "v4_tpl",
                              "datasource_ids": [ids["A"], ids["B"]],
                              "review_enabled": True})
        ck("存模板", r.status_code == 200, str(r.status_code))
        tid = r.json()["id"]
        r = client.post("/api/workorders",
                        json={"name": "v4_wo2", "from_template": tid})
        body = r.json()
        ck("从模板建单复制挂接", body.get("template_id") == tid
           and len(body.get("datasources", [])) == 2, str(body.get("datasources")))
        ck("从模板建单复制复判", body.get("review_enabled") is True,
           str(body.get("review_enabled")))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok} 项）", flush=True)


if __name__ == "__main__":
    main()
