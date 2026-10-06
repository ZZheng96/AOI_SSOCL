"""M16 冒烟：用户旅程编排层（review_用户视角操作流 P0/P1）。

覆盖：
1. M16a 品类上线 readiness 状态机：empty → ready → trained → validated →
   active 五阶段迁移 + 下一步动作/导航页名。
2. M16d 设置中心：GET /system/config 白名单结构；POST 白名单键热更新并
   写回 yaml；非白名单键拒绝。
3. M16f 数据谱系：Model.metrics.dataset_ids → /datasets/{id} 反查
   used_by_models。
4. M16b 能力透出：/system/info 含 features 五开关。

隔离环境：独立临时 sqlite/storage/cfg（同 M14/M15 模式）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # AOI_sys 根

TMP = tempfile.mkdtemp(prefix="aoi_m16_")
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


def main():
    from fastapi.testclient import TestClient
    from backend.api.app import app
    from backend.db.database import session_scope
    from backend.db.models import Dataset, EvalRun, Image as ImageRow, Model

    CAT = "m16_cat"

    with TestClient(app) as client:
        # ── 1) M16a readiness 状态机迁移 ─────────────────
        print("[M16-1] 品类上线 readiness 状态机", flush=True)

        def _stage():
            r = client.get("/api/models/readiness", params={"category": CAT})
            assert r.status_code == 200, r.text[:200]
            return r.json()["items"][0]

        it = _stage()
        ck("空品类 → empty", it["stage"] == "empty"
           and it["next_page"] == "数据管理", it["stage_cn"])

        with session_scope() as s:
            ds = Dataset(name="m16批次", category=CAT, source_type="folder")
            s.add(ds)
            s.flush()
            ds_id = ds.id
            for i in range(3):
                s.add(ImageRow(path=f"{TMP}/n{i}.png", category=CAT,
                               split="train", label="normal",
                               dataset_id=ds_id))
            s.flush()
        it = _stage()
        ck("3 张正常图 → ready", it["stage"] == "ready"
           and it["next_page"] == "模型管理", it["stage_cn"])

        with session_scope() as s:
            m = Model(name=f"{CAT}_v1", level="0", category=CAT,
                      version="v1", path=f"{TMP}/snapshots/{CAT}/v1",
                      format="demo5_snapshot", origin="demo5_fit",
                      metrics={"dataset_ids": [ds_id]}, is_active=False)
            s.add(m)
            s.flush()
            model_id = m.id
        it = _stage()
        ck("有版本未评估 → trained", it["stage"] == "trained"
           and it["next_page"] == "评估看板", it["stage_cn"])

        with session_scope() as s:
            s.add(EvalRun(name="m16验收", run_type="accuracy", category=CAT,
                          metrics={"auroc": 0.93}))
            s.flush()
        it = _stage()
        ck("有验收未激活 → validated + best_auroc",
           it["stage"] == "validated" and abs(it["best_auroc"] - 0.93) < 1e-6
           and it["next_page"] == "模型管理",
           f"{it['stage_cn']} auroc={it['best_auroc']}")

        with session_scope() as s:
            s.get(Model, model_id).is_active = True
            s.flush()
        it = _stage()
        ck("激活 → active（优先于验收状态）", it["stage"] == "active"
           and it["active_version"] == "v1"
           and it["next_page"] == "实时监控", it["stage_cn"])

        # ── 2) M16d 设置中心 ─────────────────────────────
        print("[M16-2] 设置中心白名单配置", flush=True)
        r = client.get("/api/system/config")
        ck("GET /system/config", r.status_code == 200
           and r.json().get("items"), f"{r.status_code}")
        keys = {f"{i['section']}.{i['key']}" for i in r.json()["items"]}
        ck("白名单含延迟预算/基准样本数",
           {"pipeline.latency_budget_ms",
            "evaluation.benchmark_n_images"} <= keys,
           f"n={len(keys)}")
        ck("config_path 透出", r.json().get("config_path", "").endswith("cfg.yaml"))

        r = client.post("/api/system/config", json={
            "evaluation.benchmark_n_images": 15,
            "security.api_key": "hack",       # 非白名单必须拒绝
            "badformat": 1})                  # 无 section 必须拒绝
        body = r.json()
        ck("白名单键应用", "evaluation.benchmark_n_images"
           in (body.get("applied") or []), str(body))
        ck("非白名单/坏格式拒绝",
           set(body.get("rejected") or []) >= {"security.api_key", "badformat"},
           str(body.get("rejected")))
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            persisted = yaml.safe_load(f)
        ck("写回 yaml 持久化",
           (persisted.get("evaluation") or {}).get("benchmark_n_images") == 15)
        r = client.get("/api/system/config")
        val = next(i["value"] for i in r.json()["items"]
                   if i["section"] == "evaluation"
                   and i["key"] == "benchmark_n_images")
        ck("GET 回读热生效", val == 15, f"value={val}")

        # ── 3) M16f 数据谱系反查 ─────────────────────────
        print("[M16-3] 数据谱系", flush=True)
        r = client.get(f"/api/datasets/{ds_id}")
        used = r.json().get("used_by_models") or []
        ck("批次详情反查参与版本", len(used) == 1
           and used[0]["version"] == "v1" and used[0]["is_active"],
           str(used))
        r = client.get("/api/datasets/99999")
        ck("不存在批次 404", r.status_code == 404)

        # ── 4) M16b 能力开关透出 ─────────────────────────
        print("[M16-4] 能力开关透出", flush=True)
        r = client.get("/api/system/info")
        feats = r.json().get("features") or {}
        ck("features 五开关", set(feats) >= {"webhook", "plc_trigger",
           "auto_archive", "auth", "api_docs"}, str(sorted(feats)))
        ck("默认配置：鉴权关/文档开",
           feats.get("auth") is False and feats.get("api_docs") is True,
           str(feats))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n✅ M16 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
