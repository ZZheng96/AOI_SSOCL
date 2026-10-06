"""前端反馈 v5 后端冒烟：产线状态/暂停恢复/数据流队列/回队/学习提升。

隔离环境：独立临时 sqlite/storage/cfg；数据 ORM 直插（产线消费依赖引擎
就绪，冒烟只验证状态机与 API 契约）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="aoi_v5_")
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
    from backend.db.database import session_scope
    from backend.db.models import DataSource, Dataset, Detection, Image as ImageRow

    with session_scope() as s:
        # 本地源 + 批次 + 图片（正常图供队列展示）
        # pretrain_normal=0：让 2 张图全部进检测组，形成 1 个检测批次（30图/批）
        a = DataSource(name="v5_srcA", modality="image", label_tier="L0",
                       source_type="local", pretrain_normal=0,
                       pretrain_anomaly=0)
        s.add(a); s.flush()
        da = Dataset(name="批次A", category="catA", datasource_id=a.id)
        s.add(da); s.flush()
        imgs = []
        for i in range(2):
            im = ImageRow(path=os.path.join(TMP, f"a{i}.png"), category="catA",
                          dataset_id=da.id, split="train", label="normal")
            s.add(im); imgs.append(im)
        s.flush()
        # 实时源（视频文件循环模拟）
        vid = os.path.join(TMP, "sim.mp4")
        with open(vid, "wb") as fh:
            fh.write(b"\x00" * 128)
        b = DataSource(name="v5_srcRTSP", modality="video", label_tier="L0",
                       source_type="rtsp",
                       stream_config={"path": vid, "category": "catA"})
        s.add(b); s.flush()
        return {"A": a.id, "B": b.id, "da": da.id, "img": imgs[0].id}


def main() -> None:
    from fastapi.testclient import TestClient
    from backend.api.app import app

    with TestClient(app) as client:
        ids = seed()

        # ── 1) 数据源数据流类型 ───────────────────────
        print("[V1] 数据源：source_type/stream_config", flush=True)
        r = client.get("/api/datasources")
        items = {d["name"]: d for d in r.json()["items"]}
        ck("本地源 source_type=local", items["v5_srcA"]["source_type"] == "local")
        ck("实时源 source_type=rtsp + 视频模拟",
           items["v5_srcRTSP"]["source_type"] == "rtsp"
           and (items["v5_srcRTSP"]["stream_config"] or {}).get("path"),
           str(items["v5_srcRTSP"].get("stream_config")))

        # ── 2) 工单产线：状态/暂停/恢复 ────────────────
        print("[V2] 工单产线：pipeline 控制", flush=True)
        r = client.post("/api/workorders",
                        json={"name": "v5_wo", "datasource_ids": [ids["A"], ids["B"]],
                              "auto_resume": True})
        ck("建工单默认 running + auto_resume", r.json().get("pipeline_status") == "running"
           and r.json().get("auto_resume") is True,
           str((r.json().get("pipeline_status"), r.json().get("auto_resume"))))
        wid = r.json()["id"]
        r = client.post(f"/api/workorders/{wid}/pipeline/pause")
        ck("暂停产线", r.json().get("pipeline_status") == "paused")
        r = client.post(f"/api/workorders/{wid}/pipeline/resume")
        ck("恢复产线（不重启不重配）", r.json().get("pipeline_status") == "running")
        r = client.post(f"/api/workorders/{wid}/pipeline/xyz")
        ck("非法动作 422", r.status_code == 422, str(r.status_code))

        # ── 3) 数据流队列：批次/错检统计/回队 ──────────
        print("[V3] 数据流队列：队列/回队/取消", flush=True)
        r = client.get(f"/api/workorders/{wid}/queue")
        q = r.json()
        ck("队列含检测批次 + 统计 + 回队标记",
           q.get("pipeline_status") == "running" and len(q.get("items", [])) >= 1
           and "bad_rate" in q["items"][0] and q["items"][0]["requeued"] is False,
           str(q))
        key = q["items"][0]["batch_key"]
        r = client.post(f"/api/workorders/{wid}/queue/requeue",
                        json={"batch_keys": [key]})
        ck("错检批次回队", r.json().get("requeued") == [key], str(r.json()))
        r = client.get(f"/api/workorders/{wid}/queue")
        ck("回队标记生效", r.json()["items"][0]["requeued"] is True)
        r = client.request("DELETE", f"/api/workorders/{wid}/queue/requeue",
                           json={"batch_keys": [key]})
        ck("取消回队", r.status_code == 200
           and r.json().get("requeued") == [], str(r.json()))

        # ── 4) 学习提升 ────────────────────────────────
        print("[V4] 学习提升：consolidate + auto_resume", flush=True)
        r = client.post(f"/api/workorders/{wid}/learn")
        body = r.json()
        ck("learn 返回契约", "applied_categories" in body
           and "resumed" in body and "message" in body, str(body))
        # auto_resume=True 且未准备品类：consolidate 失败但 resumed 状态保持

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n全部通过 ✅（{ok} 项）", flush=True)


if __name__ == "__main__":
    main()
