"""M10 冒烟：产线安全与集成（评审自检 §9 P0/P1/P2）。

覆盖：
1. API Key 鉴权（无头 401 / 错 Key 401 / 对 Key 200 / health 豁免）
2. /api/file 白名单（白名单外 403 / storage 内 200 / DB 登记路径 200）
3. PLC 触发检测（trigger_dir FIFO 取图 → 检测 → 移入 processed →
   OperationLog plc_trigger；空目录 404）
4. 缺陷自动归档（archive.decisions 命中后 storage/archive/{cat}/{date}/
   落盘 + Image split=archive 登记 + 当日"自动归档"批次）
5. 导出接口（/stats/export、/logs/export CSV 200 + 表头）

隔离环境：独立临时 sqlite/storage/cfg（security.api_key=test-key-m10、
plc.trigger_dir=tmp/trigger、archive.decisions=["anomaly","gray"]）。
归档测试用 gray 也归档的口径，规避小样例下 decision 不稳定。
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # AOI_sys 根

# ---- 隔离环境（必须在 import backend 之前设置 AOI_CONFIG）----
TMP = tempfile.mkdtemp(prefix="aoi_m10_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
TRIGGER_DIR = os.path.join(TMP, "trigger")
os.makedirs(TRIGGER_DIR, exist_ok=True)
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "system": {
            "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
            "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
            "engine_storage": os.path.join(TMP, "storage", "engine").replace("\\", "/"),
            "engine_base_cfg": "configs/engine_fast.yaml",
        },
        "security": {"api_key": "test-key-m10"},
        "archive": {"enabled": True, "decisions": ["anomaly", "gray"]},
        "plc": {"trigger_dir": TRIGGER_DIR.replace("\\", "/")},
    }, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

CAT = "smoke_m10"
# ---- 数据供给：自动探测可用数据集（MVTec-like 结构），无则合成兜底 ----
from tests.testdata import pick_category  # noqa: E402

DATA_DIR, _DATA_CAT, _SYNTHETIC = pick_category()
KEY = {"X-API-Key": "test-key-m10"}

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
        # ── 1) API Key 鉴权 ──────────────────────────────
        print("[M10-1] API Key 鉴权", flush=True)
        r = client.get("/api/categories")
        ck("无 Key 被拒", r.status_code == 401, f"status={r.status_code}")
        r = client.get("/api/categories", headers={"X-API-Key": "wrong"})
        ck("错 Key 被拒", r.status_code == 401)
        r = client.get("/api/categories", headers=KEY)
        ck("对 Key 放行", r.status_code == 200)
        r = client.get("/api/health")
        ck("health 豁免", r.status_code == 200)
        r = client.get("/docs")
        ck("docs 无 Key 被拒", r.status_code == 401)
        r = client.get("/openapi.json", headers=KEY)
        ck("openapi 正确 Key 放行", r.status_code == 200)
        from starlette.websockets import WebSocketDisconnect
        try:
            with client.websocket_connect("/ws/tasks"):
                unauthorized_ws = False
        except WebSocketDisconnect as exc:
            unauthorized_ws = exc.code == 4401
        ck("WebSocket 无 Key 被拒", unauthorized_ws)
        with client.websocket_connect("/ws/tasks", headers=KEY) as ws:
            ck("WebSocket 正确 Key 放行", ws is not None)

        # ── 2) /file 白名单 ──────────────────────────────
        print("[M10-2] /file 白名单", flush=True)
        outside = os.path.join(TMP, "secret.txt")
        with open(outside, "w") as f:
            f.write("x")
        r = client.get("/api/file", params={"path": outside}, headers=KEY)
        ck("白名单外 403", r.status_code == 403, f"status={r.status_code}")
        inside = os.path.join(TMP, "storage", "probe.txt")
        os.makedirs(os.path.dirname(inside), exist_ok=True)
        with open(inside, "w") as f:
            f.write("x")
        r = client.get("/api/file", params={"path": inside}, headers=KEY)
        ck("storage 内放行", r.status_code == 200, f"status={r.status_code}")

        # ── 3) 数据登记 + prepare（后续 PLC/归档要引擎）────
        print("[M10-3] 数据登记 + prepare", flush=True)
        from backend.db.database import session_scope
        from backend.db.models import Image as ImageRow
        good = _ls(os.path.join(DATA_DIR, "train", "good"), 10)
        defects = []
        for sub in sorted(os.listdir(os.path.join(DATA_DIR, "test"))):
            dd = os.path.join(DATA_DIR, "test", sub)
            if os.path.isdir(dd) and sub != "good":
                defects += _ls(dd, 3)  # 单缺陷类型数据集也能凑满 3 张
            if len(defects) >= 3:
                break
        defects = defects[:3]
        with session_scope() as s:
            for p in good:
                s.add(ImageRow(path=p, category=CAT, split="train",
                               label="normal", source="smoke"))
            for p in defects:
                s.add(ImageRow(path=p, category=CAT, split="train_anomaly",
                               label="anomaly", source="smoke"))
        # DB 登记路径应过 /file 白名单
        r = client.get("/api/file", params={"path": good[0]}, headers=KEY)
        ck("DB 登记路径放行", r.status_code == 200, f"status={r.status_code}")

        r = client.post("/api/models/prepare",
                        json={"category": CAT, "scenario": "L1a",
                              "profile": "fast", "force": True},
                        headers=KEY)
        assert r.status_code == 200, f"prepare 提交失败: {r.text}"
        t = wait_task(task_manager, r.json()["task_id"])
        ck("prepare 完成", t["status"] == "done", t.get("message", ""))

        # ── 4) PLC 触发检测 ──────────────────────────────
        print("[M10-4] PLC 触发", flush=True)
        trig_img = os.path.join(TRIGGER_DIR, "plc_001.png")
        shutil.copy2(defects[0], trig_img)
        r = client.post("/api/detect/trigger",
                        data={"category": CAT}, headers=KEY)
        ck("触发检测 200", r.status_code == 200,
           f"status={r.status_code} {r.text[:200] if r.status_code != 200 else ''}")
        out = r.json()
        ck("返回判定", out.get("decision") in ("normal", "gray", "anomaly"),
           f"decision={out.get('decision')} latency={out.get('latency_ms'):.0f}ms")
        ck("FIFO 后移入 processed", not os.path.exists(trig_img)
           and os.path.exists(os.path.join(TRIGGER_DIR, "processed",
                                           "plc_001.png")))
        r = client.post("/api/detect/trigger",
                        data={"category": CAT}, headers=KEY)
        ck("空目录 404", r.status_code == 404)
        r = client.get("/api/logs", params={"action": "plc_trigger"},
                       headers=KEY)
        ck("plc_trigger 审计日志", r.status_code == 200
           and len(r.json()["items"]) >= 1)

        # ── 5) 缺陷自动归档（decisions 含 gray，小样例确定性）──
        print("[M10-5] 缺陷自动归档", flush=True)
        r = client.post("/api/detect/image",
                        json={"path": defects[1], "category": CAT,
                              "with_heatmap": False}, headers=KEY)
        assert r.status_code == 200, f"detect 失败: {r.text}"
        det = r.json()
        arc_dir = os.path.join(TMP, "storage", "archive", CAT)
        ck("归档目录落盘", os.path.isdir(arc_dir)
           and any(os.listdir(os.path.join(arc_dir, d)) for d in os.listdir(arc_dir)),
           f"decision={det.get('decision')}")
        with session_scope() as s:
            row = (s.query(ImageRow)
                   .filter(ImageRow.category == CAT,
                           ImageRow.split == "archive").first())
        ck("归档 Image 登记", row is not None
           and row.source == "auto_archive" and row.label == "anomaly")
        r = client.get("/api/datasets", params={"category": CAT}, headers=KEY)
        names = [d["name"] for d in r.json()["items"]]
        ck("自动归档批次", any(n.startswith("自动归档") for n in names),
           f"datasets={names}")

        # ── 6) 导出接口 ──────────────────────────────────
        print("[M10-6] 产线日报/审计导出", flush=True)
        r = client.get("/api/stats/export", headers=KEY)
        ck("日报导出 CSV", r.status_code == 200
           and "n_inspected" in r.text and "defect_rate" in r.text)
        r = client.get("/api/logs/export",
                       params={"action": "plc_trigger"}, headers=KEY)
        ck("审计导出 CSV", r.status_code == 200
           and "plc_trigger" in r.text)
        r = client.get("/api/stats/export", params={"category": CAT},
                       headers=KEY)
        ck("品类过滤日报", r.status_code == 200 and CAT in r.headers.get(
            "Content-Disposition", ""))

    print(f"\n✅ M10 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
