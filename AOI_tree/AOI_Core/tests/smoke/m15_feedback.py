"""M15 冒烟：用户操作流对账补全（用户操作流/外部认识两文档）。

覆盖：
1. M15a "无法确认"反馈类型：feedback_type=uncertain → verdict=none、
   label=None、判对率曲线不计入、图像登记 label=unknown。
2. M15b 主动选样清单：/review/active_suggestions 未准备品类空态 +
   响应结构。
3. M15c 角色透出：/system/info 含 role 字段（鉴权关闭默认 admin）。
4. M15d 导出端点：/stats/export、/logs/export 返回 BOM 头 text/csv。

隔离环境：独立临时 sqlite/storage/cfg（同 M10/M11/M14 模式）。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # AOI_sys 根

TMP = tempfile.mkdtemp(prefix="aoi_m15_")
CFG_PATH = os.path.join(TMP, "cfg.yaml")
import yaml  # noqa: E402

with open(CFG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "system": {
            "database_url": "sqlite:///" + os.path.join(TMP, "aoi.db").replace("\\", "/"),
            "storage_dir": os.path.join(TMP, "storage").replace("\\", "/"),
            "engine_storage": os.path.join(TMP, "storage", "engine").replace("\\", "/"),
            "engine_base_cfg": "configs/engine_fast.yaml",
        },
        "archive": {"enabled": False},
    }, f, allow_unicode=True)
os.environ["AOI_CONFIG"] = CFG_PATH

# ---- 数据供给：自动探测可用数据集（MVTec-like 结构），无则合成兜底 ----
from tests.testdata import first_normal_image  # noqa: E402

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
    from backend.db.models import Detection, Feedback, Image as ImageRow

    CAT = "m15_cat"
    img_path = first_normal_image()

    with TestClient(app) as client:
        with session_scope() as s:
            det = Detection(category=CAT, image_path=img_path,
                            final_score=0.5, is_anomaly=False,
                            latency_ms=100.0,
                            n_tiles={"decision": "gray"})
            s.add(det)
            s.flush()
            det_id = det.id

        # ── 1) M15a uncertain 反馈 ───────────────────────
        print("[M15-1] 无法确认反馈类型", flush=True)
        r = client.post("/api/feedback", json={
            "detection_id": det_id, "feedback_type": "uncertain",
            "operator_label": -1, "operator": "smoke"})
        ck("uncertain 反馈落库", r.status_code == 200, f"{r.status_code}")
        body = r.json()
        # 引擎未 prepare → engine_update None + note，但落库成功
        ck("引擎未就绪不中断", body.get("engine_update") is None
           and body.get("note"), body.get("note", ""))
        with session_scope() as s:
            fb = s.query(Feedback).filter(Feedback.detection_id == det_id).first()
            ck("Feedback 类型记录", fb.feedback_type == "uncertain")
            img = s.query(ImageRow).filter(ImageRow.path == img_path).first()
            ck("反馈图登记 label=unknown",
               img is not None and img.label == "unknown",
               f"label={img.label if img else None}")
        # 判对率曲线不计入 uncertain
        r = client.get(f"/api/learning/online_curve/{CAT}")
        if r.status_code == 200:
            pts = r.json().get("points", [])
            ck("判对率曲线剔除 uncertain", len(pts) == 0, f"points={len(pts)}")

        # ── 2) M15b 主动选样清单 ─────────────────────────
        print("[M15-2] 主动选样清单", flush=True)
        r = client.get("/api/review/active_suggestions",
                       params={"category": CAT})
        ck("active_suggestions 空态", r.status_code == 200
           and r.json().get("items") == [], f"{r.status_code} {r.text[:120]}")

        # ── 3) M15c 角色透出 ─────────────────────────────
        print("[M15-3] 角色透出", flush=True)
        r = client.get("/api/system/info")
        ck("system/info 含 role", r.json().get("role") == "admin",
           f"role={r.json().get('role')}")

        # ── 4) M15d 导出端点 ─────────────────────────────
        print("[M15-4] 产线导出端点", flush=True)
        r = client.get("/api/stats/export")
        ck("日报 CSV", r.status_code == 200
           and r.content.startswith("﻿".encode("utf-8")),
           f"{r.status_code}")
        r = client.get("/api/logs/export")
        ck("审计 CSV", r.status_code == 200
           and "text/csv" in r.headers.get("content-type", ""))

    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n✅ M15 冒烟通过（{ok_count} 项断言）", flush=True)


if __name__ == "__main__":
    main()
