"""F1 冒烟：完整业务闭环 —— 双检落库 → 反馈即学 → 工单/数据源/产线。

依赖：server.py 已启动。
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8021/api"
ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
PAIR = SMT / "test_images" / "excess_pair" / "pair4"
NG = PAIR / "PixPin_2026-07-01_11-21-15.png"
STD = PAIR / "PixPin_2026-07-01_11-21-15_OK.png"
OK_DIR = SMT / "test_images" / "excess_pair" / "pair4"


def main() -> None:
    print("1. health:", requests.get(f"{BASE}/health", timeout=5).json()["status"])

    # 模板
    tpls = requests.get(f"{BASE}/templates", timeout=5).json()["items"]
    assert tpls, "无模板"
    tpl_id = tpls[0]["id"]
    requests.put(f"{BASE}/templates/{tpl_id}",
                 json={"engine_mode": "dual", "model_category": "smt_solder"}, timeout=5)
    print("2. 模板:", tpl_id, "已绑定 smt_solder")

    # 双检 + 落库
    r = requests.post(f"{BASE}/detect/dual",
                      json={"template_id": tpl_id, "image_path": str(NG)}, timeout=60).json()
    print("3. 双检 NG:", r["overall"], "| detection_id:", r.get("detection_id"))
    det_id = r.get("detection_id")
    assert det_id, "双检应落库"

    # 检测记录查询
    dets = requests.get(f"{BASE}/detections", timeout=5).json()
    print("4. 检测记录:", dets["total"], "条")
    d = requests.get(f"{BASE}/detections/{det_id}", timeout=5).json()
    print("   详情:", d["engine_mode"], d["traditional_overall"], d["feature_decision"])

    # 反馈即学（该 NG 图确认缺陷 → 学习）
    fb = requests.post(f"{BASE}/feedback",
                       json={"detection_id": det_id, "verdict": "correct",
                             "label": 1, "feedback_type": "confirmed"}, timeout=60).json()
    print("5. 反馈即学:", fb.get("feedback_ms"), "ms |", fb.get("action", "ok"))

    # 学习状态
    ls = requests.get(f"{BASE}/models/smt_solder/learning", timeout=5).json()
    print("6. 学习状态: normal_bank", ls["normal_bank"], "| defect_bank", ls["defect_bank"])

    # 数据源 + 导入
    src_name = f"smoke_src_{Path(NG).parent.name}"
    srcs = requests.get(f"{BASE}/datasources", timeout=5).json()["items"]
    ds = next((s for s in srcs if s["name"] == src_name), None)
    if ds is None:
        ds = requests.post(f"{BASE}/datasources",
                           json={"name": src_name, "modality": "image",
                                 "label_tier": "L1a",
                                 "pretrain_normal": 100, "pretrain_anomaly": 30,
                                 "batch_size": 30}, timeout=5).json()
    imp = requests.post(f"{BASE}/datasources/{ds['id']}/import",
                        json={"folder": str(OK_DIR), "category": "smt_solder"},
                        timeout=30).json()
    print("7. 数据源导入:", imp.get("imported"), "张 /", imp.get("total"))

    # 工单
    wo_name = "smoke_工单_1"
    wos = requests.get(f"{BASE}/workorders", timeout=5).json()["items"]
    wo = next((w for w in wos if w["name"] == wo_name), None)
    if wo is None:
        wo = requests.post(f"{BASE}/workorders",
                           json={"name": wo_name, "template_ref": tpl_id,
                                 "sources": [ds["id"]], "label_tier": "L1a"},
                           timeout=5).json()
    print("8. 工单:", wo["name"], "| pipeline:", wo["pipeline_status"],
          "| sources:", len(wo["sources"]))

    # 暂停/恢复
    requests.post(f"{BASE}/workorders/{wo['id']}/pause", timeout=5)
    paused = requests.get(f"{BASE}/workorders/{wo['id']}", timeout=5).json()
    print("9. 暂停:", paused["pipeline_status"])
    requests.post(f"{BASE}/workorders/{wo['id']}/resume", timeout=5)

    # 回队
    batches = requests.get(f"{BASE}/datasources/{ds['id']}", timeout=5).json()["batches"]
    if batches:
        rq = requests.post(f"{BASE}/workorders/{wo['id']}/requeue",
                           json={"dataset_id": batches[0]["id"]}, timeout=5).json()
        print("10. 回队:", rq.get("n_images"), "张待重检")

    # 统计
    dets2 = requests.get(f"{BASE}/detections", timeout=5).json()
    print("11. 累计检测记录:", dets2["total"], "条")
    print("=== 冒烟通过 ===")


if __name__ == "__main__":
    main()
