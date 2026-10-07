"""F1 P0-P2 综合冒烟：异步任务/SSE/板级/不良过滤/告警/CAD/坐标/SPC。

依赖：server.py 已启动。
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8022/api"
ROOT = Path(__file__).resolve().parent.parent
NG = ROOT / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"


def main() -> None:
    print("1. health:", requests.get(f"{BASE}/health", timeout=5).json()["status"])
    tpls = requests.get(f"{BASE}/templates", timeout=5).json()["items"]
    tpl_id = tpls[0]["id"]
    print("2. 模板:", tpl_id)

    # 异步任务
    r = requests.post(f"{BASE}/detect/tasks",
                      json={"template_id": tpl_id, "image_path": str(NG)},
                      timeout=5).json()
    tid = r["task_id"]
    print("3. 异步任务提交:", tid, r["status"])
    deadline = time.time() + 60
    st = "pending"
    while time.time() < deadline:
        t = requests.get(f"{BASE}/detect/tasks/{tid}", timeout=5).json()
        st = t["status"]
        if st in ("done", "failed"):
            break
        time.sleep(1)
    print("   任务状态:", st, "| overall:", (t.get("result") or {}).get("overall"))
    assert st == "done"

    # 板级检测
    insps = requests.get(f"{BASE}/inspects", timeout=5).json()
    print("4. 板级检测:", insps["total"], "条")
    if insps["items"]:
        iid = insps["items"][0]["id"]
        d = requests.get(f"{BASE}/inspects/{iid}", timeout=5).json()
        print("   板详情: ng", d["ng_count"], "pass", d["pass"], "缺陷", len(d["defects"]),
              "| 首缺陷坐标", [(x.get('board_x'), x.get('board_y')) for x in d["defects"][:1]])

    # 不良过滤：过滤 area < 4000 的 AOI 缺陷
    requests.post(f"{BASE}/ng-filters",
                  json={"defect_type": "AOI", "filter_content": "area",
                        "filter_condition": "LT", "value_max": 4000}, timeout=5)
    rules = requests.get(f"{BASE}/ng-filters", timeout=5).json()
    print("5. 不良过滤规则:", [(x["defect_type"], x["filter_condition"]) for x in rules["items"]])

    # 告警
    requests.post(f"{BASE}/alarms", json={"alarm_type": "system", "level": "INFO",
                                          "content": "冒烟告警"}, timeout=5)
    alarms = requests.get(f"{BASE}/alarms", timeout=5).json()
    print("6. 告警记录:", alarms["total"], "条")

    # CAD 导入（临时 csv）
    cad_csv = ROOT / "storage" / "smoke_cad.csv"
    with open(cad_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["位号", "物料号", "封装", "X", "Y", "角度"])
        w.writerow(["R1", "1001", "0603", "10.0", "20.0", "0"])
        w.writerow(["C1", "2002", "0805", "15.0", "25.0", "90"])
    r = requests.post(f"{BASE}/templates/{tpl_id}/cad-import",
                      json={"cad_path": str(cad_csv)}, timeout=5).json()
    print("7. CAD 导入:", r["n_items"], "个器件")
    r = requests.post(f"{BASE}/templates/{tpl_id}/panels",
                      json={"panel_dx_mm": 50.0, "panel_dy_mm": 40.0,
                            "cols": 2, "rows": 2}, timeout=5).json()
    print("   拼板复制:", r["n_items"], "个（2x2）")
    r = requests.post(f"{BASE}/templates/{tpl_id}/fov",
                      json={"board_w_mm": 100.0, "board_h_mm": 80.0,
                            "fov_w_mm": 40.0, "fov_h_mm": 30.0, "overlap": 0.1},
                      timeout=5).json()
    print("   FOV 划分:", r["n_fov"], "个网格")

    # 坐标转换单测
    from app.core.coords import Calibration
    cal = Calibration(px_per_mm_x=10.0, px_per_mm_y=10.0, origin_x_mm=5.0,
                      origin_y_mm=5.0, img_origin_x=100.0, img_origin_y=100.0)
    bx, by = cal.px_to_board(200, 150)
    back = cal.board_to_px(bx, by)
    print("8. 坐标转换: px(200,150) -> mm", (bx, by), "-> px", back)
    assert abs(back[0] - 200) < 1e-3 and abs(back[1] - 150) < 1e-3

    # SPC
    wo = requests.get(f"{BASE}/workorders", timeout=5).json()["items"][0]
    spc = requests.get(f"{BASE}/workorders/{wo['id']}/spc-summary", timeout=5).json()
    print("9. SPC:", spc["total"], "检", spc["ng"], "NG", "| types:", list(spc["defect_types"].keys()))

    print("=== P0-P2 冒烟通过 ===")


if __name__ == "__main__":
    main()
