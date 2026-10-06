"""P3 验证：REST API 冒烟 —— health / templates / detect/dual。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "http://127.0.0.1:8021/api"
ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
NG = SMT / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"
OK = SMT / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15_OK.png"


def main() -> None:
    print("health:", requests.get(f"{BASE}/health", timeout=5).json())

    r = requests.get(f"{BASE}/templates", timeout=5)
    templates = r.json()["items"]
    print("templates:", [(t["id"], t["status"], t.get("engine_mode")) for t in templates])
    assert templates, "无模板"

    tpl = templates[0]
    # 更新模板 engine_mode / model_category
    r = requests.put(f"{BASE}/templates/{tpl['id']}",
                     json={"engine_mode": "dual", "model_category": "smt_solder"}, timeout=5)
    print("update template:", r.status_code, r.json().get("engine_mode"), r.json().get("model_category"))

    # 双检 NG
    r = requests.post(f"{BASE}/detect/dual",
                      json={"template_id": tpl["id"], "image_path": str(NG)}, timeout=60)
    print("dual NG  ->", r.status_code, r.json()["overall"],
          "| trad:", r.json()["traditional"]["ng_count"],
          "| feat:", (r.json().get("feature") or {}).get("decision"),
          "| elapsed:", r.json()["elapsed_ms"], "ms")
    # 双检 OK
    r = requests.post(f"{BASE}/detect/dual",
                      json={"template_id": tpl["id"], "image_path": str(OK)}, timeout=60)
    print("dual OK  ->", r.status_code, r.json()["overall"],
          "| feat:", (r.json().get("feature") or {}).get("decision"))

    # models
    r = requests.get(f"{BASE}/models", timeout=5)
    print("models:", [(c["category"], c["current"]) for c in r.json()["categories"]])

    assert requests.post(f"{BASE}/detect/dual",
                         json={"template_id": tpl["id"], "image_path": str(NG)},
                         timeout=60).json()["overall"] == "NG"


if __name__ == "__main__":
    main()
