"""P1 验证：用 SMT 测试图创建模板 → 传统引擎检测（NG 检出 / OK 通过）。

模板 = 金样板(OK 图) + smt_pads 标定 + solder 四类检测项。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inspect.service import InspectService  # noqa: E402
from app.template.store import TemplateStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
PAIR = SMT / "test_images" / "excess_pair" / "pair4"
STD = PAIR / "PixPin_2026-07-01_11-21-15_OK.png"
NG = PAIR / "PixPin_2026-07-01_11-21-15.png"
PAD_JSON = PAIR / "pad.json"


def pad_shapes() -> list[dict]:
    data = json.loads(PAD_JSON.read_text(encoding="utf-8"))
    shapes = []
    for sh in data.get("shapes", []):
        pts = sh.get("points") or []
        if sh.get("shape_type") == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            shapes.append({"shape": "rect", "x": float(x), "y": float(y),
                           "w": float(w), "h": float(h)})
    return shapes


def main() -> None:
    store = TemplateStore()
    existing = store.find_by_standard_path(STD)
    if existing is not None:
        tpl = existing
        print("复用已有模板:", tpl.id, "version", tpl.version)
    else:
        tpl = store.create_from_standard_image(
            STD,
            display_name="SMT 焊点示例",
            category="插件器件",
            algorithm_ids=["solder_多锡", "solder_少锡", "solder_连锡", "solder_虚焊"],
            copy_image=True,
        )
        tpl.set_region_shapes("smt_pads", pad_shapes())
        issues = store.publish_checklist(tpl)
        print("发布检查:", issues)
        store.publish(tpl)
        print("模板已发布:", tpl.id, "version", tpl.version)

    # 把该测试对的 params.json 冻结进模板（模板 = 完整配方：区域+算法+参数）
    params = json.loads((PAIR / "params.json").read_text(encoding="utf-8"))
    own_keys = {
        "solder_多锡": {"excess_thresh", "excess_min_area"},
        "solder_少锡": {"insufficient_thresh", "diff_insufficient_thresh", "insufficient_use_diff"},
        "solder_连锡": {"bridge_min_area"},
        "solder_虚焊": {"pin_type", "cold_solder_dark_ratio", "cold_solder_v_dark",
                        "cold_solder_rim_ratio", "toe_metal_ratio_thresh", "cold_diff_ratio_thresh"},
    }
    by_alg: dict[str, dict] = {}
    shared: dict = {}
    for k, v in params.items():
        hit = False
        for alg, keys in own_keys.items():
            if k in keys:
                by_alg.setdefault(alg, {})[k] = v
                hit = True
                break
        if not hit:
            shared[k] = v
    if shared:
        by_alg["solder_共用参数"] = shared
    for alg, p in by_alg.items():
        tpl.upsert_params(alg, p)
    store.save(tpl)
    print("模板参数已冻结:", {a: list(p) for a, p in by_alg.items()})

    svc = InspectService()
    print("region_sets keys:", {k: len(v) for k, v in (tpl.region_sets or {}).items()})
    print("smt_pads shapes:", tpl.get_region_shapes("smt_pads"))
    run = svc.run_with_template(tpl, str(NG), allow_stub=False)
    print("=== NG 图 ===")
    print("overall:", run.summary.overall, "| ng_count:", run.summary.ng_count)
    for r in run.summary.results:
        print(f"  [{r.status}] {r.display_name}: {r.message}  boxes={len(r.boxes)}  hit={r.hit_layer}")
    run_ok = svc.run_with_template(tpl, str(STD), allow_stub=False)
    print("=== OK 图 ===")
    print("overall:", run_ok.summary.overall)
    for r in run_ok.summary.results:
        print(f"  [{r.status}] {r.display_name}: {r.message}  boxes={len(r.boxes)}")

    assert run.summary.overall == "NG", "NG 图应判 NG"
    assert run_ok.summary.overall == "OK", "OK 图应判 OK"


if __name__ == "__main__":
    main()
