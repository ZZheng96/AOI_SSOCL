"""P3 验证：双检闭环 —— 同一张图传统算法 + 特征学习并行 → 融合判定。

模板绑定品类模型：PixPin 模板（SMT 焊点）绑定 model_category=smt_solder。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inspect.service import InspectService  # noqa: E402
from app.template.store import TemplateStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SMT = ROOT / "alg_repo" / "smt"
PAIR = SMT / "test_images" / "excess_pair" / "pair4"
STD = PAIR / "PixPin_2026-07-01_11-21-15_OK.png"
NG = PAIR / "PixPin_2026-07-01_11-21-15.png"


def main() -> None:
    store = TemplateStore()
    tpl = store.find_by_standard_path(STD)
    assert tpl is not None, "模板不存在，先跑 _verify_traditional.py"
    if tpl.model_category != "smt_solder" or tpl.engine_mode != "dual":
        tpl.model_category = "smt_solder"
        tpl.engine_mode = "dual"
        store.save(tpl)
        print("模板已绑定品类模型 smt_solder（engine_mode=dual）")

    svc = InspectService()
    print("=== 双检：NG 图 ===")
    ds = svc.run_dual(tpl, str(NG), allow_stub=False)
    print("overall:", ds.overall, "| gray:", ds.gray, "| elapsed:", ds.elapsed_ms, "ms")
    trad = ds.traditional
    print("  [传统]", trad.overall, "ng=", trad.ng_count,
          "|", [(r.display_name, r.status, r.defect_count) for r in trad.results])
    if ds.feature:
        print("  [特征] decision=", ds.feature["decision"],
              "score=", ds.feature["score"],
              "slots=", ds.feature["slot_scores"])
    else:
        print("  [特征]", ds.feature_error)
    print("  融合框:", [(b["label"], b["x"], b["y"], b["w"], b["h"]) for b in ds.boxes])

    print("=== 双检：OK 图 ===")
    ds2 = svc.run_dual(tpl, str(STD), allow_stub=False)
    print("overall:", ds2.overall, "| gray:", ds2.gray)
    if ds2.feature:
        print("  [特征] decision=", ds2.feature["decision"], "score=", ds2.feature["score"])
    print("  融合框数:", len(ds2.boxes))

    assert ds.overall == "NG", "NG 图双检应判 NG"


if __name__ == "__main__":
    main()
