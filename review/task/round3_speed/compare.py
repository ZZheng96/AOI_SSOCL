"""逐图精度对照：python compare.py base opt1 → decision 一致性 + score/raw/slot 最大绝对差 + P95 对比。"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
a = json.loads((HERE / f"bench_{sys.argv[1]}.json").read_text(encoding="utf-8"))
b = json.loads((HERE / f"bench_{sys.argv[2]}.json").read_text(encoding="utf-8"))


def maxdiff(x, y):
    if not isinstance(x, dict) or not isinstance(y, dict):
        return 0.0
    m = 0.0
    for k in set(x) | set(y):
        vx, vy = x.get(k), y.get(k)
        if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
            m = max(m, abs(vx - vy))
    return m


for cat in a["cats"]:
    if cat not in b["cats"]:
        continue
    ra = {(r["image"], r["rep"]): r for r in a["cats"][cat]["rows"]}
    rb = {(r["image"], r["rep"]): r for r in b["cats"][cat]["rows"]}
    keys = sorted(set(ra) & set(rb))
    mism = [k[0] for k in keys if ra[k]["decision"] != rb[k]["decision"]]
    ds = max(abs(ra[k]["score"] - rb[k]["score"]) for k in keys)
    dr = max(maxdiff(ra[k]["raw_scores"], rb[k]["raw_scores"]) for k in keys)
    dsl = max(maxdiff(ra[k]["slot_scores"], rb[k]["slot_scores"]) for k in keys)
    nb = sum(ra[k]["n_boxes"] != rb[k]["n_boxes"] for k in keys)
    la, lb = a["cats"][cat]["agg"]["latency_ms"], b["cats"][cat]["agg"]["latency_ms"]
    print(f"[{cat}] n={len(keys)} decision_mismatch={len(mism)} {mism[:5]} "
          f"max|Δscore|={ds:.3g} max|Δraw|={dr:.3g} max|Δslot|={dsl:.3g} n_boxes_diff={nb}")
    print(f"    latency {sys.argv[1]}: p50={la['p50']} p95={la['p95']} max={la['max']}  →  "
          f"{sys.argv[2]}: p50={lb['p50']} p95={lb['p95']} max={lb['max']}")
