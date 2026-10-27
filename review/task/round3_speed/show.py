"""打印 bench_<tag>.json 的分阶段汇总：python show.py opt1 [--rows]"""
import json, sys
from pathlib import Path

d = json.loads((Path(__file__).parent / f"bench_{sys.argv[1]}.json").read_text(encoding="utf-8"))
for cat, v in d["cats"].items():
    print(f"[{cat}]")
    for k, s in v["agg"].items():
        print(f"    {k:28s} p50={s['p50']:8.1f} p95={s['p95']:8.1f} max={s['max']:8.1f}")
    if "--rows" in sys.argv:
        for r in v["rows"]:
            tk = " ".join(f"{n}={t['run']:.0f}(+{t['wait']:.0f})" for n, t in r["tasks"].items())
            print(f"    {r['shape']} lat={r['latency_ms']:.0f} {tk} tm={r['timing']}")
