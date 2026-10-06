import json
for name in ["branches_v2_summary", "branches_summary", "branches_v2_noref_summary"]:
    p = rf"d:\CGAIC\demo4\stage4\model_analysis\results\{name}.json"
    try:
        d = json.load(open(p, encoding="utf-8"))
    except FileNotFoundError:
        print("missing:", name)
        continue
    print("#" * 20, name)
    for ds, v in d["datasets"].items():
        s = v.get("summary", {})
        print("===", ds, v.get("categories"))
        print("  avg_indep:", {k: round(x, 4) for k, x in s.get("avg_independent_auc", {}).items()})
        for cat, r in v.get("results", {}).items():
            loo = r.get("leave_one_out", {})
            indep = ",".join(f"{b}{round(m['auroc'], 3)}"
                             for b, m in r.get("independent_auc", {}).items())
            print(f"  {cat}: n={r.get('n_stream')} fused={round(loo.get('pipeline_fused_auc', 0), 4)} [{indep}]")
