"""检查 demo4 data_local 四品类 t/g 分支的 mean_good（=0 即配对泄露）"""
import json

path = r"d:\CGAIC\demo4\stage4\model_analysis\results\branches_v2_summary.json"
d = json.load(open(path, encoding="utf-8"))
for cat, cv in d["datasets"]["data_local"]["results"].items():
    print(f"{cat} (n={cv['n_stream']}, def={cv['n_defect']}, good={cv['n_good']}):")
    for br in ["g", "p", "r", "s", "t"]:
        b = cv["independent_auc"].get(br, {})
        print(f"  {br}: auroc={b.get('auroc'):.4f} "
              f"mean_def={b.get('mean_defect', 0):.3f} "
              f"mean_good={b.get('mean_good', 0):.3f} "
              f"std_good={b.get('std_good', 0):.3f}")
