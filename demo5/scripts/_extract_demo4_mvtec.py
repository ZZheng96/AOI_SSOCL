"""查 demo4 branches_v2_summary 的 mvtec 各品类 t 分支成绩"""
import json

d = json.load(open(r"d:\CGAIC\demo4\stage4\model_analysis\results\branches_v2_summary.json",
                   encoding="utf-8"))
for ds_name, ds in d["datasets"].items():
    if "mvtec" not in ds_name:
        continue
    print(f"===== {ds_name} =====")
    for cat, cv in ds["results"].items():
        b = cv.get("independent_auc", {})
        fused = cv.get("pipeline_fused_auc")
        print(f"  {cat}: fused={fused} "
              + " ".join(f"{k}={v.get('auroc', '?') if isinstance(v, dict) else '?'}"
                         for k, v in b.items()))
