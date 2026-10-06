"""M4 CPU <2s 降配实测（§6.1 挑战目标）：纯手工槽位配置，CPU 推理计时。

用法：
  python scripts/exp_cpu_speed.py --category gold_finger
  python scripts/exp_cpu_speed.py --all

输出：outputs/m0/m4_cpu_speed.json
"""
import os
import sys
import json
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import torch
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m4_cpu.yaml")))
    if args.category:
        cats = {args.category: datalocal.load_category(
            cfg["datasets"]["data_local"], args.category,
            cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])}
    else:
        cats = datalocal.load_all(cfg["datasets"]["data_local"], cfg)
    out = {"device": "cpu", "slots": ["blob", "trad", "layout"], "categories": {}}
    for name, bundle in cats.items():
        print(f"=== {name} (CPU) ===", flush=True)
        backbone, slots = build(cfg, "cpu")
        pipe = Pipeline(cfg, backbone, slots)
        t0 = time.time()
        pipe.fit(bundle)
        fit_s = time.time() - t0
        test = bundle["test"][:args.max_eval] if args.max_eval else bundle["test"]
        rep = pipe.evaluate(test)
        print(f"  fit={fit_s:.1f}s  fused AUROC={rep['fused']['auroc']:.4f} "
              f"AP={rep['fused']['ap']:.4f}  {rep['ms_per_image']:.0f}ms/图", flush=True)
        out["categories"][name] = {"fit_s": round(fit_s, 1),
                                   "ms_per_image": round(rep["ms_per_image"], 1),
                                   "fused_auroc": round(rep["fused"]["auroc"], 4),
                                   "fused_ap": round(rep["fused"]["ap"], 4),
                                   "slots": {n: round(rep[n]["auroc"], 4)
                                             for n in ("blob", "trad", "layout") if n in rep},
                                   "n_test": len(test)}
        out_path = os.path.join(cfg["output_dir"], "m4_cpu_speed.json")
        prev = {}
        if os.path.exists(out_path):           # 累积写入（多品类不互相覆盖）
            prev = json.load(open(out_path))
        prev["categories"] = {**prev.get("categories", {}), name: out["categories"][name]}
        json.dump(prev, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
