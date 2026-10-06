"""U25 定位：fit 两次后各槽位 raw score 差异 → 定位非确定性槽位。

用法：python scripts/diag_slot_repro.py --category solder_smt
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="solder_smt")
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    ev_path = bundle["test"][-1][0]
    per_trial = []
    for t in range(args.trials):
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        item = pipe._get_item(ev_path)
        rec = pipe._record(item)
        per_trial.append({n: round(float(v), 6) for n, v in rec["raw_scores"].items()})
        print(f"[trial {t}] {per_trial[-1]}", flush=True)
    if args.trials >= 2:
        diff = {n: abs(per_trial[1][n] - per_trial[0][n]) for n in per_trial[0]}
        print("[diff]", {k: round(v, 6) for k, v in sorted(diff.items(), key=lambda x: -x[1])},
              flush=True)
    json.dump({"per_trial": per_trial, "path": ev_path},
              open(os.path.join(cfg["output_dir"], "diag_slot_repro.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
