"""U25 诊断：同进程内两次 fit 的 initial AUROC 是否一致（排除进程间 GPU 因素）。

用法：python scripts/diag_fit_repro.py --category solder_smt
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
from src.ssocl.learning_curve import _eval_auroc


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
    eval_items = bundle["test"][-30:]
    res = []
    for t in range(args.trials):
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        a = _eval_auroc(pipe, eval_items)
        res.append(round(a, 4))
        print(f"[trial {t}] initial={a:.4f}", flush=True)
    json.dump({"category": args.category, "trials": res,
               "same": all(abs(x - res[0]) < 1e-9 for x in res)},
              open(os.path.join(cfg["output_dir"], "diag_fit_repro.json"), "w"),
              ensure_ascii=False, indent=2)
    print("[save] diag_fit_repro.json")


if __name__ == "__main__":
    main()
