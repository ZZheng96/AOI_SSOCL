"""U24 诊断：验证 fit 确定性 + enable_ssocl 对 initial AUROC 的影响。

消融实验各组 initial_auroc 波动 0.24~0.45，需确认：
1) fit 是否确定（同 seed 两次 fit 的 initial 是否一致）；
2) enable_ssocl 本身是否改变 initial（库空时拦截应返回 0）。

用法：python scripts/diag_fit_seed.py --category solder_smt
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

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 10, "lr": 1e-4, "epochs": 5},
}


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
    test = bundle["test"]
    eval_items = test[-30:]
    res = []
    for t in range(args.trials):
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        a1 = _eval_auroc(pipe, eval_items)
        pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                          rng=np.random.default_rng(42))
        a2 = _eval_auroc(pipe, eval_items)
        res.append({"trial": t, "before_ssocl": round(a1, 4), "after_ssocl": round(a2, 4)})
        print(f"[trial {t}] fit后initial={a1:.4f}  enable_ssocl后={a2:.4f}", flush=True)
    json.dump(res, open(os.path.join(cfg["output_dir"], "diag_fit_seed.json"), "w"),
              ensure_ascii=False, indent=2)
    print("[save] diag_fit_seed.json")


if __name__ == "__main__":
    main()
