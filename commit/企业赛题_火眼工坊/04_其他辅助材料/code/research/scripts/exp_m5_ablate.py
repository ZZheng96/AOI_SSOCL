"""M5 学习机制消融（U24 诊断）：分解各在线更新机制对锚定集 AUROC 的影响。

背景（用户战略重心 2026-08-16）：核心是学习曲线与学习后 AUC。
m5_learning_curve 实测当前 SSOCL 在线更新在 AUROC 口径下反而微降（阴性结果）。
本消融定位退化源：逐项关闭 拦截加分 / 正常回流(reflow) / sem 扩展 / CDF 重估。

协议：fit → enable_ssocl → stream=test 前 42 张（feedback 0.2）→
锚定=test 后 30 张（留出只评估），测 initial/final AUROC 与 gain。

用法：
  python scripts/exp_m5_ablate.py --category gold_finger
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
from src.ssocl.learning_curve import simulate_learning

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
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--stream-n", type=int, default=42)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--feedback", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    test = bundle["test"]
    stream, eval_items = test[:args.stream_n], test[-args.eval_n:]
    groups = {
        "full": {},
        "intercept_only": {"reflow": True, "sem": True, "cdf": True},
        "reflow_only": {"intercept": True},
        "no_cdf": {"cdf": True},
        "no_sem": {"sem": True},
    }
    out = {}
    for name, abl in groups.items():
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        pipe.set_ablation(**abl)
        pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                          rng=np.random.default_rng(args.seed))
        g = simulate_learning(pipe, stream, eval_items, feedback_ratio=args.feedback,
                              eval_every=5, rng=np.random.default_rng(args.seed))
        out[name] = {k: g[k] for k in ("initial_auroc", "final_auroc", "auroc_gain",
                                       "efficiency", "n_feedback")}
        print(f"[{name}] 初始={g['initial_auroc']:.4f} 学习后={g['final_auroc']:.4f} "
              f"gain={g['auroc_gain']:+.4f} 效率={g['efficiency']:.4f} "
              f"反馈={g['n_feedback']}", flush=True)
    out_path = os.path.join(cfg["output_dir"], f"m5_ablate_{args.category}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
