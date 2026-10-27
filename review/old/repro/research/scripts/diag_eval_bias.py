"""U26 诊断：gold_finger 锚定集 fused 分布与正负构成（基线 AUROC<0.5 反向定位）。

用法：python scripts/diag_eval_bias.py --category gold_finger
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
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--eval-n", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    eval_items = bundle["test"][-args.eval_n:]
    ys, f, slots_ = [], [], {}
    for p, y in eval_items:
        rec = pipe._record(pipe._get_item(p))
        ys.append(y)
        f.append(rec["fused"])
        for n, v in rec["slot_scores"].items():
            slots_.setdefault(n, []).append(v)
    ys = np.array(ys)
    f = np.array(f)
    from sklearn.metrics import roc_auc_score
    print(f"[{args.category}] eval n={len(ys)} 正(缺陷)={ys.sum()} 负(正常)={(1-ys).sum()} "
          f"fused AUROC={roc_auc_score(ys, f):.4f}", flush=True)
    print(f"  fused: 正常 mean={f[ys==0].mean():.4f} 缺陷 mean={f[ys==1].mean():.4f} "
          f"min={f.min():.4f} max={f.max():.4f}", flush=True)
    for n, v in slots_.items():
        v = np.array(v)
        try:
            au = roc_auc_score(ys, v)
        except Exception:
            au = float("nan")
        print(f"  slot {n:8s} AUROC={au:.4f}  正常={v[ys==0].mean():.4f} 缺陷={v[ys==1].mean():.4f}",
              flush=True)
    json.dump({"category": args.category, "n": len(ys), "n_defect": int(ys.sum()),
               "fused_auroc": round(float(roc_auc_score(ys, f)), 4),
               "fused_mean_normal": round(float(f[ys == 0].mean()), 4),
               "fused_mean_defect": round(float(f[ys == 1].mean()), 4),
               "slots": {n: {"auroc": round(float(roc_auc_score(ys, np.array(v))), 4)}
                         for n, v in slots_.items()}},
              open(os.path.join(cfg["output_dir"], f"diag_eval_bias_{args.category}.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
