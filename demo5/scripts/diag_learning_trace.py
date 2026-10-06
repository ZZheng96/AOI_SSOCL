"""U26 诊断：gold_finger 学习失效定位——统计流式反馈的动作分布。

跑一次 SSOCL 学习模拟，统计 feedback_log 里各 action 的分布、
reflow 拒绝原因、拦截加分（boost）的生效情况。

用法：python scripts/diag_learning_trace.py --category gold_finger --ratio 0.5
"""
import os
import sys
import json
import argparse
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 6, "lr": 1e-4, "epochs": 5},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    test = bundle["test"]
    stream = test[:len(test) - args.eval_n]
    eval_items = test[-args.eval_n:]
    pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=bundle["init_normal"],
                      rng=np.random.default_rng(args.seed))
    rng = np.random.default_rng(args.seed)
    boosts, actions = [], Counter()
    reflow_reason = Counter()
    boost_by_label = {0: [], 1: []}
    for i, (path, y) in enumerate(stream):
        r = pipe.predict(path)
        pred = 1 if r["decision"] != "normal" else 0
        b = r.get("boost", 0.0)
        boosts.append(b)
        boost_by_label[y].append(b)
        if rng.random() < args.ratio:
            correct = (pred == y)
            entry = pipe.feedback(path, "correct" if correct else "wrong", label=y)
            actions[entry.get("action", "?")] += 1
            if entry.get("action", "").startswith("reflow"):
                reason = entry.get("action", "")
                if "drift" in entry:
                    reason += f"(drift={entry['drift']})"
                reflow_reason[reason] += 1
    from src.ssocl.learning_curve import _eval_auroc
    final = _eval_auroc(pipe, eval_items)
    b0, b1 = boost_by_label[0], boost_by_label[1]
    print(f"[{args.category}] ratio={args.ratio} 流={len(stream)} 反馈={sum(actions.values())}", flush=True)
    print(f"  actions: {dict(actions)}", flush=True)
    print(f"  reflow原因: {dict(reflow_reason)}", flush=True)
    print(f"  boost: 非零={sum(1 for b in boosts if b > 0)}/{len(boosts)} "
          f"均值={np.mean(boosts):.5f} 最大={max(boosts):.5f}", flush=True)
    if b0 or b1:
        print(f"  boost按标签: 正常 n={len(b0)} 非零={sum(1 for b in b0 if b>0)} "
              f"均值={np.mean(b0) if b0 else 0:.5f} | "
              f"缺陷 n={len(b1)} 非零={sum(1 for b in b1 if b>0)} "
              f"均值={np.mean(b1) if b1 else 0:.5f}", flush=True)
    print(f"  学习后锚定集 AUROC={final:.4f}", flush=True)
    json.dump({"category": args.category, "actions": dict(actions),
               "reflow_reason": dict(reflow_reason),
               "n_feedback": sum(actions.values()),
               "n_boost": sum(1 for b in boosts if b > 0),
               "boost_by_label": {"normal": {"n": len(b0), "nonzero": sum(1 for b in b0 if b > 0),
                                             "mean": float(np.mean(b0)) if b0 else 0.0},
                                  "defect": {"n": len(b1), "nonzero": sum(1 for b in b1 if b > 0),
                                             "mean": float(np.mean(b1)) if b1 else 0.0}},
               "boost_mean": float(np.mean(boosts)), "final_auroc": round(final, 4)},
              open(os.path.join(cfg["output_dir"], f"diag_learning_trace_{args.category}.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
