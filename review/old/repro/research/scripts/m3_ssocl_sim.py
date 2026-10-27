"""M3 学习曲线实验（§7.4）：自学习开/关对照，验证"这批错检，下批检对"。

用法：
  python scripts/m3_ssocl_sim.py --category gold_finger --feedback 0.2
  python scripts/m3_ssocl_sim.py --category solder_smt --feedback 0.05 --max-eval 60
  python scripts/m3_ssocl_sim.py --dataset gyudet --feedback 0.2

输出：outputs/m0/m3_{category}_{feedback}.json + 学习曲线 PNG
"""
import os
import sys
import json
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)   # 长任务进度实时可见（管道重定向防块缓冲）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from src.data import datalocal, gyudet
from scripts.m0_baseline import build, run_one
from src.eval.offline import Pipeline
from src.ssocl.learning_curve import simulate, curve_gap


def plot_curves(groups, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, data in groups.items():
        ax[0].plot(data["steps"], data["cum_f1"], label=f"{name} (F1={data['final_f1']})")
    ax[0].set_xlabel("steps"); ax[0].set_ylabel("cumulative F1")
    ax[0].set_title("learning curve (self-learning on/off)")
    ax[0].legend()
    for name, data in groups.items():
        ax[1].plot(data["steps"], data["cum_recall"], label=f"{name}")
        ax[1].plot(data["steps"], data["cum_precision"], ls="--", label=f"{name}(prec)")
    ax[1].set_xlabel("steps"); ax[1].set_ylabel("recall/precision")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datalocal", choices=["datalocal", "gyudet"])
    ap.add_argument("--category", default=None, help="datalocal 品类；默认全品类循环")
    ap.add_argument("--feedback", type=float, default=0.2, help="反馈率（5%/20% 两档）")
    ap.add_argument("--max-eval", type=int, default=0, help="评测集子采样（迭代期）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    cfg["seed"] = args.seed
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    ssocl_cfg = {
        "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
                  "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
        "gate": {"tol": 0.05, "anchor_size": 20},
        "active": {"top_n": 10},
        "incubate": {"min_samples": 8},
        "router_finetune": {"trigger": 10, "lr": 1e-4, "epochs": 5},
    }

    if args.dataset == "gyudet":
        bundles = {"gyudet": gyudet.load_gyudet(
            cfg["datasets"]["gyu_det"], seed=args.seed)}
    else:
        if args.category:
            bundles = {args.category: datalocal.load_category(
                cfg["datasets"]["data_local"], args.category,
                cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"],
                seed=args.seed)}
        else:
            bundles = datalocal.load_all(cfg["datasets"]["data_local"], cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    for name, bundle in bundles.items():
        if args.max_eval:
            bundle = dict(bundle)
            bundle["test"] = bundle["test"][:args.max_eval]
        print(f"=== {name}: init_normal={len(bundle['init_normal'])} "
              f"init_defect={len(bundle['init_defect'])} test={len(bundle['test'])} ===")
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        print(f"[fit] {pipe.fit_seconds:.1f}s  fused 阈值 "
              f"tau_high={pipe.decider.tau_high:.4f} tau_gray={pipe.decider.tau_gray:.4f}")
        # 对照组：自学习关闭（无反馈）——先跑，此时 handler 未启用，predict 无拦截/回流
        print("[sim] baseline (self-learning OFF)")
        baseline = simulate(pipe, bundle["test"], feedback_ratio=0.0,
                            with_feedback=False, rng=np.random.default_rng(args.seed))
        # 实验组：自学习开启（同一 pipeline，启用 handler 后 predict 才有拦截加分）
        print("[sim] ssocl (self-learning ON, feedback rate "
              f"{args.feedback})")
        pipe.enable_ssocl(ssocl_cfg, train_normal_paths=bundle["init_normal"],
                          rng=np.random.default_rng(args.seed))
        learned = simulate(pipe, bundle["test"], feedback_ratio=args.feedback,
                           with_feedback=True, cfg=ssocl_cfg,
                           rng=np.random.default_rng(args.seed))
        gap = curve_gap(learned, baseline)
        print(f"[result] {name} baseline F1={baseline['final_f1']} "
              f"learned F1={learned['final_f1']} gap={gap['final_f1_gap']:+} "
              f"feedback={learned['n_feedback']}")
        rep = {"name": name, "feedback_ratio": args.feedback,
               "baseline": {k: baseline[k] for k in ("final_f1", "tp", "fp", "tn", "fn")},
               "learned": {k: learned[k] for k in ("final_f1", "tp", "fp", "tn", "fn")},
               "gap": gap, "learned_trace": learned["trace"],
               "ssocl_stats": pipe.handler.stats,
               "rollbacks": pipe.handler.rollback_log,
               "gate_history": pipe.handler.gate.history,
               "incubate_log": pipe.head_mgr.log}
        out_json = os.path.join(cfg["output_dir"],
                                f"m3_{name}_{args.feedback}.json")
        json.dump(rep, open(out_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        plot_curves({"baseline": baseline, "learned": learned},
                    os.path.join(cfg["output_dir"], f"m3_{name}_{args.feedback}.png"))
        print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
