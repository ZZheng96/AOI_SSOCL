"""M2 消融：保底等权 vs 软路由（§5.1 消融义务）

对每品类：fit 一次（含路由训练+门控），在 test 侧分别用
  A. 保底等权：fuse(slot_scores, weights)
  B. 软路由：router.fused(slot_scores)（若门控通过）
报告两者 AUROC/AP 与门控结果，如实记录路由真实增益（可能为负——U15 已声明）。

用法：
  python scripts/exp_router.py --category gold_finger [--all]
"""
import os
import sys
import argparse
import copy
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import datalocal
from src.eval.metrics import image_metrics
from src.fusion.fixed import fuse
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def run_one(name, bundle, cfg, out_dir, max_eval=None):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if max_eval:
        bundle = dict(bundle)
        bundle["test"] = bundle["test"][:max_eval]
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    gate = pipe.router_gate or {"enabled": False, "base_auc": float("nan"),
                                "router_auc": float("nan")}

    labels, base_f, router_f, per_slot = [], [], [], {n: [] for n in pipe.slots}
    for p, y in bundle["test"]:
        r = pipe.predict(p)
        labels.append(y)
        base_f.append(fuse(r["slot_scores"], pipe.weights))
        # 路由侧：无论门控是否启用，都算出路由分供消融对比
        names = sorted(pipe.slots)
        svec = torch.tensor([[r["slot_scores"][n] for n in names]], dtype=torch.float32)
        rf, _ = pipe.router.fused(svec) if pipe.router else (torch.tensor([base_f[-1]]), None)
        router_f.append(float(rf.item()))
        for n in pipe.slots:
            per_slot[n].append(r["slot_scores"][n])

    m_base = image_metrics(labels, base_f)
    m_route = image_metrics(labels, router_f)
    gain = m_route["auroc"] - m_base["auroc"]
    print(f"[{name}] gate_enabled={gate['enabled']} "
          f"(train: base={gate['base_auc']:.4f} router={gate['router_auc']:.4f})")
    print(f"  test: base={m_base['auroc']:.4f} router={m_route['auroc']:.4f} "
          f"gain={gain:+.4f} | router{'启用' if gate['enabled'] else '未启用(回退保底)'}")
    return {"name": name, "gate": gate, "base": m_base, "router": m_route,
            "gain": round(gain, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"))
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        results = []
        for name in datalocal.CATEGORIES:
            bundle = datalocal.load_category(cfg["datasets"]["data_local"], name, 100, 30, cfg["seed"])
            results.append(run_one(name, bundle, cfg, out_dir, args.max_eval))
        import json
        with open(os.path.join(out_dir, "m2_router_ablation.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print("\n===== M2 保底 vs 路由 消融汇总 =====")
        print(f"{'品类':<14}{'base':>8}{'router':>8}{'gain':>8}  门控")
        for r in results:
            print(f"{r['name']:<14}{r['base']['auroc']:.4f}  {r['router']['auroc']:.4f}  "
                  f"{r['gain']:+.4f}  {'启用' if r['gate']['enabled'] else '未启用'}")
    else:
        bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category, 100, 30, cfg["seed"])
        run_one(args.category, bundle, cfg, out_dir, args.max_eval)


if __name__ == "__main__":
    main()
