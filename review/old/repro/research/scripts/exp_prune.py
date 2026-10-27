"""M4 特征裁剪回归（§6.1 工程杠杆）：按贡献档案裁剪低贡献槽位 → 融合 AUROC 对比。

策略：全槽等权 vs drop 单槽 AUROC<0.5 / <0.6 vs keep-top2 vs 共识权重（若启用）。
只消费 eval 侧分数（校准分已缓存），不重复 fit。

用法：
  python scripts/exp_prune.py --category gold_finger --max-eval 40
  python scripts/exp_prune.py --all --max-eval 40

输出：outputs/m0/m4_prune.json
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
from src.eval.metrics import image_metrics
from src.fusion.fixed import fuse


def fused_auroc(per_slot, labels, keep):
    w = {n: 1.0 / len(keep) for n in keep}
    fused = [fuse({n: float(per_slot[n][i]) for n in keep}, w)
             for i in range(len(labels))]
    return image_metrics(labels, fused)["auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.category:
        cats = {args.category: datalocal.load_category(
            cfg["datasets"]["data_local"], args.category,
            cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])}
    else:
        cats = datalocal.load_all(cfg["datasets"]["data_local"], cfg)
    out_path = os.path.join(cfg["output_dir"], "m4_prune.json")
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))
    out.setdefault("categories", {})

    for name, bundle in cats.items():
        print(f"=== {name} ===", flush=True)
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        test = bundle["test"][:args.max_eval] if args.max_eval else bundle["test"]
        pipe.evaluate(test)
        le = pipe._last_eval
        labels = le["labels"]; per_slot = le["per_slot"]
        names = list(per_slot)
        single = {n: round(image_metrics(labels, per_slot[n])["auroc"], 4) for n in names}
        full = fused_auroc(per_slot, labels, names)
        strategies = {"all_equal": (full, names)}
        for th in (0.5, 0.6):
            keep = [n for n in names if single[n] >= th]
            strategies[f"drop_lt_{th}"] = (
                fused_auroc(per_slot, labels, keep) if keep else 0.0, keep)
        top2 = sorted(names, key=lambda n: -single[n])[:2]
        strategies["keep_top2"] = (fused_auroc(per_slot, labels, top2), top2)
        if pipe.consensus_gate is not None:
            keep_c = list(pipe.weights)
            strategies["consensus"] = (fused_auroc(per_slot, labels, keep_c), keep_c)
        print(f"  单槽: {single}")
        best = None
        for k, (au, keep) in strategies.items():
            print(f"  {k}: AUROC={au:.4f} keep={keep}")
            if best is None or au > best[1]:
                best = (k, au)
        print(f"  >>> 最佳策略: {best[0]} ({best[1]:.4f})", flush=True)
        out["categories"][name] = {"single": single, "strategies": {
            k: {"auroc": round(v[0], 4), "keep": v[1]} for k, v in strategies.items()},
            "best": best[0]}
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
