"""M4/U9 精度回归：PDN 蒸馏学生 vs DINO 主干，逐槽位 AUROC + 速度对比。

用法：
  python scripts/exp_pdn_regress.py --category gold_finger --max-eval 40
  python scripts/exp_pdn_regress.py --all --max-eval 40

输出：outputs/m0/m4_pdn_regress.json
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


def run_one(name, bundle, cfg, device, backbone_name, max_eval=None):
    c = dict(cfg)
    c["backbone"] = dict(cfg["backbone"])
    c["backbone"]["name"] = backbone_name
    if backbone_name == "pdn":
        ckpt = os.path.join(cfg["output_dir"], f"pdn_distill_{name}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"PDN 蒸馏 checkpoint 缺失: {ckpt}（先跑 train_pdn_distill.py）")
        c["backbone"]["checkpoint"] = ckpt
    backbone, slots = build(c, device)
    pipe = Pipeline(c, backbone, slots)
    pipe.fit(bundle)
    test = bundle["test"][:max_eval] if max_eval else bundle["test"]
    rep = pipe.evaluate(test)
    return {"fused": rep["fused"], "ms_per_image": rep["ms_per_image"],
            "slots": {n: rep[n]["auroc"] for n in rep
                      if n not in ("fused", "ms_per_image")}}


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
    out = {"categories": {}, "rule": "损失>0.01 AUC 不采用（U9）"}
    for name, bundle in cats.items():
        print(f"=== {name} ===", flush=True)
        print(f"  [DINO] fit+eval ...", flush=True)
        d = run_one(name, bundle, cfg, device, "dino", args.max_eval)
        print(f"  [pdn ] fit+eval ...", flush=True)
        p = run_one(name, bundle, cfg, device, "pdn", args.max_eval)
        fused_gap = p["fused"]["auroc"] - d["fused"]["auroc"]
        adopt = abs(fused_gap) <= 0.01
        print(f"  DINO fused={d['fused']['auroc']:.4f} {d['ms_per_image']:.0f}ms | "
              f"PDN fused={p['fused']['auroc']:.4f} {p['ms_per_image']:.0f}ms | "
              f"gap={fused_gap:+.4f} {'采纳' if adopt else '不采纳(U9)'}", flush=True)
        out_path = os.path.join(cfg["output_dir"], "m4_pdn_regress.json")
        out = {}
        if os.path.exists(out_path):           # 累积写入（多品类/多轮不互相覆盖）
            out = json.load(open(out_path))
        out.setdefault("categories", {})
        out["categories"][name] = {
            "dino": {"fused": d["fused"]["auroc"], "ms": d["ms_per_image"], "slots": d["slots"]},
            "pdn": {"fused": p["fused"]["auroc"], "ms": p["ms_per_image"], "slots": p["slots"]},
            "gap": round(fused_gap, 4), "adopt": adopt}
        json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
