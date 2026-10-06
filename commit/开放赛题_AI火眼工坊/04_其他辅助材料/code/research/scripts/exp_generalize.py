"""M4 泛化验证（§9.1）：跨品类 leave-one-out——A 品类 fit，B 品类 test 上测（跨域）。

报告品类矩阵：对角=域内（in-domain），非对角=跨域（cross-domain）fused AUROC。
跨域衰减 = cross - in_domain（诚实回答"自行验证泛化能力"，赛题要求 2）。

用法：
  python scripts/exp_generalize.py --max-eval 60
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


def run_fit_eval(name_fit, name_eval, cfg, device, max_eval):
    bf = datalocal.load_category(cfg["datasets"]["data_local"], name_fit,
                                 cfg["protocol"]["n_init_normal"],
                                 cfg["protocol"]["n_init_defect"], cfg["seed"])
    be = datalocal.load_category(cfg["datasets"]["data_local"], name_eval,
                                 cfg["protocol"]["n_init_normal"],
                                 cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bf)
    test = be["test"][:max_eval] if max_eval else be["test"]
    rep = pipe.evaluate(test)
    return round(rep["fused"]["auroc"], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=60, help="eval 侧子采样")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = ["component", "extra_part", "gold_finger", "solder_smt"]
    out = {"categories": cats, "max_eval": args.max_eval, "matrix": {}}
    for fit in cats:
        for ev in cats:
            au = run_fit_eval(fit, ev, cfg, device, args.max_eval)
            out["matrix"].setdefault(fit, {})[ev] = au
            tag = "域内" if fit == ev else "跨域"
            print(f"  fit={fit:12s} eval={ev:12s} fused={au:.4f} [{tag}]", flush=True)
    # 跨域衰减统计
    diag = {c: out["matrix"][c][c] for c in cats}
    cross = [out["matrix"][a][b] for a in cats for b in cats if a != b]
    out["in_domain_mean"] = round(float(np.mean(list(diag.values()))), 4)
    out["cross_domain_mean"] = round(float(np.mean(cross)), 4)
    out["cross_decay"] = round(out["cross_domain_mean"] - out["in_domain_mean"], 4)
    print(f"\n[generalize] 域内均值={out['in_domain_mean']} "
          f"跨域均值={out['cross_domain_mean']} 衰减={out['cross_decay']:+}")
    out_path = os.path.join(cfg["output_dir"], "m4_generalize.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
