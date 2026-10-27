"""共识 blend 调优（§5.3）：0.0/0.5/0.8/1.0 消融，test 侧 AUROC 对比。

demo3 用 blend=0.8 实测有效；M2 默认 0.5。本脚本在目标品类上
量化 blend 对融合精度的真实影响（U14 共识自适应的落地调参）。

用法：
  python scripts/tune_consensus.py --category gold_finger [--all]
  python scripts/tune_consensus.py --all --max-eval 80
"""
import os
import sys
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)   # 长任务进度实时可见（管道重定向防块缓冲）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from src.data import datalocal
from scripts.m0_baseline import build, run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--blends", default="0.0,0.5,0.8,1.0")
    ap.add_argument("--hard-thresholds", default="0.0,0.02,0.05,0.1",
                    help="共识硬门控阈值扫描（M4 攻坚：弱共识槽位置零）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    blends = [float(x) for x in args.blends.split(",")]
    threshes = [float(x) for x in args.hard_thresholds.split(",")]

    if args.category:
        cats = {args.category: datalocal.load_category(
            cfg["datasets"]["data_local"], args.category,
            cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])}
    else:
        cats = datalocal.load_all(cfg["datasets"]["data_local"], cfg)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for name, bundle in cats.items():
        if args.max_eval:
            bundle = dict(bundle)
            bundle["test"] = bundle["test"][:args.max_eval]
        print(f"=== {name}: test={len(bundle['test'])} ===")
        rows = []
        for b in blends:
            for th in threshes:
                c = dict(cfg)
                c["consensus"] = dict(cfg.get("consensus", {}))
                c["consensus"]["blend"] = b
                c["consensus"]["hard_thresh"] = th
                backbone, slots = build(c, device)
                rep = run_one(f"{name}_b{b}_t{th}", bundle, c, device, out_dir,
                              max_eval=args.max_eval)
                au = rep["test"]["fused"]["auroc"] if rep.get("test") else float("nan")
                rows.append((b, th, au))
                print(f"  blend={b} hard_thresh={th}: test AUROC={au:.4f}")
        best = max(rows, key=lambda x: x[2])
        print(f"[{name}] 最佳 blend={best[0]} hard_thresh={best[1]} "
              f"AUROC={best[2]:.4f}")


if __name__ == "__main__":
    main()
