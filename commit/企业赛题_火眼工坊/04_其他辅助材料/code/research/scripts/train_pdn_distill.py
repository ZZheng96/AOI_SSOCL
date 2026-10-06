"""M4/U9 PDN 蒸馏训练：PDN-S 学生对齐 DINO 特征（train/good，无泄漏）。

用法：
  python scripts/train_pdn_distill.py --category gold_finger --epochs 20
  python scripts/train_pdn_distill.py --all --epochs 20

输出：outputs/m0/pdn_distill_{category}.pt + pdn_distill_{category}.json（loss 曲线）
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)   # 长任务进度实时可见
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.common.io import load_image
from src.backbone.dino import FrozenDINO
from src.backbone.pdn import PDNBackbone, distill_pdn
from src.data import datalocal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-tiles", type=int, default=0, help="蒸馏用 train/good 张数上限（0=全量）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.category:
        cats = {args.category: datalocal.load_category(
            cfg["datasets"]["data_local"], args.category,
            cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])}
    else:
        cats = datalocal.load_all(cfg["datasets"]["data_local"], cfg)

    dino = FrozenDINO(output_layer=cfg["backbone"]["output_layer"],
                      grid=cfg["backbone"]["grid"], device=device)
    for name, bundle in cats.items():
        paths = bundle["init_normal"]
        if args.max_tiles:
            rng = np.random.default_rng(args.seed)
            paths = sorted(rng.choice(paths, min(args.max_tiles, len(paths)),
                                      replace=False).tolist())
        print(f"=== {name}: 蒸馏 train/good {len(paths)} 张，epochs={args.epochs} ===", flush=True)
        imgs = [load_image(p) for p in paths]
        student = PDNBackbone(grid=cfg["backbone"]["grid"], device=device)
        student, info = distill_pdn(student, dino, imgs, epochs=args.epochs,
                                    device=device, progress_every=5)
        ckpt = os.path.join(out_dir, f"pdn_distill_{name}.pt")
        torch.save(student.state_dict(), ckpt)
        rep = {"category": name, "epochs": args.epochs, "n_tiles": info["n_tiles"],
               "losses": info["losses"], "n_params": student.model.n_params,
               "checkpoint": ckpt}
        json.dump(rep, open(os.path.join(out_dir, f"pdn_distill_{name}.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"[{name}] 蒸馏完成 loss={info['losses'][-1]} 参数={student.model.n_params} "
              f"-> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
