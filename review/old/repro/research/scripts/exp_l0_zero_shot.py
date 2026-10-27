"""L0 零样本启动验证（U103，2026-08-24，评审 §13"L0 零样本启动无独立实证"可补项）

L0 定义（设计文档 §1）：仅正常图（train/good），槽位 = sem + inp + trad + open，
无校准缺陷集。本脚本禁用 disc/shead/blob/layout（需真实缺陷/伪异常），只用正常图
fit，评测 test 域 AUROC——量化"新品打样冷启动"（零缺陷样本）的诚实水平。

用法: python scripts/exp_l0_zero_shot.py [--dataset mvtec] [--category bottle]
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import numpy as np
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
L0_KEEP = {"sem", "inp", "trad", "open"}


def load(dataset, category, root):
    if dataset == "btad":
        return mvtec_like.load_btad(root, category, 100, 30, 42, n_eval_good=0, max_test=0)
    return mvtec_like.load_category(root, category, 100, 30, 42,
                                    n_eval_good=0, max_test=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--config", default=CFG)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    # L0：只留零样本槽位
    for n in list(cfg["slots"]):
        cfg["slots"][n]["enabled"] = n in L0_KEEP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = load(args.dataset, args.category, cfg["datasets"][args.dataset])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)

    y_true, y_score = [], []
    for p, y in b["test"]:
        r = pipe.predict(p)
        y_true.append(y)
        y_score.append(r["fused"])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)
    n_test = len(y_true)
    print(f"[U103] {args.dataset}/{args.category} L0零样本: AUROC={auc:.4f} "
          f"（test {n_test} 张，槽位 sem+inp+trad+open，仅 train/good 100 张）", flush=True)


if __name__ == "__main__":
    main()
