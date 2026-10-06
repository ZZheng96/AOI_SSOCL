"""机制C诊断：域差品类各槽位跨域 AUROC + blob 权重扫描（2026-08-26，小样本快速迭代）

确定"blob 融合权重强化"的最优配置：对 data_local 域差品类（solder/gold_finger）
fit 完整 pipeline，测 test 域各槽位 AUROC（哪些正向、哪些反向），并扫描
fused 中 blob 权重（其余等权归一）对 AUROC 的影响。

用法: python scripts/diag_slot_domain.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import datalocal
from sklearn.metrics import roc_auc_score
from src.fusion.fixed import fuse

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for cat in ["solder_smt", "gold_finger"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=60)
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(b)
        print(f"\n[slot-domain] {cat}: train_auroc={pipe.train_auroc} "
              f"slots={sorted(pipe._score_slots)}", flush=True)

        # test 域逐槽位 + fused AUROC
        names = list(pipe._score_slots)
        per = {n: [] for n in names}
        fused_ew = []
        ys = []
        for p, y in b["test"]:
            rec = pipe._record(pipe._get_item(p))
            for n in names:
                per[n].append(float(rec["slot_scores"][n]))
            fused_ew.append(float(rec["fused"]))
            ys.append(y)
        print("  test 域槽位 AUROC（跨域判别）:", flush=True)
        for n in names:
            au = roc_auc_score(ys, per[n])
            print(f"    {n:8s} {au:.4f}", flush=True)
        print(f"  fused（等权）: {roc_auc_score(ys, fused_ew):.4f}", flush=True)

        # blob 权重扫描（其余等权归一）
        w_rest = 1.0 / max(len(names) - 1, 1)
        print("  blob 权重扫描（其余等权归一）:", flush=True)
        for wb in [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65]:
            w = {n: w_rest * (1.0 - wb) for n in names}
            if "blob" in w:
                w["blob"] = wb
            sc = [fuse({n: per[n][i] for n in names}, w) for i in range(len(ys))]
            au = roc_auc_score(ys, sc)
            mark = " <=" if "blob" in w and abs(wb - 0.15) < 1e-6 else ""
            print(f"    blob={wb:.2f}: fused AUROC={au:.4f}{mark}", flush=True)


if __name__ == "__main__":
    main()
