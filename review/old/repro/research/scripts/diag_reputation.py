"""诊断 v3：反馈驱动的槽位信誉权重（目标模式核心机制验证，2026-08-16）

背景：solder tiles36 blob AUROC=0.833 但 fused=0.486（sem/disc 反向拖累）。
缺陷原型相似度路线已证伪（DINO patch 层缺陷不聚集，AUROC 0.23-0.32）。

新机制（v4 在线学习）：反馈样本带真值标签 -> 在线估计各槽位判别力（信誉）->
动态融合权重。这本身就是持续学习：冷启动等权（诚实），随反馈学会"该品类
该信谁"。数学：槽位信誉 = Mann-Whitney U（缺陷分校验分>正常分概率），
w_i ∝ relu(U_i - 0.5)，按有效样本对数收缩到等权（少反馈=不武断）。

协议：流=test shuffle，反馈率 0.5（模拟操作员）；eval=未反馈图（排除标签
已进权重的图，无泄漏）。每 5 反馈重算 eval AUROC。对照=等权固定。

输出：outputs/m0/diag_reputation_{category}_{mode}.json
"""
import os
import sys
import json
import argparse
import copy

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
}


class ReputationWeights:
    """反馈驱动的槽位信誉权重（v4 机制原型）。

    slot_scores：反馈样本的校准分按槽位累积；U = P(defect > normal)；
    w ∝ relu(U - 0.5)，n_pairs < n0 时收缩到等权。
    """

    def __init__(self, names, n0=30):
        self.names = names
        self.n0 = n0
        self.D = {n: [] for n in names}     # 确认缺陷的槽位校准分
        self.N = {n: [] for n in names}     # 确认正常的槽位校准分

    def update(self, slot_scores, label):
        tgt = self.D if label == 1 else self.N
        for n in self.names:
            tgt[n].append(float(slot_scores[n]))

    def weights(self):
        n_pairs = min(sum(len(v) for v in self.D.values()) / max(len(self.names), 1),
                      sum(len(v) for v in self.N.values()) / max(len(self.names), 1))
        lam = n_pairs / (n_pairs + self.n0)
        w = {}
        for n in self.names:
            U = 0.5
            if self.D[n] and self.N[n]:
                d, nn = np.array(self.D[n]), np.array(self.N[n])
                U = float((d[:, None] > nn[None, :]).mean() + 0.5 *
                          (d[:, None] == nn[None, :]).mean())
            s = max(U - 0.5, 0.0)
            w[n] = lam * s + (1 - lam) * 0.0
        tot = sum(w.values())
        if tot <= 1e-9:
            k = len(self.names)
            return {n: 1.0 / k for n in self.names}, 0.0
        # 归一化后与等权混合（lam 控制），保持和为 1
        base = {n: w[n] / tot for n in self.names}
        k = len(self.names)
        return {n: lam * base[n] + (1 - lam) / k for n in self.names}, lam

    def stats(self):
        out = {}
        for n in self.names:
            if self.D[n] and self.N[n]:
                d, nn = np.array(self.D[n]), np.array(self.N[n])
                U = float((d[:, None] > nn[None, :]).mean())
                out[n] = round(U, 3)
        return out


def eval_auroc(pipe, items, weights):
    """eval 集上用指定权重融合校准分的 AUROC。"""
    from src.fusion.fixed import fuse
    ys, fs = [], []
    for p, y in items:
        rec = pipe._record(pipe._get_item(p))
        ys.append(y)
        fs.append(fuse(rec["slot_scores"], weights))
    return float(roc_auc_score(ys, fs)), ys, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="solder_smt")
    ap.add_argument("--mode", default="tiles36")
    ap.add_argument("--fb-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    cfg = copy.deepcopy(cfg)
    cfg["tiling"]["mode"] = args.mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(
        cfg["datasets"]["data_local"], args.category,
        cfg["protocol"]["n_init_normal"], cfg["protocol"]["n_init_defect"], cfg["seed"])

    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    names = sorted(pipe.slots)
    print(f"[{args.category}/{args.mode}] 槽位: {names}", flush=True)

    rng = np.random.default_rng(args.seed)
    test = list(bundle["test"])
    rng.shuffle(test)

    rep = ReputationWeights(names)
    unfb = []
    curve, curve_eq = [], []
    n_fb = 0
    for i, (p, y) in enumerate(test):
        rec = pipe._record(pipe._get_item(p))
        if rng.random() < args.fb_ratio:
            rep.update(rec["slot_scores"], y)
            n_fb += 1
            if n_fb % 5 == 0:
                w, lam = rep.weights()
                au, _, _ = eval_auroc(pipe, unfb, w) if len(unfb) >= 8 else (None,) * 3
                if au is not None:
                    curve.append({"n_fb": n_fb, "auroc": round(au, 4), "lam": round(lam, 3)})
                    print(f"  [rep] 反馈={n_fb} AUROC={au:.4f} lam={lam:.3f} "
                          f"U={rep.stats()}", flush=True)
        else:
            unfb.append((p, y))
        if (i + 1) % 30 == 0:
            print(f"  [stream] {i + 1}/{len(test)} 反馈={n_fb} eval池={len(unfb)}",
                  flush=True)

    w_final, lam = rep.weights()
    if len(unfb) >= 8:
        au_final, ys, fs = eval_auroc(pipe, unfb, w_final)
        au_eq, _, _ = eval_auroc(pipe, unfb, {n: 1.0 / len(names) for n in names})
        au_init, _, _ = eval_auroc(pipe, unfb, pipe.weights)   # fit 权重（共识/等权）
    else:
        au_final = au_eq = au_init = None
        ys, fs = [], []

    out = {"category": args.category, "mode": args.mode,
           "n_test": len(test), "n_fb": n_fb, "n_eval": len(unfb),
           "n_eval_defect": int(sum(y for _, y in unfb)),
           "reputation_U": rep.stats(),
           "final_weights": {k: round(v, 4) for k, v in w_final.items()},
           "lam": round(lam, 3),
           "auroc_fit_weights": round(au_init, 4) if au_init else None,
           "auroc_equal": round(au_eq, 4) if au_eq else None,
           "auroc_reputation": round(au_final, 4) if au_final else None,
           "curve": curve}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = os.path.join(cfg["output_dir"],
                            f"diag_reputation_{args.category}_{args.mode}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
