"""诊断 v4：在线线性融合器（Online Router，目标模式核心，2026-08-16）

背景：信誉权重 AUROC 仅 0.6125（软加权稀释+贝叶斯收缩过保守），
但 fit_weights=0.6687 反而最好 -> fit 域 router 知识在 test 域有效。
blob 单槽位 0.833 是上限参考。

机制：反馈对（槽位校准分向量 -> 真值标签）训练逻辑回归：
    logit = b + Σ a_i s_i,  P(defect) 为融合分
- online_cold：先验=等权弱收缩（纯在线，冷启动诚实）
- online_prior：MAP 微调，先验=fit router 权重（迁移+在线结合点）
    loss = NLL(balanced) + 0.5·λ·||a − a_prior||²
与 fit router 的本质区别：反馈样本来自生产分布，无域差失配。

协议：流=test shuffle、反馈率 0.5、eval=未反馈图（无泄漏）。
预计算全部 test 槽位分 -> 流式模拟纯 numpy。
对照：equal / fit_weights / blob 单槽位上限。

输出：outputs/m0/diag_online_router_{category}_{mode}.json
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


def fit_lr_map(X, y, a_prior, b_prior, lam, n_iter=200):
    """MAP 逻辑回归：NLL(balanced) + 0.5·lam·||a-a_prior||² + 0.5·lam·(b-b_prior)²"""
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    sw = torch.where(yt > 0.5, torch.tensor(1.0 / max(n1, 1e-9)),
                     torch.tensor(1.0 / max(n0, 1e-9)))
    a = torch.nn.Parameter(torch.tensor(a_prior, dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(b_prior, dtype=torch.float64))
    ap = torch.tensor(a_prior, dtype=torch.float64)
    bp = torch.tensor(b_prior, dtype=torch.float64)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=n_iter,
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        logit = Xt @ a + b
        nll = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, yt, weight=sw)
        reg = 0.5 * lam * (((a - ap) ** 2).sum() + (b - bp) ** 2)
        loss = nll + reg
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().numpy(), float(b.detach().numpy())


class OnlineRouter:
    def __init__(self, names, w_prior, lam):
        self.names = names
        k = len(names)
        w = np.array([w_prior.get(n, 1.0 / k) for n in names])
        scale = 4.0  # 融合分 0.5 决策 -> logit=0
        self.a_prior = scale * w
        self.b_prior = -scale * 0.5
        self.lam = lam
        self.a = self.a_prior.copy()
        self.b = self.b_prior

    def update(self, X, y):
        self.a, self.b = fit_lr_map(X, y, self.a_prior, self.b_prior, self.lam)

    def scores(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.a + self.b)))

    def weights(self):
        s = np.abs(self.a).sum()
        return {n: float(abs(ai) / s) for n, ai in zip(self.names, self.a)}


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

    # 预计算全部 test 槽位校准分（一次 _record/图）
    print("[precompute] 槽位分 ...", flush=True)
    S_all, y_all = [], []
    for i, (p, y) in enumerate(test):
        rec = pipe._record(pipe._get_item(p))
        S_all.append([float(rec["slot_scores"][n]) for n in names])
        y_all.append(y)
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)
    S_all = np.array(S_all)
    y_all = np.array(y_all, dtype=np.float64)
    w_eq = {n: 1.0 / len(names) for n in names}
    eq_vec = np.array([w_eq[n] for n in names])

    def auroc_fused(idx, vec, bias=0.0):
        s = S_all[idx] @ vec + bias
        if len(np.unique(y_all[idx])) < 2:
            return None
        return float(roc_auc_score(y_all[idx], s))

    # 变体
    k = len(names)
    w_fit = pipe.weights
    fit_vec = np.array([w_fit.get(n, 1.0 / k) for n in names])
    variants = {
        "online_cold": OnlineRouter(names, w_eq, lam=4.0),
        "online_prior": OnlineRouter(names, w_fit, lam=1.0),
    }

    Xfb, yfb = [], []
    unfb = []
    curve = {v: [] for v in variants}
    n_fb = 0
    for i in range(len(test)):
        if rng.random() < args.fb_ratio:
            Xfb.append(S_all[i])
            yfb.append(y_all[i])
            n_fb += 1
            if n_fb % 5 == 0 and sum(yfb) >= 3 and (len(yfb) - sum(yfb)) >= 2:
                X, y = np.array(Xfb), np.array(yfb)
                for vn, v in variants.items():
                    v.update(X, y)
                idx = np.array(unfb)
                if len(idx) >= 8 and len(np.unique(y_all[idx])) == 2:
                    for vn, v in variants.items():
                        au = auroc_fused(idx, v.a, v.b)
                        if au is not None:
                            curve[vn].append({"n_fb": n_fb, "auroc": round(au, 4)})
                    msg = " ".join(f"{vn}={curve[vn][-1]['auroc']:.4f}"
                                   for vn in variants if curve[vn])
                    print(f"  [fb={n_fb}] {msg}", flush=True)
        else:
            unfb.append(i)
        if (i + 1) % 30 == 0:
            print(f"  [stream] {i + 1}/{len(test)} fb={n_fb} eval={len(unfb)}",
                  flush=True)

    idx = np.array(unfb)
    final = {}
    if len(idx) >= 8 and len(np.unique(y_all[idx])) == 2:
        final["equal"] = round(auroc_fused(idx, eq_vec), 4)
        final["fit_weights"] = round(auroc_fused(idx, fit_vec), 4)
        for vn, v in variants.items():
            final[vn] = round(auroc_fused(idx, v.a, v.b), 4)
        # 各单槽位（上限参考）
        for j, n in enumerate(names):
            v_onehot = np.zeros(k)
            v_onehot[j] = 1.0
            final[f"slot_{n}"] = round(auroc_fused(idx, v_onehot), 4)

    out = {"category": args.category, "mode": args.mode,
           "n_test": len(test), "n_fb": n_fb, "n_eval": len(unfb),
           "n_eval_defect": int(y_all[idx].sum()) if len(idx) else None,
           "fit_weights": {n: round(float(fit_vec[j]), 4) for j, n in enumerate(names)},
           "online_cold_weights": {n: round(float(w), 4) for n, w in
                                   zip(names, np.abs(variants["online_cold"].a))},
           "online_prior_weights": {n: round(float(w), 4) for n, w in
                                    zip(names, np.abs(variants["online_prior"].a))},
           "final_auroc": final,
           "curve": curve}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    out_path = os.path.join(cfg["output_dir"],
                            f"diag_online_router_{args.category}_{args.mode}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
