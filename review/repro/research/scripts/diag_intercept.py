"""U105 诊断：易品类学习负增益根因——缺陷库拦截加分误伤正常样本（2026-08-25）

m5 实测（子代理重跑，2026-08-25）：transistor -0.0311 / tubes -0.0489 /
bracket_black -0.0622，负增益品类 eval 集上 eval_boost_gt0=21/20（其中正常样本
大量 boost>0）。本脚本对 transistor 学习流后，统计 eval 集正常/缺陷样本的
拦截加分分量：sims_max（缺陷库近邻 cos）、n_sim（正常库近邻 cos）、
rel=sims_max-n_sim、boost——精确定位误加分机制与修复方向。

用法: python scripts/diag_intercept.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")

SSOCL_CFG = {
    "banks": {"ext_max": 64, "cluster_k": 3, "sim_thresh": 0.75,
              "intercept_topk": 8, "intercept_boost": 0.25, "intercept_sim": 0.15},
    "gate": {"tol": 0.05, "anchor_size": 20},
    "active": {"top_n": 10},
    "incubate": {"min_samples": 8},
    "router_finetune": {"trigger": 6, "lr": 1e-4, "epochs": 5},
    "height_suppress": False,
    "height_recal": True,
    "head_ft": {"enabled": True, "min_pos": 20, "min_neg": 40,
                "lr": 1e-4, "epochs": 3, "anchor_n": 20, "anchor_lambda": 1.0,
                "opt": "adam", "tol_down": 0.35},
    "disc_ft": {"enabled": True, "min_pos": 60, "min_neg": 120,
                "lr": 1e-4, "epochs": 3, "batch": 128, "tol_down": 0.35},
    "thresh_recal": {"enabled": True, "min_n": 20, "max_fp_rate": 0.5,
                     "cooldown": 10},
}


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[U105] 加载 mvtec/transistor（100+30）...", flush=True)
    b = mvtec_like.load_category(cfg["datasets"]["mvtec"], "transistor",
                                 100, 30, 42, max_test=150)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)

    test = b["test"]
    eval_n = 30
    eval_items = test[-eval_n:]
    stream = test[:len(test) - eval_n]
    pipe.enable_ssocl(SSOCL_CFG, train_normal_paths=b["init_normal"],
                      rng=np.random.default_rng(42))
    # 学习流：同 simulate_learning（ratio 0.5），但只跑反馈不重算 AUROC
    rng = np.random.default_rng(42)
    n_fb = 0
    for i, (path, y) in enumerate(stream):
        r = pipe.predict(path)
        pred = 1 if r["decision"] != "normal" else 0
        if rng.random() < 0.5:
            correct = (pred == y)
            pipe.feedback(path, "correct" if correct else "wrong", label=y)
            n_fb += 1
        if (i + 1) % 50 == 0:
            print(f"  [stream] {i + 1}/{len(stream)} 反馈={n_fb}", flush=True)
    print(f"[U105] 学习完成，反馈={n_fb}，缺陷库={len(pipe.handler.defect_bank.samples)} 样本",
          flush=True)

    db = pipe.handler.defect_bank
    nb = pipe.handler.normal_bank
    # 对 eval 集逐张算拦截分量
    rows = []
    for p, y in eval_items:
        it = pipe._get_item(p)
        f = db._norm_patch(it["feats"]) if hasattr(db, "_norm_patch") else None
        # 复用 intercept_score 内部逻辑：取 rel 分量
        feats = torch.as_tensor(np.asarray(it["feats"], dtype=np.float32))
        fn = torch.nn.functional.normalize(
            feats.flatten(2).permute(0, 2, 1).reshape(-1, 384), dim=1).half()
        # 全 CPU 计算（normal_bank core/ext 在 CPU 驻留，避免设备不匹配）
        all_ref = torch.cat([s[0] for s in db.samples], dim=0).float()
        sims_max = torch.empty(fn.shape[0], dtype=torch.float32)
        for p0 in range(0, fn.shape[0], 1024):
            s = fn[p0:p0 + 1024].float() @ all_ref.T
            sims_max[p0:p0 + 1024] = s.max(dim=1).values
        n_sim = torch.as_tensor(nb._sim_to(fn), dtype=torch.float32)
        rel = sims_max - n_sim
        top_rel = rel.topk(min(8, rel.shape[0])).values
        boost = db.intercept_score(it["feats"], normal_bank=nb)
        rows.append({"label": int(y), "sims_max_mean": float(sims_max.mean()),
                     "sims_max_max": float(sims_max.max()),
                     "n_sim_mean": float(n_sim.mean()),
                     "rel_mean": float(rel.mean()),
                     "rel_top8_mean": float(top_rel.mean()),
                     "rel_top8_max": float(top_rel.max()),
                     "n_gt015": int((rel >= 0.15).sum()),
                     "boost": boost})
        print(f"[U105] label={y} boost={boost:.4f} "
              f"sims_max[mean/max]={sims_max.mean():.3f}/{sims_max.max():.3f} "
              f"n_sim={n_sim.mean():.3f} rel_top8[mean/max]="
              f"{top_rel.mean():.3f}/{top_rel.max():.3f} "
              f"n_patch_rel>=0.15={int((rel >= 0.15).sum())}", flush=True)

    # 汇总：正常 vs 缺陷
    norm = [r for r in rows if r["label"] == 0]
    defs = [r for r in rows if r["label"] == 1]
    print("\n[U105] 汇总（正常 vs 缺陷）：", flush=True)
    for name, grp in [("正常", norm), ("缺陷", defs)]:
        if not grp:
            continue
        print(f"  {name} n={len(grp)} "
              f"boost>0={sum(1 for r in grp if r['boost'] > 0)}/{len(grp)} "
              f"boost均值={np.mean([r['boost'] for r in grp]):.4f} "
              f"rel_top8均值={np.mean([r['rel_top8_mean'] for r in grp]):.4f} "
              f"rel_top8_max均值={np.mean([r['rel_top8_max'] for r in grp]):.4f} "
              f"rel>=0.15 patch数={np.mean([r['n_gt015'] for r in grp]):.1f}", flush=True)


if __name__ == "__main__":
    main()
