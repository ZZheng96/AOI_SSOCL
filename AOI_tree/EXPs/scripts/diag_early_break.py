"""U106 诊断：易品类学习负增益早期破坏源定位（2026-08-25）

m5 实测（U105 消融）：transistor 学习曲线在 feedback=6 处 AUROC 暴跌至 ~0.75，
拦截加分误伤修复（min_hit）后负增益不消除（-0.037）——破坏源在早期学习机制。
本脚本逐条反馈追踪：记录每条反馈的动作（reflow/权重更新/虚高重估/阈值重估/
缺陷入库）+ 每条反馈后 eval 锚定集 AUROC + 关键样本 fused 变化，定位掉点机制。

用法: python scripts/diag_early_break.py
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
from src.ssocl.learning_curve import _eval_auroc

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
    print("[U106] 加载 mvtec/transistor（100+30）...", flush=True)
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
    # 学习前基线
    base = _eval_auroc(pipe, eval_items)
    print(f"[U106] 学习前 eval AUROC={base:.4f}", flush=True)
    rng = np.random.default_rng(42)
    n_fb = 0
    last_au = base
    prev_weights = dict(pipe.weights) if hasattr(pipe, "weights") else None
    for i, (path, y) in enumerate(stream):
        r = pipe.predict(path)
        pred = 1 if r["decision"] != "normal" else 0
        if rng.random() < 0.5:
            correct = (pred == y)
            entry = pipe.feedback(path, "correct" if correct else "wrong", label=y)
            n_fb += 1
            # 每次反馈后重算 eval AUROC（细粒度定位掉点）
            au = _eval_auroc(pipe, eval_items)
            delta = au - last_au
            # 权重变化检测
            w_now = dict(pipe.weights) if hasattr(pipe, "weights") else None
            w_chg = ""
            if prev_weights and w_now:
                ch = {k: round(w_now[k] - prev_weights[k], 3)
                      for k in w_now if abs(w_now[k] - prev_weights[k]) > 0.01}
                if ch:
                    w_chg = " w:" + ",".join(f"{k}{v:+.3f}" for k, v in ch.items())
            prev_weights = w_now
            act = entry.get("action", "?")
            # 机制事件（thresh_recal / height_recal / head_ft / disc_ft）
            ev = ""
            for key in ["thresh_recal", "height_recal", "head_ft", "disc_ft"]:
                n = getattr(pipe.handler, "stats", {}).get(key, 0)
                if n > 0:
                    ev += f" [{key}#{n}]"
            print(f"[U106] fb#{n_fb} y={y} pred={pred} act={act} AUROC={au:.4f}"
                  f" Δ={delta:+.4f}{w_chg}{ev}", flush=True)
            last_au = au
            if n_fb >= 15:      # 只追踪前 15 条反馈（掉点在 6 附近）
                break
    print(f"[U106] 学习后 eval AUROC={_eval_auroc(pipe, eval_items):.4f}", flush=True)


if __name__ == "__main__":
    main()
