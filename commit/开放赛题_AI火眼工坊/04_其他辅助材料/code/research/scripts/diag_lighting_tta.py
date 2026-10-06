"""光照扰动 TTA 验证（U101，2026-08-24，评审 §13 光照鲁棒性 -3 的对症攻坚）

U95 发现 bottle gamma/亮度扰动误报 0.47-0.67（U99 决策阈值修复后基线 0.13，但光照
扰动仍高）。本脚本验证光照 TTA（多个 gamma 候选打分取最小）能否修复光照误报。

用法: python scripts/diag_lighting_tta.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def gamma(img, g):
    return np.clip((img.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)


def bright(img, d):
    return np.clip(img.astype(np.float32) + d, 0, 255).astype(np.uint8)


def tta_gamma(pipe, img, gammas=(0.8, 1.0, 1.25)):
    best = float("inf")
    for g in gammas:
        r = pipe.predict_frame(gamma(img, g))
        best = min(best, r["fused"])
    return best


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = mvtec_like.load_category(cfg["datasets"]["mvtec"], "bottle", 100, 30, 42,
                                 n_eval_good=0, max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)

    rng = np.random.default_rng(42)
    goods = sorted(rng.choice([p for p, y in b["test"] if y == 0], 15, replace=False).tolist())
    defs = sorted(rng.choice([p for p, y in b["test"] if y == 1], 15, replace=False).tolist())

    for name, pert in [("gamma07", lambda i: gamma(i, 0.7)),
                       ("gamma13", lambda i: gamma(i, 1.3)),
                       ("bright-25", lambda i: bright(i, -25)),
                       ("bright+25", lambda i: bright(i, 25))]:
        fp_plain = fp_tta = fn = 0
        for pp in goods:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r = pipe.predict_frame(imgp)
            if r["decision"] != "normal":
                fp_plain += 1
            s = tta_gamma(pipe, imgp)
            if s >= pipe.decider.tau_gray:
                fp_tta += 1
        for pp in defs:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r = pipe.predict_frame(imgp)
            if r["decision"] == "normal":
                fn += 1
        print(f"[U101] {name}: 误报 无TTA {fp_plain}/15 vs gamma-TTA {fp_tta}/15 | 漏检 {fn}/15",
              flush=True)


if __name__ == "__main__":
    main()
