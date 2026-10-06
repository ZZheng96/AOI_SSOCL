"""TTA 全候选（gamma+亮度+平移）误报抑制验证（U104，2026-08-24）

评审 §14.2 核心关切：扰动下误报 fp 升高（γ1.3 0.667、平移 0.933），TTA 已降但残余
（gamma13 33%、bright-25 40%）仍高于基线（13%）。本脚本验证"gamma+亮度+平移"全候选
联合 TTA 能否把扰动误报压到接近基线。

用法: python scripts/diag_lighting_full.py
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


def tta_full(pipe, img):
    best = float("inf")
    cands = []
    for g in [0.8, 1.25]:
        cands.append(gamma(img, g))
    for dd in [-25, 25]:
        cands.append(bright(img, dd))
    for dx in [-10, 10]:
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        cands.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                    borderMode=cv2.BORDER_REPLICATE))
    for c in cands:
        r = pipe.predict_frame(c)
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

    for name, pert in [("bright-25", lambda i: bright(i, -25)),
                       ("bright+25", lambda i: bright(i, 25)),
                       ("gamma13", lambda i: gamma(i, 1.3))]:
        fp_plain = fp_tta = fn = 0
        for pp in goods:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r = pipe.predict_frame(imgp)
            if r["decision"] != "normal":
                fp_plain += 1
            s = tta_full(pipe, imgp)
            if s >= pipe.decider.tau_gray:
                fp_tta += 1
        for pp in defs:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r = pipe.predict_frame(imgp)
            if r["decision"] == "normal":
                fn += 1
        print(f"[U104] {name}: 误报 无TTA {fp_plain}/15 vs 全候选TTA {fp_tta}/15 | 漏检 {fn}/15",
              flush=True)


if __name__ == "__main__":
    main()
