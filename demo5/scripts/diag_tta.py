"""对位偏移 TTA 修复验证（U100，2026-08-24，评审 §11 鲁棒性 45 分的对症攻坚）

U95 发现 bottle 平移 10px 误报 93%（DINO+记忆库位置敏感）。本脚本验证 TTA
（测试时平移搜索，取最小 fused 分数——正确对齐时分数最低）能否修复对位偏移误报。

用法: python scripts/diag_tta.py
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


def shifted(img, dx, dy):
    """平移图像（BORDER_REPLICATE 填充）。"""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def tta_score(pipe, img, offsets):
    """TTA：对多个平移打分，取最小 fused。"""
    best = float("inf")
    for dx, dy in offsets:
        r = pipe.predict_frame(shifted(img, dx, dy))
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

    goods = [p for p, y in b["test"] if y == 0]
    defects = [p for p, y in b["test"] if y == 1]
    rng = np.random.default_rng(42)
    goods = sorted(rng.choice(goods, min(15, len(goods)), replace=False).tolist())
    defects = sorted(rng.choice(defects, min(15, len(defects)), replace=False).tolist())

    offsets = [(0, 0), (-10, 0), (10, 0), (0, -10), (0, 10),
               (-10, -10), (10, 10), (-10, 10), (10, -10)]

    # 基线（无扰动）误报/漏检
    base_fp = sum(1 for p in goods if pipe.predict(p)["decision"] != "normal")
    base_fn = sum(1 for p in defects if pipe.predict(p)["decision"] == "normal")

    # 平移 10px 扰动后：无 TTA vs 有 TTA
    for shift in [10, 20]:
        fp_plain = fn_plain = fp_tta = fn_tta = 0
        for p in goods:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = pipe.predict_frame(shifted(img, shift, 0))
            if r["decision"] != "normal":
                fp_plain += 1
            s = tta_score(pipe, img, offsets)
            if s >= pipe.decider.tau_gray:
                fp_tta += 1
        for p in defects:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = pipe.predict_frame(shifted(img, shift, 0))
            if r["decision"] == "normal":
                fn_plain += 1
            s = tta_score(pipe, img, offsets)
            if s < pipe.decider.tau_gray:
                fn_tta += 1
        n = len(goods); nd = len(defects)
        print(f"[U100] 平移 {shift}px: 误报 无TTA {fp_plain}/{n} vs TTA {fp_tta}/{n} | "
              f"漏检 无TTA {fn_plain}/{nd} vs TTA {fn_tta}/{nd}", flush=True)

    print(f"[U100] 基线（无扰动）: 误报 {base_fp}/{len(goods)} 漏检 {base_fn}/{len(defects)}", flush=True)


if __name__ == "__main__":
    main()
