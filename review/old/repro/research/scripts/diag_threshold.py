"""决策阈值失效精确诊断（U99，2026-08-24，目标模式：标准数据集真实可用性）

U95 发现 bottle 基线误报 0.6（AUROC 0.9985 排序好但决策阈值失效）。本脚本定位：
哪个槽位导致 test/good 的 fused 分数右移（超过 train/good 分位阈值），并验证几种修复。

用法: python scripts/diag_threshold.py
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
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = mvtec_like.load_category(cfg["datasets"]["mvtec"], "bottle", 100, 30, 42,
                                 n_eval_good=0, max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)

    goods = [p for p, y in b["test"] if y == 0]
    # 各槽位原始分 + 校准分：train/good vs test/good
    train_raw = {n: [] for n in pipe._score_slots}
    test_raw = {n: [] for n in pipe._score_slots}
    # 用 fit 时 train_items 的分数（重新打分）
    for p in b["init_normal"]:
        r = pipe.predict(p)
        for n in pipe._score_slots:
            train_raw[n].append(r["raw_scores"][n])
    for p in goods:
        r = pipe.predict(p)
        for n in pipe._score_slots:
            test_raw[n].append(r["raw_scores"][n])

    print(f"[U99] bottle: train/good {len(b['init_normal'])} vs test/good {len(goods)}", flush=True)
    print(f"[U99] tau_gray={pipe.decider.tau_gray:.4f} tau_high={pipe.decider.tau_high:.4f}", flush=True)
    print("[U99] 各槽位 原始分(mean) / 校准分(mean) / test超train-max比例:", flush=True)
    for n in pipe._score_slots:
        tr = np.array(train_raw[n]); te = np.array(test_raw[n])
        tr_cal = pipe.calibrators[n].transform(tr)
        te_cal = pipe.calibrators[n].transform(te)
        over = (te > tr.max()).mean()
        print(f"  {n:8s} train_raw={tr.mean():.4f} test_raw={te.mean():.4f} "
              f"| train_cal={tr_cal.mean():.3f} test_cal={te_cal.mean():.3f} "
              f"| test超train-max {over:.2f}", flush=True)

    # fused 分布
    fs_g = [pipe.predict(p)["fused"] for p in goods]
    print(f"\n[U99] test/good fused: mean={np.mean(fs_g):.3f} max={np.max(fs_g):.3f} "
          f">tau_gray {sum(1 for x in fs_g if x >= pipe.decider.tau_gray)}/{len(fs_g)}", flush=True)


if __name__ == "__main__":
    main()
