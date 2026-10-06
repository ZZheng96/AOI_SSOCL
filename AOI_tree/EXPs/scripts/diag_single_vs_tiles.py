"""sem 槽位粒度对照：single（整图单前向，demo4 式） vs tiles36

目标：
1. 速度：single 模式是否回到 ~200ms 区间（demo4 161.9ms 兜底验证）
2. 分数：sem-single vs sem-tiles36 在 GYU-DET / gold_finger 上的诚实 AUROC
3. 验证 §6.1 论断：记忆库匹配不是瓶颈，前向次数才是
"""
import os
import sys
import time
import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone.dino import FrozenDINO
from src.slots.sem import SemSlot
from src.data import datalocal, gyudet
from src.common.tiling import compute_tiles, extract_tiles
from src.common.io import load_image
from src.fusion.calibrate import CDFCalibrator


def single_sem_scores(bb, sem, items, with_timing=False):
    """single 模式：整图一次前向 → patch kNN → 图像 top-k"""
    scores, times = [], []
    for it in items:
        img = load_image(it["path"])
        t0 = time.time()
        feats = bb.extract_tiles([img]).cpu()          # (1,384,32,32)
        ts, _ = sem.score_tiles(feats)
        scores.append(float(np.sort(ts)[-3:].mean()))  # patch top-3 聚合
        times.append((time.time() - t0) * 1000)
    return scores, times


def tiles36_sem_scores(bb, sem, items):
    scores = []
    for it in items:
        img = load_image(it["path"])
        ti = extract_tiles(img, compute_tiles(img.shape[:2], "tiles36", 512, 448))
        feats = bb.extract_tiles(ti).cpu()
        ts, _ = sem.score_tiles(feats)
        scores.append(float(np.sort(ts)[-3:].mean()))
    return scores


def run(name, bundle, n_eval=120):
    print(f"\n===== {name} =====")
    bb = FrozenDINO(output_layer=9, grid=32, device="cuda")
    sem = SemSlot({"coreset_size": 512, "tile_topk_ratio": 0.05}, "cuda")
    # fit：single 前向的库（与 demo4 一致的整图 patch 库）
    ctx = {"train_tile_feats": []}
    t0 = time.time()
    for p in bundle["init_normal"][:60]:
        img = load_image(p)
        ctx["train_tile_feats"].append(bb.extract_tiles([img]).cpu())
    sem.fit(ctx)
    print(f"  fit({len(bundle['init_normal'][:60])}张)={time.time()-t0:.1f}s")

    eval_items = [{"path": p, "label": y} for p, y in bundle["test"][:n_eval]]
    labels = [it["label"] for it in eval_items]

    # single
    s1, t1 = single_sem_scores(bb, sem, eval_items, with_timing=True)
    print(f"  single: AUROC={roc_auc_score(labels, s1):.4f} "
          f"ms/img={np.mean(t1):.0f} (中位 {np.median(t1):.0f})")

    # tiles36
    t0 = time.time()
    s36 = tiles36_sem_scores(bb, sem, eval_items)
    dt = (time.time() - t0) / len(eval_items) * 1000
    print(f"  tiles36: AUROC={roc_auc_score(labels, s36):.4f} ms/img={dt:.0f}")


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = gyudet.load_gyudet(cfg["datasets"]["gyu_det"], 100, 30, 42)
    run("gyudet", g, n_eval=120)
    gf = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger", 100, 30, 42)
    run("gold_finger", gf, n_eval=120)


if __name__ == "__main__":
    main()
