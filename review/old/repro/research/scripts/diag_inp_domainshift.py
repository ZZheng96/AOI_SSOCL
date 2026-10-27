"""inp 域差免疫验证：GYU-DET 上 sem vs inp 的 val→test 掉点对比

假设：sem 依赖外部参考库，域差下 val→test 掉点大；inp 图内自找原型，
不依赖外部库，val→test 应基本不掉。同时对比等权融合（全 6 槽位）效果。
迭代用 max_eval 120。
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
from src.slots.inp import InpSlot
from src.data import gyudet, datalocal
from src.common.tiling import compute_tiles, extract_tiles
from src.common.io import load_image
from src.fusion.calibrate import CDFCalibrator
from src.fusion.fixed import fuse


def score_batch(bb, slots, items):
    slot_raw = {n: [] for n in slots}
    for it in items:
        img = load_image(it["path"])
        ti = extract_tiles(img, compute_tiles(img.shape[:2], "single", 512, 448))
        feats = bb.extract_tiles(ti).cpu()
        for n, s in slots.items():
            sc, _ = s.score_tiles(feats, ti)
            slot_raw[n].append(float(np.sort(sc)[-3:].mean()))
    return slot_raw


def run(name, bundle, device, n_eval=120):
    print(f"\n===== {name} =====")
    bb = FrozenDINO(output_layer=9, grid=32, device=device)
    sem = SemSlot({"coreset_size": 512, "tile_topk_ratio": 0.05}, device)
    inp = InpSlot({}, device)
    slots = {"sem": sem, "inp": inp}
    ctx = {"train_tile_feats": []}
    for p in bundle["init_normal"][:60]:
        img = load_image(p)
        ctx["train_tile_feats"].append(bb.extract_tiles([img]).cpu())
    sem.fit(ctx)
    inp.fit(ctx)

    cal = {}
    train_raw = score_batch(bb, slots, [{"path": p} for p in bundle["init_normal"][:60]])
    for n in slots:
        cal[n] = CDFCalibrator(256).fit(train_raw[n])

    splits = []
    if bundle["val"]:
        splits.append(("val", bundle["val"][:n_eval]))
    if bundle["test"]:
        splits.append(("test", bundle["test"][:n_eval]))
    for tag, items in splits:
        labels = [y for _, y in items]
        raw = score_batch(bb, slots, [{"path": p} for p, _ in items])
        s2 = {n: cal[n].transform(raw[n]) for n in slots}
        f2 = [fuse({n: s2[n][i] for n in slots}) for i in range(len(labels))]
        print(f"  {tag}: fused={roc_auc_score(labels, f2):.4f} "
              f"sem={roc_auc_score(labels, s2['sem']):.4f} "
              f"inp={roc_auc_score(labels, s2['inp']):.4f}")


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = gyudet.load_gyudet(cfg["datasets"]["gyu_det"], 100, 30, 42)
    run("gyudet", g, "cuda", n_eval=120)
    gf = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger", 100, 30, 42)
    run("gold_finger", gf, "cuda", n_eval=120)


if __name__ == "__main__":
    main()
