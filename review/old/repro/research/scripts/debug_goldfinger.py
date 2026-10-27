"""M0 诊断脚本：量化 gold_finger 三问题（blob 常数化 / sem 反转 / 速度分布）

只做研究性诊断，不进主干。
"""
import os
import sys
import time
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.tiling import compute_tiles, extract_tiles
from src.common.io import load_image
from src.backbone.dino import FrozenDINO
from src.data import datalocal


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger",
                                     100, 30, cfg["seed"])
    good = [p for p, y in bundle["test"] if y == 0][:8]
    bad = [p for p, y in bundle["test"] if y == 1][:8]
    train = bundle["init_normal"][:8]

    # ---------- 1. blob 原始响应诊断 ----------
    from skimage.feature import blob_log
    import cv2

    def blob_probe(path, thr):
        img = load_image(path)
        tiles = compute_tiles(img.shape[:2], "tiles36", 512, 448)
        out = []
        for t in extract_tiles(img, tiles):
            g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            b = blob_log(g, min_sigma=1, max_sigma=16, num_sigma=5, threshold=thr, overlap=0.3)
            out.append(0.0 if len(b) == 0 else float(np.sort(b[:, 2] ** 2)[-3:].mean()))
        return max(out)

    print("== blob 原始分数（tiles36 全分辨率） ==")
    for thr in (0.03, 0.01, 0.005):
        g = [blob_probe(p, thr) for p in good]
        d = [blob_probe(p, thr) for p in bad]
        print(f"thr={thr}: good={np.round(g, 3)} defect={np.round(d, 3)}")

    # ---------- 2. sem 分数分布（好 vs 坏） ----------
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    from src.slots.sem import SemSlot

    def extract(paths):
        items = []
        for p in paths:
            img = load_image(p)
            tiles = compute_tiles(img.shape[:2], "tiles36", 512, 448)
            ti = extract_tiles(img, tiles)
            items.append((p, backbone.extract_tiles(ti).cpu()))
        return items

    tr_items = extract(train)
    sem = SemSlot(cfg["slots"]["sem"], device).fit(
        {"train_tile_feats": [f for _, f in tr_items]})
    print("\n== sem tile 分数统计 ==")
    for tag, paths in (("train", [p for p, _ in tr_items]), ("test_good", good), ("test_defect", bad)):
        items = extract(paths)
        per_img = []
        for _, f in items:
            ts, _ = sem.score_tiles(f)
            per_img.append(float(np.sort(ts)[-3:].mean()))
        print(f"{tag}: {np.round(per_img, 4)}")

    # ---------- 3. 速度分解（单图 tiles36） ----------
    print("\n== 单图耗时分解 (tiles36) ==")
    p = good[0]
    t0 = time.time(); img = load_image(p); t_decode = time.time() - t0
    t0 = time.time()
    tiles = compute_tiles(img.shape[:2], "tiles36", 512, 448)
    ti = extract_tiles(img, tiles)
    t_tile = time.time() - t0
    torch.cuda.synchronize(); t0 = time.time()
    feats = backbone.extract_tiles(ti).cpu(); torch.cuda.synchronize()
    t_dino = time.time() - t0
    t0 = time.time(); sem.score_tiles(feats); t_sem = time.time() - t0
    t0 = time.time()
    for t in ti:
        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blob_log(g, min_sigma=1, max_sigma=16, num_sigma=5, threshold=0.01, overlap=0.3)
    t_blob = time.time() - t0
    print(f"decode={t_decode:.2f}s tile={t_tile:.2f}s dino={t_dino:.2f}s "
          f"sem={t_sem:.2f}s blob={t_blob:.2f}s (n_tiles={len(ti)})")


if __name__ == "__main__":
    main()
