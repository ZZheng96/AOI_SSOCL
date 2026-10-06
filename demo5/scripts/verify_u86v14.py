"""U86v15 对照验证：u86v14_head.pt 用 v14 相同 crop_scores 逻辑复现 test_final。"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _parse_yolo_boxes,
                             _sliding_grid)
from crop_score_retrain import pick_test, load_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "m0", "u86v14_head.pt"), map_location=device)
    print(f"[U86v15 对照验证] crop_size={ckpt['crop_size']}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    crop_size = int(ckpt["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    valid_paths = {p for p, _ in pick_test(15, 15, seed=5)}
    test_set = pick_test(30, 30, seed=123, excl=valid_paths)
    test_y = [l for _, l in test_set]
    print(f"  test {len(test_set)} 张 (缺陷 {sum(test_y)})", flush=True)

    # v14 crop_scores 逻辑（逐图）
    t0 = time.time()
    scores = []
    for p, _ in test_set:
        img, _ = load_boxes(p)
        crops, ny, nx, trunc = _sliding_grid(img, crop_size, 0.3, 200)
        f = backbone.extract_tiles(crops, batch=batch)
        x = _image_level_feats(f.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        s = shead.head(x.to(device)).reshape(-1).float()
        if not trunc:
            smap = s.reshape(ny, nx)[None, None]
            pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
            scores.append(float(F.conv2d(pad, k33).max().item()))
        else:
            scores.append(float(s.topk(min(3, len(s))).values.mean().item()))
    auc = float(roc_auc_score(test_y, scores))
    print(f"  [复现 crop_scores] AUROC={auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  元数据 test_final={ckpt['test_final']:.4f}")


if __name__ == "__main__":
    main()
