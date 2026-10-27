"""U86v7 速度实测（2026-08-20）：crop 打分全量速度复核 + 优化空间。

verify_u86v4_full 实测 2.7s/张超红线。本脚本抽 30 张全量 test 图（大图为主），
对比：a) crop 打分 ov0.5（117 窗）b) crop 打分 ov0.3（63 窗）c) 16 块打分。
分别计时，确认超红线原因与优化空间。
"""
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

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _parse_yolo_boxes,
                             _sliding_grid)

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
GRID = 4
SCORE_MAX_CROPS = 200


def grid4_blocks(img):
    h, w = img.shape[:2]
    gh, gw = h // GRID, w // GRID
    return [img[r * gh:min(h, (r + 1) * gh), c * gw:min(w, (c + 1) * gw)]
            for r in range(GRID) for c in range(GRID)]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "m0", "u86v4_head.pt"), map_location=device)
    print(f"[U86v7 速度实测] device={device}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    crop_size = int(ckpt["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    img_dir = os.path.join(GYU_ROOT, "test", "images")
    paths = sorted(os.listdir(img_dir))[:30]
    paths = [os.path.join(img_dir, f) for f in paths
             if f.lower().endswith((".jpg", ".png", ".jpeg"))][:30]
    print(f"  抽样 {len(paths)} 张", flush=True)

    # warmup
    img0 = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    def score_crop(img, ov):
        crops, ny, nx, trunc = _sliding_grid(img, crop_size, ov, SCORE_MAX_CROPS)
        f = backbone.extract_tiles(crops, batch=batch)
        x = _image_level_feats(f.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        s = shead.head(x.to(device)).reshape(-1).float()
        if not trunc:
            smap = s.reshape(ny, nx)[None, None]
            pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
            return float(F.conv2d(pad, k33).max().item()), len(crops)
        return float(s.topk(min(3, len(s))).values.mean().item()), len(crops)

    def score_block(img):
        blocks = grid4_blocks(img)
        f = backbone.extract_tiles(blocks, batch=batch)
        x = _image_level_feats(f.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        s = shead.head(x.to(device)).reshape(-1)
        return float(s.topk(3).values.mean().item())

    for ov, tag in [(0.5, "ov0.5"), (0.3, "ov0.3")]:
        t0 = time.time()
        nw = 0
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            _, n = score_crop(img, ov)
            nw += n
        dt = time.time() - t0
        print(f"  crop {tag}: {len(paths)} 张 {dt:.0f}s -> {dt / len(paths) * 1000:.0f}ms/张 "
              f"(平均 {nw / len(paths):.0f} 窗)", flush=True)

    t0 = time.time()
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        _ = score_block(img)
    dt = time.time() - t0
    print(f"  block 16块: {len(paths)} 张 {dt:.0f}s -> {dt / len(paths) * 1000:.0f}ms/张",
          flush=True)


if __name__ == "__main__":
    main()
