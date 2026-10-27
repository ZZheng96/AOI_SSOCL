"""U86v13 速度预检：修正后的 adaptive_overlap 窗数 + 打分耗时（30 张抽样）。"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
import torch

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _sliding_grid)

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"


def n_windows(img, cs, ov):
    h, w = img.shape[:2]
    if h <= cs or w <= cs:
        return 1
    stride = max(1, int(cs * (1 - ov)))
    y_pos = list(range(0, max(1, h - cs + 1), stride))
    x_pos = list(range(0, max(1, w - cs + 1), stride))
    if y_pos[-1] + cs < h:
        y_pos.append(h - cs)
    if x_pos[-1] + cs < w:
        x_pos.append(w - cs)
    return len(y_pos) * len(x_pos)


def adaptive_overlap(img, cs, target=43):
    lo, hi = 0.1, 0.8
    for _ in range(14):
        mid = (lo + hi) / 2
        if n_windows(img, cs, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo if n_windows(img, cs, lo) <= target else hi


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_size = 699
    print(f"[U86v13 速度预检] crop_size={crop_size}", flush=True)
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    batch = 8

    img_dir = os.path.join(GYU_ROOT, "test", "images")
    paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
             if f.lower().endswith((".jpg", ".png", ".jpeg"))][:30]
    img0 = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    t0 = time.time()
    nw = 0
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ov = adaptive_overlap(img, crop_size)
        crops, ny, nx, _ = _sliding_grid(img, crop_size, ov, 200)
        nw += len(crops)
        _ = backbone.extract_tiles(crops, batch=batch)
    dt = time.time() - t0
    print(f"  adaptive target=43: 30 张 {dt:.0f}s -> {dt / 30 * 1000:.0f}ms/张 "
          f"(平均 {nw / 30:.0f} 窗)", flush=True)
    # 窗数分布
    ws = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ov = adaptive_overlap(img, crop_size)
        ws.append(n_windows(img, crop_size, ov))
    print(f"  窗数分布: min={min(ws)} max={max(ws)} median={sorted(ws)[15]}")


if __name__ == "__main__":
    main()
