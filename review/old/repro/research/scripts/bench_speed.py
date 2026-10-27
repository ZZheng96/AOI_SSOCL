"""U86 速度红线实测（2026-08-20）：DINO 前向单批耗时 + 16 块打分单图耗时。

目的：确定冲 0.9 方案的耗时预算（块打分 vs 滑窗 crop 打分）。
协议：3 张不同尺寸 test 图（大 3456x4608 / 中 1200x1600 / 小 816x1088），
16 块提取 + 滑窗 crop 提取分别计时。纯实测，不涉及训练。
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import cv2

from src.backbone.dino import FrozenDINO

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42
GRID = 4


def grid4_blocks(img):
    h, w = img.shape[:2]
    gh, gw = h // GRID, w // GRID
    blocks = []
    for r in range(GRID):
        for c in range(GRID):
            blocks.append(img[r * gh:min(h, (r + 1) * gh),
                             c * gw:min(w, (c + 1) * gw)])
    return blocks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U86 速度实测] device={device}", flush=True)
    backbone = FrozenDINO(output_layer=11, grid=16, device=device)
    paths = [os.path.join(GYU_ROOT, "test", "images", "12320.jpg"),   # 大
             os.path.join(GYU_ROOT, "test", "images", "12321.jpg"),   # 中
             os.path.join(GYU_ROOT, "test", "images", "12322.jpg")]   # 小

    # DINO 单批（8 张 448 输入）耗时
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    t0 = time.time()
    _ = backbone.extract_tiles([cv2.resize(i, (448, 448)) for i in imgs], batch=8)
    t_dino = time.time() - t0
    print(f"  DINO 单批(8张)前向: {t_dino * 1000:.0f}ms")

    for p, img in zip(paths, imgs):
        h, w = img.shape[:2]
        # load 计时
        t0 = time.time()
        _ = cv2.imread(p)
        t_load = time.time() - t0
        # 16 块提取（2 批）
        blocks = grid4_blocks(img)
        t0 = time.time()
        _ = backbone.extract_tiles(blocks, batch=8)
        t_blocks = time.time() - t0
        # 滑窗 crop 699 计时（overlap 0.5 / stride 0.35）
        from src.slots.shead import _sliding_grid
        for ov, tag in [(0.5, "ov0.5"), (0.3, "ov0.3")]:
            crops, ny, nx, trunc = _sliding_grid(img, 699, ov, max_crops=200)
            t0 = time.time()
            _ = backbone.extract_tiles(crops, batch=8)
            t_crop = time.time() - t0
            print(f"  {w}x{h} load={t_load * 1000:.0f}ms 16块={t_blocks * 1000:.0f}ms "
                  f"| crop699 {tag} {len(crops)}窗={t_crop * 1000:.0f}ms "
                  f"(单图总计~{t_load * 1000 + t_crop * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
