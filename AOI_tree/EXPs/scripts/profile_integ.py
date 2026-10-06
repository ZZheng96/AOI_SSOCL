"""U86v16 集成速度 profile：定位 pipeline 单图耗时构成（5 张 test）。"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
import cv2

from m0_baseline import build
from src.data import gyudet
from src.eval.offline import Pipeline
from src.common.io import load_image as pipe_load
from src.common.tiling import compute_tiles, extract_tiles

U86_HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                        "u86v8_head.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    cfg["slots"]["shead"]["score_overlap"] = 0.3
    cfg["slots"]["shead"]["extract_batch"] = 16
    cfg["slots"]["shead"]["adaptive_score"] = True
    cfg["slots"]["shead"]["adaptive_target"] = 36
    cfg["slots"]["open"]["enabled"] = False
    cfg["fusion"]["skip_zero_weight"] = True
    bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], cfg["seed"],
                                max_test=60)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    pipe.hier = False
    ck = torch.load(U86_HEAD, map_location=device)
    shead = pipe.slots["shead"]
    shead.head.load_state_dict(ck["head"])
    shead.crop_size = int(ck["crop_size"])
    shead.chan_stats = ck["chan_stats"]
    shead.patch_stats = ck["patch_stats"]

    test = bundle["test"][:5]
    t_load = t_tile = t_dino = t_crop = t_rec = 0.0
    for p, _ in test:
        t0 = time.time(); img = pipe_load(p); t_load += time.time() - t0
        t0 = time.time()
        tiles = compute_tiles(img.shape[:2], cfg["tiling"]["mode"],
                              cfg["tiling"]["tile_size"], cfg["tiling"]["stride"])
        tile_imgs = extract_tiles(img, tiles)
        t_tile += time.time() - t0
        t0 = time.time()
        bb = cfg.get("backbone", {})
        feats = backbone.extract_tiles(tile_imgs, batch=int(bb.get("extract_batch", 16))).cpu()
        t_dino += time.time() - t0
        t0 = time.time()
        item = {"path": p, "img": img, "tiles": tiles, "tile_imgs": tile_imgs, "feats": feats}
        s, ts, hms = pipe._slot_image_score(shead, item)
        t_crop += time.time() - t0
        # 窗数验证（adaptive 是否生效）
        ov = shead._adaptive_overlap(img, 36)
        from src.slots.shead import _sliding_grid
        crops, ny, nx, _ = _sliding_grid(img, shead.crop_size, ov, 200)
        print(f"    {os.path.basename(p)}: ov={ov:.2f} 窗={len(crops)} "
              f"crop_size={shead.crop_size}", flush=True)
        t0 = time.time()
        rec = pipe._record(item)
        t_rec += time.time() - t0
    n = len(test)
    print(f"[profile {n} 张] load={t_load/n*1000:.0f}ms tile切图={t_tile/n*1000:.0f}ms "
          f"整图DINO={t_dino/n*1000:.0f}ms shead-crop打分={t_crop/n*1000:.0f}ms "
          f"_record(含crop?重叠)={t_rec/n*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
