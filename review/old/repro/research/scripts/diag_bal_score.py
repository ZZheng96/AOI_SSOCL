"""U86v17 平衡口径打分对比：u86v8_head 在平衡 test（30+30）上不同打分方式。

目标：找 >0.8856（v8 的 ov0.3 成绩）的打分方式，无需重训。
打分方式：ov0.3 / adaptive(target=40) / ov0.5 / crop800-ov0.1。
协议：test = pick_test(30,30,seed=123,excl valid)，纯评测不学习。
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
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _parse_yolo_boxes,
                             _sliding_grid)
from crop_score_retrain import pick_test, load_boxes, adaptive_overlap

U86_HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                        "u86v8_head.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ck = torch.load(U86_HEAD, map_location=device)
    print(f"[U86v17 平衡口径打分对比] crop_size={ck['crop_size']}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ck["head"])
    shead.chan_stats, shead.patch_stats = ck["chan_stats"], ck["patch_stats"]
    crop_size = int(ck["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    valid_paths = {p for p, _ in pick_test(15, 15, seed=5)}
    test = pick_test(30, 30, seed=123, excl=valid_paths)
    ys = [l for _, l in test]
    imgs = [load_boxes(p)[0] for p, _ in test]
    print(f"  平衡 test: {len(test)} 张 (缺陷 {sum(ys)} + 正常 {len(ys) - sum(ys)})",
          flush=True)
    img0 = imgs[0]
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    def score_all(cs, ov, tag):
        scores = []
        t0 = time.time()
        for img in imgs:
            csm = min(cs, img.shape[0], img.shape[1])
            if ov is None:
                ov_use = adaptive_overlap(img, csm, target=40)
            else:
                ov_use = ov
            crops, ny, nx, trunc = _sliding_grid(img, csm, ov_use, 200)
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
        auc = float(roc_auc_score(ys, scores))
        print(f"  [{tag}] AUROC={auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return auc

    score_all(crop_size, 0.3, "crop699-ov0.3 (v8基线, 期望0.8856)")
    score_all(crop_size, None, "crop699-adaptive40")
    score_all(crop_size, 0.5, "crop699-ov0.5")
    score_all(800, 0.1, "crop800-ov0.1 (集成配置)")


if __name__ == "__main__":
    main()
