"""U86v16 诊断：crop_size 对速度与 AUROC 的影响（fit 一次，多 crop_size 打分）。

目标：找 <1s 红线内可用的 crop_size（大图窗数 = ceil 无重叠网格，crop 越大窗越少）。
协议：fit 全量一次，用 u86v8 head 对 pipeline test 60 张，分别用 crop_size
699/800/900 打分（ov 固定最小 0.1 无重叠），报告 AUROC + 平均窗数 + 耗时。
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

from m0_baseline import build
from src.data import gyudet
from src.eval.offline import Pipeline
from src.slots.shead import _image_level_feats, _sliding_grid

U86_HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                        "u86v8_head.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    cfg["slots"]["shead"]["extract_batch"] = 16
    cfg["slots"]["open"]["enabled"] = False
    cfg["fusion"]["skip_zero_weight"] = True
    bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], cfg["seed"],
                                max_test=60)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    ck = torch.load(U86_HEAD, map_location=device)
    shead = pipe.slots["shead"]
    shead.head.load_state_dict(ck["head"])
    shead.chan_stats = ck["chan_stats"]
    shead.patch_stats = ck["patch_stats"]
    batch = 16
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0
    test = bundle["test"]
    ys = [l for _, l in test]
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p, _ in test]

    img0 = imgs[0]
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    for cs in (699, 800, 900, 1000):
        scores, nws = [], []
        t0 = time.time()
        for img in imgs:
            h, w = img.shape[:2]
            csm = min(cs, h, w)
            ov = 0.1   # 无重叠最小窗（crop 越大窗越少）
            crops, ny, nx, trunc = _sliding_grid(img, csm, ov, 200)
            nws.append(len(crops))
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
        dt = time.time() - t0
        auc = float(roc_auc_score(ys, scores))
        print(f"  crop_size={cs}: AUROC={auc:.4f} 平均窗={np.mean(nws):.0f} "
              f"(min={min(nws)} max={max(nws)}) 打分 {dt/len(test)*1000:.0f}ms/张",
              flush=True)


if __name__ == "__main__":
    main()
