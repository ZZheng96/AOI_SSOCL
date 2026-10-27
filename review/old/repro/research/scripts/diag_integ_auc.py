"""U86 集成诊断：pipeline test 口径（分层 15+45）下 u86v8 head 的 crop 打分 AUROC。

对比：pipeline evaluate 报告 0.7785（fused）。本脚本用 U86 相同 crop_scores 逻辑
（ov0.3 + 3x3 平滑 max）在同一个 test 集上打分，确认差异来源是口径还是实现。
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
    cfg["slots"]["shead"]["score_overlap"] = 0.3
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
    shead.crop_size = int(ck["crop_size"])
    shead.chan_stats = ck["chan_stats"]
    shead.patch_stats = ck["patch_stats"]
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    test = bundle["test"]
    print(f"  pipeline test 集: {len(test)} 张 "
          f"(缺陷 {sum(1 for _, l in test if l == 1)} + 正常 "
          f"{sum(1 for _, l in test if l == 0)})", flush=True)

    t0 = time.time()
    scores = []
    ys = []
    for p, lab in test:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        crops, ny, nx, trunc = _sliding_grid(img, shead.crop_size, 0.3, 200)
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
        ys.append(lab)
    auc = float(roc_auc_score(ys, scores))
    print(f"  [U86 crop_scores 同 test 集] AUROC={auc:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    # 分正常/缺陷报告
    for lab, tag in ((1, "缺陷"), (0, "正常")):
        ss = [sc for sc, y in zip(scores, ys) if y == lab]
        print(f"    {tag}图分数 mean={np.mean(ss):.3f} std={np.std(ss):.3f} "
              f"median={np.median(ss):.3f}", flush=True)


if __name__ == "__main__":
    main()
