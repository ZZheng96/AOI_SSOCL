"""U86v4 全量验证（2026-08-20）：u86v4_head.pt 在全 test 域上的图级 AUROC。

U86v4 训练：valid(seed=5, 30张) 门控选出 best head（valid 0.8222），
在 test(seed=123, 60张) 上报告 0.8911。
本脚本：同一 head 在 (a) test(seed=123) 60 张复现 (b) 全 test 域 1113 张
（1053 缺陷 + 60 正常）最终报告。纯评测不学习。
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

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SCORE_OVERLAP = 0.5
SCORE_MAX_CROPS = 200


def load_boxes(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def pick_test(n_def, n_nor, seed=123, excl=None):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    defect, normal = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir, f)
        if excl and p in excl:
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            defect.append(p)
        else:
            normal.append(p)
    di = rng.choice(len(defect), min(n_def, len(defect)), replace=False)
    ni = rng.choice(len(normal), min(n_nor, len(normal)), replace=False)
    return [(defect[i], 1) for i in di] + [(normal[i], 0) for i in ni]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "m0", "u86v4_head.pt"), map_location=device)
    print(f"[U86v4 全量验证] device={device} crop_size={ckpt['crop_size']} "
          f"best_valid={ckpt['best_valid']:.4f} test_base={ckpt['test_base']:.4f} "
          f"test_final={ckpt['test_final']:.4f}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    crop_size = int(ckpt["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    def crop_scores_paths(paths, tag=""):
        out = []
        for i, p in enumerate(paths):
            img = load_boxes(p)
            crops, ny, nx, trunc = _sliding_grid(img, crop_size,
                                                 SCORE_OVERLAP, SCORE_MAX_CROPS)
            f = backbone.extract_tiles(crops, batch=batch)
            x = _image_level_feats(f.float().to(device),
                                   shead.chan_stats, shead.patch_stats)
            s = shead.head(x.to(device)).reshape(-1).float()
            if not trunc:
                smap = s.reshape(ny, nx)[None, None]
                pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
                out.append(float(F.conv2d(pad, k33).max().item()))
            else:
                out.append(float(s.topk(min(3, len(s))).values.mean().item()))
            if tag and (i + 1) % 100 == 0:
                print(f"    {tag} 进度 {i + 1}/{len(paths)}", flush=True)
        return out

    # (a) 复现 test(seed=123) 60 张（排除 valid，与训练 test_set 一致）
    valid_paths = {p for p, _ in pick_test(15, 15, seed=5)}
    rep_set = pick_test(30, 30, seed=123, excl=valid_paths)
    rep_y = [l for _, l in rep_set]
    t0 = time.time()
    rep_s = crop_scores_paths([p for p, _ in rep_set])
    rep_auc = float(roc_auc_score(rep_y, rep_s))
    print(f"  [复现 test(seed=123)] {len(rep_set)}张 AUROC={rep_auc:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # (b) 全 test 域（可选，约 40 分钟；已跑过 AUROC=0.8620，默认跳过）
    if os.environ.get("RUN_FULL") != "1":
        print(f"  [跳过全量] 已跑过: 1113张 AUROC=0.8620 (RUN_FULL=1 时重跑)", flush=True)
        return
    full = pick_test(100000, 100000, seed=999)
    full_y = [l for _, l in full]
    t0 = time.time()
    full_s = crop_scores_paths([p for p, _ in full])
    full_auc = float(roc_auc_score(full_y, full_s))
    # 分类准确率（阈值无关的额外参考）：正常图分数 vs 缺陷图分数中位数
    import numpy as np
    full_s = np.array(full_s)
    full_y = np.array(full_y)
    def_med = float(np.median(full_s[full_y == 1]))
    nor_med = float(np.median(full_s[full_y == 0]))
    print(f"  [全量 test] {len(full)}张 (缺陷 {sum(full_y)} + 正常 {len(full_y) - sum(full_y)}) "
          f"AUROC={full_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  分数中位数: 缺陷={def_med:.3f} 正常={nor_med:.3f}", flush=True)
    print(f"  结论: 全量 AUROC {'>=0.9 达标' if full_auc >= 0.9 else '%.4f (差 %.4f)' % (full_auc, 0.9 - full_auc)}")


if __name__ == "__main__":
    main()
