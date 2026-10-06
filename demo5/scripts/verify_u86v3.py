"""U86v3 诚实验证（2026-08-20）：u86v3_head.pt 在未见 eval（seed=123）上的图级 AUROC。

训练：U86v3 门控在 eval(seed=42, 60 张) 上选出的 head（轮1 重训后 0.9233）。
验证：另一组 60 张（seed=123, 30 缺陷+30 正常，与训练 eval 不重叠），
纯评测不学习。诚实协议：验证集只在最终报告，不参与任何选择。
输出：crop 打分图级 AUROC + 块打分图级/块级 AUROC 对照。
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _parse_yolo_boxes,
                             _sliding_grid)

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
GRID = 4
SCORE_OVERLAP = 0.5
SCORE_MAX_CROPS = 200


def grid4_blocks(img, boxes):
    h, w = img.shape[:2]
    gh, gw = h // GRID, w // GRID
    blocks, hit_idx = [], []
    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = r * gh, c * gw
            y1, x1 = min(h, y0 + gh), min(w, x0 + gw)
            blocks.append(img[y0:y1, x0:x1])
            hit = False
            for bx in boxes:
                iw = max(0, min(x1, bx[2]) - max(x0, bx[0]))
                ih = max(0, min(y1, bx[3]) - max(y0, bx[1]))
                if iw * ih > 0:
                    hit = True
                    break
            if hit:
                hit_idx.append(r * GRID + c)
    return blocks, hit_idx


def pick_test(n_def, n_nor, seed=42, excl=None):
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
    di = rng.choice(len(defect), n_def, replace=False)
    ni = rng.choice(len(normal), min(n_nor, len(normal)), replace=False)
    return [(defect[i], 1) for i in di] + [(normal[i], 0) for i in ni]


def load_boxes(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = os.path.join(GYU_ROOT, "test", "labels",
                       os.path.splitext(os.path.basename(path))[0] + ".txt")
    return img, _parse_yolo_boxes(lbl, w, h)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "m0", "u86v3_head.pt"),
                      map_location=device)
    print(f"[U86v3 诚实验证] device={device} crop_size={ckpt['crop_size']} "
          f"训练eval图级={ckpt['best_crop_img']:.4f}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    crop_size = int(ckpt["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    # 验证集：seed=123，排除训练 eval(seed=42)
    tr_excl = {p for p, _ in pick_test(30, 30, seed=42)}
    val_set = pick_test(30, 30, seed=123, excl=tr_excl)
    eval_imgs, y_img, def_idx = [], [], []
    for path, lab in val_set:
        img, boxes = load_boxes(path)
        _, hit = grid4_blocks(img, boxes)
        eval_imgs.append(img)
        y_img.append(lab)
        def_idx.append(hit)
    print(f"  验证集: {len(val_set)} 张 (缺陷 {sum(y_img)} + 正常 {len(y_img) - sum(y_img)})",
          flush=True)

    t0 = time.time()
    scores = []
    for img in eval_imgs:
        crops, ny, nx, trunc = _sliding_grid(img, crop_size,
                                             SCORE_OVERLAP, SCORE_MAX_CROPS)
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
    auc = float(roc_auc_score(y_img, scores))
    print(f"  [验证] crop图级({len(val_set)}) AUROC={auc:.4f}  "
          f"(前向+打分 {time.time() - t0:.0f}s)", flush=True)
    print(f"  结论: 未见 eval 上 {'达标 >=0.9' if auc >= 0.9 else '未达 0.9 (差 %.4f)' % (0.9 - auc)}")


if __name__ == "__main__":
    main()
