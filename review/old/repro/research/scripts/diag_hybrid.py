"""U86v15 混合打分诊断（2026-08-20）：块快筛召回 + 精查打分基线。

用 u86v14_head.pt（学习后 head）在 test 60 张（seed=123 excl valid）上：
1. 16 块打分 -> top-k(k=1..5) 对缺陷块的召回率（块快筛可靠性，U77 教训）
2. 混合打分（块 top-3 内滑窗 crop 精查 -> max）图级 AUROC vs 纯 crop 打分
3. 速度实测
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
GRID = 4
CS = 699


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
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "m0", "u86v8_head.pt"), map_location=device)
    print(f"[U86v15 混合打分诊断] crop_size={CS} head=u86v8(best 0.8856)",
          flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ckpt["head"])
    shead.chan_stats, shead.patch_stats = ckpt["chan_stats"], ckpt["patch_stats"]
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    valid_paths = {p for p, _ in pick_test(15, 15, seed=5)}
    test_set = pick_test(30, 30, seed=123, excl=valid_paths)
    print(f"  test {len(test_set)} 张", flush=True)

    # warmup
    img0, _ = load_boxes(test_set[0][0])
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    recall = {k: [] for k in (1, 2, 3, 4, 5)}
    y_img, s_pure, s_hyb, s_hyb2 = [], [], [], []
    t0 = time.time()
    for p, lab in test_set:
        img, boxes = load_boxes(p)
        blocks, hit = grid4_blocks(img, boxes)
        f4 = backbone.extract_tiles(blocks, batch=batch)
        x = _image_level_feats(f4.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        s_blk = shead.head(x.to(device)).reshape(-1)
        y_img.append(lab)
        # 纯 crop 打分（ov0.3，对齐 v8）
        crops, ny, nx, trunc = _sliding_grid(img, CS, 0.3, 200)
        f = backbone.extract_tiles(crops, batch=batch)
        x = _image_level_feats(f.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        sc = shead.head(x.to(device)).reshape(-1).float()
        if not trunc:
            smap = sc.reshape(ny, nx)[None, None]
            pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
            s_pure.append(float(F.conv2d(pad, k33).max().item()))
        else:
            s_pure.append(float(sc.topk(3).values.mean().item()))
        # 混合打分：top-3 块内滑窗精查
        top_idx = torch.topk(s_blk, 3).indices.cpu().tolist()
        refined = []
        for bi in top_idx:
            blk = blocks[bi]
            bh, bw = blk.shape[:2]
            cs2 = min(CS, bh, bw)
            c2 = _sliding_grid(blk, cs2, 0.4, 20)[0]
            if c2:
                f = backbone.extract_tiles(c2, batch=batch)
                x = _image_level_feats(f.float().to(device),
                                       shead.chan_stats, shead.patch_stats)
                refined.extend(shead.head(x.to(device)).reshape(-1).tolist())
        s_hyb.append(max(refined) if refined else float(s_blk.topk(3).values.mean()))
        # 变体 D：图级 = 块 top1 分 与 精查 max 的融合（加权 0.5/0.5，z 归一化避免尺度）
        if refined:
            z_blk = float(s_blk.topk(3).values.mean())
            z_ref = max(refined)
            s_hyb2.append(0.5 * (z_blk - float(s_blk.mean())) / (float(s_blk.std()) + 1e-9)
                          + 0.5 * (z_ref - np.mean(refined)) / (np.std(refined) + 1e-9))
        else:
            s_hyb2.append(0.0)
        # 召回统计（缺陷图）
        if lab == 1 and hit:
            for k in recall:
                recall[k].append(1 if any(i in top_idx for i in hit) else 0)
    dt = time.time() - t0
    print(f"  耗时 {len(test_set)} 张 {dt:.0f}s -> {dt / len(test_set) * 1000:.0f}ms/张",
          flush=True)
    for k in sorted(recall):
        r = recall[k]
        print(f"  top-{k} 缺陷块召回: {np.mean(r):.3f} ({sum(r)}/{len(r)})", flush=True)
    auc_pure = float(roc_auc_score(y_img, s_pure))
    auc_hyb = float(roc_auc_score(y_img, s_hyb))
    auc_hyb2 = float(roc_auc_score(y_img, s_hyb2))
    print(f"  [纯 crop 打分 ov0.3] AUROC={auc_pure:.4f}", flush=True)
    print(f"  [混合打分 max精查] AUROC={auc_hyb:.4f}", flush=True)
    print(f"  [混合变体D 融合块分] AUROC={auc_hyb2:.4f}", flush=True)
    # 误判分布：正常/缺陷图的分数中位数
    y = np.array(y_img); sp = np.array(s_pure); sh = np.array(s_hyb)
    print(f"  纯crop: 缺陷中位 {np.median(sp[y==1]):.3f} vs 正常中位 {np.median(sp[y==0]):.3f}",
          flush=True)
    print(f"  混合: 缺陷中位 {np.median(sh[y==1]):.3f} vs 正常中位 {np.median(sh[y==0]):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
