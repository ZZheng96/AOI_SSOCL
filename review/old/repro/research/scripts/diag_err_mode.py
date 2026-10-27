"""U86v19 错误模式诊断（2026-08-20）：主要问题是漏检还是误报？

分析 u86v8 head（平衡口径 test 60 张，0.8856）的错误构成：
1. 每图 crop 打分（ov0.3，与 v8 一致）+ 图级分数
2. 阈值分析：最优 Youden 阈值下 FP（正常判异常）/FN（异常判正常）计数
3. 分数排序：误报 Top（正常图高分）vs 漏检 Bottom（缺陷图低分）具体样本
4. 错误样本特征：图尺寸、框数 vs 分数（找漏检/误报的结构性原因）

协议：平衡 test（seed=123 excl valid，30+30），纯评测不学习。
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
from crop_score_retrain import pick_test, load_boxes

U86_HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                        "u86v8_head.pt")
GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ck = torch.load(U86_HEAD, map_location=device)
    print(f"[U86v19 错误模式诊断] crop_size={ck['crop_size']}", flush=True)
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
    img0, _ = load_boxes(test[0][0])
    _ = backbone.extract_tiles([cv2.resize(img0, (448, 448))], batch=8)

    rows = []   # (path, label, score, h, w, n_boxes)
    t0 = time.time()
    for p, lab in test:
        img, boxes = load_boxes(p)
        h, w = img.shape[:2]
        crops, ny, nx, trunc = _sliding_grid(img, crop_size, 0.3, 200)
        f = backbone.extract_tiles(crops, batch=batch)
        x = _image_level_feats(f.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        s = shead.head(x.to(device)).reshape(-1).float()
        if not trunc:
            smap = s.reshape(ny, nx)[None, None]
            pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
            score = float(F.conv2d(pad, k33).max().item())
        else:
            score = float(s.topk(min(3, len(s))).values.mean().item())
        rows.append((os.path.basename(p), lab, score, h, w, len(boxes)))
    print(f"  打分完成 ({time.time() - t0:.0f}s)", flush=True)

    # 1. 分数分布
    def_scores = [r[2] for r in rows if r[1] == 1]
    nor_scores = [r[2] for r in rows if r[1] == 0]
    print(f"\n=== 分数分布 ===", flush=True)
    print(f"  缺陷图(30): mean={np.mean(def_scores):.3f} std={np.std(def_scores):.3f} "
          f"min={min(def_scores):.3f} median={np.median(def_scores):.3f} "
          f"max={max(def_scores):.3f}", flush=True)
    print(f"  正常图(30): mean={np.mean(nor_scores):.3f} std={np.std(nor_scores):.3f} "
          f"min={min(nor_scores):.3f} median={np.median(nor_scores):.3f} "
          f"max={max(nor_scores):.3f}", flush=True)

    # 2. 最优 Youden 阈值下 FP/FN
    from sklearn.metrics import roc_curve
    ys = np.array([r[1] for r in rows])
    ss = np.array([r[2] for r in rows])
    fpr, tpr, thrs = roc_curve(ys, ss)
    youden = tpr - fpr
    bi = int(np.argmax(youden))
    th = thrs[bi]
    pred = (ss >= th).astype(int)
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    tp = int(((pred == 1) & (ys == 1)).sum())
    tn = int(((pred == 0) & (ys == 0)).sum())
    print(f"\n=== 最优阈值判定 (Youden 阈值={th:.3f}) ===", flush=True)
    print(f"  TP={tp} FN={fn}(漏检) FP={fp}(误报) TN={tn}", flush=True)
    print(f"  漏检率(FN/(FN+TP))={fn/(fn+tp):.2%}  误报率(FP/(FP+TN))={fp/(fp+tn):.2%}",
          flush=True)
    print(f"  → 主要问题: {'漏检为主' if fn > fp else ('误报为主' if fp > fn else '均衡')}",
          flush=True)

    # 3. 具体错误样本
    def_rows = sorted([r for r in rows if r[1] == 1], key=lambda r: r[2])
    nor_rows = sorted([r for r in rows if r[1] == 0], key=lambda r: -r[2])
    print(f"\n=== 漏检样本（缺陷图分数最低 {min(len(def_rows), 7)} 张）===", flush=True)
    for p, lab, sc, h, w, nb in def_rows[:7]:
        mark = " <阈" if sc < th else ""
        print(f"  {p}: 分数={sc:.3f} 尺寸={w}x{h} 框数={nb}{mark}", flush=True)
    print(f"\n=== 误报样本（正常图分数最高 {min(len(nor_rows), 4)} 张）===", flush=True)
    for p, lab, sc, h, w, nb in nor_rows[:4]:
        mark = " >阈" if sc >= th else ""
        print(f"  {p}: 分数={sc:.3f} 尺寸={w}x{h}{mark}", flush=True)

    # 4. 结构性特征：尺寸/框数 vs 分数
    print(f"\n=== 结构性分析 ===", flush=True)
    big = [r for r in rows if max(r[3], r[4]) >= 2500]
    small = [r for r in rows if max(r[3], r[4]) < 2500]
    for tag, grp in (("大图(>=2500)", big), ("中小图(<2500)", small)):
        if grp:
            print(f"  {tag}: {len(grp)} 张 缺陷图分数中位="
                  f"{np.median([r[2] for r in grp if r[1]==1]):.3f} "
                  f"正常图分数中位="
                  f"{np.median([r[2] for r in grp if r[1]==0]):.3f}", flush=True)
    for nb in (1, 2, 3):
        grp = [r for r in rows if r[1] == 1 and r[5] == nb]
        if grp:
            print(f"  缺陷图框数={nb}: {len(grp)} 张 分数中位="
                  f"{np.median([r[2] for r in grp]):.3f}", flush=True)
    one = [r for r in rows if r[1] == 1 and r[5] >= 4]
    if one:
        print(f"  缺陷图框数>=4: {len(one)} 张 分数中位="
              f"{np.median([r[2] for r in one]):.3f}", flush=True)


if __name__ == "__main__":
    main()
