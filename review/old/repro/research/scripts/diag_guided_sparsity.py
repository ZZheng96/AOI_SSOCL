"""U77c（2026-08-19）：sem 热力图中缺陷 patch 的稀疏度诊断

U77 说框内最高 patch 分位 <0.1 达 90%，但候选框 IoU 覆盖仅 0.13。
假设：缺陷框内高分 patch 稀疏（1-3 个），被背景噪声淹没，连通域无法成簇。
本脚本统计：每个 GT 框内"高分 patch（分位<0.1 / <0.2）"的数量 + 框内 patch 总数。
"""
import os
import sys
import json

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from src.backbone.dino import FrozenDINO
from src.slots.shead import _parse_yolo_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.diag_guided_localize import load_good_paths, load_defect_paths
    good_paths = load_good_paths("train", 40, SEED)
    def_paths = load_defect_paths("test", 30, SEED)
    backbone = FrozenDINO(model_name="vits14", output_layer=9, grid=32, device=device)
    G = backbone.grid

    # 建库（同 diag_guided_candidates）
    rng = np.random.default_rng(SEED)
    flat_all = []
    for p in good_paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        f = backbone.extract_tiles([img], batch=8)[0]
        flat_all.append(f.reshape(f.shape[0], -1).T)
    feats = torch.cat(flat_all, dim=0)
    feats = F.normalize(feats.float(), dim=1)
    n = feats.shape[0]
    idx = [int(rng.integers(n))]
    d = 1 - feats @ feats[idx[0]]
    for _ in range(255):
        i = int(torch.argmax(d).item())
        idx.append(i)
        d = torch.minimum(d, 1 - feats @ feats[i])
    bank = feats[idx]
    print(f"  库: {bank.shape}", flush=True)

    n_boxes = 0
    in_p10, in_p20 = [], []      # 框内高分 patch 数（分位<0.1 / <0.2）
    in_total = []                # 框内总 patch 数
    hit_p10 = hit_p20 = 0
    for p in def_paths:
        lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                           os.path.splitext(os.path.basename(p))[0] + ".txt")
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = _parse_yolo_boxes(lbl, w, h)
        if not boxes:
            continue
        f = backbone.extract_tiles([img], batch=8)[0].cpu()
        flat_t = F.normalize(f.reshape(f.shape[0], -1).T.float(), dim=1)
        sims = (flat_t @ bank.cpu().T).max(dim=1).values
        dmap = (1 - sims).reshape(G, G).numpy()
        thr10 = np.quantile(dmap, 0.90)   # 分位 0.10 阈值
        thr20 = np.quantile(dmap, 0.80)   # 分位 0.20 阈值
        for bx in boxes:
            y0 = int(bx[1] / h * G); y1 = max(y0 + 1, int(bx[3] / h * G))
            x0 = int(bx[0] / w * G); x1 = max(x0 + 1, int(bx[2] / w * G))
            y1 = min(G, y1); x1 = min(G, x1)
            sub = dmap[y0:y1, x0:x1]
            total = sub.size
            n10 = int((sub >= thr10).sum())
            n20 = int((sub >= thr20).sum())
            in_total.append(total)
            in_p10.append(n10)
            in_p20.append(n20)
            if n10 >= 1:
                hit_p10 += 1
            if n20 >= 1:
                hit_p20 += 1
            n_boxes += 1

    in_p10 = np.array(in_p10); in_p20 = np.array(in_p20); in_total = np.array(in_total)
    print(f"\n=== U77c 缺陷框内高分 patch 稀疏度（框数={n_boxes}）===")
    print(f"  框内总 patch 数: mean={in_total.mean():.1f} median={np.median(in_total):.0f}")
    print(f"  框内 top-10% 高分 patch 数: mean={in_p10.mean():.2f} median={np.median(in_p10):.0f} "
          f"≥1 占比={hit_p10 / n_boxes:.3f}")
    print(f"  框内 top-20% 高分 patch 数: mean={in_p20.mean():.2f} median={np.median(in_p20):.0f} "
          f"≥1 占比={hit_p20 / n_boxes:.3f}")
    for k in (1, 2, 4, 6):
        print(f"    top-10% 高分 patch≥{k}: {(in_p10 >= k).mean():.3f}")

    out = {"n_boxes": n_boxes,
           "in_patch_median": int(np.median(in_total)),
           "in_p10_mean": round(float(in_p10.mean()), 2),
           "in_p10_median": int(np.median(in_p10)),
           "in_p10_ge1": round(float(hit_p10 / n_boxes), 3),
           "in_p10_ge2": round(float((in_p10 >= 2).mean()), 3),
           "in_p10_ge4": round(float((in_p10 >= 4).mean()), 3)}
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                            "diag_guided_sparsity.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
