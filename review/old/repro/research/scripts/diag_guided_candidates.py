"""U77b（2026-08-19）：guided 粗筛候选框 vs GT 框重合诊断

guided smoke AUROC=0.5994 << ablB 0.8447，训练组成相同（pos=896 ctx=448），
差异只在打分路径。怀疑：粗筛候选框未覆盖缺陷（框中心偏移/只覆盖缺陷一角），
导致精查 crop 中缺陷占比低或缺失 -> 判别头（在"缺陷居中"crop 上训练）漏检。

度量（test 缺陷图）：
  1. 每 GT 框 vs 最佳候选框的 IoU（覆盖率）
  2. 候选框中心 vs GT 框中心的偏移（归一化）
诚实边界：train 正常建库（协议内），test 只评价。
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from src.backbone.dino import FrozenDINO
from src.slots.shead import _candidates_from_heatmap, _iou, _parse_yolo_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42


def load_defect_paths(split, n, seed=42):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, split, "images")
    lbl_dir = os.path.join(GYU_ROOT, split, "labels")
    defect = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            defect.append(os.path.join(img_dir, f))
    idx = rng.choice(len(defect), min(n, len(defect)), replace=False)
    return [defect[i] for i in idx]


def build_bank(backbone, good_paths, target=256, seed=42):
    rng = np.random.default_rng(seed)
    flat_all = []
    for p in good_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f = backbone.extract_tiles([img], batch=8)[0]
        flat_all.append(f.reshape(f.shape[0], -1).T)
    feats = torch.cat(flat_all, dim=0)
    feats = F.normalize(feats.float(), dim=1)
    n = feats.shape[0]
    idx = [int(rng.integers(n))]
    d = 1 - feats @ feats[idx[0]]
    for _ in range(target - 1):
        i = int(torch.argmax(d).item())
        idx.append(i)
        d = torch.minimum(d, 1 - feats @ feats[i])
    return feats[idx]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U77b 候选框诊断] device={device}", flush=True)
    from scripts.diag_guided_localize import load_good_paths
    good_paths = load_good_paths("train", 40, SEED)
    def_paths = load_defect_paths("test", 30, SEED)
    print(f"  good={len(good_paths)} defect={len(def_paths)}", flush=True)

    backbone = FrozenDINO(model_name="vits14", output_layer=9, grid=32, device=device)
    G = backbone.grid
    bank = build_bank(backbone, good_paths, 256, SEED)
    print(f"  粗筛库: {bank.shape}", flush=True)

    # 参数扫描：topk x recall_p -> IoU>0.3 覆盖率
    for topk in (8, 16, 24):
        for recall_p in (0.10, 0.20, 0.30):
            ious, offs = [], []
            n_boxes = n_hit = 0
            for i, p in enumerate(def_paths):
                lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                                   os.path.splitext(os.path.basename(p))[0] + ".txt")
                img = cv2.imread(p)
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                boxes = _parse_yolo_boxes(lbl, w, h)
                if not boxes:
                    continue
                f = backbone.extract_tiles([img_rgb], batch=8)[0].cpu()
                flat_t = f.reshape(f.shape[0], -1).T
                flat_t = F.normalize(flat_t.float(), dim=1)
                sims = (flat_t @ bank.cpu().T).max(dim=1).values
                heatmap = (1 - sims).reshape(G, G).numpy()
                cands = _candidates_from_heatmap(heatmap, (h, w), 699,
                                                 topk=topk, recall_p=recall_p)
                for bx in boxes:
                    n_boxes += 1
                    best_iou = max((_iou(bx, c) for c in cands), default=0.0)
                    ious.append(best_iou)
                    if best_iou > 0.3:
                        n_hit += 1
                    best_c = None
                    for c in cands:
                        if _iou(bx, c) == best_iou:
                            best_c = c
                            break
                    if best_c is not None:
                        bcx = (bx[0] + bx[2]) / 2 / w
                        bcy = (bx[1] + bx[3]) / 2 / h
                        ccx = (best_c[0] + best_c[2]) / 2 / w
                        ccy = (best_c[1] + best_c[3]) / 2 / h
                        offs.append(np.hypot(ccx - bcx, ccy - bcy))
            ious = np.array(ious)
            offs = np.array(offs)
            print(f"  topk={topk:2d} recall_p={recall_p:.2f}: "
                  f"IoU>0.3={n_hit / n_boxes:.3f} IoU_med={np.median(ious):.3f} "
                  f"off_med={np.median(offs):.3f} (框={n_boxes})", flush=True)
            if topk == 16 and recall_p == 0.20:
                best_cfg = {"topk": topk, "recall_p": recall_p,
                            "iou_mean": round(float(ious.mean()), 3),
                            "iou_median": round(float(np.median(ious)), 3),
                            "cover_0.3": round(float((ious > 0.3).mean()), 3),
                            "off_median": round(float(np.median(offs)), 3)}
    out = {"best_cfg": best_cfg}
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                            "diag_guided_candidates.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
