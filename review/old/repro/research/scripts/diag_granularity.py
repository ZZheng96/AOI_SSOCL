"""U79（2026-08-19）：评测单位变换实验 -- 整图 vs 16 小图的 F1/AUC 对比

用户设计（验证 U76 核心论断"评测单位影响"）：
  1. 取 GYU-DET 10 张图（5 缺陷 + 5 正常）整图检测 -> F1/AUC
  2. 每张裁 4x4=16 小图，含标注区域的为异常、不含的为正常 -> 160 小图单独检测 -> F1/AUC
  3. 比较前后，看评测粒度对检测难度的启发

同一模型（sheadA image 模式，fit 只用 train 域协议内 init_normal+init_defect，
test 只验不选），唯一变量 = 评测单位（整图 vs 16 小图）。
预期：小图缺陷信噪比高（缺陷在小图中占比高）-> 小图级 AUROC > 整图级，
实证"demo4 crop 级 0.9 = 评测粒度红利"（U76）。
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
import cv2
import torch
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import SheadSlot
from src.slots.sem import SemSlot
from src.slots.disc import DiscSlot
from src.eval.offline import Pipeline
from src.data import gyudet
from src.slots.shead import _parse_yolo_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42
N_IMGS = 10          # 10 张图
GRID = 4             # 4x4 = 16 小图


def pick_test_images(n, seed=42):
    """test 域选 n/2 缺陷 + n/2 正常图。"""
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    defect, normal = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir, f)
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            defect.append(p)
        else:
            normal.append(p)
    half = n // 2
    di = rng.choice(len(defect), half, replace=False)
    ni = rng.choice(len(normal), half, replace=False)
    return [defect[i] for i in di], [normal[i] for i in ni]


def best_f1(y, s):
    """最优阈值 F1（扫描所有分数点）。"""
    y = np.array(y)
    s = np.array(s)
    order = np.argsort(s)
    best = 0.0
    for i in range(len(s)):
        th = s[order[i]]
        pred = (s >= th).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        if tp + fp + fn == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        best = max(best, f1)
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U79 评测粒度实验] device={device}", flush=True)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_sheadA.yaml"),
                              encoding="utf-8"))
    # fit 用协议内 train 域（同 m0_baseline）
    bundle = gyudet.load_gyudet(GYU_ROOT, cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], SEED)
    print(f"  init_normal={len(bundle['init_normal'])} init_defect={len(bundle['init_defect'])}",
          flush=True)

    # 构建管线（image 模式 sheadA）
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    bcfg = cfg["backbone"]
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    slots = []
    s = cfg["slots"]
    if s["sem"]["enabled"]:
        slots.append(SemSlot(s["sem"], device))
    if s["disc"]["enabled"]:
        slots.append(DiscSlot({**s["disc"], "seed": SEED}, backbone, device))
    if s["shead"]["enabled"]:
        slots.append(SheadSlot({**s["shead"], "seed": SEED}, backbone, device))
    if s["blob"]["enabled"]:
        from src.slots.blob import BlobSlot
        slots.append(BlobSlot(s["blob"], device))
    if s["trad"]["enabled"]:
        from src.slots.trad import TradSlot
        slots.append(TradSlot(s["trad"]))
    if s["layout"]["enabled"]:
        from src.slots.layout import LayoutSlot
        slots.append(LayoutSlot(s["layout"]))
    if s["inp"]["enabled"]:
        from src.slots.inp import InpSlot
        slots.append(InpSlot(s["inp"], device))
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    print(f"  fit 完成 ({time.time() - t0:.0f}s)", flush=True)

    # 选 10 张 test 图（5 缺陷 + 5 正常）
    def_paths, nor_paths = pick_test_images(N_IMGS, SEED)
    all_paths = def_paths + nor_paths
    y_img = np.array([1] * len(def_paths) + [0] * len(nor_paths))
    print(f"  选图: 缺陷={len(def_paths)} 正常={len(nor_paths)}", flush=True)

    # 1. 整图检测
    img_scores = []
    for p in all_paths:
        rec = pipe.predict(p)
        img_scores.append(rec["slot_scores"]["shead"])
    img_scores = np.array(img_scores)
    auc_img = float(roc_auc_score(y_img, img_scores))
    f1_img = best_f1(y_img, img_scores)
    print(f"\n=== 整图（{N_IMGS} 张）===")
    print(f"  shead AUROC = {auc_img:.4f}  最优F1 = {f1_img:.4f}")

    # 2. 16 小图检测
    tile_scores, tile_labels = [], []
    detail = []
    for i, p in enumerate(all_paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                           os.path.splitext(os.path.basename(p))[0] + ".txt")
        boxes = _parse_yolo_boxes(lbl, w, h)
        gh, gw = h // GRID, w // GRID
        for r in range(GRID):
            for c in range(GRID):
                y0, x0 = r * gh, c * gw
                y1, x1 = min(h, y0 + gh), min(w, x0 + gw)
                small = img[y0:y1, x0:x1]
                # 标签：含标注区域（框与单元交集>0）
                hit = False
                for bx in boxes:
                    iw = max(0, min(x1, bx[2]) - max(x0, bx[0]))
                    ih = max(0, min(y1, bx[3]) - max(y0, bx[1]))
                    if iw * ih > 0:
                        hit = True
                        break
                rec = pipe.predict_frame(small)
                tile_scores.append(rec["slot_scores"]["shead"])
                tile_labels.append(1 if hit else 0)
                detail.append((os.path.basename(p), r, c, 1 if hit else 0,
                               rec["slot_scores"]["shead"]))
    tile_scores = np.array(tile_scores)
    tile_labels = np.array(tile_labels)
    auc_tile = float(roc_auc_score(tile_labels, tile_scores))
    f1_tile = best_f1(tile_labels, tile_scores)
    n_pos_tile = int(tile_labels.sum())
    print(f"\n=== 16 小图（{N_IMGS * 16} 张，含缺陷 {n_pos_tile}）===")
    print(f"  shead AUROC = {auc_tile:.4f}  最优F1 = {f1_tile:.4f}")

    # 3. 对比 + 小图分数分布
    print(f"\n=== 对比 ===")
    print(f"  AUROC: 整图 {auc_img:.4f} -> 小图 {auc_tile:.4f} "
          f"(Δ {auc_tile - auc_img:+.4f})")
    print(f"  最优F1: 整图 {f1_img:.4f} -> 小图 {f1_tile:.4f} "
          f"(Δ {f1_tile - f1_img:+.4f})")
    pos_s = tile_scores[tile_labels == 1]
    neg_s = tile_scores[tile_labels == 0]
    print(f"  小图得分: 含缺陷 mean={pos_s.mean():.4f} std={pos_s.std():.4f} "
          f"| 不含 mean={neg_s.mean():.4f} std={neg_s.std():.4f}")
    print(f"  含缺陷小图分 > 不含中位数 比例: "
          f"{(pos_s > np.median(neg_s)).mean():.3f}")

    out = {"auc_img": round(auc_img, 4), "f1_img": round(f1_img, 4),
           "auc_tile": round(auc_tile, 4), "f1_tile": round(f1_tile, 4),
           "n_tiles": len(tile_labels), "n_pos_tiles": int(n_pos_tile),
           "pos_mean": round(float(pos_s.mean()), 4),
           "neg_mean": round(float(neg_s.mean()), 4),
           "detail": detail}
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                            "diag_granularity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
