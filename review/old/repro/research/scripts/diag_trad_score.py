"""诊断 trad 打分方式（2026-08-16）：整图 vs tile 级 + 聚合策略对照

背景：demo4 t 分支在 data_local 诚实单分支 0.81-1.0（solder 0.993），
demo5 trad 槽位反向（solder 0.4688）。两者用同一个 200 维提取器，
差异只在打分：
  demo4：整图特征 vs 整图正常库 -> kNN 距离取最近 3 个邻居的均值（平滑）
  demo5：每 tile 特征 vs tile 正常库 -> 每 tile min-kNN -> 图像分=最差 3 tile 均值

变体矩阵（隔离两个因素）：
  whole_3nn     整图 + 3 近邻均值（demo4 原版）
  whole_min     整图 + 1 近邻
  tile_min_top3 tile min-kNN + 最差 3 tile 均值（demo5 现状）
  tile_min_mean tile min-kNN + 全 tile 均值
  tile_3nn_top3 tile 3 近邻均值 + 最差 3 tile 均值
  tile_3nn_mean tile 3 近邻均值 + 全 tile 均值

协议：诚实--正常库只用 init_normal（train/good），评测 test 全量。
输出：outputs/m0/diag_trad_score_{category}.json
"""
import os
import sys
import json
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data import datalocal
from src.common.tiling import compute_tiles, extract_tiles
from src.slots.trad import TraditionalExtractor

VARIANTS = ["whole_3nn", "whole_min",
            "tile_min_top3", "tile_min_mean",
            "tile_3nn_top3", "tile_3nn_mean"]


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def tile_vecs(img, mode, tile_size, stride, ext):
    tiles = compute_tiles(img.shape[:2], mode, tile_size, stride)
    return [ext.extract(t).astype(np.float32) for t in extract_tiles(img, tiles)]


def run_category(root, category, cfg, mode, seed):
    prot = cfg["protocol"]
    bundle = datalocal.load_category(root, category, prot["n_init_normal"],
                                     prot["n_init_defect"], seed)
    ext = TraditionalExtractor()
    tl = cfg.get("tiling", {})
    tile_size, stride = tl.get("tile_size", 512), tl.get("stride", 448)

    # ---- 正常库（init_normal，诚实）----
    t0 = time.time()
    bank_w, bank_t = [], []
    for i, p in enumerate(bundle["init_normal"]):
        img = load_rgb(p)
        bank_w.append(ext.extract(img).astype(np.float32))
        bank_t.extend(tile_vecs(img, mode, tile_size, stride, ext))
        if (i + 1) % 20 == 0:
            print(f"  [bank] {i + 1}/{len(bundle['init_normal'])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    bank_w = np.stack(bank_w)
    bank_t = np.stack(bank_t)
    mw, sw = bank_w.mean(0), bank_w.std(0) + 1e-6
    mt, st = bank_t.mean(0), bank_t.std(0) + 1e-6
    bank_wz = (bank_w - mw) / sw
    bank_tz = (bank_t - mt) / st
    print(f"[bank] {category}: whole={bank_w.shape} tile={bank_t.shape} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 评测 ----
    S, y = [], []
    t0 = time.time()
    for i, (p, lab) in enumerate(bundle["test"]):
        img = load_rgb(p)
        fw = (ext.extract(img).astype(np.float32) - mw) / sw
        dw = np.linalg.norm(bank_wz - fw, axis=1)
        scores = {
            "whole_3nn": float(np.sort(dw)[:3].mean()),
            "whole_min": float(dw.min()),
        }
        fts = tile_vecs(img, mode, tile_size, stride, ext)
        mins, n3s = [], []
        for ft in fts:
            fz = (ft - mt) / st
            d = np.linalg.norm(bank_tz - fz, axis=1)
            mins.append(float(d.min()))
            n3s.append(float(np.sort(d)[:3].mean()))
        mins, n3s = np.asarray(mins), np.asarray(n3s)
        scores["tile_min_top3"] = float(np.sort(mins)[::-1][:3].mean())
        scores["tile_min_mean"] = float(mins.mean())
        scores["tile_3nn_top3"] = float(np.sort(n3s)[::-1][:3].mean())
        scores["tile_3nn_mean"] = float(n3s.mean())
        S.append([scores[v] for v in VARIANTS])
        y.append(lab)
        if (i + 1) % 20 == 0:
            print(f"  [test] {i + 1}/{len(bundle['test'])} ({time.time() - t0:.0f}s)",
                  flush=True)

    S, y = np.asarray(S), np.asarray(y)
    auroc = {v: round(float(roc_auc_score(y, S[:, j])), 4)
             for j, v in enumerate(VARIANTS)}
    out = {"category": category, "mode": mode, "n_test": len(y),
           "n_defect": int(y.sum()), "bank": {"whole": bank_w.shape[0],
                                              "tile": bank_t.shape[0]},
           "auroc": auroc, "eval_sec": round(time.time() - t0, 1)}
    print(f"[{category}] " + " ".join(f"{k}={v}" for k, v in auroc.items()),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="solder_smt",
                    help="逗号分隔或 all")
    ap.add_argument("--mode", default="tiles36")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    root = cfg["datasets"]["data_local"]
    cats = (datalocal.CATEGORIES if args.categories == "all"
            else args.categories.split(","))

    results = {}
    for c in cats:
        c = c.strip()
        results[c] = run_category(root, c, cfg, args.mode, args.seed)
        p_out = os.path.join(cfg["output_dir"], f"diag_trad_score_{c}.json")
        with open(p_out, "w", encoding="utf-8") as f:
            json.dump(results[c], f, ensure_ascii=False, indent=2)
    if len(cats) > 1:
        out_path = os.path.join(cfg["output_dir"], "diag_trad_score_all.json")
        json.dump(results, open(out_path, "w", ensure_ascii=False), indent=2)
        print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
