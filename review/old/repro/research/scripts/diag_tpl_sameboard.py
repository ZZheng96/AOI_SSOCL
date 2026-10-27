"""U88 同板模板差分验证（2026-08-22）：demo4 r 分支（差分）机理的 L3 合法复现。

背景：data_local train/good 是 {id}_repaired.png（修复后正常图），test/defect 是 {id}.png
（同一板修复前含缺陷图）——同板同构图。demo4 靠文件名隐式配对（泄漏）拿 r=1.0；
本脚本 = 用户显式提供模板（L3 层次，--templates 目录/列表）→ tpl 逐位置差分 → 只验
tpl 单槽 AUROC（不经全槽融合，聚焦机理验证）+ 计时。

用法: python scripts/diag_tpl_sameboard.py --category gold_finger [--tpl-max 68]
"""
import os
import sys
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import cv2

from src.backbone.dino import FrozenDINO
from src.slots.tpl import TplSlot
from src.data.datalocal import _split_good_defect

ROOT = r"D:\CGAIC\data_local"


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--tpl-max", type=int, default=68)
    ap.add_argument("--topk-ratio", type=float, default=0.1)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cat = args.category

    tg = sorted(os.path.join(ROOT, cat, "train", "good", f)
                for f in os.listdir(os.path.join(ROOT, cat, "train", "good"))
                if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.tpl_max]
    cat_dir = os.path.join(ROOT, cat)
    te_good, te_def = _split_good_defect(cat_dir, "test")   # 兼容 solder 类型子目录
    # 对齐 datalocal.load_category 协议：init_defect 取 test/defect 一半（seed 42），其余 eval
    rng = np.random.default_rng(42)
    n_init = min(30, max(1, len(te_def) // 2))
    sel = set(rng.choice(len(te_def), n_init, replace=False).tolist())
    eval_def = [p for i, p in enumerate(te_def) if i not in sel]
    test = [(p, 0) for p in te_good] + [(p, 1) for p in eval_def]
    print(f"[U88] {cat}: 模板 {len(tg)} | test {len(test)}（正常 {len(te_good)} / 缺陷 {len(eval_def)}）",
          flush=True)

    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    tpl = TplSlot({"topk_ratio": args.topk_ratio}, backbone, device)
    t0 = time.time()
    tpl.fit({"templates": tg})
    print(f"[U88] 模板特征提取+缓存 {time.time() - t0:.0f}s", flush=True)

    # 每板配对命中率（诊断：test/defect 是否真有同板 repaired 模板）
    ids = {os.path.splitext(os.path.basename(p))[0].split("_")[0]
           for p in tg if "_repaired" in os.path.basename(p)}
    hit = sum(1 for p in eval_def if os.path.splitext(os.path.basename(p))[0] in ids)
    print(f"[U88] eval 缺陷中 {hit}/{len(eval_def)} 有同板 repaired 模板", flush=True)

    scores, ts = [], []
    for p, y in test:
        t = time.perf_counter()
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        feats = backbone.extract_tiles([img]).cpu()[0].unsqueeze(0)   # (1,384,32,32)
        # 像素差分须在 448 网格（与 tpl fit 缓存的模板 pix 对齐）——喂原图会在 2500² 上
        # 逐模板 float 差分（68×25MB CPU 操作/图），是 U88 首跑卡死根因
        img448 = cv2.resize(img, (448, 448)) if img.shape[:2] != (448, 448) else img
        s, _ = tpl.score_tiles(feats, [img448])
        scores.append(float(s[0]))
        ts.append((time.perf_counter() - t) * 1000)
    print(f"[U88] tpl 打分 {np.mean(ts):.0f}ms/图（{len(test)} 图, {len(tg)} 模板）", flush=True)
    au = auroc([y for _, y in test], scores)
    print(f"[U88] tpl 单槽 AUROC = {au:.4f}", flush=True)
    # 对照 blob 已知水平（U55 实测）与随机
    print(f"[U88] 对照：blob 单槽 gold 0.7075 / solder 0.7458；tpl 3 随机模板 0.38/0.35", flush=True)


if __name__ == "__main__":
    main()
