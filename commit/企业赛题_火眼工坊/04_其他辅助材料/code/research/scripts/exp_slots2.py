"""M0 扩展槽位对比：blob / trad / sem 单槽 vs 融合（data_local 四品类 + gyudet）

目标：确认 blob(域不变) 与 trad(200维传统特征) 是否能补 sem 的弱项。
用 max-eval 40 快速出数字（迭代用，正式数字需全量）。
校准：先用 train/good 拟合 CDF，再在 eval 上评分（与 M0 管线一致，红线 3）。
"""
import os
import sys
import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone.dino import FrozenDINO
from src.slots.sem import SemSlot
from src.slots.blob import BlobSlot
from src.slots.trad import TradSlot
from src.data import datalocal, gyudet
from src.common.tiling import compute_tiles, extract_tiles
from src.common.io import load_image
from src.fusion.calibrate import CDFCalibrator
from src.fusion.fixed import fuse


def score_batch(backbone, slots, items):
    """返回 {slot: raw_scores(list)}"""
    slot_raw = {n: [] for n in slots}
    for it in items:
        img = load_image(it["path"])
        ti = extract_tiles(img, compute_tiles(img.shape[:2], "tiles36", 512, 448))
        feats = backbone.extract_tiles(ti).cpu() if any(
            s.needs_dino for s in slots.values()) else None
        for n, s in slots.items():
            sc, _ = s.score_tiles(feats, ti) if s.needs_dino else s.score_tiles(None, ti)
            slot_raw[n].append(float(np.sort(sc)[-3:].mean()))
    return slot_raw


def run(name, bundle, device):
    bb = FrozenDINO(output_layer=9, grid=32, device=device)
    sem = SemSlot({"coreset_size": 512, "tile_topk_ratio": 0.05}, device)
    blob = BlobSlot({})
    trad = TradSlot({"extractor": "demo4"})
    slots = {"sem": sem, "blob": blob, "trad": trad}

    # fit（只用 train/good 前 30 张）
    tr_imgs, tr_feats = [], []
    for p in bundle["init_normal"][:30]:
        img = load_image(p)
        ti = extract_tiles(img, compute_tiles(img.shape[:2], "tiles36", 512, 448))
        tr_imgs.append(ti)
        tr_feats.append(bb.extract_tiles(ti).cpu())
    ctx = {"train_tile_feats": tr_feats, "train_tile_imgs": tr_imgs}
    for s in slots.values():
        s.fit(ctx)

    # 校准：train/good raw -> CDF
    cal = {}
    train_raw = score_batch(bb, slots,
                            [{"path": p} for p in bundle["init_normal"][:30]])
    for n in slots:
        cal[n] = CDFCalibrator(256).fit(train_raw[n])

    # 评测
    eval_items = [{"path": p, "label": y} for p, y in bundle["test"][:40]]
    labels = [it["label"] for it in eval_items]
    raw = score_batch(bb, slots, eval_items)
    s2 = {n: cal[n].transform(raw[n]) for n in slots}
    f2 = [fuse({n: s2[n][i] for n in slots}) for i in range(len(labels))]
    print(f"  [{name}] fused={roc_auc_score(labels, f2):.4f} "
          f"| sem={roc_auc_score(labels, s2['sem']):.4f} "
          f"blob={roc_auc_score(labels, s2['blob']):.4f} "
          f"trad={roc_auc_score(labels, s2['trad']):.4f}")


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    for c in ("gold_finger", "extra_part", "solder_smt", "component"):
        b = datalocal.load_category(cfg["datasets"]["data_local"], c, 100, 30, 42)
        print(f"== {c} ==")
        run(c, b, device)
    g = gyudet.load_gyudet(cfg["datasets"]["gyu_det"], 100, 30, 42)
    print("== gyudet ==")
    run("gyudet", g, device)


if __name__ == "__main__":
    main()
