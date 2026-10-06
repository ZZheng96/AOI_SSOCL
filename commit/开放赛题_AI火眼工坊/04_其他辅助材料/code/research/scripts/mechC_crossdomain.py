"""跨域机制C验证：域不变特征（blob/DoG）跨域判别力（2026-08-26，小样本快速迭代）

理论（跨域攻坚方案 v1 机制 C）：U97 实证"data_local 突破靠 blob（DoG 局部亮度
结构，域不变）"——component 0.9286 达标靠 blob 裁剪。blob 是**零训练**特征（无判别
方向可漂移），判别力来自特征本身（斑点响应），理论上跨域直接可用，只需 CDF
校准分数偏移（U62 已有）。

机制 A 首测失败（gold_finger DINO 图像级判别 AUROC 0.0，内容级反向）——语义特征
的判别方向在跨域完全不可用。本脚本验证：**blob 分数（DoG top-10 均值）在跨域下
能否直接区分 test 正常/缺陷**（不依赖 train 域判别方向）。

小样本（用户规则：每批 ≤10 图随机取，快速迭代）。

用法: python scripts/mechC_crossdomain.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml

from src.slots.blob import BlobSlot
from src.data import datalocal
from sklearn.metrics import roc_auc_score

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def blob_scores(paths, slot):
    """整图 blob 分数（DoG top-10 均值，高=异常）。"""
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    s, _ = slot.score_tiles(tile_imgs=imgs)
    return s


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    slot = BlobSlot(cfg.get("slots", {}).get("blob", {}), device="cpu")
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_n = sorted(rng.choice(b["init_normal"], 10, replace=False).tolist())
        te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
        te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())
        s_trn = blob_scores(tr_n, slot)
        s_ten = blob_scores(te_n, slot)
        s_ted = blob_scores(te_d, slot)
        auroc = roc_auc_score([0] * len(s_ten) + [1] * len(s_ted),
                              list(s_ten) + list(s_ted))
        print(f"\n[mechC] {cat}: train正常 {len(tr_n)} test正常 {len(te_n)} "
              f"test缺陷 {len(te_d)}", flush=True)
        print(f"  blob 分数: train正常 {np.mean(s_trn):.4f} "
              f"test正常 {np.mean(s_ten):.4f} test缺陷 {np.mean(s_ted):.4f}", flush=True)
        print(f"  跨域判别 AUROC（test 10 图）: {auroc:.4f}", flush=True)


if __name__ == "__main__":
    main()
