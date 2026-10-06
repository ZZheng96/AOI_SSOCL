"""跨域机制H验证：tpl per-position min 差分 + 完整模板库（train/good + test/good
修复图）在 data_local 的表现（2026-08-26，用户决策：用户配置提供模板时 test 可用）

关键区别 vs mechG：mechG 用"整图最近邻模板"（argmax mean 相似度）逐位置差分
（反向 0.068）；tpl 是 **per-position min**（每个 patch 位置独立在模板库中选
差分最小的模板）——模板库含 test/good 后，test 正常图（repaired）的每位置在
库中能对齐到自身/同类 repaired（差分≈0），test 缺陷图的配对 repaired（train
或 test 域）逐位置差分在缺陷区域大。只测 DINO 特征差分（feat_w=1，跳过像素
差分避免慢）。

用法: python scripts/mechH_tpl_full.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
import torch

from src.backbone.dino import FrozenDINO
from src.slots.tpl import TplSlot
from src.data import datalocal
from sklearn.metrics import roc_auc_score

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    tpl_cfg = dict(cfg.get("slots", {}).get("tpl", {}), feat_weight=1.0, pix_weight=0.0)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_n = sorted(b["init_normal"])                          # train/good 修复图
        te_all = sorted(b["test"])
        te_good = sorted([p for p, y in te_all if y == 0])       # test/good（repaired）
        te_def = sorted([p for p, y in te_all if y == 1])
        templates = tr_n + te_good                               # L3 用户配置模板库
        print(f"\n[mechH] {cat}: 模板库 {len(tr_n)}(train)+{len(te_good)}(test)="
              f"{len(templates)} test正常 {len(te_good)} 缺陷 {len(te_def)}", flush=True)

        tpl = TplSlot(tpl_cfg, backbone, device)
        tpl.fit({"templates": templates})
        # 注意：模板含 test/good（t_fov_13_repaired 等），这些同时是评测正常样本
        #（用户决策：L3 模板模式 test 可用）

        def scores(paths):
            out = []
            for p in paths:
                from src.common.io import load_image
                img = load_image(p)
                feats = backbone.extract_tiles([img]).cpu()
                s, _ = tpl.score_tiles(feats, None)   # 仅特征差分（pix_w=0）
                out.append(float(s[0]))
            return np.array(out)

        s_n, s_d = scores(te_good), scores(te_def)
        au = roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))
        print(f"  正常均值={s_n.mean():.4f} 缺陷均值={s_d.mean():.4f} "
              f"tpl 完整模板库 per-position min AUROC = {au:.4f}", flush=True)


if __name__ == "__main__":
    main()
