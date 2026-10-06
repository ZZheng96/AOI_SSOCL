"""跨域机制F验证：tpl 模板差分（train/good 修复图当模板）在 data_local 的表现
（2026-08-26，用户质疑"tpl 不就是为了跨域加的吗"——回应 + 补测像素差分通道）

背景：tpl 是设计上的域差防线（差分消全局域差），但依赖 L3 显式模板，data_local
无模板契约 → 跨域攻坚从未启用。等效机制 U116（空间对齐记忆库=DINO 特征差分无
配准版）已实测无效（train 修复图 vs test 原始图内容级反向）。但 tpl 还有**像素
差分通道（0.3 权重）**未测。本实验：train/good（修复图）作模板 fit TplSlot，
test 正常 vs 缺陷打分 AUROC（gold_finger/solder_smt）。

用法: python scripts/mechF_tpl_cross.py
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
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        # M14b：模板 = 全量 train/good（修复图库，包含每张 test 图的配对修复图）
        # ——内容最近邻 min 差分（非文件名配对，L3 显式模板目录契约）。
        # test 先小样本（5+5）快速验证方向（模板全库是本次关键变量）。
        tr_n = sorted(b["init_normal"])
        te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
        te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())
        print(f"\n[mechF] {cat}: 模板 {len(tr_n)} 张（train 修复图全库） "
              f"test 正常 {len(te_n)} 缺陷 {len(te_d)}", flush=True)

        # tpl fit：全量模板
        tpl = TplSlot(cfg.get("slots", {}).get("tpl", {}), backbone, device)
        tpl.fit({"templates": tr_n})
        n_tpl = len(tpl.templates)

        def scores(paths):
            out = []
            for p in paths:
                from src.common.io import load_image
                img = load_image(p)
                feats = backbone.extract_tiles([img]).cpu()
                s, _ = tpl.score_tiles(feats, [img])
                out.append(float(s[0]))
            return np.array(out)

        s_n, s_d = scores(te_n), scores(te_d)
        au = roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))
        print(f"  正常均值={s_n.mean():.4f} 缺陷均值={s_d.mean():.4f}", flush=True)
        print(f"  tpl 全库模板差分 AUROC = {au:.4f}", flush=True)


if __name__ == "__main__":
    main()
