"""跨域机制B验证：判别头 TTA（soft-voting + 一致性微调）跨域判别力（2026-08-26，小样本）

理论（跨域攻坚方案 v1 机制 B）：U98 证明判别方向跨域漂移，TTA 用 test 域无标签
样本在线纠正判别头。本脚本测两种 TTA 形式：
  base        无 TTA：单图判别头输出
  vote        soft-voting TTA：每图 K 增强（亮度/模糊/平移）proba 取均值（最简单 TTA）
  ft          一致性微调 TTA：K 增强输出一致性损失微调 LR 判别头 1 步（带锚定保护）

对"缺陷率不敏感"（一致性约束不需要正常/缺陷先验）。预期：gold_finger 内容级反向
（方向全反）TTA 无法纠正（阴性确认）；solder 若 base 正向则 TTA 可能提升。

小样本（用户规则：每批 ≤10 图）。

用法: python scripts/mechB_crossdomain.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from src.backbone.dino import FrozenDINO
from src.data import datalocal
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def img_feat(backbone, img, device):
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        toks = backbone(x)
    f = toks.flatten(2).squeeze(0).cpu().numpy()
    return np.concatenate([f.mean(0), f.std(0)])


def augs(img, rng, k=4):
    outs = [img]
    for _ in range(k):
        r = rng.integers(4)
        if r == 0:
            b = int(rng.integers(-25, 26))
            outs.append(np.clip(img.astype(np.float32) + b, 0, 255).astype(np.uint8))
        elif r == 1:
            g = rng.choice([0.8, 1.25])
            outs.append(np.clip((img.astype(np.float32) / 255.0) ** g * 255.0,
                                0, 255).astype(np.uint8))
        elif r == 2:
            ks = rng.choice([3, 5])
            outs.append(cv2.GaussianBlur(img, (ks, ks), 0))
        else:
            dx = int(rng.integers(-6, 7))
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            outs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                       borderMode=cv2.BORDER_REPLICATE))
    return outs


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_n = sorted(rng.choice(b["init_normal"], 30, replace=False).tolist())
        tr_d = sorted(rng.choice(b["init_defect"], 15, replace=False).tolist())
        te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
        te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())

        def feats(paths, n_aug=0):
            out = []
            for p in paths:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                for im in augs(img, rng)[: n_aug + 1]:
                    out.append(img_feat(backbone, im, device))
            return np.array(out)

        def feats_grouped(paths, n_aug=0):
            """每张图一组增强特征：list[array(n_var, 768)]"""
            out = []
            for p in paths:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                g = [img_feat(backbone, im, device)
                     for im in augs(img, rng)[: n_aug + 1]]
                out.append(np.array(g))
            return out

        Xn, Xd = feats(tr_n), feats(tr_d)
        X = np.concatenate([Xn, Xd])
        y = np.array([0] * len(Xn) + [1] * len(Xd))
        clf = LogisticRegression(max_iter=500).fit(X, y)

        # base：单图 proba
        s_n = clf.predict_proba(feats(te_n))[:, 1]
        s_d = clf.predict_proba(feats(te_d))[:, 1]
        auroc_base = roc_auc_score([0] * len(s_n) + [1] * len(s_d),
                                   list(s_n) + list(s_d))
        print(f"[mechB] {cat}: 判别头 base AUROC={auroc_base:.4f}", flush=True)

        # vote：K 增强 proba 均值（按图分组，无硬编码尺寸）
        v_n = np.array([clf.predict_proba(g)[:, 1].mean() for g in feats_grouped(te_n, n_aug=4)])
        v_d = np.array([clf.predict_proba(g)[:, 1].mean() for g in feats_grouped(te_d, n_aug=4)])
        auroc_vote = roc_auc_score([0] * len(v_n) + [1] * len(v_d),
                                   list(v_n) + list(v_d))
        print(f"[mechB] {cat}: soft-vote TTA AUROC={auroc_vote:.4f} "
              f"(Δ{auroc_vote - auroc_base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
