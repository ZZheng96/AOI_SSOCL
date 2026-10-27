"""跨域机制G验证：内容最近邻模板匹配能否找到配对修复图（2026-08-26）

核心假设（用户主张"用户配置提供模板时允许配对"）：data_local 的 train/good 是
修复图库（每张 test 原始图都有同名配对修复图，如 0598_repaired ↔ 0598）。若
"内容匹配最近邻"（特征最相似的模板）能找到配对修复图（非文件名配对，合法），
则 tpl 模板差分 ≈ 配对差分（demo4 0.9983 的合法版）——数据层面的前提验证。

只算 DINO 特征（无像素差分，快）：test 图整图特征 vs 模板库整图特征最近邻，
统计"最近邻是否为同名配对修复图"；并算"最近邻模板差分"的判别 AUROC。

用法: python scripts/mechG_tpl_match.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
import torch

from src.backbone.dino import FrozenDINO
from src.data import datalocal
from sklearn.metrics import roc_auc_score

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def stem_key(path):
    """文件名主干（去扩展名、去 _repaired 后缀、去目录）——仅用于统计配对命中
    是否正确（诊断用，不用于任何算法路径）。"""
    base = os.path.basename(path).rsplit(".", 1)[0]
    return base.replace("_repaired", "")


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_n = sorted(b["init_normal"])                       # 模板库（修复图）
        te_n = sorted([p for p, y in b["test"] if y == 0])
        te_d = sorted([p for p, y in b["test"] if y == 1])
        print(f"\n[mechG] {cat}: 模板 {len(tr_n)} test正常 {len(te_n)} 缺陷 {len(te_d)}",
              flush=True)

        # 提取特征（整图 patch 特征拉平后池化：mean-pool 32x32x384 -> 384）
        def emb(paths):
            out = []
            for p in paths:
                from src.common.io import load_image
                img = load_image(p)
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    toks = backbone(x)
                f = toks.flatten(2).squeeze(0).mean(0).cpu().numpy()   # (384,)
                out.append(f)
            return np.stack(out)

        E_tr = emb(tr_n)
        E_n = emb(te_n)
        E_d = emb(te_d)

        # 归一化 + 最近邻
        def nn_sim(q):
            qn = q / (np.linalg.norm(q) + 1e-8)
            trn = E_tr / (np.linalg.norm(E_tr, axis=1, keepdims=True) + 1e-8)
            return trn @ qn                                   # (68,)

        # 配对命中率：test 图的最近邻模板是否为同名 repaired
        def pair_hit(paths, embs):
            hit = 0
            for p, q in zip(paths, embs):
                sim = nn_sim(q)
                best = tr_n[int(np.argmax(sim))]
                if stem_key(best) == stem_key(p):
                    hit += 1
            return hit / len(paths)

        hn = pair_hit(te_n, E_n)
        hd = pair_hit(te_d, E_d)
        print(f"  最近邻=同名配对修复图: test正常 {hn:.3f} ({len(te_n)}张) "
              f"test缺陷 {hd:.3f} ({len(te_d)}张)", flush=True)

        # 最近邻模板差分判别力：test 与最近邻模板的整图特征距离
        def nn_dist(embs):
            out = []
            for q in embs:
                out.append(1.0 - nn_sim(q).max())
            return np.array(out)

        s_n, s_d = nn_dist(E_n), nn_dist(E_d)
        au = roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))
        print(f"  最近邻模板整图差分 AUROC = {au:.4f} "
              f"（正常均值 {s_n.mean():.4f} / 缺陷均值 {s_d.mean():.4f}）", flush=True)

        # 逐位置 patch 差分（tpl 机理）：test 图与最近邻模板的 patch 级 L2 差分，
        # 图像分 = 差分图 top 5% 均值（捕获局部缺陷信号，整图 mean 会稀释）
        def _patches(paths):
            out = []
            for p in paths:
                from src.common.io import load_image
                img = load_image(p)
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    toks = backbone(x)
                out.append(toks.squeeze(0).permute(1, 2, 0).cpu().numpy())  # (G,G,384)
            return out

        def patch_diff_auroc():
            tr_p = _patches(tr_n)                      # list[(32,32,384)]
            tr_mean = np.array([p.mean(0).ravel() for p in tr_p])
            tr_flat = np.array([p.reshape(-1, 384) for p in tr_p])   # (68,1024,384)
            s_nn, s_dd = [], []
            for q in _patches(te_n):
                qm = q.mean(0).ravel()
                t = int(np.argmax(tr_mean @ (qm / (np.linalg.norm(qm) + 1e-8))))
                d = np.abs(q - tr_flat[t].reshape(32, 32, 384)).mean(axis=-1)  # (32,32)
                s_nn.append(np.partition(d.ravel(), -50)[-50:].mean())
            for q in _patches(te_d):
                qm = q.mean(0).ravel()
                t = int(np.argmax(tr_mean @ (qm / (np.linalg.norm(qm) + 1e-8))))
                d = np.abs(q - tr_flat[t].reshape(32, 32, 384)).mean(axis=-1)
                s_dd.append(np.partition(d.ravel(), -50)[-50:].mean())
            return roc_auc_score([0] * len(s_nn) + [1] * len(s_dd),
                                 list(s_nn) + list(s_dd)), np.array(s_nn), np.array(s_dd)

        au_p, sn_p, sd_p = patch_diff_auroc()
        print(f"  最近邻模板逐位置差分 AUROC = {au_p:.4f} "
              f"（正常均值 {sn_p.mean():.4f} / 缺陷均值 {sd_p.mean():.4f}）", flush=True)


if __name__ == "__main__":
    main()
