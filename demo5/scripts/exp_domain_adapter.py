"""判别投影/Adapter 验证（U98，2026-08-24，放宽红线 6 的语义级域适配攻坚）

评审 §8 建议"可微调轻量主干 + 适配层"解决少样本跨域可塑性不足。放宽红线 6 后，
本脚本验证：用 init 数据（normal + defect）训练特征投影/适配器，能否提升 test 域
（跨域）的特征可分性。

方案（由弱到强）：
  LDA        线性判别投影（sklearn，找"正常 vs 缺陷"最大可分方向，可解释基线）
  MLP-Adapter 384→256→384 bottleneck + 判别损失（对比/BCE），恒等初始化残差

关键问题：gyudet 的 init 是 train 域（纯跨域，无 test 域信息），投影泛化到 test 域
能否有效？data_local 的 init_defect 是 test 域缺陷（有 test 域信息），应更有效。

用法: python scripts/exp_domain_adapter.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn

from src.backbone.dino import FrozenDINO
from src.data import gyudet, datalocal
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def extract_patches(backbone, paths, device, per_img=64):
    feats, gidx = [], []
    for gi, p in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            toks = backbone(x)
        f = toks.flatten(2).squeeze(0).permute(1, 0).cpu().numpy()
        sel = np.random.default_rng(gi).choice(len(f), min(per_img, len(f)), replace=False)
        feats.append(f[sel])
        gidx += [gi] * len(sel)
    return np.concatenate(feats), np.array(gidx)


def nn_auroc(train_f, te_n_f, te_d_f):
    s_n = 1 - cosine_similarity(te_n_f, train_f).max(1)
    s_d = 1 - cosine_similarity(te_d_f, train_f).max(1)
    return roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))


def train_mlp_adapter(pos, neg, dim=384, hidden=256, epochs=50, seed=42, device="cuda"):
    """训练 MLP Adapter（bottleneck + 残差，恒等初始化），中心对比损失让 pos/neg 分离。"""
    torch.manual_seed(seed)
    adapter = nn.Sequential(
        nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)).to(device)
    for m in adapter:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    P = torch.tensor(pos, dtype=torch.float32).to(device)
    N = torch.tensor(neg, dtype=torch.float32).to(device)
    for ep in range(epochs):
        opt.zero_grad()
        zp = adapter(P)
        zn = adapter(N)
        cp = zp.mean(0, keepdim=True).detach()
        cn = zn.mean(0, keepdim=True).detach()
        # 中心对比损失：pos 靠近 pos 中心远离 neg 中心，neg 反之；最大化中心距离
        loss = (zp - cp).pow(2).mean() + (zn - cn).pow(2).mean() \
               - (cp - cn).pow(2).sum() * 0.05
        loss.backward()
        opt.step()
    adapter.eval()
    def apply(f):
        t = torch.tensor(f, dtype=torch.float32).to(device)
        with torch.no_grad():
            return (t + adapter(t)).cpu().numpy()
    return apply


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    rng = np.random.default_rng(42)

    # ---- gyudet（纯跨域：init 是 train 域，test 是跨域）----
    root = cfg["datasets"]["gyu_det"]
    tr_n, tr_d = gyudet._load_split(root, "train")
    te_n, te_d = gyudet._load_split(root, "test")
    tr_n = sorted(rng.choice(tr_n, min(60, len(tr_n)), replace=False).tolist())
    tr_d = sorted(rng.choice(tr_d, min(30, len(tr_d)), replace=False).tolist())
    te_n = sorted(rng.choice(te_n, min(30, len(te_n)), replace=False).tolist())
    te_d = sorted(rng.choice(te_d, min(60, len(te_d)), replace=False).tolist())

    tr_n_f, _ = extract_patches(backbone, tr_n, device)
    tr_d_f, _ = extract_patches(backbone, tr_d, device)
    te_n_f, _ = extract_patches(backbone, te_n, device)
    te_d_f, _ = extract_patches(backbone, te_d, device)
    print(f"[U98] gyudet: train正常 {len(tr_n)} / train缺陷 {len(tr_d)} / test正常 {len(te_n)} / test缺陷 {len(te_d)}", flush=True)

    # LDA 判别投影（用 train 域 normal+defect 训练方向）
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    X_tr = np.concatenate([tr_n_f, tr_d_f])
    y_tr = np.array([0] * len(tr_n_f) + [1] * len(tr_d_f))
    lda = LDA(n_components=1).fit(X_tr, y_tr)
    print(f"[U98] gyudet LDA: raw={nn_auroc(tr_n_f, te_n_f, te_d_f):.4f} "
          f"→ LDA投影={nn_auroc(lda.transform(tr_n_f), lda.transform(te_n_f), lda.transform(te_d_f)):.4f}", flush=True)

    # MLP Adapter
    apply_adapt = train_mlp_adapter(tr_d_f[:2000], tr_n_f[:2000], device=device)
    print(f"[U98] gyudet MLP-Adapter: raw={nn_auroc(tr_n_f, te_n_f, te_d_f):.4f} "
          f"→ Adapter={nn_auroc(apply_adapt(tr_n_f), apply_adapt(te_n_f), apply_adapt(te_d_f)):.4f}", flush=True)

    # ---- data_local gold_finger（init_defect 是 test 域缺陷）----
    dl_root = cfg["datasets"]["data_local"]
    b = datalocal.load_category(dl_root, "gold_finger", 100, 30, 42, max_test=0)
    tr_n2 = sorted(rng.choice(b["init_normal"], min(40, len(b["init_normal"])), replace=False).tolist())
    tr_d2 = sorted(rng.choice(b["init_defect"], min(30, len(b["init_defect"])), replace=False).tolist())
    te_n2 = sorted(rng.choice([p for p, y in b["test"] if y == 0], min(15, 17), replace=False).tolist())
    te_d2 = sorted(rng.choice([p for p, y in b["test"] if y == 1], min(40, 85), replace=False).tolist())

    tr_n2_f, _ = extract_patches(backbone, tr_n2, device)
    tr_d2_f, _ = extract_patches(backbone, tr_d2, device)
    te_n2_f, _ = extract_patches(backbone, te_n2, device)
    te_d2_f, _ = extract_patches(backbone, te_d2, device)
    print(f"\n[U98] data_local/gold_finger: train正常 {len(tr_n2)} / test域缺陷 {len(tr_d2)} / test正常 {len(te_n2)} / test缺陷 {len(te_d2)}", flush=True)

    X_tr2 = np.concatenate([tr_n2_f, tr_d2_f])
    y_tr2 = np.array([0] * len(tr_n2_f) + [1] * len(tr_d2_f))
    lda2 = LDA(n_components=1).fit(X_tr2, y_tr2)
    print(f"[U98] data_local LDA: raw={nn_auroc(tr_n2_f, te_n2_f, te_d2_f):.4f} "
          f"→ LDA投影={nn_auroc(lda2.transform(tr_n2_f), lda2.transform(te_n2_f), lda2.transform(te_d2_f)):.4f}", flush=True)

    apply_adapt2 = train_mlp_adapter(tr_d2_f[:2000], tr_n2_f[:2000], device=device)
    print(f"[U98] data_local MLP-Adapter: raw={nn_auroc(tr_n2_f, te_n2_f, te_d2_f):.4f} "
          f"→ Adapter={nn_auroc(apply_adapt2(tr_n2_f), apply_adapt2(te_n2_f), apply_adapt2(te_d2_f)):.4f}", flush=True)


if __name__ == "__main__":
    main()
