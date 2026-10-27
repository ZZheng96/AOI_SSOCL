"""U70 聚焦特征机制诊断（2026-08-18，临时，20 张级少量图）：
跨域（train→test）下"全图统计特征 vs 聚焦统计特征"的特征空间判别力。

协议（诚实）：train/good 30 张建正常库（协议内 fit 同源）；
test 分层抽样 20 正常 + 20 缺陷（U43 口径）；init_defect 30 张（协议内）。
指标：test 图特征（390 维）→ train_normal 库最近邻 L2 距离 = 异常分 → AUROC。
对比 focus_topk = None / 0.05 / 0.10（同口径聚焦特征本身，绕过头训练噪声）。
回答：聚焦是否破坏跨域特征语义（判别力下降在特征层而非训练层）。
"""
import os
import sys
import yaml
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone.dino import FrozenDINO
from src.slots.shead import _image_level_feats
from src.data import gyudet
from src.eval.metrics import image_metrics


def load_imgs(paths):
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_sheadA.yaml"),
                              encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=cfg["backbone"]["output_layer"],
                          grid=cfg["backbone"]["grid"], device=device)
    rng = np.random.default_rng(42)
    root = cfg["datasets"]["gyu_det"]
    tr_n, tr_d = gyudet._load_split(root, "train")
    te_n, te_d = gyudet._load_split(root, "test")
    # 协议内库：train/good 30（与 fit max_base 同级）+ init_defect 30
    n_base, n_def, n_te_n, n_te_d = 30, 30, 20, 20
    base = sorted(rng.choice(tr_n, n_base, replace=False).tolist())
    defect = sorted(rng.choice(tr_d, min(n_def, len(tr_d)), replace=False).tolist())
    te_n_sel = sorted(rng.choice(te_n, min(n_te_n, len(te_n)), replace=False).tolist())
    te_d_sel = sorted(rng.choice(te_d, min(n_te_d, len(te_d)), replace=False).tolist())
    print(f"[u70diag] base={len(base)} defect={len(defect)} "
          f"test_normal={len(te_n_sel)} test_defect={len(te_d_sel)}", flush=True)

    def feats_of(imgs, focus_topk, bs=16):
        out = []
        for i in range(0, len(imgs), bs):
            f = backbone.extract_tiles(imgs[i:i + bs], batch=bs)
            out.append(_image_level_feats(f, False, False, focus_topk))
        return torch.cat(out, dim=0).numpy()

    base_imgs = load_imgs(base)
    te_imgs = load_imgs(te_n_sel + te_d_sel)
    labels = np.array([0] * len(te_n_sel) + [1] * len(te_d_sel))
    print(f"[u70diag] 图加载完成 base={len(base_imgs)} test={len(te_imgs)}", flush=True)
    for focus in (None, 0.01, 0.05, 0.10):
        Fb = feats_of(base_imgs, focus)
        Ft = feats_of(te_imgs, focus)
        # 最近邻 L2 距离
        d = np.linalg.norm(Ft[:, None, :] - Fb[None, :, :], axis=-1).min(axis=1)
        m = image_metrics(labels, d)
        # 正常/缺陷均值向量分离度（类间距离 proxy）
        sep = np.linalg.norm(Ft[labels == 0].mean(0) - Ft[labels == 1].mean(0))
        # 聚焦特征 vs 全图特征余弦一致性（看聚焦改变了多少）
        F_full = feats_of(te_imgs, None)
        c = np.mean([np.dot(Ft[i], F_full[i]) /
                     (np.linalg.norm(Ft[i]) * np.linalg.norm(F_full[i]) + 1e-9)
                     for i in range(len(Ft))])
        print(f"[u70diag] focus_topk={focus}: 最近邻 AUROC={m['auroc']:.4f} "
              f"类间分离={sep:.3f} 特征余弦(聚焦vs全图)={c:.4f}", flush=True)


if __name__ == "__main__":
    main()
