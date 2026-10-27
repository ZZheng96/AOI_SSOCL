"""跨域机制E验证：拦截库伪框化（blob 定位缺陷 patch 入拦截库 vs 整图，2026-08-26，小样本）

U120 发现：跨域学习后 0.71 平台由缺陷库拦截（intercept）主导，与起点解耦——
要突破需改进拦截本身。demo4 裁切 0.9 证明"缺陷区域 patch 入库比整图纯净"
（框内 patch 信噪比高）。本机制：无 GT 框时用 blob 响应图（域不变特征）定位
top 异常 patch 作为"伪框内 patch"入拦截库，替代整图入库。

验证：train 缺陷图（init_defect）→ 拦截库 A=整图 feats / B=blob top 25% patch →
test 域（正常 vs 缺陷）拦截加分分布的判别 AUROC（学习前静态拦截力对比）。

用法: python scripts/mechE_intercept_crop.py
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch
import torch.nn.functional as F

from src.backbone.dino import FrozenDINO
from src.data import datalocal
from src.ssocl.banks import DefectBank, NormalBank, _norm_patch
from sklearn.metrics import roc_auc_score

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def blob_top_patches(blob, tile_img, feats, ratio=0.25):
    """blob 响应图 → top ratio patch 特征 (K, D)。feats: (1,D,G,G)。"""
    t = tile_img
    g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    r = blob._dog_response(g)
    h, w = r.shape
    G = feats.shape[-1]
    r32 = np.zeros((G, G), dtype=np.float32)
    for i in range(G):
        y0_, y1_ = i * h // G, (i + 1) * h // G
        for j in range(G):
            r32[i, j] = r[y0_:y1_, j * w // G:(j + 1) * w // G].max()
    K = max(1, int(G * G * ratio))
    top_idx = np.argpartition(r32.ravel(), -K)[-K:]
    f = torch.as_tensor(feats, dtype=torch.float32)
    if f.ndim == 3:
        f = f.unsqueeze(0)
    return f[0].permute(1, 2, 0).reshape(-1, f.shape[1])[top_idx].numpy()


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    from src.slots.blob import BlobSlot
    blob = BlobSlot(cfg.get("slots", {}).get("blob", {}), device="cpu")
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_d = sorted(rng.choice(b["init_defect"], 10, replace=False).tolist())
        te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
        te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())

        # 提取特征
        def feats_of(paths):
            out = []
            for p in paths:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    toks = backbone(x)
                out.append({"path": p, "img": img, "feats": toks.cpu().numpy()})
            return out

        tr_items = feats_of(tr_d)
        te_n_items = feats_of(te_n)
        te_d_items = feats_of(te_d)

        # 拦截库 A：整图 feats（现状）
        bankA = DefectBank(topk=8, boost=0.25, sim_thresh=0.15, min_hit=1)
        for it in tr_items:
            bankA.add(it["feats"])
        # 拦截库 B：blob top 25% patch（伪框）——直接 patch 特征入库
        bankB = DefectBank(topk=8, boost=0.25, sim_thresh=0.15, min_hit=1)
        for it in tr_items:
            patch = blob_top_patches(blob, it["img"], it["feats"])   # (K, D)
            f = F.normalize(torch.as_tensor(patch, dtype=torch.float32), dim=1).half()
            bankB.samples.append([f, time.time(), f.shape[0]])
            bankB.version += 1

        def intercept_auroc(bank):
            s_n = [bank.intercept_score(it["feats"]) for it in te_n_items]
            s_d = [bank.intercept_score(it["feats"]) for it in te_d_items]
            return roc_auc_score([0] * 5 + [1] * 5, list(s_n) + list(s_d)), s_n, s_d

        auA, snA, sdA = intercept_auroc(bankA)
        auB, snB, sdB = intercept_auroc(bankB)
        print(f"\n[mechE] {cat}:", flush=True)
        print(f"  拦截库A 整图: AUROC={auA:.4f} 正常boost均值={np.mean(snA):.4f} "
              f"缺陷boost均值={np.mean(sdA):.4f}", flush=True)
        print(f"  拦截库B blob裁剪: AUROC={auB:.4f} 正常boost均值={np.mean(snB):.4f} "
              f"缺陷boost均值={np.mean(sdB):.4f}", flush=True)


if __name__ == "__main__":
    main()
