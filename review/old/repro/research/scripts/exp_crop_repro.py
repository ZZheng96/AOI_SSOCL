"""U71（2026-08-18）：标注裁切图复现 demo4 0.9 实验

目的：用 GYU-DET GT 框裁切图（对标 demo4 的 crop 训练），在 demo5 框架内
验证 shead 判别头能否复现 demo4 的 0.9 AUROC。

根因链（demo4 0.9 调查结论）：
  - demo4 用 GT 框+pad 裁切缺陷 patch + 同图随机裁切正常 patch
  - crop 图信噪比高（框内区域，无背景干扰）-> DINO 特征判别力 0.86-0.93
  - 整图仅 0.5（背景 patch 稀释缺陷信号）

实验流程：
  1. 从 train 裁切 crop（缺陷框+pad / 正常随机框 IoU<0.05）-> 训练 shead 头
  2. 从 test 裁切 crop -> 评测 AUROC
  3. 对照：整图 shead AUROC（已知 ~0.75）
  4. 对照：crop 图数量 vs AUROC（少量 crop 能到多少）

诚实边界：GT 框 = test 域标注信息，本实验是"离线全量裁切监督"（对标 demo4），
验证特征上限；非在线学习路径（在线路径走 DefectBank.add_box + head_ft）。
信息层次：有标注（用户提供的缺陷框标注），诚实记录。
"""
import os
import sys
import json
import argparse
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import SheadHead, _image_level_feats, DeviationLoss


def parse_yolo_boxes(lbl_path, img_w, img_h):
    """解析 YOLO txt -> 像素 xyxy 框列表。"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, w, h = (float(v) for v in parts[:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            x0 = (cx - w / 2) * img_w
            y0 = (cy - h / 2) * img_h
            x1 = (cx + w / 2) * img_w
            y1 = (cy + h / 2) * img_h
            boxes.append([x0, y0, x1, y1])
    return boxes


def iou(a, b):
    """两 xyxy 框 IoU。"""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + bb - inter + 1e-9)


def build_crops(root, split, rng, normal_per_image=3, pad_ratio=0.2,
                min_side=24, max_defect_crops=None, max_normal_crops=None):
    """从 {split} 裁切 crop 图。

    缺陷 crop：每个 YOLO 框 +pad_ratio pad 裁切，最短边 >= min_side。
    正常 crop：同图内与所有框 IoU<0.05 的随机正方形裁剪，边长取框中位数 0.8-1.5 倍，
              每图 normal_per_image 张。
    返回 (defect_crops, normal_crops)，每个是 list[ndarray HxWx3 uint8]。
    """
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    defect_crops, normal_crops = [], []
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    for f in files:
        if max_defect_crops and len(defect_crops) >= max_defect_crops and \
           max_normal_crops and len(normal_crops) >= max_normal_crops:
            break
        img_path = os.path.join(img_dir, f)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        boxes = parse_yolo_boxes(lbl_path, W, H)
        # 缺陷 crop
        if boxes and (not max_defect_crops or len(defect_crops) < max_defect_crops):
            for bx in boxes:
                if max_defect_crops and len(defect_crops) >= max_defect_crops:
                    break
                bw = bx[2] - bx[0]; bh = bx[3] - bx[1]
                pw = bw * pad_ratio; ph = bh * pad_ratio
                x0 = max(0, int(bx[0] - pw)); y0 = max(0, int(bx[1] - ph))
                x1 = min(W, int(bx[2] + pw)); y1 = min(H, int(bx[3] + ph))
                if x1 - x0 < min_side or y1 - y0 < min_side:
                    continue
                crop = np.array(img.crop((x0, y0, x1, y1)))
                defect_crops.append(crop)
        # 正常 crop
        if not boxes:
            # 无框图 = 正常图，随机裁切
            boxes_for_iou = []
        else:
            boxes_for_iou = boxes
        if max_normal_crops and len(normal_crops) >= max_normal_crops:
            continue
        # 边长参考框中位数
        if boxes_for_iou:
            sides = [max(b[2]-b[0], b[3]-b[1]) for b in boxes_for_iou]
            med_side = float(np.median(sides))
        else:
            med_side = min(W, H) * 0.3
        for _ in range(normal_per_image):
            if max_normal_crops and len(normal_crops) >= max_normal_crops:
                break
            side = int(med_side * rng.uniform(0.8, 1.5))
            side = max(min_side, min(side, min(W, H) - 1))
            if side >= min(W, H):
                continue
            x0 = rng.integers(0, W - side)
            y0 = rng.integers(0, H - side)
            cand = [x0, y0, x0 + side, y0 + side]
            if all(iou(cand, b) < 0.05 for b in boxes_for_iou):
                normal_crops.append(np.array(img.crop((cand[0], cand[1], cand[2], cand[3]))))
    return defect_crops, normal_crops


def extract_feats(backbone, crops, batch=8):
    """crop 列表 -> (N, D+6) 特征。"""
    feats = []
    for i in range(0, len(crops), batch):
        f = backbone.extract_tiles(crops[i:i+batch], batch=batch)
        feats.append(_image_level_feats(f).cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


def train_shead(neg_feats, pos_feats, device, epochs=100, lr=1e-3, batch=32,
                margin=5.0, n_pseudo=3, rng=None):
    """训练 shead 头（对标 demo4 _train_discriminative_head）。

    负样本 = 正常 crop 特征（y=0）
    伪异常 = 正常特征 + randn*0.1（y=1，特征空间噪声）
    真实异常 = 缺陷 crop 特征（y=1）
    DeviationLoss + Adam。
    """
    rng = rng or np.random.default_rng(42)
    # 伪异常
    pseudo = []
    for _ in range(n_pseudo):
        noise = neg_feats + torch.randn_like(neg_feats) * 0.1
        pseudo.append(noise)
    pseudo = torch.cat(pseudo, dim=0)
    pos = torch.cat([pseudo, pos_feats], dim=0)
    neg = neg_feats
    X = torch.cat([neg, pos]).float().to(device)
    y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(device)
    in_dim = X.shape[1]
    head = SheadHead(in_dim, 256, 0.3).to(device)
    loss_fn = DeviationLoss(margin=margin)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    print(f"[crop-shead] 训练: neg={len(neg)} pseudo={len(pseudo)} "
          f"real_defect={len(pos_feats)} pos_total={len(pos)} X={len(X)}", flush=True)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            xb = X[perm[i:i+batch]]; yb = y[perm[i:i+batch]]
            scores = head(xb)
            loss = loss_fn(scores, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                s = head(X)
                l_n = s[y == 0].pow(2).mean().item()
                l_a = (margin - s[y == 1]).clamp(min=0).pow(2).mean().item()
            print(f"  epoch {ep+1}/{epochs} loss_n={l_n:.4f} loss_a={l_a:.4f}", flush=True)
    head.eval()
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\CGAIC\data_origin\GYU-DET")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n-pseudo", type=int, default=3)
    ap.add_argument("--normal-per-image", type=int, default=3)
    ap.add_argument("--max-train-defect", type=int, default=500,
                    help="train 缺陷 crop 上限（控制耗时）")
    ap.add_argument("--max-train-normal", type=int, default=500)
    ap.add_argument("--max-test-defect", type=int, default=300)
    ap.add_argument("--max-test-normal", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..",
                                                     "configs", "m0_gyudet_sheadA.yaml"))
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    print(f"[U71 crop 复现] root={args.root} device={device}", flush=True)

    # 1. 裁切 crop
    print("[1/4] 裁切 train crop...", flush=True)
    tr_def, tr_nor = build_crops(args.root, "train", rng,
                                 normal_per_image=args.normal_per_image,
                                 max_defect_crops=args.max_train_defect,
                                 max_normal_crops=args.max_train_normal)
    print(f"  train: 缺陷crop={len(tr_def)} 正常crop={len(tr_nor)}", flush=True)
    print("[1/4] 裁切 test crop...", flush=True)
    te_def, te_nor = build_crops(args.root, "test", rng,
                                 normal_per_image=args.normal_per_image,
                                 max_defect_crops=args.max_test_defect,
                                 max_normal_crops=args.max_test_normal)
    print(f"  test: 缺陷crop={len(te_def)} 正常crop={len(te_nor)}", flush=True)

    # 2. 构建 backbone
    print("[2/4] 构建 DINOv2 backbone...", flush=True)
    bcfg = cfg["backbone"]
    backbone = FrozenDINO(model_name=bcfg.get("name", "dinov2_vits14").replace("dinov2_", ""),
                          output_layer=bcfg["output_layer"], grid=bcfg["grid"], device=device)

    # 3. 提特征 + 训练
    print("[3/4] 提取 crop 特征...", flush=True)
    tr_nor_feats = extract_feats(backbone, tr_nor)
    tr_def_feats = extract_feats(backbone, tr_def)
    print(f"  train 正常特征{tr_nor_feats.shape} 缺陷特征{tr_def_feats.shape}", flush=True)
    print("[3/4] 训练 shead 头...", flush=True)
    head = train_shead(tr_nor_feats, tr_def_feats, device, epochs=args.epochs,
                       n_pseudo=args.n_pseudo, rng=rng)

    # 4. 评测
    print("[4/4] 评测 test crop...", flush=True)
    te_nor_feats = extract_feats(backbone, te_nor)
    te_def_feats = extract_feats(backbone, te_def)
    with torch.no_grad():
        s_nor = head(te_nor_feats.to(device)).cpu().numpy()
        s_def = head(te_def_feats.to(device)).cpu().numpy()
    labels = np.concatenate([np.zeros(len(s_nor)), np.ones(len(s_def))])
    scores = np.concatenate([s_nor, s_def])
    auroc = float(roc_auc_score(labels, scores))
    # 正常/缺陷得分分布
    print(f"\n=== U71 crop 复现结果 ===")
    print(f"  test crop: 正常={len(te_nor)} 缺陷={len(te_def)}")
    print(f"  正常得分: mean={s_nor.mean():.4f} std={s_nor.std():.4f}")
    print(f"  缺陷得分: mean={s_def.mean():.4f} std={s_def.std():.4f}")
    print(f"  AUROC = {auroc:.4f}")
    print(f"  (对照: 整图 shead AUROC ~0.7458, demo4 crop AUROC 0.86-0.93)")
    elapsed = time.time() - t0
    print(f"\n[done] 耗时 {elapsed:.0f}s")

    # 保存结果
    out = {"experiment": "U71_crop_repro",
           "auroc": round(auroc, 4),
           "n_train_defect": len(tr_def), "n_train_normal": len(tr_nor),
           "n_test_defect": len(te_def), "n_test_normal": len(te_nor),
           "epochs": args.epochs, "n_pseudo": args.n_pseudo,
           "normal_score_mean": float(s_nor.mean()),
           "defect_score_mean": float(s_def.mean()),
           "elapsed": round(elapsed, 1)}
    out_path = os.path.join(cfg["output_dir"], "diag_crop_repro.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
