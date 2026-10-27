"""U82（2026-08-19）：在线标注裁切学习 -- 重训判别头（等效 U71 离线 crop 训练）

U81 结论：crop 特征入缺陷库修正块分无效（跨分布）；"等效离线裁切训练"的
正确实现 = 用反馈框内 crop 特征**在线重训判别头**（U71 离线：500 缺陷 crop
+500 正常 crop 训练判别头达 0.9651）。

本实验闭环：
1. fit ablB 判别头（train 域 crop）
2. 缓存 train crop 特征（pos=init_defect 框+pad+上下文 crop；neg=正常随机 crop）
   -> 重训时保留防灾难性遗忘
3. 学习轮次：反馈带框 -> 框内 crop 特征 (D+6) 入 pos 缓冲；反馈正常 ->
   正常块特征入 neg 缓冲；缓冲达标 -> DeviationLoss 重训 shead 头
   （合并 train crop + 在线 crop，等效"离线全量裁切训练 + 在线增量"）
4. eval：块级（30 张 test x 16 块 = 480）+ 图级（topk 聚合），每轮评测

诚实边界：初始 fit + train crop 只用 train 域；在线 crop 来自反馈真值
（操作员标注框，U64 有标注层次）；eval 集 30 张全程只评测不学习。
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import (SheadSlot, _image_level_feats, _parse_yolo_boxes,
                             DeviationLoss, _extract_box_crops,
                             _extract_context_crops, _extract_random_crops)
from src.ssocl.block_learning import BlockLearner
from src.data import gyudet

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42
GRID = 4
N_EVAL_DEF = 15       # U85：30 张 eval（test 正常图稀少，分层抽样）
N_EVAL_NOR = 15
N_ROUNDS = 8
POOL_PER_ROUND = 40   # U85：加大反馈量（每轮 40 张）
RT_MIN_POS = 30       # 重训触发：在线 pos 缓冲数
RT_MIN_NEG = 60
RT_EPOCHS = 30        # U85：恢复 U82 有效参数（epochs 30）
RT_LR = 1e-3
RT_MIN_CROP_SIDE = 30  # U85：只过滤极小的 crop（30px）


def grid4_blocks(img, boxes):
    h, w = img.shape[:2]
    gh, gw = h // GRID, w // GRID
    blocks, hit_idx = [], []
    for r in range(GRID):
        for c in range(GRID):
            y0, x0 = r * gh, c * gw
            y1, x1 = min(h, y0 + gh), min(w, x0 + gw)
            blocks.append(img[y0:y1, x0:x1])
            hit = False
            for bx in boxes:
                iw = max(0, min(x1, bx[2]) - max(x0, bx[0]))
                ih = max(0, min(y1, bx[3]) - max(y0, bx[1]))
                if iw * ih > 0:
                    hit = True
                    break
            if hit:
                hit_idx.append(r * GRID + c)
    return blocks, hit_idx


def pick_test(n_def, n_nor, seed=42):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    defect, normal = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir, f)
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            defect.append(p)
        else:
            normal.append(p)
    di = rng.choice(len(defect), n_def, replace=False)
    ni = rng.choice(len(normal), min(n_nor, len(normal)), replace=False)
    return [(defect[i], 1) for i in di] + [(normal[i], 0) for i in ni]


def load_boxes(path):
    split = os.path.basename(os.path.dirname(os.path.dirname(path)))
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = os.path.join(GYU_ROOT, split, "labels",
                       os.path.splitext(os.path.basename(path))[0] + ".txt")
    return img, _parse_yolo_boxes(lbl, w, h)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U82 在线标注裁切学习-重训判别头] device={device}", flush=True)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bundle = gyudet.load_gyudet(GYU_ROOT, cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], SEED)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    bcfg = cfg["backbone"]
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({**cfg["slots"]["shead"], "seed": SEED}, backbone, device)
    ctx = {"train_tile_imgs": [], "defect_tile_imgs": [], "defect_paths": []}
    for p in bundle["init_normal"]:
        ctx["train_tile_imgs"].append([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)])
    for p in bundle["init_defect"]:
        ctx["defect_tile_imgs"].append([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)])
        ctx["defect_paths"].append(p)
    t0 = time.time()
    shead.fit(ctx)
    print(f"  shead fit 完成 ({time.time() - t0:.0f}s)", flush=True)
    crop_size = shead.crop_size
    batch = 8

    def feats_d6(imgs):
        f = backbone.extract_tiles(imgs, batch=batch)
        return _image_level_feats(f.float().to(device),
                                  shead.chan_stats, shead.patch_stats).cpu()

    # 1. 缓存 train crop 特征（重训保留防遗忘；与 _fit_crop 同配方）
    rng = np.random.default_rng(SEED)
    train_pos, train_neg = [], []
    # pos: init_defect 框+pad + 上下文 crop
    for i, p in enumerate(bundle["init_defect"]):
        img, boxes = load_boxes(p)
        if boxes:
            crops = _extract_box_crops(img, boxes, 0.2) + \
                _extract_context_crops(img, boxes, crop_size)
            if crops:
                train_pos.append(feats_d6(crops))
    # neg: init_normal 每图 10 随机 crop
    for i in range(len(bundle["init_normal"])):
        img = ctx["train_tile_imgs"][i][0]
        crops = _extract_random_crops(img, crop_size, 10, rng)
        if crops:
            train_neg.append(feats_d6(crops))
    X_tr_pos = torch.cat(train_pos, dim=0) if train_pos else torch.empty(0, 390)
    X_tr_neg = torch.cat(train_neg, dim=0) if train_neg else torch.empty(0, 390)
    print(f"  train crop 缓存: pos={X_tr_pos.shape[0]} neg={X_tr_neg.shape[0]} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 2. eval 集（30 张 test，全程只评测不学习）+ train 域门控集（重训回滚用）
    eval_set = pick_test(N_EVAL_DEF, N_EVAL_NOR, seed=SEED)
    eval_tiles, eval_labels, eval_def_idx = [], [], []
    for path, lab in eval_set:
        img, boxes = load_boxes(path)
        blocks, hit = grid4_blocks(img, boxes)
        eval_tiles.append(backbone.extract_tiles(blocks, batch=8).cpu())
        eval_labels.extend([1 if i in hit else 0 for i in range(GRID * GRID)])
        eval_def_idx.append(hit)
    eval_cat = torch.cat(eval_tiles, dim=0)
    eval_means = eval_cat.mean(dim=(2, 3))          # (480, D) 块均值（库修正用）

    # U85：BlockLearner 初始库（同分布块特征：train 正常块 + init_defect 含框块）
    rng_b = np.random.default_rng(SEED + 7)
    b_sel_n = sorted(rng_b.choice(len(bundle["init_normal"]), 20,
                                  replace=False).tolist())
    seed_n_imgs, seed_d_imgs = [], []
    for i in b_sel_n:
        img = ctx["train_tile_imgs"][i][0]
        blocks, _ = grid4_blocks(img, [])
        seed_n_imgs.extend(blocks)
    for i in range(len(bundle["init_defect"])):
        img, boxes = load_boxes(bundle["init_defect"][i])
        blocks, hit = grid4_blocks(img, boxes)
        for j in hit:
            seed_d_imgs.append(blocks[j])
    seed_n = list(backbone.extract_tiles(seed_n_imgs, batch=16)
                  .mean(dim=(2, 3)).cpu())
    seed_d = list(backbone.extract_tiles(seed_d_imgs, batch=16)
                  .mean(dim=(2, 3)).cpu()) if seed_d_imgs else []
    learner = BlockLearner(dim=backbone.dim, lambda_boost=5.0,
                           shrink_thresh=0.92, max_shrink=20).seed(seed_n, seed_d)
    print(f"  块级库初始: normal={len(learner.normal)} defect={len(learner.defect)}",
          flush=True)

    def head_scores(f4):
        """分批打分（控显存）。"""
        outs = []
        for i in range(0, len(f4), 128):
            x = _image_level_feats(f4[i:i + 128].float().to(device),
                                   shead.chan_stats, shead.patch_stats)
            outs.append(shead.head(x.to(device)).detach().cpu())
        return torch.cat(outs)

    def eval_all(tag):
        # 单级 16 块打分 + BlockLearner 库修正（U80 同分布有效）
        s = head_scores(eval_cat)
        s_adj = learner.adjust(eval_means, s)
        auc = float(roc_auc_score(eval_labels, s_adj.numpy()))
        y_img = [1 if any(h) else 0 for h in eval_def_idx]
        iaucs = []
        for i in range(len(eval_set)):
            iaucs.append(float(s_adj[i * 16:(i + 1) * 16].topk(3).values.mean()))
        auc_img = float(roc_auc_score(y_img, iaucs))
        print(f"  [eval-{tag}] 块级(480) AUROC={auc:.4f}  图级({len(eval_set)}) "
              f"AUROC={auc_img:.4f}", flush=True)
        return auc, auc_img

    # 3. 在线学习（重训判别头）
    online_pos, online_neg = [], []
    results = [{"round": 0}]
    b_b, b_i = eval_all("轮0-基线")
    results[0].update(block_auc=round(b_b, 4), img_auc=round(b_i, 4))

    test_all = pick_test(200, 80, seed=SEED + 1)   # 候选池（充足缺陷 + 正常）
    eval_paths = {p for p, _ in eval_set}
    pool = [(p, l) for p, l in test_all if p not in eval_paths]
    rng_pool = np.random.default_rng(SEED + 2)
    n_rt = 0
    for r in range(1, N_ROUNDS + 1):
        sel = rng_pool.choice(len(pool), POOL_PER_ROUND, replace=False)
        for k in sel:
            path, label = pool[k]
            img, boxes = load_boxes(path)
            blocks, hit = grid4_blocks(img, boxes)
            f4 = backbone.extract_tiles(blocks, batch=8).cpu()
            scores = head_scores(f4)
            # 块级库学习（U80 同分布：误报块扩正常库/漏检块入缺陷库+收缩）
            learner.learn(f4.mean(dim=(2, 3)), scores, label, hit)
            # 标注裁切学习（U82 重训：反馈框内 crop 特征入 pos 缓冲）
            if label == 1 and boxes:
                crops = _extract_box_crops(img, boxes, 0.2)
                crops = [c for c in crops
                         if min(c.shape[0], c.shape[1]) >= RT_MIN_CROP_SIDE]
                if crops:
                    online_pos.append(feats_d6(crops))
            elif label == 0:
                online_neg.append(feats_d6(blocks[:4]))
        n_pos = int(sum(x.shape[0] for x in online_pos))
        n_neg = int(sum(x.shape[0] for x in online_neg))
        # 重训触发：在线缓冲达标 -> 合并 train crop 重训判别头
        if n_pos >= RT_MIN_POS and n_neg >= RT_MIN_NEG:
            Xp = torch.cat([X_tr_pos] + online_pos, dim=0).float()
            Xn = torch.cat([X_tr_neg] + online_neg, dim=0).float()
            X = torch.cat([Xn, Xp]).detach().to(device)
            y = torch.cat([torch.zeros(len(Xn)), torch.ones(len(Xp))]).to(device)
            loss_fn = DeviationLoss(margin=float(
                cfg["slots"]["shead"].get("margin", 5.0)))
            opt = torch.optim.Adam(shead.head.parameters(), lr=RT_LR)
            bs = 64
            for ep in range(RT_EPOCHS):
                perm = torch.randperm(len(X))
                for i in range(0, len(X), bs):
                    xb = X[perm[i:i + bs]]; yb = y[perm[i:i + bs]]
                    s = shead.head(xb)
                    loss = loss_fn(s, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
            n_rt += 1
            online_pos, online_neg = [], []     # 重训后清空缓冲
            print(f"  轮{r}: 重训判别头 #{n_rt} (train pos={len(X_tr_pos)} "
                  f"neg={len(X_tr_neg)} + 在线 pos={n_pos} neg={n_neg})",
                  flush=True)
        st = f"轮{r}"
        ba, ia = eval_all(st)
        results.append({"round": r, "block_auc": round(ba, 4),
                        "img_auc": round(ia, 4), "n_retrain": n_rt,
                        "n_pos_buf": n_pos, "n_neg_buf": n_neg})
        json.dump({"results": results},
                  open(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                    "m0", "crop_learn_retrain.json"), "w"),
                  ensure_ascii=False, indent=2)

    print(f"\n=== 汇总 ===")
    for res in results:
        print(f"  轮{res['round']}: 块级={res.get('block_auc')} "
              f"图级={res.get('img_auc')} 重训={res.get('n_retrain', 0)}")
    print(f"  基线 块级={b_b:.4f} 图级={b_i:.4f} -> 末轮 块级="
          f"{results[-1]['block_auc']:.4f} 图级={results[-1]['img_auc']:.4f}")


if __name__ == "__main__":
    main()
