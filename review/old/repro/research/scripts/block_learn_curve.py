"""U80（2026-08-19）：块级持续学习实验 -- 用户 U75 方案验证

闭环：fit ablB crop 判别头 -> BlockLearner 初始库（train 正常块 + init_defect
缺陷块）-> 学习轮次（test 流反馈：误报块扩正常库 / 漏检块入缺陷库+收缩正常库）
-> 每轮评测：块级（10 test 图 x 16 块 = 160 小图，对齐 U79）+ 图级（10 整图）。

指标：块级 AUROC（含缺陷块 vs 不含）+ 图级 AUROC（top-k 块聚合）。
期望：学习后块级/图级 AUROC 提升（误报块被正常库覆盖、漏检块被缺陷库捕获）。
诚实边界：初始库只用 train 域；学习用 test 流操作员真值（U67 协议：eval 集
从不反馈，其余 test 图反馈）；eval 集 10 张全程只评测不学习。
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml
import cv2
from sklearn.metrics import roc_auc_score

from src.backbone.dino import FrozenDINO
from src.slots.shead import SheadSlot, _image_level_feats, _parse_yolo_boxes
from src.ssocl.block_learning import BlockLearner
from src.data import gyudet

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42
GRID = 4            # 4x4 = 16 块
N_EVAL_DEF = 15     # eval 集缺陷 15 + 正常 15 = 30 张 -> 480 块
N_EVAL_NOR = 15
N_ROUNDS = 10       # 学习轮次（U81：更多轮验证收敛）
POOL_PER_ROUND = 10  # 每轮反馈 10 张 test 图
LAMBDA = 5.0        # 库修正缩放
PAD_RATIO = 0.2     # 框内 crop pad（U71 同款）


def box_crops(img, boxes, pad_ratio=PAD_RATIO):
    """框+pad 裁切（对齐 U71 exp_crop_repro.build_crops 缺陷 crop）。"""
    h, w = img.shape[:2]
    out = []
    for bx in boxes:
        bw = bx[2] - bx[0]; bh = bx[3] - bx[1]
        pw = bw * pad_ratio; ph = bh * pad_ratio
        x0 = max(0, int(bx[0] - pw)); y0 = max(0, int(bx[1] - ph))
        x1 = min(w, int(bx[2] + pw)); y1 = min(h, int(bx[3] + ph))
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        out.append(img[y0:y1, x0:x1])
    return out


def grid4_blocks(img, boxes):
    """4x4 网格裁块 + 含框块索引。img HxWx3; boxes 像素 xyxy。"""
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
    ni = rng.choice(len(normal), n_nor, replace=False)
    return [(defect[i], 1) for i in di] + [(normal[i], 0) for i in ni]


def load_boxes(path):
    """按路径所在 split 读取 GT 框（train/test/valid 通用）。"""
    split = os.path.basename(os.path.dirname(os.path.dirname(path)))
    img_dir = os.path.join(GYU_ROOT, split, "images")
    lbl_dir = os.path.join(GYU_ROOT, split, "labels")
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = os.path.join(lbl_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
    return img, _parse_yolo_boxes(lbl, w, h)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U80 块级持续学习] device={device}", flush=True)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bundle = gyudet.load_gyudet(GYU_ROOT, cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], SEED)

    # 1. fit ablB 判别头（只 fit shead slot，绕过管线校准/sanity ~7min）
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    bcfg = cfg["backbone"]
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({**cfg["slots"]["shead"], "seed": SEED}, backbone, device)
    # 构造 _fit_crop 所需 ctx（single tile = 整图）
    ctx = {"train_tile_imgs": [], "defect_tile_imgs": [], "defect_paths": []}
    for p in bundle["init_normal"]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ctx["train_tile_imgs"].append([img])
    for p in bundle["init_defect"]:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ctx["defect_tile_imgs"].append([img])
        ctx["defect_paths"].append(p)
    t0 = time.time()
    shead.fit(ctx)
    print(f"  shead fit 完成 ({time.time() - t0:.0f}s)", flush=True)

    # 2. BlockLearner 初始库：train 正常块 + init_defect 缺陷块（4x4）
    def block_feats4(imgs):
        """块 4D 特征 (B,D,G,G)。"""
        return backbone.extract_tiles(imgs, batch=8).cpu()

    def head_scores(f4):
        """块 4D 特征 -> 判别头分数 (B,)。"""
        x = _image_level_feats(f4.float().to(device),
                               shead.chan_stats, shead.patch_stats)
        return shead.head(x.to(device)).detach().cpu()

    n_blocks_seed = 0
    seed_normal_imgs, seed_defect_imgs = [], []
    # 正常块：init_normal 取 30 张（批量收集后一次提取）
    rng = np.random.default_rng(SEED)
    sel_n = sorted(rng.choice(len(bundle["init_normal"]), 30, replace=False).tolist())
    for i in sel_n:
        img = cv2.cvtColor(cv2.imread(bundle["init_normal"][i]), cv2.COLOR_BGR2RGB)
        blocks, _ = grid4_blocks(img, [])
        seed_normal_imgs.extend(blocks)
    # 缺陷块：init_defect 框内 crop 特征（U81：缺陷库用高 SNR 框内特征）
    for i, p in enumerate(bundle["init_defect"]):
        img, boxes = load_boxes(p)
        for c in box_crops(img, boxes):
            seed_defect_imgs.append(c)
    print(f"  种子块: 正常={len(seed_normal_imgs)} 缺陷crop={len(seed_defect_imgs)}",
          flush=True)
    seed_normal = list(block_feats4(seed_normal_imgs).mean(dim=(2, 3)))
    seed_defect = (list(block_feats4(seed_defect_imgs).mean(dim=(2, 3)))
                   if seed_defect_imgs else [])
    learner = BlockLearner(dim=backbone.dim, lambda_boost=LAMBDA).seed(
        seed_normal, seed_defect)
    print(f"  初始库: normal={len(learner.normal)} defect_crop={len(learner.defect)}",
          flush=True)

    # 3. eval 集（30 张 test = 15 缺陷 + 15 正常，480 块，全程只评测不学习）
    eval_set = pick_test(N_EVAL_DEF, N_EVAL_NOR, seed=SEED)
    eval_tiles, eval_labels = [], []
    eval_def_idx = []
    for path, lab in eval_set:
        img, boxes = load_boxes(path)
        blocks, hit = grid4_blocks(img, boxes)
        f4 = block_feats4(blocks)
        eval_tiles.append(f4)
        eval_labels.extend([1 if i in hit else 0 for i in range(GRID * GRID)])
        eval_def_idx.append(hit)
    eval_tiles_cat = torch.cat(eval_tiles, dim=0)   # (160, D, G, G)
    eval_means = eval_tiles_cat.mean(dim=(2, 3))    # (160, D)

    def eval_blocks(tag):
        s = head_scores(eval_tiles_cat)
        s_adj = learner.adjust(eval_means, s)
        auc = float(roc_auc_score(eval_labels, s_adj.numpy()))
        # 图级：每图 top-k 块聚合
        y_img = [1 if any(h) else 0 for h in eval_def_idx]
        img_aucs = []
        for i in range(len(eval_set)):
            s16 = s_adj[i * 16:(i + 1) * 16]
            img_aucs.append(float(s16.topk(3).values.mean()))
        auc_img = float(roc_auc_score(y_img, img_aucs))
        print(f"  [eval-{tag}] 块级(160) AUROC={auc:.4f}  图级(10) AUROC={auc_img:.4f}")
        return auc, auc_img

    # 4. 学习轮次
    results = [{"round": 0, "block_auc": None, "img_auc": None}]
    base_block, base_img = eval_blocks("轮0-基线")
    results[0]["block_auc"] = round(base_block, 4)
    results[0]["img_auc"] = round(base_img, 4)

    # 学习 pool：test 图（除 eval 30 张外），分轮次随机取
    test_all = pick_test(100, 50, seed=SEED + 1)   # 缺陷 100 + 正常 50 候选池
    eval_paths = {p for p, _ in eval_set}
    pool = [(p, l) for p, l in test_all if p not in eval_paths]
    print(f"  学习 pool: {len(pool)} 张（每轮取 {POOL_PER_ROUND}）", flush=True)
    rng_pool = np.random.default_rng(SEED + 2)
    for r in range(1, N_ROUNDS + 1):
        sel = rng_pool.choice(len(pool), POOL_PER_ROUND, replace=False)
        for k in sel:
            path, label = pool[k]
            img, boxes = load_boxes(path)
            blocks, hit = grid4_blocks(img, boxes)
            f4 = block_feats4(blocks)
            scores = head_scores(f4)
            # 标注裁切学习（U81）：缺陷反馈时框内 crop 特征入缺陷库
            crop_feats = None
            if label == 1 and boxes:
                crops = box_crops(img, boxes)
                if crops:
                    crop_feats = list(block_feats4(crops).mean(dim=(2, 3)))
            learner.learn(f4.mean(dim=(2, 3)), scores, label, hit, crop_feats)
        st = learner.state()
        print(f"  轮{r}: 库 normal={st['n_normal']} defect={st['n_defect']} "
              f"stats={st['stats']}", flush=True)
        ba, ia = eval_blocks(f"轮{r}")
        results.append({"round": r, "block_auc": round(ba, 4),
                        "img_auc": round(ia, 4)})
        # 留痕落盘（每轮）
        json.dump({"results": results, "state": learner.state()},
                  open(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                    "m0", "block_learn_curve.json"), "w"),
                  ensure_ascii=False, indent=2)

    print(f"\n=== 汇总 ===")
    for r in results:
        print(f"  轮{r['round']}: 块级={r['block_auc']} 图级={r['img_auc']}")
    print(f"  学习增益: 块级 {base_block:.4f} -> {results[-1]['block_auc']:.4f} "
          f"(Δ{results[-1]['block_auc'] - base_block:+.4f})")


if __name__ == "__main__":
    main()
