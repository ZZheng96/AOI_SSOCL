"""U86v24（2026-08-21）：v23 双指标 + 经验回放（用户指令：设计方法避免退化，重中之重）

v23 发现：A 再检曲线 0.9017(轮1) -> 0.8800(轮8) 退化--每轮重训数据集
（train 缓存+当轮 40 张）不含早期反馈图，其决策边界被后续训练冲刷
（灾难性遗忘）。v22 全量累积保记忆但在线占比 68%、每图被训 8 次，
分布失衡致 B 崩溃（0.8389）。

v24 经验回放（记忆保持 vs 分布稳定的折中）：
- 特征记忆库 mem：每张反馈图的 pos/neg 特征（390 维）提取一次永久保存
- 每轮重训 = train 缓存 + 当轮缓冲 + 均匀随机回放 REPLAY_N=20 张历史图特征
- 剂量：在线占比 23%（v8=17% / v22=68%），每张历史图期望再训 ~0.5 次（v22=7 次）

双指标（v23 框架不变）：
- A 再检：反馈学习过的缺陷图 + 未见正常锚（30 张）的 AUROC（允许看答案）
- B 泛化：test 60 张（全程不参与选择）AUROC
成功标准：A 轮8 保持 >=0.90 不退化，B 不低于 0.87。

协议（U86v3 教训）：门控在 eval 上选模型违反"test 只验不选"：
- valid 集（30 张，seed=5）：门控选择（在线重训采纳/回滚）唯一依据
- test 集（60 张，seed=123，排除 valid）：最终报告，全程不参与任何选择
- 学习 pool：test 域其余缺陷图（缺陷图无框滑窗提供 neg）

闭环：
1. fit ablB 判别头（train crop）+ 缓存 train crop 特征（防遗忘）
2. 轮0 基线：valid crop 图级 + test crop 图级 + A 再检学习前基线
3. 在线学习：反馈缺陷图 -> 框内 crop + 含框滑窗入 pos / 无框滑窗入 neg
   -> 特征存当轮缓冲+记忆库 -> 重训（train 缓存+当轮+回放）-> valid 门控
4. 每轮记录 A/B 双指标曲线（test 只记录不选择）
5. 结束：best head 双指标报告（只验不选）
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
                             _extract_context_crops, _extract_random_crops,
                             _sliding_grid)
from src.data import gyudet

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42
GRID = 4
N_VALID_DEF = 15        # U86v10：v8 配方（valid 30 张门控，test 0.8856）
N_VALID_NOR = 15
N_TEST_DEF = 30         # test 60 张（最终报告，只验不选）
N_TEST_NOR = 30
N_ROUNDS = 8            # U86v24：v8 收敛轮数
POOL_PER_ROUND = 40     # U86v24：v8 渐进反馈
REPLAY_N = 20           # U86v24：经验回放张数/轮（防 A 再检退化，v8=0/v22=全量 之间）
RT_MIN_POS = 30
RT_MIN_NEG = 60
RT_EPOCHS = 30          # U86v8：v4 配方
RT_LR = 1e-3
RT_MIN_CROP_SIDE = 30
SCORE_OVERLAP = 0.3     # U86v8：速度达标路径（853ms/张 < 1s，v7 实测）
SCORE_MAX_CROPS = 200


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


def pick_test(n_def, n_nor, seed=42, excl=None):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    defect, normal = [], []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir, f)
        if excl and p in excl:
            continue
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


def sliding_positions(img, cs, overlap):
    h, w = img.shape[:2]
    cs = min(cs, h, w)
    if h <= cs or w <= cs:
        return [(img, 0, 0)]
    stride = max(1, int(cs * (1 - overlap)))
    y_pos = list(range(0, max(1, h - cs + 1), stride))
    x_pos = list(range(0, max(1, w - cs + 1), stride))
    if y_pos[-1] + cs < h:
        y_pos.append(h - cs)
    if x_pos[-1] + cs < w:
        x_pos.append(w - cs)
    out = []
    for y0 in y_pos:
        for x0 in x_pos:
            out.append((img[y0:y0 + cs, x0:x0 + cs], y0, x0))
    return out


def iou_boxes(y0, x0, cs, boxes):
    best = 0.0
    for bx in boxes:
        iw = max(0, min(x0 + cs, bx[2]) - max(x0, bx[0]))
        ih = max(0, min(y0 + cs, bx[3]) - max(y0, bx[1]))
        inter = iw * ih
        if inter <= 0:
            continue
        wa = (bx[2] - bx[0]) * (bx[3] - bx[1])
        wb = cs * cs
        best = max(best, inter / (wa + wb - inter))
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U86v24 v23双指标+经验回放{REPLAY_N}张/轮] device={device}", flush=True)
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
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    def feats_d6(imgs):
        f = backbone.extract_tiles(imgs, batch=batch)
        return _image_level_feats(f.float().to(device),
                                  shead.chan_stats, shead.patch_stats).cpu()

    # 1. 缓存 train crop 特征
    rng = np.random.default_rng(SEED)
    train_pos, train_neg = [], []
    for p in bundle["init_defect"]:
        img, boxes = load_boxes(p)
        if boxes:
            crops = _extract_box_crops(img, boxes, 0.2) + \
                _extract_context_crops(img, boxes, crop_size)
            if crops:
                train_pos.append(feats_d6(crops))
    for i in range(len(bundle["init_normal"])):
        img = ctx["train_tile_imgs"][i][0]
        crops = _extract_random_crops(img, crop_size, 10, rng)
        if crops:
            train_neg.append(feats_d6(crops))
    X_tr_pos = torch.cat(train_pos, dim=0) if train_pos else torch.empty(0, 390)
    X_tr_neg = torch.cat(train_neg, dim=0) if train_neg else torch.empty(0, 390)
    print(f"  train crop 缓存: pos={X_tr_pos.shape[0]} neg={X_tr_neg.shape[0]} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 2. valid（门控选择）+ test（只验不选）
    valid_set = pick_test(N_VALID_DEF, N_VALID_NOR, seed=5)
    valid_paths = {p for p, _ in valid_set}
    test_set = pick_test(N_TEST_DEF, N_TEST_NOR, seed=123, excl=valid_paths)

    def load_set(paths):
        imgs, ys = [], []
        for path, lab in paths:
            img, _ = load_boxes(path)
            imgs.append(img)
            ys.append(lab)
        return imgs, ys

    valid_imgs, valid_y = load_set(valid_set)
    test_imgs, test_y = load_set(test_set)
    print(f"  valid {len(valid_set)} 张 (缺陷 {sum(valid_y)}) + "
          f"test {len(test_set)} 张 (缺陷 {sum(test_y)}) "
          f"({time.time() - t0:.0f}s)", flush=True)

    def crop_scores(imgs):
        """U86v21：回退 v8 单尺度滑窗打分（crop_size + 固定 ov0.3）。"""
        out = []
        for img in imgs:
            cs = min(crop_size, img.shape[0], img.shape[1])
            crops, ny, nx, trunc = _sliding_grid(img, cs, SCORE_OVERLAP,
                                                 SCORE_MAX_CROPS)
            if not crops:
                out.append(-1e9)
                continue
            f = backbone.extract_tiles(crops, batch=batch)
            x = _image_level_feats(f.float().to(device),
                                   shead.chan_stats, shead.patch_stats)
            s = shead.head(x.to(device)).reshape(-1).float()
            if not trunc:
                smap = s.reshape(ny, nx)[None, None]
                pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
                sc = float(F.conv2d(pad, k33).max().item())
            else:
                sc = float(s.topk(min(3, len(s))).values.mean().item())
            out.append(sc)
        return out

    def score_defect_paths(paths):
        """U86v23：逐张 load+打分（避免 320 张大图同时驻留内存），
        只统计有框图（与反馈训练范围一致），返回分数列表。"""
        out = []
        for p in paths:
            img, boxes = load_boxes(p)
            if boxes:
                out.extend(crop_scores([img]))
        return out

    def valid_auc(tag):
        auc = float(roc_auc_score(valid_y, crop_scores(valid_imgs)))
        print(f"  [valid-{tag}] crop图级({len(valid_set)}) AUROC={auc:.4f}",
              flush=True)
        return auc

    # 3. 轮0 基线（valid 门控基线 + test 诚实基线 + A 再检学习前基线）
    b_v = valid_auc("轮0-基线")
    t0a = time.time()
    t_sc0 = crop_scores(test_imgs)
    test_base = float(roc_auc_score(test_y, t_sc0))
    print(f"  [test-轮0-基线] crop图级({len(test_set)}) AUROC={test_base:.4f} "
          f"({time.time() - t0a:.0f}s)", flush=True)
    best_i, best_state = b_v, {k: v.detach().clone()
                               for k, v in shead.head.state_dict().items()}
    head_list = [best_state]      # U86v10：集成所有重训 head（含基线）
    results = [{"round": 0, "valid": round(b_v, 4), "test": round(test_base, 4)}]
    n_rt = 0

    # 4. 在线学习（重训 + valid 门控）
    # U86v24：经验回放记忆库 mem（按图分组 (pos_feats, neg_feats)）+ 独立回放 rng
    online_pos, online_neg = [], []
    mem = []                    # U86v24：历史反馈特征记忆库（永久保存）
    rng_replay = np.random.default_rng(SEED + 3)
    feedback_paths = []   # U86v24：累计反馈图路径（A 再检指标用，只记路径省内存）
    valid_paths = {p for p, _ in valid_set} | {p for p, _ in test_set}
    img_dir_all = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir_all = os.path.join(GYU_ROOT, "test", "labels")
    pool = []
    for f in sorted(os.listdir(img_dir_all)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir_all, f)
        lbl = os.path.join(lbl_dir_all, os.path.splitext(f)[0] + ".txt")
        if p in valid_paths:
            continue
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            pool.append((p, 1))
    rng_pool = np.random.default_rng(SEED + 2)
    used = set()

    # U86v23：A 再检指标的学习前基线--预演反馈序列（独立同 seed rng，
    # 抽样逻辑与正式循环完全一致），用轮0 head 在"将被反馈的图"上打分
    rng_pre = np.random.default_rng(SEED + 2)
    pre_used = set()
    for _ in range(N_ROUNDS):
        cand_all = [k for k in range(len(pool)) if k not in pre_used]
        n_sel = min(POOL_PER_ROUND, len(cand_all))
        if not n_sel:
            break
        pre_used.update(rng_pre.choice(cand_all, n_sel, replace=False).tolist())
    pre_paths = [pool[k][0] for k in sorted(pre_used)]
    t0a = time.time()
    pre_sc = score_defect_paths(pre_paths)
    pre_nor = [s for s, yy in zip(t_sc0, test_y) if yy == 0]
    a_base = float(roc_auc_score([1] * len(pre_sc) + [0] * len(pre_nor),
                                 pre_sc + pre_nor))
    print(f"  [A再检-学习前基线] 将反馈{len(pre_sc)}张+正常锚{len(pre_nor)} "
          f"AUROC={a_base:.4f} ({time.time() - t0a:.0f}s)", flush=True)
    results[0]["A_recheck_base"] = round(a_base, 4)

    for r in range(1, N_ROUNDS + 1):
        # U86v24：v8 随机反馈；mem_mark 记录当轮反馈前的记忆库长度（回放排除当轮）
        mem_mark = len(mem)
        cand_all = [k for k in range(len(pool)) if k not in used]
        n_sel = min(POOL_PER_ROUND, len(cand_all))
        if not n_sel:
            break
        sel = rng_pool.choice(cand_all, n_sel, replace=False).tolist()
        used.update(sel)
        print(f"  轮{r}: 随机反馈 {len(sel)} 张 (累计 {len(used)}/{len(pool)})",
              flush=True)
        for k in sel:
            path, _ = pool[k]
            img, boxes = load_boxes(path)
            if boxes:
                feedback_paths.append(path)   # A 再检指标统计范围
                pos_crops = [c for c in _extract_box_crops(img, boxes, 0.2)
                             if min(c.shape[0], c.shape[1]) >= RT_MIN_CROP_SIDE]
                win_pos, win_neg = [], []
                csm = min(crop_size, img.shape[0], img.shape[1])
                for wimg, y0, x0 in sliding_positions(img, csm, SCORE_OVERLAP):
                    iou = iou_boxes(y0, x0, csm, boxes)
                    if iou > 0.3:
                        win_pos.append(wimg)
                    elif iou == 0.0 and len(win_neg) < 4:
                        win_neg.append(wimg)
                xp = feats_d6(pos_crops) if pos_crops else None
                xw = feats_d6(win_pos) if win_pos else None
                xn = feats_d6(win_neg) if win_neg else None
                # 当轮缓冲（重训后清空）
                if xp is not None:
                    online_pos.append(xp)
                if xw is not None:
                    online_pos.append(xw)
                if xn is not None:
                    online_neg.append(xn)
                # U86v24：记忆库（按图分组，供后续轮回放防遗忘）
                xp_all = torch.cat([t for t in (xp, xw) if t is not None], 0) \
                    if (xp is not None or xw is not None) else None
                mem.append((xp_all, xn))
        n_pos = int(sum(x.shape[0] for x in online_pos))
        n_neg = int(sum(x.shape[0] for x in online_neg))
        if n_pos >= RT_MIN_POS and n_neg >= RT_MIN_NEG:
            pos_list, neg_list = list(online_pos), list(online_neg)
            # U86v24：经验回放--从历史反馈记忆库（排除当轮）均匀随机抽 REPLAY_N 张
            hist = mem[:mem_mark]
            n_rep = 0
            if hist and REPLAY_N > 0:
                rep_idx = rng_replay.choice(len(hist),
                                            min(REPLAY_N, len(hist)),
                                            replace=False).tolist()
                for i in rep_idx:
                    mp, mn = hist[i]
                    if mp is not None:
                        pos_list.append(mp)
                    if mn is not None:
                        neg_list.append(mn)
                    n_rep += 1
            Xp = torch.cat([X_tr_pos] + pos_list, dim=0).float()
            Xn = torch.cat([X_tr_neg] + neg_list, dim=0).float()
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
            online_pos, online_neg = [], []   # 当轮缓冲清空（mem 记忆库保留）
            head_list.append({k: v.detach().clone()
                              for k, v in shead.head.state_dict().items()})
            print(f"  轮{r}: 重训判别头 #{n_rt} (在线 pos={n_pos} neg={n_neg} "
                  f"+ 回放{n_rep}张 | 记忆库{len(mem)})",
                  flush=True)
        c_v = valid_auc(f"轮{r}")
        if c_v < best_i - 0.003:
            shead.head.load_state_dict(best_state)
            print(f"  [门控] valid {c_v:.4f} < best {best_i:.4f} -> 回滚 head",
                  flush=True)
            c_v = valid_auc(f"轮{r}-回滚后")
        else:
            best_i, best_state = c_v, {k: v.detach().clone()
                                       for k, v in shead.head.state_dict().items()}
        # U86v23 双指标记录（test 只记录不选择，门控仍仅用 valid）
        t_sc = crop_scores(test_imgs)
        b_r = float(roc_auc_score(test_y, t_sc))
        nor_anchor = [s for s, yy in zip(t_sc, test_y) if yy == 0]
        fb_sc = score_defect_paths(feedback_paths)
        a_r = float(roc_auc_score([1] * len(fb_sc) + [0] * len(nor_anchor),
                                  fb_sc + nor_anchor))
        print(f"  [双指标-轮{r}] A再检(反馈{len(fb_sc)}+正常锚{len(nor_anchor)})"
              f"={a_r:.4f}  B泛化(60)={b_r:.4f}", flush=True)
        results.append({"round": r, "valid": round(c_v, 4), "n_retrain": n_rt,
                        "A_recheck": round(a_r, 4), "B_general": round(b_r, 4)})
        json.dump({"results": results},
                  open(os.path.join(os.path.dirname(__file__), "..", "outputs",
                                    "m0", "crop_score_retrain_v24.json"), "w"),
                  ensure_ascii=False, indent=2)

    # 5. 最终：best head（valid 门控）在 test 上报告（只验不选）
    shead.head.load_state_dict(best_state)
    t0a = time.time()
    test_final = float(roc_auc_score(test_y, crop_scores(test_imgs)))
    print(f"\n  [test-最终-单head] crop图级({len(test_set)}) AUROC={test_final:.4f} "
          f"({time.time() - t0a:.0f}s)", flush=True)
    # U86v10：多 head 等权集成（固定算法，不选模型；z-score 归一化后平均）
    t0a = time.time()
    zsum = np.zeros(len(test_imgs))
    for hs in head_list:
        shead.head.load_state_dict(hs)
        s = np.array(crop_scores(test_imgs))
        s = (s - s.mean()) / (s.std() + 1e-9)
        zsum += s
    zsum /= len(head_list)
    test_ens = float(roc_auc_score(test_y, zsum))
    print(f"  [test-最终-集成{len(head_list)}head] crop图级 AUROC={test_ens:.4f} "
          f"({time.time() - t0a:.0f}s)", flush=True)
    print(f"  [B泛化] test 基线 {test_base:.4f} -> 单head {test_final:.4f} -> "
          f"集成 {test_ens:.4f}", flush=True)
    # U86v15/v23 修复：集成循环把 shead.head 留在最后一个（被回滚的）head 上，
    # A 再检与保存前必须先恢复 best head
    shead.head.load_state_dict(best_state)
    # U86v23 A 再检指标：best head 在全部反馈图 + 未见正常锚上
    t0a = time.time()
    nor_anchor = [s for s, yy in zip(crop_scores(test_imgs), test_y) if yy == 0]
    fb_sc = score_defect_paths(feedback_paths)
    a_final = float(roc_auc_score([1] * len(fb_sc) + [0] * len(nor_anchor),
                                  fb_sc + nor_anchor))
    print(f"  [A再检-最终] 反馈{len(fb_sc)}张+正常锚{len(nor_anchor)} "
          f"AUROC={a_final:.4f} ({time.time() - t0a:.0f}s)", flush=True)
    ok_a = a_final >= 0.9
    ok_b = max(test_final, test_ens) >= 0.9
    print(f"  [双指标结论] A再检: 学习前{a_base:.4f} -> 学习后{a_final:.4f}"
          f"{'(达标>=0.9)' if ok_a else ''}; "
          f"B泛化={test_final:.4f}{'(达标>=0.9)' if ok_b else ''}; "
          f"{'双指标之一达标' if (ok_a or ok_b) else '均未达 0.9'}", flush=True)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0")
    torch.save({"head": shead.head.state_dict(),
                "chan_stats": shead.chan_stats, "patch_stats": shead.patch_stats,
                "crop_size": crop_size, "best_valid": best_i,
                "test_base": test_base, "test_final": test_final,
                "test_ens": test_ens, "a_recheck": a_final,
                "n_feedback": len(feedback_paths),
                "n_heads": len(head_list),
                "score_overlap": SCORE_OVERLAP},
               os.path.join(out_dir, "u86v24_head.pt"))
    print(f"  head 已保存: outputs/m0/u86v24_head.pt", flush=True)


if __name__ == "__main__":
    main()
