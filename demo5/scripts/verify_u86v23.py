"""U86v23 A 再检指标独立复测：u86v23_head.pt（best head）在
反馈 320 张（复现 v8 反馈序列）+ test 30 张正常锚上的 AUROC。

背景：主脚本首次运行时 A_final=0.8089 是 bug（集成循环把 head 留在
被回滚的轮 8 head 上），轮 8 记录的 0.8800 才是 best head 真值。
本脚本加载保存的 best head 独立复算，验证 0.88 并作为正式 A_final。
"""
import os
import sys
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
from src.slots.shead import (SheadSlot, _image_level_feats, _sliding_grid)
from crop_score_retrain import pick_test, load_boxes, GYU_ROOT, SEED, \
    N_ROUNDS, POOL_PER_ROUND, N_VALID_DEF, N_VALID_NOR, N_TEST_DEF, N_TEST_NOR

HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                    "u86v23_head.pt")
SCORE_OVERLAP = 0.3
SCORE_MAX_CROPS = 200


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "m0_gyudet_cropv4_ablB.yaml"),
                              encoding="utf-8"))
    bcfg = cfg["backbone"]
    ck = torch.load(HEAD, map_location=device)
    print(f"[U86v23 A再检复测] crop_size={ck['crop_size']}", flush=True)
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    shead = SheadSlot({"seed": 42}, backbone, device)
    shead.head.load_state_dict(ck["head"])
    shead.chan_stats, shead.patch_stats = ck["chan_stats"], ck["patch_stats"]
    crop_size = int(ck["crop_size"])
    batch = 8
    k33 = torch.ones(1, 1, 3, 3, device=device) / 9.0

    def crop_scores(imgs):
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
                out.append(float(F.conv2d(pad, k33).max().item()))
            else:
                out.append(float(s.topk(min(3, len(s))).values.mean().item()))
        return out

    # 1. test 60 张（正常 30 作锚）
    valid_paths = {p for p, _ in pick_test(N_VALID_DEF, N_VALID_NOR, seed=5)}
    test = pick_test(N_TEST_DEF, N_TEST_NOR, seed=123, excl=valid_paths)
    test_y = [l for _, l in test]
    t0 = time.time()
    t_sc = crop_scores([load_boxes(p)[0] for p, _ in test])
    b_final = float(roc_auc_score(test_y, t_sc))
    nor_anchor = [s for s, yy in zip(t_sc, test_y) if yy == 0]
    print(f"  [B泛化复测] test60 AUROC={b_final:.4f} ({time.time() - t0:.0f}s)",
          flush=True)

    # 2. 复现反馈序列（与主脚本预演逻辑一致：rng seed 42+2，8 轮 x 40）
    all_paths = valid_paths | {p for p, _ in test}
    img_dir = os.path.join(GYU_ROOT, "test", "images")
    lbl_dir = os.path.join(GYU_ROOT, "test", "labels")
    pool = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        p = os.path.join(img_dir, f)
        if p in all_paths:
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            pool.append(p)
    rng = np.random.default_rng(SEED + 2)
    used = set()
    for _ in range(N_ROUNDS):
        cand = [k for k in range(len(pool)) if k not in used]
        n_sel = min(POOL_PER_ROUND, len(cand))
        if not n_sel:
            break
        used.update(rng.choice(cand, n_sel, replace=False).tolist())
    fb_paths = [pool[k] for k in sorted(used)]
    print(f"  反馈序列复现: {len(fb_paths)} 张 (pool {len(pool)})", flush=True)

    # 3. best head 逐张打分反馈图（只计有框图）
    fb_sc = []
    t0 = time.time()
    for p in fb_paths:
        img, boxes = load_boxes(p)
        if boxes:
            fb_sc.extend(crop_scores([img]))
    a_final = float(roc_auc_score([1] * len(fb_sc) + [0] * len(nor_anchor),
                                  fb_sc + nor_anchor))
    print(f"  [A再检复测] 反馈{len(fb_sc)}+锚{len(nor_anchor)} "
          f"AUROC={a_final:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  [结论] A再检={a_final:.4f}{'(达标>=0.9)' if a_final >= 0.9 else ''} / "
          f"B泛化={b_final:.4f}{'(达标>=0.9)' if b_final >= 0.9 else ''}; "
          f"{'双指标之一达标' if (a_final >= 0.9 or b_final >= 0.9) else '均未达 0.9'}",
          flush=True)


if __name__ == "__main__":
    main()
