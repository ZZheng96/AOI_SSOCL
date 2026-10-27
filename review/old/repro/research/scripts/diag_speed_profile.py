"""速度画像：fit 后对少量 test 图逐组件计时，定位 467-629ms/图 的耗时分布。

用法: python scripts/diag_speed_profile.py --dataset mvtec --category bottle
输出: 每张图平均 ms：load+tiles / DINO 前向 / 各槽位打分 / open / hier
"""
import os
import sys
import time
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like, gyudet, datalocal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mvtec", choices=["mvtec", "btad", "mpdd", "gyudet", "datalocal"])
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch_available() else "cpu"

    if args.dataset in ("mvtec", "btad", "mpdd"):
        from src.data import mvtec_like
        root = cfg["datasets"][args.dataset]
        loader = mvtec_like.load_btad if args.dataset == "btad" else mvtec_like.load_category
        bundle = loader(root, args.category,
                        n_init_normal=cfg["protocol"]["n_init_normal"],
                        n_init_defect=cfg["protocol"]["n_init_defect"],
                        seed=cfg["seed"], n_eval_good=25, max_test=args.n * 4)
    elif args.dataset == "gyudet":
        bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                    cfg["protocol"]["n_init_normal"],
                                    cfg["protocol"]["n_init_defect"], cfg["seed"],
                                    max_test=args.n * 4)
    else:
        bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                         cfg["protocol"]["n_init_normal"],
                                         cfg["protocol"]["n_init_defect"], cfg["seed"])

    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    print(f"[profile] fit {pipe.fit_seconds:.0f}s, 槽位: {list(pipe.slots)}", flush=True)

    items = bundle["test"][:args.n]
    agg = {"load": 0.0, "dino": 0.0, "tiles": 0.0}
    slot_ms = {n: 0.0 for n in pipe.slots}
    open_ms, hier_ms, cal_ms = 0.0, 0.0, 0.0
    predict_ms = []
    seg = {"get_item": 0.0, "record": 0.0, "finalize": 0.0}
    for i, (p, y) in enumerate(items):
        # ---- 端到端推理分段 ----
        t = time.time()
        it = pipe._get_item(p)
        seg["get_item"] += time.time() - t
        t = time.time()
        rec = pipe._record(it)
        seg["record"] += time.time() - t
        t = time.time()
        pipe._finalize(it, rec)
        seg["finalize"] += time.time() - t
        t = time.time()
        r = pipe.predict(p)
        predict_ms.append((time.time() - t) * 1000)
        # ---- 特征提取段 ----
        t = time.time()
        item = pipe._get_item(p)
        agg["load"] += time.time() - t
        # ---- 各槽位打分 ----
        for n, slot in pipe.slots.items():
            t = time.time()
            pipe._slot_image_score(slot, item)
            slot_ms[n] += time.time() - t
        # ---- 校准 ----
        t = time.time()
        for n, cal in pipe.calibrators.items():
            cal.transform([0.5])
        cal_ms += time.time() - t
        # ---- open ----
        if pipe.open_det is not None:
            t = time.time()
            try:
                pipe.open_det.score(item["feats"])
            except Exception:
                pass
            open_ms += time.time() - t
        if (i + 1) % 5 == 0:
            print(f"  [profile] {i + 1}/{len(items)}", flush=True)

    n = len(items)
    pred_avg = np.mean(predict_ms) if predict_ms else 0
    print("\n=== 每图平均耗时 (ms) ===")
    print(f"  predict 端到端   : {pred_avg:7.1f}   <-- 走 _record（U56 CPU/GPU 并行）")
    print(f"    _get_item(load+DINO): {seg['get_item'] / n * 1000:7.1f}")
    print(f"    _record(槽位并行)   : {seg['record'] / n * 1000:7.1f}")
    print(f"    _finalize(决策+hier): {seg['finalize'] / n * 1000:7.1f}")
    print(f"  load_image      : {agg['load'] / n * 1000:7.1f}")
    for n_ in slot_ms:
        print(f"  slot {n_:<8}    : {slot_ms[n_] / n * 1000:7.1f}")
    print(f"  calibrate       : {cal_ms / n * 1000:7.1f}")
    print(f"  open_det        : {open_ms / n * 1000:7.1f}")
    total = agg["load"] / n * 1000 + sum(slot_ms.values()) / n * 1000 + cal_ms / n * 1000 + open_ms / n * 1000
    print(f"  ~total          : {total:7.1f}")


def torch_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
