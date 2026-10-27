"""速度 profile（§6.1）：逐阶段计时 predict，定位 954ms 构成。

用法：
  python scripts/diag_speed.py --category gold_finger
"""
import os
import sys
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.common.io import load_image
from src.common.tiling import compute_tiles, extract_tiles
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--n", type=int, default=5, help="计时图片数（取均值）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    print(f"[fit] {time.time() - t0:.1f}s", flush=True)

    paths = [p for p, _ in bundle["test"]][:args.n]
    sizes = set()
    normal_ms, anomaly_ms = [], []
    for p in paths:
        img = load_image(p)
        sizes.add(img.shape[:2])
        t = time.time()
        r = pipe.predict(p)
        dt = (time.time() - t) * 1000
        if r["decision"] == "normal":
            normal_ms.append(dt)
        else:
            anomaly_ms.append(dt)
        print(f"  {r['decision']:7s} {dt:7.1f}ms  {os.path.basename(p)}", flush=True)
    print(f"[size] 测试图尺寸: {sorted(sizes)}  mode={pipe.tiling['mode']}")
    if normal_ms:
        print(f"[perf] 正常图 均摊 {np.mean(normal_ms):.1f}ms/图 (n={len(normal_ms)})")
    if anomaly_ms:
        print(f"[perf] 异常图 均摊 {np.mean(anomaly_ms):.1f}ms/图 (n={len(anomaly_ms)}, 含分层输出)")
    if normal_ms and anomaly_ms:
        print(f"[perf] 混合均摊 {np.mean(normal_ms + anomaly_ms):.1f}ms/图")


if __name__ == "__main__":
    main()

