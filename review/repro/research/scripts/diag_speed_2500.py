"""标准 2500×2500 速度基准：demo5 全槽位 predict 端到端耗时（赛题口径 2500²@2060 <200ms）

用法: python scripts/diag_speed_2500.py
输出: 2500² 图 predict 端到端 ms（多次平均）+ 分段
"""
import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import datalocal

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
SRC = r"D:\CGAIC\data_local\gold_finger\ground_truth\defect\0598_mask.png"  # 任意真实图做底
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0", "bench_2500.png")


def make_2500(src=SRC, out=OUT, side=2500):
    data = np.fromfile(src, dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=None, help="槽位子集，逗号分隔（如 sem,disc,shead）")
    ap.add_argument("--config", default=CFG, help="配置文件路径（默认 m0.yaml；fast 用 m0_fast.yaml）")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bench_path = make_2500()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.slots:
        keep = set(x.strip() for x in args.slots.split(",") if x.strip())
        for k in cfg["slots"]:
            cfg["slots"][k]["enabled"] = k in keep
        print(f"[bench] 槽位子集: {sorted(keep)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger",
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    print(f"[bench] fit {time.time()-t0:.0f}s, 槽位: {list(pipe.slots)}", flush=True)

    # 真实流式端到端：两张不同 2500² 图交替 predict（每张都走 load+DINO，无缓存命中）
    import glob
    p2 = OUT.replace(".png", "_b.png")
    make_2500(SRC, p2, 2500)
    paths = [bench_path, p2]
    n = 10
    ts = []
    for i in range(n):
        p = paths[i % 2]
        t = time.time()
        r = pipe.predict(p)
        ts.append((time.time() - t) * 1000)
        print(f"  [stream] {i+1}/{n} {ts[-1]:.0f}ms fused={r['fused']:.4f}", flush=True)
    print(f"\n=== 2500×2500 真实流式端到端（{n} 次，两张图交替，含 load+DINO）===")
    print(f"  平均 {np.mean(ts):.0f}ms | min {np.min(ts):.0f} | max {np.max(ts):.0f}")
    print(f"  赛题目标 <200ms → {'✅ 达标' if np.mean(ts) < 200 else '❌ 未达标'}")
    os.remove(p2)


if __name__ == "__main__":
    main()
