"""微基准：load_image 开销 + shead quantile 优化对照"""
import time
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.io import load_image


def bench_load(path, n=10):
    ts = []
    for _ in range(n):
        t = time.time()
        img = load_image(path)
        ts.append((time.time() - t) * 1000)
    print(f"load_image: {np.mean(ts):.1f}ms/张 (min {np.min(ts):.1f}) shape={img.shape}")


def bench_quantile(shape=(1, 384, 32, 32), n=20):
    f = torch.randn(shape, device="cuda")
    flat = f.reshape(f.shape[0], -1)
    # 现版：torch.quantile 全量
    ts = []
    for _ in range(n):
        t = time.time()
        torch.quantile(flat, 0.25, dim=1)
        torch.quantile(flat, 0.75, dim=1)
        torch.cuda.synchronize()
        ts.append((time.time() - t) * 1000)
    print(f"quantile 全量 (393216 元素): {np.mean(ts):.1f}ms")

    # 采样近似 8192
    idx = torch.randint(0, flat.shape[1], (8192,), device="cuda")
    ts = []
    for _ in range(n):
        t = time.time()
        s = flat[:, idx]
        torch.quantile(s, 0.25, dim=1)
        torch.quantile(s, 0.75, dim=1)
        torch.cuda.synchronize()
        ts.append((time.time() - t) * 1000)
    print(f"quantile 采样 8192: {np.mean(ts):.1f}ms")

    # torch.sort 全量一次
    ts = []
    for _ in range(n):
        t = time.time()
        v, _ = torch.sort(flat, dim=1)
        torch.cuda.synchronize()
        ts.append((time.time() - t) * 1000)
    print(f"torch.sort 全量: {np.mean(ts):.1f}ms")

    # numpy 全量（CPU）
    f_cpu = flat.cpu().numpy()
    ts = []
    for _ in range(10):
        t = time.time()
        np.percentile(f_cpu, [25, 75], axis=1)
        ts.append((time.time() - t) * 1000)
    print(f"np.percentile 全量 CPU: {np.mean(ts):.1f}ms")

    # 仅 mean/std/min/max（去掉 quantile）
    ts = []
    for _ in range(n):
        t = time.time()
        flat.mean(dim=1); flat.std(dim=1)
        flat.min(dim=1).values; flat.max(dim=1).values
        torch.cuda.synchronize()
        ts.append((time.time() - t) * 1000)
    print(f"仅 mean/std/min/max: {np.mean(ts):.1f}ms")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"D:\CGAIC\data_origin\mvtec\bottle\test\good\000.png"
    bench_load(p)
    if torch.cuda.is_available():
        bench_quantile()
