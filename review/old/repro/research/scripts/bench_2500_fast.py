# -*- coding: utf-8 -*-
"""2500×2500 速度工作点原始测量（评审问题#3/#5/#8 补强，2026-08-31）。

针对评审意见"69ms@2500² 无包内原始记录 / bench 样本仅 12 次 / CPU '2500²级'
口径不实（实为原生小图）"，本脚本在真实 2500×2500 图上输出带时间戳、
硬件指纹、逐次明细的 JSON 原始记录：

  GPU fast 工作点（configs/m0_fast.yaml，sem/disc/shead 三槽）：
    - model_ms：predict_frame（帧已在内存，resize448+前向+融合，无磁盘 IO）
                —— 即文档"算法推理耗时/模型运行时间"口径
    - e2e_ms  ：predict(path)（含 imdecode 读盘）—— 完整链路口径
  CPU 降配（--cpu，configs/m4_cpu.yaml，blob/trad/layout 手工槽位）：
    - 同双口径，回答"2500² CPU 到底多少毫秒"

用法：
  python scripts/bench_2500_fast.py                      # GPU fast，5 图×6 轮
  python scripts/bench_2500_fast.py --cpu                # CPU 降配，5 图×3 轮
输出：outputs/m0/bench_2500_fast.json / bench_2500_cpu.json
"""
import os
import sys
import json
import time
import random
import argparse
import platform
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "outputs" / "m0" / "_bench2500"


def hw_fingerprint():
    info = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cpu_cores": os.cpu_count()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def make_2500_images(category, n, side=2500):
    """从 data_local 真实图随机取 n 张，重采样到 2500² 落临时目录（测完清理）。"""
    src_root = Path(r"D:\CGAIC\data_local") / category
    pool = [p for p in src_root.rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
            and "_mask" not in p.stem and "ground_truth" not in str(p)]
    rng = random.Random(42)
    rng.shuffle(pool)
    pool = pool[:n]
    if not pool:
        raise SystemExit(f"data_local/{category} 无可用图")
    TMP.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(pool):
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        dst = TMP / f"bench_{i}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.append(dst)
        print(f"  [mkimg] {dst.name} <= {p.name}", flush=True)
    return out


def _stats(ts):
    s = sorted(ts)
    return {"n": len(s), "mean": round(float(np.mean(s)), 1),
            "p50": round(float(np.median(s)), 1),
            "p95": round(float(s[max(0, int(len(s) * 0.95) - 1)]), 1),
            "min": round(s[0], 1), "max": round(s[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="CPU 降配（m4_cpu.yaml）")
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None, help="默认 GPU 6 轮 / CPU 3 轮")
    ap.add_argument("--side", type=int, default=2500)
    args = ap.parse_args()
    rounds = args.rounds or (3 if args.cpu else 6)

    cfg_name = "m4_cpu.yaml" if args.cpu else "m0_fast.yaml"
    cfg = yaml.safe_load(open(ROOT / "configs" / cfg_name, encoding="utf-8"))
    # 槽位键补齐（m4_cpu.yaml 等精简配置缺 shead/disc 等键，build 直接索引）
    for _k in ("sem", "disc", "shead", "blob", "trad", "layout", "inp", "open", "tpl"):
        cfg.setdefault("slots", {}).setdefault(_k, {"enabled": False})
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    from m0_baseline import build
    from src.eval.offline import Pipeline
    from src.data import datalocal

    print(f"[bench2500] config={cfg_name} device={device} "
          f"images={args.n_images} rounds={rounds}", flush=True)
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    t0 = time.time()
    pipe.fit(bundle)
    fit_s = time.time() - t0
    print(f"[bench2500] fit {fit_s:.0f}s, 槽位: {list(pipe.slots)}", flush=True)

    paths = make_2500_images(args.category, args.n_images, args.side)
    # 帧缓冲预读（不计时）：模拟相机帧已在内存
    frames = []
    for p in paths:
        data = np.fromfile(str(p), dtype=np.uint8)
        frames.append(cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB))

    # warmup（2 次，不计入）
    for i in range(2):
        pipe.predict_frame(frames[i % len(frames)])
    print("[bench2500] warmup done", flush=True)

    model_ts, e2e_ts = [], []
    detail = []
    k = 0
    for rnd in range(1, rounds + 1):
        for i in range(len(paths)):
            t = time.perf_counter()
            pipe.predict_frame(frames[i])
            m_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            pipe.predict(str(paths[i]))
            e_ms = (time.perf_counter() - t) * 1000
            model_ts.append(m_ms)
            e2e_ts.append(e_ms)
            k += 1
            detail.append({"round": rnd, "image": paths[i].name,
                           "model_ms": round(m_ms, 1), "e2e_ms": round(e_ms, 1)})
            print(f"  [{k}/{rounds * len(paths)}] model {m_ms:.0f}ms | "
                  f"e2e {e_ms:.0f}ms", flush=True)

    report = {
        "bench": "2500x2500 speed working point",
        "config": cfg_name,
        "category": args.category,
        "side": args.side,
        "slots": list(pipe.slots),
        "fit_seconds": round(fit_s, 1),
        "hardware": hw_fingerprint(),
        "model_ms": _stats(model_ts),   # 模型运行时间口径（内存帧，无磁盘 IO）
        "e2e_ms": _stats(e2e_ts),       # 完整链路口径（含 imdecode 读盘）
        "detail": detail,
    }
    out_name = "bench_2500_cpu.json" if args.cpu else "bench_2500_fast.json"
    out_path = ROOT / "outputs" / "m0" / out_name
    json.dump(report, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n=== {args.side}x{args.side} {cfg_name} ({device}) ===")
    print(f"  model_ms: {report['model_ms']}")
    print(f"  e2e_ms  : {report['e2e_ms']}")
    print(f"[save] {out_path}", flush=True)
    for p in paths:
        p.unlink(missing_ok=True)
    TMP.rmdir()


if __name__ == "__main__":
    main()
