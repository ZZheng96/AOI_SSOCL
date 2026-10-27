"""鲁棒性测试（G4，评审自检"鲁棒性空白"补齐，2026-08-24）：光照波动 + 对位偏移

对 fit 后的 pipeline，把 test 正常图/缺陷图施加合成扰动，统计：
  1. 正常图扰动后误报率（decision != normal 比例，应低）
  2. 缺陷图扰动后漏检率（decision == normal 比例，应低）
扰动类型：gamma 亮度（0.7/1.3）、对比度缩放、平移（+对位偏移）。
诚实边界：这是合成扰动（非真实相机/震动/镜头脏污），作"环境波动下检测稳定性的
下限证据"，真实产线鲁棒性（24h 长稳/并发）仍需现场数据。

用法: python scripts/exp_robustness.py [--category bottle] [--dataset mvtec]
"""
import os
import sys
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like, datalocal

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0", "robustness.json")


def perturb(img, kind):
    """返回扰动后的 RGB uint8 图。"""
    img = img.astype(np.float32)
    if kind == "gamma07":
        img = (img / 255.0) ** 0.7 * 255.0
    elif kind == "gamma13":
        img = (img / 255.0) ** 1.3 * 255.0
    elif kind == "bright+25":
        img = img + 25.0
    elif kind == "bright-25":
        img = img - 25.0
    elif kind == "contrast06":
        img = (img - 127.5) * 0.6 + 127.5
    elif kind == "shift10":
        M = np.float32([[1, 0, 10], [0, 1, 10]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mvtec", choices=["mvtec", "datalocal"])
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--config", default=CFG)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "mvtec":
        bundle = mvtec_like.load_category(cfg["datasets"]["mvtec"], args.category,
                                          cfg["protocol"]["n_init_normal"],
                                          cfg["protocol"]["n_init_defect"],
                                          cfg["seed"], n_eval_good=0, max_test=0)
    else:
        bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                         cfg["protocol"]["n_init_normal"],
                                         cfg["protocol"]["n_init_defect"],
                                         cfg["seed"], max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)

    goods = [p for p, y in bundle["test"] if y == 0][:args.n]
    defects = [p for p, y in bundle["test"] if y == 1][:args.n]
    kinds = ["gamma07", "gamma13", "bright+25", "bright-25", "contrast06", "shift10"]

    result = {"category": args.category, "dataset": args.dataset,
              "n_good": len(goods), "n_defect": len(defects), "perturbations": {}}
    print(f"[鲁棒性] {args.dataset}/{args.category}: 正常 {len(goods)} / 缺陷 {len(defects)}",
          flush=True)

    for kind in kinds:
        fp = fn = 0
        for p in goods:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = pipe.predict_frame(perturb(img, kind))
            if r["decision"] != "normal":
                fp += 1
        for p in defects:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            r = pipe.predict_frame(perturb(img, kind))
            if r["decision"] == "normal":
                fn += 1
        fp_r = fp / max(len(goods), 1)
        fn_r = fn / max(len(defects), 1)
        result["perturbations"][kind] = {"fp": fp, "fn": fn,
                                         "fp_rate": round(fp_r, 3),
                                         "fn_rate": round(fn_r, 3)}
        print(f"  {kind:12s} 误报率 {fp_r:.3f} ({fp}/{len(goods)}) | "
              f"漏检率 {fn_r:.3f} ({fn}/{len(defects)})", flush=True)

    # 基线（无扰动）
    fp = sum(1 for p in goods if pipe.predict(p)["decision"] != "normal")
    fn = sum(1 for p in defects if pipe.predict(p)["decision"] == "normal")
    result["baseline"] = {"fp_rate": round(fp / max(len(goods), 1), 3),
                          "fn_rate": round(fn / max(len(defects), 1), 3)}
    print(f"  基线(无扰动) 误报率 {result['baseline']['fp_rate']:.3f} | "
          f"漏检率 {result['baseline']['fn_rate']:.3f}", flush=True)

    import json
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[鲁棒性] 结果落盘 {OUT}")


if __name__ == "__main__":
    main()
