"""U104 集成后端到端验证（2026-08-25）：开启 cfg decision.tta（enabled+lighting+brightness）
验证集成进 offline.py 的 _tta_refine/_tta_min_fused 路径实际运行且误报抑制与独立脚本一致。

用法: python scripts/diag_tta_integrated.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def bright(img, d):
    return np.clip(img.astype(np.float32) + d, 0, 255).astype(np.uint8)


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    # 开启全 TTA（平移+光照+亮度），验证集成路径
    cfg["decision"]["tta"] = {
        "enabled": True, "shift": 10,
        "lighting": True, "gammas": [0.8, 1.25],
        "brightness": True, "brights": [-25, 25],
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U104] device={device}，加载 bottle（100+30）...", flush=True)
    b = mvtec_like.load_category(cfg["datasets"]["mvtec"], "bottle", 100, 30, 42,
                                 n_eval_good=0, max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)
    print(f"[U104] fit 完成，tta_enabled={pipe.tta_enabled} "
          f"lighting={pipe.tta_lighting} brightness={pipe.tta_brightness}", flush=True)

    rng = np.random.default_rng(42)
    goods = sorted(rng.choice([p for p, y in b["test"] if y == 0], 15, replace=False).tolist())
    defs = sorted(rng.choice([p for p, y in b["test"] if y == 1], 15, replace=False).tolist())

    # TTA 关闭对照：重建一个关 TTA 的 pipe（同 backbone/slots，仅 cfg 差异）
    cfg_off = yaml.safe_load(open(CFG, encoding="utf-8"))
    pipe_off = Pipeline(cfg_off, backbone, slots)
    pipe_off.fit(b)

    for name, pert in [("bright-25", lambda i: bright(i, -25)),
                       ("bright+25", lambda i: bright(i, 25))]:
        fp_on = fp_off = fn = 0
        for pp in goods:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r_off = pipe_off.predict_frame(imgp)
            r_on = pipe.predict_frame(imgp)
            if r_off["decision"] != "normal":
                fp_off += 1
            if r_on["decision"] != "normal":
                fp_on += 1
        for pp in defs:
            img = cv2.cvtColor(cv2.imread(pp), cv2.COLOR_BGR2RGB)
            imgp = pert(img)
            r_on = pipe.predict_frame(imgp)
            if r_on["decision"] == "normal":
                fn += 1
        print(f"[U104] {name}: 误报 TTA关 {fp_off}/15 -> TTA开 {fp_on}/15 | 漏检 {fn}/15",
              flush=True)


if __name__ == "__main__":
    main()
