"""U94 归因诊断：缺件少件 + 色彩变化样本的各槽位校准分与归因结果（2026-08-24）

定位归因仍 0% 的根因：layout 是否 fire（缺件信号在不在 layout）、color 是否 fire
（色彩信号在不在 color 槽位）、top-1 为何落在"外观缺陷"。
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def probe(root, cat, defect_dir, device, n=5):
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    bundle = mvtec_like.load_category(root, cat, cfg["protocol"]["n_init_normal"],
                                      cfg["protocol"]["n_init_defect"], cfg["seed"],
                                      n_eval_good=0, max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    d = os.path.join(root, cat, "test", defect_dir)
    imgs = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg")))[:n]
    print(f"\n=== {cat}/{defect_dir}（各槽位校准分 + 归因 top-1）===", flush=True)
    for f in imgs:
        r = pipe.predict(os.path.join(d, f))
        ss = {k: round(v, 3) for k, v in sorted(r["slot_scores"].items(),
                                                key=lambda kv: -kv[1])}
        top = r["types"][0]["type"] if r["types"] else "无(漏检)"
        print(f"  {f}: 归因={top:8s} 各槽={ss}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = yaml.safe_load(open(CFG, encoding="utf-8"))["datasets"]["mvtec"]
    # 色彩变化样本（MVTec 有 color 缺陷类型的品类）
    for cat, d in [("carpet", "color"), ("leather", "color"), ("metal_nut", "color"),
                   ("pill", "color"), ("wood", "color")]:
        probe(root, cat, d, device)
    # 缺件样本（MVTec cable missing）
    probe(root, "cable", "missing_cable", device)
    probe(root, "cable", "missing_wire", device)


if __name__ == "__main__":
    main()
