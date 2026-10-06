"""调试 U27b：定位 m0_baseline 与 _dbg_trad 流程差异（trad 0.4286 vs 0.6429）
逐步二分：load_all vs load_category / 预热 _record 循环 / evaluate 顺序
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
import torch
from sklearn.metrics import roc_auc_score

from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline

cfg = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "configs", "m0.yaml")))
cfg["tiling"]["mode"] = "tiles36"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- A: load_all（m0 同款） + 直接 evaluate（无预热循环） ---
bundles = datalocal.load_all(cfg["datasets"]["data_local"], cfg)
bundle = bundles["component"]
backbone, slots = build(cfg, device)
pipe = Pipeline(cfg, backbone, slots)
pipe.fit(bundle)
rep = pipe.evaluate(bundle["test"])
print(f"[A load_all+直接evaluate] trad={rep['trad']['auroc']:.4f} "
      f"fused={rep['fused']['auroc']:.4f}", flush=True)
raws = [pipe._record(pipe._get_item(p))["raw_scores"]["trad"] for p, _ in bundle["test"]]
print(f"  raw trad: {[round(v, 4) for v in raws]}", flush=True)

# --- B: load_category + 直接 evaluate（隔离加载路径） ---
bundle2 = datalocal.load_category(cfg["datasets"]["data_local"], "component",
                                  cfg["protocol"]["n_init_normal"],
                                  cfg["protocol"]["n_init_defect"], cfg["seed"])
print(f"  bundle 相同: {bundle['test'] == bundle2['test']}", flush=True)
backbone, slots = build(cfg, device)
pipe2 = Pipeline(cfg, backbone, slots)
pipe2.fit(bundle2)
rep2 = pipe2.evaluate(bundle2["test"])
print(f"[B load_category+直接evaluate] trad={rep2['trad']['auroc']:.4f}", flush=True)
raws2 = [pipe2._record(pipe2._get_item(p))["raw_scores"]["trad"] for p, _ in bundle2["test"]]
print(f"  raw trad: {[round(v, 4) for v in raws2]}", flush=True)

# --- C: 正常分数（校准边缘）对账 ---
trad_slot = pipe.slots["trad"]
n_raw = []
from src.common.io import load_image
for p in bundle["init_normal"]:
    r = trad_slot.score_image(load_image(p))
    n_raw.append(round(float(r[0]), 4))
print(f"[train normal raw trad] {n_raw}", flush=True)
trad_slot2 = pipe2.slots["trad"]
n_raw2 = [round(float(trad_slot2.score_image(load_image(p))[0]), 4)
          for p in bundle2["init_normal"]]
print(f"[train normal raw trad B] {n_raw2}", flush=True)
