"""调试 U27：管线内 trad 整图打分与 diag_trad_score 对账（component 0.4286 vs 0.8571）"""
import os
import sys
import json

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
bundle = datalocal.load_category(cfg["datasets"]["data_local"], "component",
                                 cfg["protocol"]["n_init_normal"],
                                 cfg["protocol"]["n_init_defect"], cfg["seed"])

backbone, slots = build(cfg, device)
pipe = Pipeline(cfg, backbone, slots)
pipe.fit(bundle)
trad = pipe.slots["trad"]
print(f"bank_w={'有' if trad.bank_w is not None else '无'} "
      f"shape={None if trad.bank_w is None else trad.bank_w.shape} "
      f"bank(tile)={'有' if trad.bank is not None else '无'}", flush=True)

# 管线内逐图 trad 原始分
raw, cal, y = [], [], []
for p, lab in bundle["test"]:
    item = pipe._get_item(p)
    r = trad.score_image(item["img"])
    raw.append(float(r[0]))
    cal.append(pipe._record(item)["slot_scores"]["trad"])
    y.append(lab)
raw, cal, y = np.asarray(raw), np.asarray(cal), np.asarray(y)
print(f"n={len(y)} 缺陷={int(y.sum())}", flush=True)
print(f"raw AUROC={roc_auc_score(y, raw):.4f}  cal AUROC={roc_auc_score(y, cal):.4f}",
      flush=True)

# 与诊断脚本对账
diag = json.load(open(os.path.join(cfg["output_dir"], "diag_trad_score_component.json"),
                     encoding="utf-8"))
print(f"diag: {diag['auroc']}", flush=True)

# 独立复算（与 diag 相同的代码路径）
from PIL import Image
from src.slots.trad import TraditionalExtractor
ext = TraditionalExtractor()
bw = np.stack([ext.extract(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
               for p in bundle["init_normal"]])
m, s = bw.mean(0), bw.std(0) + 1e-6
bwz = (bw - m) / s
raw2 = []
for p, lab in bundle["test"]:
    v = (ext.extract(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32) - m) / s
    d = np.linalg.norm(bwz - v, axis=1)
    raw2.append(float(np.sort(d)[:3].mean()))
raw2 = np.asarray(raw2)
print(f"独立复算 AUROC={roc_auc_score(y, raw2):.4f}", flush=True)
print(f"raw vs raw2 max|diff|={np.abs(raw - raw2).max():.6f}", flush=True)

# evaluate 全路径复现
rep = pipe.evaluate(bundle["test"])
print(f"evaluate: fused={rep['fused']['auroc']:.4f} trad={rep['trad']['auroc']:.4f} "
      f"sem={rep['sem']['auroc']:.4f} blob={rep['blob']['auroc']:.4f}", flush=True)
cal_ev = []
for p, lab in bundle["test"]:
    cal_ev.append(pipe.predict(p)["slot_scores"]["trad"])
print(f"predict trad cal: {[round(v, 3) for v in cal_ev]}", flush=True)
print(f"_record trad cal: {[round(v, 3) for v in cal]}", flush=True)
print(f"y: {y.astype(int).tolist()}", flush=True)
