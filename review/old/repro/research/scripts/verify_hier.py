"""验证 predict 分层输出 L2/L3（§6.2）：boxes/types 是否产出。"""
import sys, os, yaml
sys.path.insert(0, r"d:\CGAIC\demo5")
sys.path.insert(0, r"d:\CGAIC\demo5\scripts")
import torch
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline

with open(r"d:\CGAIC\demo5\configs\m0.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
device = "cuda"
bundle = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger", 100, 30, cfg["seed"])
backbone, slots = build(cfg, device)
pipe = Pipeline(cfg, backbone, slots)
pipe.fit(bundle)

n_box = n_type = n_anom = 0
for p, y in bundle["test"][:20]:
    r = pipe.predict(p)
    if r["decision"] != "normal":
        n_anom += 1
        n_box += len(r["boxes"])
        n_type += len(r["types"])
        if n_anom <= 3:
            tinfo = [t["type"] for t in r["types"]]
            print("  [{}] fused={:.3f} boxes={} types={}".format(r["decision"], r["fused"], len(r["boxes"]), tinfo))
print("总计: anomaly={}/20, 框数={}, 类型数={}".format(n_anom, n_box, n_type))
