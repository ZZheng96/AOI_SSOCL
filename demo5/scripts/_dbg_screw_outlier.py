"""定位 screw good outlier：哪个图、哪个特征维爆掉"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
from PIL import Image

from src.data import mvtec_like
from src.slots.trad import TraditionalExtractor

cfg = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "configs", "m0.yaml")))
root = cfg["datasets"]["mvtec"]
ext = TraditionalExtractor()
np.random.seed(0)


def vec(p):
    img = np.asarray(Image.open(p).convert("RGB"))
    return ext.extract(img).astype(np.float32)


b = mvtec_like.load_category(root, "screw", 100, 30, 42, n_eval_good=None)
bank = np.stack([vec(p) for p in b["init_normal"]])
m, s = bank.mean(0), bank.std(0) + 1e-6
bankz = (bank - m) / s

print("train/good 特征极端值（按 |z|>10 计数，前 15 维）：")
cnt = (np.abs(bankz) > 10).sum(axis=0)
order = np.argsort(-cnt)[:15]
for d in order:
    print(f"  dim {d}: |z|>10 有 {cnt[d]} 个, max|z|={np.abs(bankz[:, d]).max():.1f}")

print("\ntest 逐样本距离排查：")
for p, lab in b["test"]:
    v = (vec(p) - m) / s
    d = np.linalg.norm(bankz - v, axis=1)
    d3 = float(np.sort(d)[:3].mean())
    if d3 > 100:
        # 找出贡献最大的维度
        dm = np.abs(bankz - v).max(axis=0)
        top = np.argsort(-dm)[:5]
        print(f"  {lab} {os.path.basename(p)}: d3={d3:.1f} 极端维="
              + " ".join(f"{t}(v={v[t]:.1f},bankmax={np.abs(bankz[:, t]).max():.1f})"
                         for t in top))
