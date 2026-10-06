"""诊断 mvtec screw trad 反向（0.2998）：raw 3NN vs 校准分 vs 库大小/划分"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data import mvtec_like
from src.slots.trad import TraditionalExtractor

cfg = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "configs", "m0.yaml")))
root = cfg["datasets"]["mvtec"]
ext = TraditionalExtractor()


def vec(p):
    img = np.asarray(Image.open(p).convert("RGB"))
    return ext.extract(img).astype(np.float32)


def run(n_init_normal, seed):
    b = mvtec_like.load_category(root, "screw", n_init_normal, 30, seed,
                                 n_eval_good=None)
    bank = np.stack([vec(p) for p in b["init_normal"]])
    m, s = bank.mean(0), bank.std(0) + 1e-6
    bankz = (bank - m) / s
    ys, raws = [], []
    for p, lab in b["test"]:
        v = (vec(p) - m) / s
        d = np.linalg.norm(bankz - v, axis=1)
        raws.append(float(np.sort(d)[:3].mean()))
        ys.append(lab)
    ys, raws = np.asarray(ys), np.asarray(raws)
    good = raws[ys == 0]
    bad = raws[ys == 1]
    print(f"[n_init={n_init_normal}, seed={seed}] n_test={len(ys)} "
          f"good={len(good)} bad={len(bad)}")
    print(f"  raw AUROC={roc_auc_score(ys, raws):.4f}  "
          f"good: mean={good.mean():.4f} max={good.max():.4f} | "
          f"bad: mean={bad.mean():.4f} max={bad.max():.4f}")
    print(f"  good 前 5 高: {np.sort(good)[-5:].round(3).tolist()}")
    print(f"  bad  前 5 高: {np.sort(bad)[-5:].round(3).tolist()}")


for n in (100, 30):
    for seed in (42, 7):
        run(n, seed)
