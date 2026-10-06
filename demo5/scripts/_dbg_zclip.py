"""验证 zclip 是否生效（screw 009.png dim72）"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import yaml
from PIL import Image

from src.data import mvtec_like
from src.slots.trad import TradSlot

cfg = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "configs", "m0.yaml")))
print("slots.trad cfg:", cfg["slots"]["trad"])
slot = TradSlot(cfg["slots"]["trad"])
print("zclip =", slot.zclip)

b = mvtec_like.load_category(cfg["datasets"]["mvtec"], "screw", 100, 30, 42,
                             n_eval_good=None)
from src.common.io import load_image
bank = np.stack([slot._vec(load_image(p)) for p in b["init_normal"]])
m, s = bank.mean(0), bank.std(0) + 1e-6
z = slot._z(bank, (m, s))
print("bank z max|dim72|:", np.abs(z[:, 72]).max())

from src.common.io import load_image
p009 = r"D:\CGAIC\data_origin\mvtec\screw\test\good\009.png"
v = slot._vec(load_image(p009))
z9 = slot._z(v, (m, s))
print("009.png dim72 raw=%.4f z=%.2f (未 clip 应为 %.0f)" % (v[72], z9[72], (v[72]-m[72])/s[72]))

# 直接对比 clip 前后 009 的 3NN 距离
d_clip = np.linalg.norm(z - z9, axis=1)
print("clip 后 009 3NN:", np.sort(d_clip)[:3], "mean=%.3f" % np.sort(d_clip)[:3].mean())
z9_nc = (v - m) / s
d_nc = np.linalg.norm(z - z9_nc, axis=1)
print("未 clip 009 3NN:", np.sort(d_nc)[:3], "mean=%.3f" % np.sort(d_nc)[:3].mean())
