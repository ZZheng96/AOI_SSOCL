"""打印 009.png dim 60-80 的特征值，确认 dim72 来源"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image

from src.slots.trad import TraditionalExtractor

ext = TraditionalExtractor()
np.random.seed(0)
p = r"D:\CGAIC\data_origin\mvtec\screw\test\good\009.png"
img = np.asarray(Image.open(p).convert("RGB"))
f = ext.extract(img).astype(np.float32)
print("009.png dim 55-90:")
for i in range(55, 90):
    print(f"  dim {i}: {f[i]:.4f}")
print("非有限值:", np.isnan(f).sum(), np.isinf(f).sum())
print("|f|>1000 的维:", [(i, f[i]) for i in range(200) if abs(f[i]) > 1000])
