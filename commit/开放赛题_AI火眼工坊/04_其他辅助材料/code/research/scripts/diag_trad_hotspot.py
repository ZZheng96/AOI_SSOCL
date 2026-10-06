"""trad 特征提取热点分析（cProfile）"""
import cProfile
import pstats
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "demo4"))
from src.features.traditional import TraditionalFeatureExtractor

data = np.fromfile(r"D:\CGAIC\data_origin\mvtec\bottle\test\good\000.png", dtype=np.uint8)
img = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
ext = TraditionalFeatureExtractor()

# 预热
_ = ext.extract(img)

p = cProfile.Profile()
p.enable()
for _ in range(5):
    v = ext.extract(img)
p.disable()
print(f"特征维: {len(v)}  5 次总耗时: ", end="")
import time
t0 = time.time()
for _ in range(5):
    ext.extract(img)
print(f"{time.time()-t0:.2f}s ({ (time.time()-t0)/5*1000:.0f}ms/张)")

pstats.Stats(p).sort_stats("cumulative").print_stats(18)
