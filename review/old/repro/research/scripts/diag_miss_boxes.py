"""U86v20 前置诊断：漏检/误报样本的框面积与缺陷占比（验证信噪比假设）。

7 个漏检 + 4 个误报样本的 GT 框统计：框面积、占图比、框相对 crop_size(699) 占比。
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2

from src.slots.shead import _parse_yolo_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"

# 漏检样本（缺陷图被判正常）+ 误报样本（正常图）
MISS = ["12336.jpg", "12542.jpg", "12347.jpg", "12581.jpg", "7224.JPG",
        "12617.jpg", "12610.jpg"]
FP = ["12402.jpg", "12520.jpg", "12383.jpg", "12405.jpg"]


def analyze(fname):
    img = cv2.cvtColor(cv2.imread(os.path.join(GYU_ROOT, "test", "images", fname)),
                       cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = os.path.join(GYU_ROOT, "test", "labels",
                       os.path.splitext(fname)[0] + ".txt")
    boxes = _parse_yolo_boxes(lbl, w, h)
    area_ratio = []
    for bx in boxes:
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        area_ratio.append((bw * bh) / (w * h))
    return h, w, boxes, area_ratio


print("=== 漏检样本（缺陷图被判正常）===", flush=True)
for f in MISS:
    h, w, boxes, ar = analyze(f)
    if boxes:
        sides = sorted(min(b[2]-b[0], b[3]-b[1]) for b in boxes)
        print(f"  {f}: {w}x{h} 框={len(boxes)} 框面积占图比 mean={np.mean(ar):.2%} "
              f"框最小边 median={np.median(sides):.0f}px "
              f"(crop699占比={np.median(sides)/699:.0%})", flush=True)
    else:
        print(f"  {f}: {w}x{h} 无框!", flush=True)

print("\n=== 误报样本（正常图被判异常）===", flush=True)
for f in FP:
    h, w, boxes, ar = analyze(f)
    print(f"  {f}: {w}x{h} 框数={len(boxes)}", flush=True)

# 对照：正常检出的缺陷图（高分）框占比
print("\n=== 对照：高分缺陷图（正确检出）===", flush=True)
import random
rng = random.Random(7)
for f in ["12320.jpg", "12321.jpg", "12322.jpg", "12330.jpg", "12335.jpg"]:
    p = os.path.join(GYU_ROOT, "test", "images", f)
    if not os.path.exists(p):
        continue
    h, w, boxes, ar = analyze(f)
    if boxes:
        sides = sorted(min(b[2]-b[0], b[3]-b[1]) for b in boxes)
        print(f"  {f}: {w}x{h} 框={len(boxes)} 框面积占图比 mean={np.mean(ar):.2%} "
              f"框最小边 median={np.median(sides):.0f}px", flush=True)
