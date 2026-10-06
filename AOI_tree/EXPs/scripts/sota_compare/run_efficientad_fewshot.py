# -*- coding: utf-8 -*-
"""EfficientAD-S few-shot（100 正常）批处理复测 —— SOTA 对比（S2）。

协议：train/good 随机抽 100 张（seed=42，--fewshot 100），train_steps=2500
（≈25 epochs × 100 张，EfficientAD 论文 25 epoch 语义），test 全量评估。
逐品类顺序执行，输出到 outputs/sota/efficientad。
"""
import subprocess
import sys
import os
import json

sys.stdout.reconfigure(line_buffering=True)

MVTEC = r"D:\CGAIC\data_origin\mvtec"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientad.py")
CATS = ["bottle", "capsule", "cable", "transistor", "screw"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sota")
os.makedirs(OUT, exist_ok=True)

results = {}
for cat in CATS:
    print(f"\n========== EfficientAD-S few-shot 100 | {cat} ==========")
    cmd = [sys.executable, "-X", "utf8", SCRIPT,
           "--dataset", "mvtec_ad", "--subdataset", cat,
           "--mvtec_ad_path", MVTEC,
           "--weights", os.path.join(os.path.dirname(SCRIPT), "teacher_small.pth"),
           "--fewshot", "100", "--train_steps", "2500",
           "--output_dir", os.path.join(OUT, "efficientad")]
    r = subprocess.run(cmd, cwd=os.path.dirname(SCRIPT))
    print(f"[{cat}] exit={r.returncode}")

json.dump({"note": "EfficientAD-S few-shot100, train_steps=2500",
           "log": os.path.join(OUT, "efficientad")},
          open(os.path.join(OUT, "efficientad_fewshot100_meta.json"), "w"),
          ensure_ascii=False, indent=2)
print("\nAll done.")
