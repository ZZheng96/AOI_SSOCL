"""TEST-03 子进程：共享同一 GPU 的第二实例推理循环。"""
import os, sys, time
from pathlib import Path

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
os.environ["AOI_CONFIG"] = sys.argv[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.pipeline.service import get_detection_service

svc = get_detection_service()
img, category, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
for i in range(n):
    svc.detect_image(img, category, persist=False, with_heatmap=False)
    time.sleep(0.02)
print("child done")
