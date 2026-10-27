"""OPT-DECODE 第 1 步：原生 2500² PNG 解码段逐步计时（开发自测，10 张）。

只做解码相关步骤，不加载模型。对比 PIL 路径（现状）与 cv2 路径的像素等价性。
"""
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAN = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency\raw\images.json")
recs = json.loads(MAN.read_text(encoding="utf-8"))["images"]
paths = list(dict.fromkeys(r["path"] for r in recs))[:10]

def ms(a, b):
    return (b - a) * 1000

rows = []
for i, p in enumerate(paths):
    for _ in range(2):  # 第 1 次热文件缓存，取第 2 次
        t0 = time.perf_counter()
        with Image.open(p) as pil:
            fmt, mode = pil.format, pil.mode
            t1 = time.perf_counter()
            pil.load()
            t2 = time.perf_counter()
            conv = pil.convert("RGB") if mode != "RGB" else pil
            t3 = time.perf_counter()
            a_pil = np.array(conv)
            t4 = time.perf_counter()
        data = np.fromfile(p, dtype=np.uint8)
        t5 = time.perf_counter()
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        t6 = time.perf_counter()
        a_cv = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t7 = time.perf_counter()
        a_rgb = cv2.imdecode(data, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION)
        t8 = time.perf_counter()
    row = {"image": Path(p).name, "fmt": fmt, "mode": mode, "bytes": int(data.size),
           "pil_open": ms(t0, t1), "pil_load": ms(t1, t2), "pil_convert": ms(t2, t3),
           "np_array": ms(t3, t4), "pil_total": ms(t0, t4),
           "fromfile": ms(t4, t5), "cv_imdecode": ms(t5, t6), "cv_cvtcolor": ms(t6, t7),
           "cv_total": ms(t4, t7), "cv_rgb_direct": ms(t7, t8) + ms(t4, t5),
           "maxdiff_cv_vs_pil": int(np.abs(a_cv.astype(int) - a_pil).max()),
           "maxdiff_rgbdirect_vs_pil": int(np.abs(a_rgb.astype(int) - a_pil).max())}
    rows.append(row)
    print(f"[{i+1}/{len(paths)}] " + " ".join(
        f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

keys = [k for k, v in rows[0].items() if isinstance(v, float)]
med = {k: float(np.median([r[k] for r in rows])) for k in keys}
print("MEDIAN", json.dumps(med, indent=1))
out = Path(__file__).with_name("micro_decode.json")
out.write_text(json.dumps({"rows": rows, "median": med,
                           "note": "开发自测，热文件缓存，不能替代 TEST-01"},
                          ensure_ascii=False, indent=2), encoding="utf-8")
