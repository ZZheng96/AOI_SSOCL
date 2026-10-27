"""下采样解码微基准（AC-066 / §69）：PNG/JPEG 各解码方案耗时与输出尺寸。

方案：full=现行 cv2 IMREAD_COLOR_RGB；reduced2/4=IMREAD_REDUCED_COLOR_2/4（BGR，需 cvtColor）；
full+area{1024,512}=全解码后 INTER_AREA；pil_draft{1024,512}=PIL JPEG draft（DCT 缩放）。
图源：TEST-01 清单 2500² PNG（去重）+ 稳定版归档 2500² JPG + data_local gold_finger 4096x3000/5120² JPG。
只读，不写任何库/快照。
"""
import json, statistics, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
ARCH = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")
DL = Path(r"D:\CGAIC\data_local\gold_finger\test")
RGB = getattr(cv2, "IMREAD_COLOR_RGB", None)
IGN = cv2.IMREAD_IGNORE_ORIENTATION


def rd(p, flags):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flags)


def area(img, side):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def pil_draft(p, side):
    with Image.open(str(p)) as im:
        im.draft("RGB", (side, side))
        return np.asarray(im.convert("RGB"))


METHODS = {
    "full": lambda p: rd(p, IGN | RGB),
    "reduced2": lambda p: cv2.cvtColor(rd(p, IGN | cv2.IMREAD_REDUCED_COLOR_2), cv2.COLOR_BGR2RGB),
    "reduced4": lambda p: cv2.cvtColor(rd(p, IGN | cv2.IMREAD_REDUCED_COLOR_4), cv2.COLOR_BGR2RGB),
    "full+area1024": lambda p: area(rd(p, IGN | RGB), 1024),
    "full+area512": lambda p: area(rd(p, IGN | RGB), 512),
    "pil_draft1024": lambda p: pil_draft(p, 1024),
    "pil_draft512": lambda p: pil_draft(p, 512),
}


def main():
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, png = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); png.append(Path(r["path"]))
    groups = {
        "png2500": png[:6],
        "jpg2500": sorted(ARCH.glob("*.jpg"))[:6],
        "jpg4096x3000": [p for p in sorted(DL.rglob("*.jpg")) if Image.open(p).size == (4096, 3000)][:4],
        "jpg5120": [p for p in sorted(DL.rglob("*.jpg")) if Image.open(p).size == (5120, 5120)][:4],
    }
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {}
    for g, paths in groups.items():
        out[g] = {"n": len(paths)}
        for m, fn in METHODS.items():
            if m.startswith("pil_draft") and g.startswith("png"):
                continue  # draft 仅对 JPEG 有效
            fn(paths[0])  # warm
            ts, shp = [], None
            for _ in range(reps):
                for p in paths:
                    t = time.perf_counter(); a = fn(p); ts.append((time.perf_counter() - t) * 1000)
                    shp = a.shape
            out[g][m] = {"p50": round(statistics.median(ts), 1),
                         "p95": round(float(np.percentile(ts, 95)), 1), "shape": list(shp)}
            print(f"[{g}] {m:14s} p50={out[g][m]['p50']:6.1f} p95={out[g][m]['p95']:6.1f} shape={shp}", flush=True)
    (TASK / "dev_downsample" / "decode_bench.json").write_text(
        json.dumps({"cv2": cv2.__version__, "reps": reps, "results": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
