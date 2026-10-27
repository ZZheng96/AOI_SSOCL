"""OPT-DECODE：拆分 _decode_rgb 各子步耗时，并对比候选快路径像素一致性。

用法（AOI_Core 下）：
  "C:\\Program Files\\Python311\\python.exe" ..\\..\\review\\task\\test01_e2e_latency\\dev_opt\\bench_decode.py --n 20 --reps 5
只读 TEST-01 清单中的图片（≤20 张去重），不写任何生产文件。
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
sys.path.insert(0, os.getcwd())


def t_ms(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); out.append((time.perf_counter() - t0) * 1000)
    return r, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="before")
    args = ap.parse_args()
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, paths = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); paths.append(r["path"])
    paths = paths[:args.n]
    from backend.pipeline.service import _decode_rgb
    flag = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    has_rgb = hasattr(cv2, "IMREAD_COLOR_RGB")
    agg = {k: [] for k in ("pil_header", "fromfile", "imdecode_bgr", "cvtColor",
                           "imdecode_rgb", "pil_full", "decode_rgb_current")}
    rows = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            fmt, mode, size = im.format, im.mode, im.size
            info = {k: im.info.get(k) for k in ("gamma", "transparency", "icc_profile") if k in im.info}

        def hdr():
            with Image.open(p) as im:
                return im.format, im.mode
        _, a = t_ms(hdr, args.reps)
        data, b = t_ms(lambda: np.fromfile(p, dtype=np.uint8), args.reps)
        bgr, c = t_ms(lambda: cv2.imdecode(data, flag), args.reps)
        rgb_cv, d = t_ms(lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.reps)
        e = [0.0]
        rgb_direct = None
        if has_rgb:
            rgb_direct, e = t_ms(lambda: cv2.imdecode(data, cv2.IMREAD_COLOR_RGB | cv2.IMREAD_IGNORE_ORIENTATION), args.reps)

        def pil_full():
            with Image.open(p) as im:
                return np.array(im.convert("RGB"))
        rgb_pil, f = t_ms(pil_full, args.reps)
        cur, g = t_ms(lambda: _decode_rgb(p), args.reps)
        md_cv = int(np.abs(rgb_cv.astype(np.int16) - rgb_pil).max())
        md_direct = (int(np.abs(rgb_direct.astype(np.int16) - rgb_pil).max())
                     if rgb_direct is not None else None)
        md_cur = int(np.abs(cur.astype(np.int16) - rgb_pil).max())
        for k, v in zip(agg, (a, b, c, d, e, f, g)):
            agg[k].extend(v)
        row = {"image": Path(p).name, "format": fmt, "mode": mode, "size": size,
               "file_kb": round(os.path.getsize(p) / 1024, 1), "png_info_keys": sorted(info),
               "med_ms": {k: round(statistics.median(v), 2) for k, v in
                          zip(agg, (a, b, c, d, e, f, g))},
               "maxdiff_cv2_vs_pil": md_cv, "maxdiff_cv2rgb_vs_pil": md_direct,
               "maxdiff_current_vs_pil": md_cur}
        rows.append(row)
        print(f"[{i+1}/{len(paths)}] {row['image']} {fmt}/{mode} cur={row['med_ms']['decode_rgb_current']}ms "
              f"pil={row['med_ms']['pil_full']}ms cv2={row['med_ms']['imdecode_bgr']}+{row['med_ms']['cvtColor']}ms "
              f"cv2rgb={row['med_ms']['imdecode_rgb']}ms md={md_cv}/{md_direct}", flush=True)
    summ = {k: {"median": round(statistics.median(v), 2),
                "p95": round(float(np.percentile(v, 95)), 2)} for k, v in agg.items()}
    out = {"tag": args.tag, "n_images": len(paths), "reps": args.reps, "cv2": cv2.__version__,
           "has_IMREAD_COLOR_RGB": has_rgb, "summary_ms": summ, "rows": rows}
    op = TASK / "dev_opt" / f"decode_breakdown_{args.tag}.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
