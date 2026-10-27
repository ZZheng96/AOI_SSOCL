"""dev_async 步骤1：_finish_detect 关键路径分段计时（render / io 等待 / persist）。

隔离同 dev_opt/bench_e2e_ab.py：TEMP storage/engine/SQLite，stable 归档复制 14 张做 init_normal，
测试图 = TEST-01 清单去重 6 张。不触碰生产 DB/快照。
分段：decode=total-infer；render 子步由包装函数计时；io_wait=finally 中 f.result() 阻塞；
persist=_persist；imwrite_bg=线程池内编码落盘耗时（不在关键路径，仅作参考）。
"""
import argparse, json, os, random, shutil, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")


def pct(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(np.mean(v)), 1)}


def isolate(prepare_n):
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); tests.append(r["path"])
    outdir = Path(tempfile.mkdtemp(prefix="aoi_async_"))
    print(f"[isolate] {outdir}", flush=True)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(outdir / "config.yaml")
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from backend.engine import get_engine
    engine = get_engine()
    test_set = set(tests)
    pool = []
    for p in sorted((STABLE / "storage" / "archive" / "gold_finger" / "20261011").iterdir()):
        if p.suffix.lower() in (".jpg", ".png") and str(p) not in test_set:
            with Image.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(str(p))
    random.seed(123); random.shuffle(pool)
    train = outdir / "train"; train.mkdir(parents=True)
    normals = []
    for i, src in enumerate(pool[:prepare_n]):
        dst = train / f"normal_{i:03d}{Path(src).suffix}"; shutil.copy2(src, dst); normals.append(str(dst))
    print(f"[prepare] {len(normals)} normals + {len(tests)} test images", flush=True)
    t0 = time.perf_counter()
    engine.prepare("gold_finger", {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}", flush=True))
    print(f"[prepare] done {time.perf_counter()-t0:.1f}s", flush=True)
    return outdir, tests, len(normals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prepare-n", type=int, default=14)
    args = ap.parse_args()
    outdir, tests, n_prep = isolate(args.prepare_n)
    import cv2
    from backend.pipeline import service as svc

    service = svc.get_detection_service()
    rec = {}
    lock = threading.Lock()

    def timed(name, fn):
        def w(*a, **k):
            st = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                with lock:
                    rec.setdefault(name, 0.0)
                    rec[name] += (time.perf_counter() - st) * 1000
        return w

    orig = {n: getattr(svc, n) for n in ("render_heatmap_overlay", "quantize_raw_heatmap", "imwrite", "_save_npy")}
    svc.render_heatmap_overlay = timed("render_overlay", orig["render_heatmap_overlay"])
    svc.quantize_raw_heatmap = timed("quantize", orig["quantize_raw_heatmap"])
    svc.imwrite = timed("imwrite_bg", orig["imwrite"])
    svc._save_npy = timed("save_npy_bg", orig["_save_npy"])
    orig_persist = service._persist
    service._persist = timed("persist", orig_persist)
    orig_alarm = service._record_alarm
    service._record_alarm = timed("alarm", orig_alarm)

    class _F:
        def __init__(self, f): self.f = f
        def add_done_callback(self, cb): self.f.add_done_callback(lambda _f: cb(self))
        def done(self): return self.f.done()
        def result(self, timeout=None):
            st = time.perf_counter()
            try:
                return self.f.result(timeout)
            finally:
                with lock:
                    rec["io_wait"] = rec.get("io_wait", 0.0) + (time.perf_counter() - st) * 1000

    real_exec = svc._io_exec

    class _E:
        def submit(self, *a, **k): return _F(real_exec.submit(*a, **k))
    svc._io_exec = _E()

    def run(p):
        rec.clear()
        st = time.perf_counter()
        r = service.detect_image(p, "gold_finger", persist=True, with_heatmap=True)
        wall = (time.perf_counter() - st) * 1000
        return r, wall, dict(rec)

    print("[warmup]", flush=True)
    for i in range(3):
        run(tests[i % len(tests)])
    rows = []
    for rnd in range(args.rounds):
        for i, p in enumerate(tests):
            r, wall, seg = run(p)
            infer = float(r["latency_ms"]); total = float(r["total_latency_ms"]); e2e = float(r["latency_e2e_ms"])
            row = {"image": Path(p).name, "decode_ms": total - infer, "infer_ms": infer,
                   "render_ms": e2e - total, "e2e_ms": e2e, "wall_ms": wall,
                   "post_e2e_ms": wall - e2e,
                   **{f"{k}_ms": v for k, v in seg.items()}}
            rows.append(row)
            print(f"  r{rnd+1} {i+1}/{len(tests)} e2e={e2e:.0f} render={row['render_ms']:.1f} "
                  f"ov={seg.get('render_overlay', 0):.1f} persist={seg.get('persist', 0):.1f} "
                  f"io_wait={seg.get('io_wait', 0):.1f} wall={wall:.0f}", flush=True)

    # render 子步离线细分（同输入，非关键路径计时）
    print("[substeps]", flush=True)
    from backend.engine import get_engine
    engine = get_engine()
    subs = []
    for p in tests:
        rgb = svc._decode_rgb(p)
        res = engine.predict_ndarray("gold_finger", rgb, path_hint=p)
        hm = svc._best_heatmap(res)
        for _ in range(3):
            t = {}
            s = time.perf_counter(); h0, w0 = rgb.shape[:2]; sc = 1280 / max(h0, w0)
            d = cv2.resize(rgb, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA); t["display_resize"] = time.perf_counter() - s
            s = time.perf_counter(); bgr = cv2.cvtColor(d, cv2.COLOR_RGB2BGR); t["cvt_bgr"] = time.perf_counter() - s
            s = time.perf_counter(); q = svc.quantize_raw_heatmap(hm); t["quantize"] = time.perf_counter() - s
            s = time.perf_counter(); u8 = svc._heatmap_u8(q, bgr.shape[1::-1]); t["heatmap_u8"] = time.perf_counter() - s
            s = time.perf_counter(); hc = cv2.applyColorMap(u8, cv2.COLORMAP_JET); t["colormap"] = time.perf_counter() - s
            s = time.perf_counter(); ov = cv2.addWeighted(bgr, 0.5, hc, 0.5, 0); t["addweighted"] = time.perf_counter() - s
            s = time.perf_counter(); svc._mask_and_boxes(u8, 0.6, 16, 3); t["mask_boxes"] = time.perf_counter() - s
            s = time.perf_counter(); cv2.imencode(".jpg", hc); t["enc_heat_jpg"] = time.perf_counter() - s
            s = time.perf_counter(); cv2.imencode(".jpg", ov); t["enc_overlay_jpg"] = time.perf_counter() - s
            subs.append({k: v * 1000 for k, v in t.items()})

    svc._io_exec = real_exec
    summ = {k: pct([r.get(k) for r in rows]) for k in sorted({k for r in rows for k in r if k.endswith("_ms")})}
    sub_summ = {k: pct([s[k] for s in subs]) for k in subs[0]}
    out = {"isolated_temp": str(outdir), "n_test_images": len(tests), "n_prepare": n_prep,
           "rounds": args.rounds, "segments": summ, "render_substeps": sub_summ, "rows": rows}
    (TASK / "dev_async" / "breakdown.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SEGMENTS", json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    print("SUBSTEPS", json.dumps(sub_summ, ensure_ascii=False, indent=1), flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)


if __name__ == "__main__":
    main()
