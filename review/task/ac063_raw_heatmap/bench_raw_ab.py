"""AC-063 原始热图留存 A/B：overlay_save_raw_heatmap on vs off（同进程交替）。

隔离：TEMP storage/engine/SQLite；init_normal 从 stable 归档复制 6 张，测试图取
TEST-01 清单去重前 4 张（合计 10 张）。不触碰生产 DB/快照。
口径同 TEST-01：e2e=latency_e2e_ms（不含 join），wall 含 persist + 写盘 join。
另测：raw .npy 单独写盘耗时、文件大小、默认参数复算与在线 mask/框一致性。
"""
import argparse, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
T01 = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\ac063_raw_heatmap")


def pct(v):
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 2),
            "p95": round(float(np.percentile(v, 95)), 2), "mean": round(float(np.mean(v)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--prepare-n", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=4)
    args = ap.parse_args()
    man = json.loads((T01 / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); tests.append(r["path"])
    tests = tests[:args.n_test]
    outdir = Path(tempfile.mkdtemp(prefix="aoi_ac063raw_"))
    print(f"[1/5] isolate -> {outdir}", flush=True)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(outdir / "config.yaml")
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from backend.core.imaging import imread
    from backend.pipeline import service as svc
    from backend.engine import get_engine
    service = svc.get_detection_service()
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
    for i, src in enumerate(pool[:args.prepare_n]):
        dst = train / f"normal_{i:03d}{Path(src).suffix}"; shutil.copy2(src, dst); normals.append(str(dst))
    print(f"[2/5] prepare {len(normals)} normals + {len(tests)} test images", flush=True)
    t0 = time.perf_counter()
    engine.prepare("gold_finger", {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}", flush=True))
    print(f"  prepare done {time.perf_counter()-t0:.1f}s", flush=True)

    real_get = service.settings.get
    state = {"raw": True}

    def patched_get(sec, key, default=None):
        if sec == "pipeline" and key == "overlay_save_raw_heatmap":
            return state["raw"]
        return real_get(sec, key, default)
    service.settings.get = patched_get

    def run(tag, path):
        state["raw"] = tag == "on"
        st = time.perf_counter()
        r = service.detect_image(path, "gold_finger", persist=True, with_heatmap=True)
        return r, (time.perf_counter() - st) * 1000

    print("[3/5] warmup", flush=True)
    for i in range(4):
        run("on" if i % 2 else "off", tests[i % len(tests)])

    print("[4/5] raw 单独写盘耗时/大小 + 默认参数复算一致性", flush=True)
    hm_dir = service.settings.storage("heatmaps")
    io_rows = []
    for p in tests:
        r, _ = run("on", p)
        pp = (r.get("n_tiles") or {}).get("localization", {}).get("heatmap_postproc")
        if not pp or "raw_heatmap_path" not in r:
            print(f"  {Path(p).name} 无热图（status={r.get('status')} keys={sorted(r)[:12]}），跳过", flush=True)
            io_rows.append({"image": Path(p).name, "skipped": "no_heatmap", "status": r.get("status")})
            continue
        raw = np.load(r["raw_heatmap_path"], allow_pickle=False)
        ts = []
        for k in range(5):
            tp = Path(tempfile.gettempdir()) / f"ac063_w{k}.npy"
            s = time.perf_counter(); svc._save_npy(tp, raw); ts.append((time.perf_counter() - s) * 1000)
            tp.unlink()
        rec = svc.recompute_from_postproc(pp, hm_dir)
        mask_eq = bool(np.array_equal(rec["mask"], imread(r["mask_path"], 0)))
        boxes_eq = rec["boxes_image"] == [h["bbox"] for h in r["n_tiles"]["localization"]["heatmap_boxes"]]
        sizes = {k: Path(r[k]).stat().st_size for k in ("raw_heatmap_path", "mask_path",
                                                        "heatmap_path", "overlay_path")}
        row = {"image": Path(p).name, "raw_shape": pp["raw_shape"], "raw_dtype": pp["raw_dtype"],
               "display_size": pp["display_size"], "write_ms": pct(ts), "bytes": sizes,
               "recompute_mask_equal": mask_eq, "recompute_boxes_equal": boxes_eq,
               "n_heatmap_boxes": len(rec["boxes_image"])}
        io_rows.append(row)
        print(f"  {row['image']} shape={row['raw_shape']} {row['raw_dtype']} raw={sizes['raw_heatmap_path']}B "
              f"write_p50={row['write_ms']['p50']}ms mask_eq={mask_eq} boxes_eq={boxes_eq}", flush=True)

    print(f"[5/5] latency A/B {args.rounds} rounds x {len(tests)} x 2 (交替)", flush=True)
    rows = []
    for rnd in range(args.rounds):
        for i, p in enumerate(tests):
            order = ("on", "off") if (rnd + i) % 2 == 0 else ("off", "on")
            for tag in order:
                r, wall = run(tag, p)
                rows.append({"tag": tag, "image": Path(p).name, "e2e_ms": float(r["latency_e2e_ms"]),
                             "infer_ms": float(r["latency_ms"]), "wall_ms": wall,
                             "raw": "raw_heatmap_path" in r})
            last = {x["tag"]: x["e2e_ms"] for x in rows[-2:]}
            print(f"  round {rnd+1}/{args.rounds} img {i+1}/{len(tests)} "
                  f"on_e2e={last['on']:.0f} off_e2e={last['off']:.0f}", flush=True)
    summ = {tag: {k: pct([r[k] for r in rows if r["tag"] == tag]) for k in ("infer_ms", "e2e_ms", "wall_ms")}
            for tag in ("on", "off")}
    out = {"isolated_temp": str(outdir), "n_test_images": len(tests), "n_prepare": len(normals),
           "rounds": args.rounds, "io": io_rows, "latency_summary": summ, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)


if __name__ == "__main__":
    main()
