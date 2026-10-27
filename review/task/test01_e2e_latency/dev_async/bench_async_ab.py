"""G1 异步 render 同进程 A/B：旧 _finish_detect（service_before.py，主线程渲染）vs 新（线程池渲染）。

隔离：TEMP storage/engine/SQLite；init_normal 从 stable 归档复制 14 张，测试图=TEST-01 去重 6 张。
口径：e2e=latency_e2e_ms（decode+infer+render，不含 persist）；wall=detect_image 调用墙钟（含 join）。
一致性：同图 old/new 的 decision/score/defect_boxes/heatmap_boxes 及 heat/overlay/mask 文件字节逐张比对。
"""
import argparse, hashlib, importlib.util, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
OUT = TASK / "dev_async"


def pct(v):
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(np.mean(v)), 1)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if p and Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prepare-n", type=int, default=14)
    args = ap.parse_args()
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); tests.append(r["path"])
    outdir = Path(tempfile.mkdtemp(prefix="aoi_g1async_"))
    print(f"[1/5] isolate -> {outdir}", flush=True)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(outdir / "config.yaml")
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from backend.pipeline import service as svc
    from backend.engine import get_engine
    spec = importlib.util.spec_from_file_location("backend.pipeline._service_before", OUT / "service_before.py")
    old_mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = old_mod
    spec.loader.exec_module(old_mod)
    impl = {"old": old_mod.DetectionService._finish_detect, "new": svc.DetectionService._finish_detect}
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

    def run(tag, path):
        svc.DetectionService._finish_detect = impl[tag]
        st = time.perf_counter()
        r = service.detect_image(path, "gold_finger", persist=True, with_heatmap=True)
        return r, (time.perf_counter() - st) * 1000

    print("[3/5] warmup", flush=True)
    for i in range(4):
        run("old" if i % 2 else "new", tests[i % len(tests)])

    print("[4/5] consistency old vs new (同图逐张)", flush=True)
    prec = []
    for p in tests:
        ra, _ = run("old", p)
        fa = {k: sha(ra.get(k)) for k in ("heatmap_path", "overlay_path", "mask_path")}
        rb, _ = run("new", p)
        fb = {k: sha(rb.get(k)) for k in ("heatmap_path", "overlay_path", "mask_path")}
        loc_a, loc_b = ra.get("localization") or {}, rb.get("localization") or {}
        row = {"image": Path(p).name, "decision_old": ra["decision"], "decision_new": rb["decision"],
               "score_old": ra["final_score"], "score_new": rb["final_score"],
               "keys_equal": set(ra) == set(rb),
               "defect_boxes_equal": ra.get("defect_boxes") == rb.get("defect_boxes"),
               "localization_equal": json.dumps(loc_a, sort_keys=True, default=str)
               == json.dumps(loc_b, sort_keys=True, default=str),
               "files_old": fa, "files_new": fb, "files_equal": fa == fb,
               "loc_diff": {k: [str(loc_a.get(k))[:160], str(loc_b.get(k))[:160]]
                            for k in set(loc_a) | set(loc_b)
                            if json.dumps(loc_a.get(k), sort_keys=True, default=str)
                            != json.dumps(loc_b.get(k), sort_keys=True, default=str)}}
        row["identical"] = (row["decision_old"] == row["decision_new"] and row["score_old"] == row["score_new"]
                            and row["keys_equal"] and row["defect_boxes_equal"] and row["files_equal"])
        prec.append(row)
        print(f"  {row['image']} {row['decision_old']}/{row['decision_new']} "
              f"score {row['score_old']:.6g}/{row['score_new']:.6g} boxes_eq={row['defect_boxes_equal']} "
              f"files_eq={row['files_equal']} keys_eq={row['keys_equal']} loc_eq={row['localization_equal']}",
              flush=True)

    print(f"[5/5] latency A/B {args.rounds} rounds x {len(tests)} x 2 (交替)", flush=True)
    rows = []
    for rnd in range(args.rounds):
        for i, p in enumerate(tests):
            order = ("old", "new") if (rnd + i) % 2 == 0 else ("new", "old")
            cur = {}
            for tag in order:
                r, wall = run(tag, p)
                infer = float(r["latency_ms"]); total = float(r["total_latency_ms"]); e2e = float(r["latency_e2e_ms"])
                cur[tag] = {"tag": tag, "image": Path(p).name, "decode_ms": total - infer, "infer_ms": infer,
                            "render_ms": e2e - total, "e2e_ms": e2e, "wall_ms": wall,
                            "decision": r["decision"], "score": r["final_score"]}
                rows.append(cur[tag])
            print(f"  round {rnd+1}/{args.rounds} img {i+1}/{len(tests)} "
                  f"old e2e={cur['old']['e2e_ms']:.0f} render={cur['old']['render_ms']:.1f} | "
                  f"new e2e={cur['new']['e2e_ms']:.0f} render={cur['new']['render_ms']:.1f}", flush=True)
    svc.DetectionService._finish_detect = impl["new"]
    summ = {}
    for tag in ("old", "new"):
        rr = [r for r in rows if r["tag"] == tag]
        summ[tag] = {k: pct([r[k] for r in rr]) for k in ("decode_ms", "infer_ms", "render_ms", "e2e_ms", "wall_ms")}
    dec_same = all(a["decision"] == b["decision"] and a["score"] == b["score"]
                   for a in rows if a["tag"] == "old"
                   for b in rows if b["tag"] == "new" and b["image"] == a["image"])
    out = {"isolated_temp": str(outdir), "n_test_images": len(tests), "n_prepare": len(normals),
           "rounds": args.rounds, "consistency": prec,
           "consistency_all_identical": all(r["identical"] for r in prec),
           "latency_summary": summ, "ab_rows_decision_and_score_identical": dec_same, "rows": rows}
    (OUT / "ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False, indent=1),
          "\nCONSISTENCY_ALL", out["consistency_all_identical"], "\nAB_IDENTICAL", dec_same, flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)


if __name__ == "__main__":
    main()
