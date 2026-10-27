"""TEST-02 同 GPU 推理+学习并发测量（AC-069）。

隔离 TEMP 环境，prepare 两个品类（gold_finger / gold_finger_b）。
前台并发 1/2/4（ThreadPoolExecutor），两品类交替请求；
负载两种：idle（无后台学习）、learning（后台持续 submit_feedback，
生产默认 learning_parallel=False 串行让行）。
逐请求记录 t_submit/t_start/t_done、queue（=t_start-t_submit）、wall、
响应内 infer/e2e 分段；学习任务时间线单独记录。
"""
import argparse, csv, json, os, random, shutil, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test02_gpu_concurrency")
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests-per-level", type=int, default=24)
    ap.add_argument("--max-learning-events", type=int, default=12)
    args = ap.parse_args()

    outdir = Path(tempfile.mkdtemp(prefix="aoi_test02_"))
    print(f"[isolate] {outdir}", flush=True)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"),
                         engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                         host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    cfgp = outdir / "config.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(cfgp)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    from PIL import Image
    pool = []
    for p in sorted(ARCHIVE.iterdir()):
        if p.suffix.lower() not in (".jpg", ".png"):
            continue
        with Image.open(p) as im:
            if im.size == (2500, 2500):
                pool.append(p)
    random.seed(777)
    random.shuffle(pool)

    from backend.pipeline.service import get_detection_service
    from backend.engine import get_engine
    engine = get_engine()
    engine.learning_parallel = False
    service = get_detection_service()

    def prep(category, imgs, tag):
        train_dir = outdir / "train" / category
        train_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, src in enumerate(imgs):
            dst = train_dir / f"n{i:03d}{src.suffix}"
            shutil.copy2(src, dst)
            paths.append(str(dst))
        t0 = time.perf_counter()
        ver = engine.prepare(category,
                             {"init_normal": paths, "init_defect": [], "val": [], "test": []},
                             activate=True, progress_cb=lambda f, m: None)
        print(f"[prepare] {category} v{ver} {len(paths)} imgs {time.perf_counter()-t0:.1f}s",
              flush=True)

    prep("gold_finger", pool[:40], "a")
    prep("gold_finger_b", pool[40:70], "b")

    cats = ["gold_finger", "gold_finger_b"]
    inputs = {c: [str(p) for p in pool[i::2][:20]] for i, c in enumerate(cats)}
    # 热身
    for c in cats:
        for p in inputs[c][:5]:
            service.detect_image(p, c, persist=True, with_heatmap=True)
    print("[warmup] done", flush=True)

    rows = []
    learn_log = []
    learn_stop = threading.Event()
    counter = {"i": 0}
    lock = threading.Lock()

    def learn_loop():
        i = 0
        while not learn_stop.is_set() and i < args.max_learning_events:
            cat = cats[i % 2]
            path = inputs[cat][(i * 3) % len(inputs[cat])]
            done = threading.Event()
            out = {}
            t_submit = time.perf_counter()
            engine.submit_feedback(cat, path, "correct", label=0,
                                   on_result=lambda res, err: (out.update(error=err), done.set()))
            while not done.wait(5) and not learn_stop.is_set():
                pass
            learn_log.append({"event": i, "category": cat,
                              "t_submit": round(t_submit, 3),
                              "t_done": round(time.perf_counter(), 3),
                              "wall_ms": round((time.perf_counter() - t_submit) * 1000, 1),
                              "error": str(out.get("error")) if out.get("error") else None})
            print(f"  learning {i+1} wall={learn_log[-1]['wall_ms']}ms "
                  f"{learn_log[-1]['error'] or 'OK'}", flush=True)
            i += 1

    def one_request(cat, path, conc, load, idx):
        t_submit = time.perf_counter()
        t_start = time.perf_counter()
        r = service.detect_image(path, cat, persist=True, with_heatmap=True)
        t_done = time.perf_counter()
        row = {"idx": idx, "category": cat, "concurrency": conc, "load": load,
               "t_submit": round(t_submit, 3), "t_lock_acquired": round(t_start, 3),
               "t_done": round(t_done, 3),
               "queue_ms": round((t_start - t_submit) * 1000, 3),
               "wall_ms": round((t_done - t_submit) * 1000, 3),
               "infer_ms": round(float(r["latency_ms"]), 3),
               "e2e_ms": round(float(r.get("latency_e2e_ms") or r["latency_ms"]), 3),
               "decision": r["decision"]}
        with lock:
            rows.append(row)
        return row

    for conc in (1, 2, 4):
        for load in ("idle", "learning"):
            print(f"[run] concurrency={conc} load={load} "
                  f"requests={args.requests_per_level}", flush=True)
            learn_thread = None
            if load == "learning":
                learn_stop.clear()
                learn_thread = threading.Thread(target=learn_loop, daemon=True)
                learn_thread.start()
                time.sleep(0.3)  # 让学习先进入运行态
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futs = []
                for i in range(args.requests_per_level):
                    cat = cats[i % 2]
                    path = inputs[cat][i % len(inputs[cat])]
                    futs.append(ex.submit(one_request, cat, path, conc, load, i))
                for f in futs:
                    f.result(timeout=600)
            if learn_thread:
                learn_stop.set()
                learn_thread.join(timeout=600)
            done_rows = [r for r in rows if r["concurrency"] == conc and r["load"] == load]
            walls = sorted(r["wall_ms"] for r in done_rows)
            if walls:
                import numpy as np
                print(f"  -> n={len(walls)} p50={np.percentile(walls,50):.1f} "
                      f"p95={np.percentile(walls,95):.1f} max={max(walls):.1f}ms", flush=True)

    with open(TASK / "raw" / "per_request.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["concurrency"], r["load"], r["idx"])))
    (TASK / "raw" / "learning_timeline.json").write_text(
        json.dumps(learn_log, ensure_ascii=False, indent=2), encoding="utf-8")

    import numpy as np
    summary = {"temp": str(outdir), "categories": cats, "requests_per_level": args.requests_per_level,
               "learning_events": len(learn_log), "levels": {}}
    for conc in (1, 2, 4):
        for load in ("idle", "learning"):
            sel = [r["wall_ms"] for r in rows if r["concurrency"] == conc and r["load"] == load]
            q = [r["queue_ms"] for r in rows if r["concurrency"] == conc and r["load"] == load]
            if not sel:
                continue
            summary["levels"][f"c{conc}_{load}"] = {
                "n": len(sel), "wall_p50": float(np.percentile(sel, 50)),
                "wall_p95": float(np.percentile(sel, 95)),
                "wall_max": float(max(sel)),
                "queue_p95": float(np.percentile(q, 95)) if q else None}
    (TASK / "raw" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
