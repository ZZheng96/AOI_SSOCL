"""TEST-02 复测（AC-070 缩小品类锁临界区）。

基于 raw/bench_test02.py：ROOT 换开发版，输出到 dev_lock/<tag>/；
--prefetch 0/1 切换 DetectionEngine.predict_prefetch（0=改前整段持锁，1=改后锁外预提取）。
新增：accuracy 阶段（学习前：串行 40 张 + c4 并发 40 张，记录 score/decision 供跨档对比）、
逐请求 lock_wait_ms、进度输出。输入集仍取 stable 归档（只读）以与基线同口径。
"""
import argparse, csv, json, os, random, shutil, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test02_gpu_concurrency")
ARCHIVE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--requests-per-level", type=int, default=24)
    ap.add_argument("--max-learning-events", type=int, default=12)
    args = ap.parse_args()
    tag = "after" if args.prefetch else "before"
    out_dir = TASK / "dev_lock" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    outdir = Path(tempfile.mkdtemp(prefix="aoi_test02_lock_"))
    print(f"[isolate] {outdir} prefetch={args.prefetch}", flush=True)
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

    import numpy as np
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
    from backend.engine import get_engine, DetectionEngine
    DetectionEngine.predict_prefetch = bool(args.prefetch)
    engine = get_engine()
    engine.learning_parallel = False
    service = get_detection_service()

    def prep(category, imgs):
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

    prep("gold_finger", pool[:40])
    prep("gold_finger_b", pool[40:70])

    cats = ["gold_finger", "gold_finger_b"]
    inputs = {c: [str(p) for p in pool[i::2][:20]] for i, c in enumerate(cats)}
    for c in cats:
        for p in inputs[c][:5]:
            service.detect_image(p, c, persist=True, with_heatmap=True)
    print("[warmup] done", flush=True)

    # 精度阶段（学习前）：串行一遍 + c4 并发一遍，同一批 40 张
    acc_items = [(c, p) for c in cats for p in inputs[c]]
    acc = {"serial": {}, "c4": {}}
    for k, (c, p) in enumerate(acc_items):
        r = service.detect_image(p, c, persist=True, with_heatmap=True)
        acc["serial"][f"{c}|{Path(p).name}"] = [float(r["final_score"]), r["decision"]]
        if (k + 1) % 10 == 0:
            print(f"  [acc serial] {k+1}/{len(acc_items)}", flush=True)

    def acc_one(c, p):
        r = service.detect_image(p, c, persist=True, with_heatmap=True)
        return f"{c}|{Path(p).name}", [float(r["final_score"]), r["decision"]]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for key, v in ex.map(lambda a: acc_one(*a), acc_items):
            acc["c4"][key] = v
    diff = [k for k in acc["serial"]
            if acc["serial"][k][1] != acc["c4"][k][1]
            or abs(acc["serial"][k][0] - acc["c4"][k][0]) > 1e-6]
    print(f"[acc] c4 vs serial mismatches={len(diff)}", flush=True)
    (out_dir / "accuracy.json").write_text(
        json.dumps({**acc, "c4_vs_serial_mismatch": diff}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    rows, learn_log = [], []
    learn_stop = threading.Event()
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
        r = service.detect_image(path, cat, persist=True, with_heatmap=True)
        t_done = time.perf_counter()
        row = {"idx": idx, "category": cat, "concurrency": conc, "load": load,
               "t_submit": round(t_submit, 3), "t_done": round(t_done, 3),
               "wall_ms": round((t_done - t_submit) * 1000, 3),
               "infer_ms": round(float(r["latency_ms"]), 3),
               "e2e_ms": round(float(r.get("total_latency_ms") or r["latency_ms"]), 3),
               "lock_wait_ms": round(float(r.get("lock_wait_ms") or 0.0), 3),
               "score": float(r["final_score"]), "decision": r["decision"]}
        with lock:
            rows.append(row)
            n = sum(1 for x in rows if x["concurrency"] == conc and x["load"] == load)
        if n % 8 == 0:
            print(f"    c{conc}_{load} {n}/{args.requests_per_level}", flush=True)
        return row

    for conc in (1, 2, 4):
        for load in ("idle", "learning"):
            print(f"[run] concurrency={conc} load={load} requests={args.requests_per_level}",
                  flush=True)
            learn_thread = None
            if load == "learning":
                learn_stop.clear()
                learn_thread = threading.Thread(target=learn_loop, daemon=True)
                learn_thread.start()
                time.sleep(0.3)
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(one_request, cats[i % 2], inputs[cats[i % 2]][i % 20], conc, load, i)
                        for i in range(args.requests_per_level)]
                for f in futs:
                    f.result(timeout=600)
            if learn_thread:
                learn_stop.set()
                learn_thread.join(timeout=600)
            walls = [r["wall_ms"] for r in rows if r["concurrency"] == conc and r["load"] == load]
            print(f"  -> n={len(walls)} p50={np.percentile(walls,50):.1f} "
                  f"p95={np.percentile(walls,95):.1f} max={max(walls):.1f}ms", flush=True)

    with open(out_dir / "per_request.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["concurrency"], r["load"], r["idx"])))
    (out_dir / "learning_timeline.json").write_text(
        json.dumps(learn_log, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"temp": str(outdir), "prefetch": bool(args.prefetch), "categories": cats,
               "requests_per_level": args.requests_per_level,
               "learning_events": len(learn_log),
               "learning_errors": sum(1 for x in learn_log if x["error"]),
               "acc_c4_vs_serial_mismatch": len(diff), "levels": {}}
    for conc in (1, 2, 4):
        for load in ("idle", "learning"):
            sel = [r for r in rows if r["concurrency"] == conc and r["load"] == load]
            wl = [r["wall_ms"] for r in sel]
            lw = [r["lock_wait_ms"] for r in sel]
            summary["levels"][f"c{conc}_{load}"] = {
                "n": len(wl), "wall_p50": float(np.percentile(wl, 50)),
                "wall_p95": float(np.percentile(wl, 95)), "wall_max": float(max(wl)),
                "lock_wait_p95": float(np.percentile(lw, 95)),
                "infer_p50": float(np.percentile([r["infer_ms"] for r in sel], 50))}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
