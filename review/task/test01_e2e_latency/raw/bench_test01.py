"""TEST-01 原生 2500x2500 端到端耗时测量（直接调用 / HTTP 两种入口）。

隔离方式：复制 stable 版 gold_finger 快照到 TEMP，AOI_CONFIG 指向 TEMP
storage/engine/SQLite，不触碰稳定版 DB 与快照。每个进程第一次请求记为冷启动。
负载两种：idle（空闲）；learning（submit_feedback 后台学习事件进行中，默认
串行让行 learning_parallel=False，与生产默认一致）。
"""
import argparse, csv, hashlib, json, os, random, shutil, socket, sys, tempfile, threading, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("direct", "http"), required=True)
    ap.add_argument("--manifest", type=Path, default=TASK / "raw" / "images.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--learning-events", type=int, default=2)
    ap.add_argument("--prepare-n", type=int, default=0,
                    help=">0 时在隔离环境中用归档正常图重新 prepare（旧快照缺 layout 统计无法加载）")
    ap.add_argument("--prepare-seed", type=int, default=123)
    args = ap.parse_args()

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert man.get("native") is True
    records = man["images"]
    for row in records:
        assert row["sha256"] == digest(row["path"]), row["path"]

    outdir = Path(tempfile.mkdtemp(prefix="aoi_test01_"))
    print(f"[1/4] isolate snapshot/storage/SQLite -> {outdir}", flush=True)
    if not args.prepare_n:
        shutil.copytree(ROOT / "storage" / "engine" / "snapshots" / "gold_finger",
                        outdir / "engine" / "snapshots" / "gold_finger")
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

    from backend.pipeline.service import get_detection_service
    from backend.engine import get_engine

    service = get_detection_service()
    engine = get_engine()
    engine.learning_parallel = False  # 生产默认：学习让行前台

    if args.prepare_n:
        # 旧生产快照缺 layout 拟合统计无法被新代码加载（详见 result.md），
        # 在隔离环境中用归档正常图重新 prepare（与测试样本不同种子，避免泄漏）。
        test_set = {r["path"] for r in records}
        from PIL import Image as _Img
        pool = []
        for p in sorted((ROOT / "storage" / "archive" / "gold_finger" / "20261011").iterdir()):
            if p.suffix.lower() not in (".jpg", ".png") or str(p) in test_set:
                continue
            with _Img.open(p) as _im:
                if _im.size == (2500, 2500):
                    pool.append(str(p))
        random.seed(args.prepare_seed)
        random.shuffle(pool)
        normals = []
        train_dir = outdir / "train"  # guard_bundle 要求 init_normal 位于 train 域
        train_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(pool[:args.prepare_n]):
            dst = train_dir / f"normal_{i:03d}{Path(src).suffix}"
            shutil.copy2(src, dst)
            normals.append(str(dst))
        print(f"[prepare] {len(normals)} normal images (seed={args.prepare_seed})", flush=True)
        t0 = time.perf_counter()
        ver = engine.prepare("gold_finger",
                             {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                             activate=True,
                             progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}",
                                                            flush=True))
        print(f"[prepare] done v{ver} in {time.perf_counter()-t0:.1f}s", flush=True)

    base = None
    server = server_thread = None
    client = None
    if args.mode == "http":
        from backend.db import database
        from backend.db.models import Model
        import uvicorn
        from backend.api.app import create_app
        with database.session_scope() as session:
            for v in (outdir / "engine" / "snapshots" / "gold_finger").glob("v*"):
                if v.is_dir():
                    session.add(Model(name=f"bench-gold_finger-{v.name}", category="gold_finger",
                                      version=v.name, path=str(v), format="engine", origin="uploaded"))
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(128)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
        server_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        server_thread.start()
        deadline = time.monotonic() + 30
        while not server.started:
            if not server_thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("isolated HTTP startup failed")
            time.sleep(0.1)
        base = f"http://127.0.0.1:{port}"
        import requests
        client = requests.Session()
        print(f"[http] isolated server on {base}", flush=True)

    rows = []

    def detect(path, phase, load, idx):
        start = time.perf_counter()
        if args.mode == "direct":
            r = service.detect_image(path, "gold_finger", persist=True, with_heatmap=True)
        else:
            resp = client.post(base + "/api/detect/image",
                               json={"path": str(path), "category": "gold_finger",
                                     "with_heatmap": True}, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            r = resp.json()
        wall = (time.perf_counter() - start) * 1000
        infer = float(r["latency_ms"])
        total = float(r.get("total_latency_ms") or infer)
        e2e = float(r.get("latency_e2e_ms") or total)
        row = {"image": Path(path).name, "entry": args.mode, "load": load, "phase": phase,
               "decode_ms": round(total - infer, 3), "infer_ms": round(infer, 3),
               "render_ms": round(e2e - total, 3), "persist_ms": round(max(wall - e2e, 0.0), 3),
               "e2e_ms": round(e2e, 3), "wall_ms": round(wall, 3),
               "decision": r["decision"], "model_version": r["model_version"]}
        rows.append(row)
        print(f"  {phase}/{load} {idx}: wall={wall:.1f}ms infer={infer:.1f}ms {r['decision']}",
              flush=True)

    inputs = [r["path"] for r in records]
    print("[2/4] cold first request", flush=True)
    detect(inputs[0], "cold", "idle", 0)
    print(f"[3/4] warmup x{args.warmup} (excluded from hot stats)", flush=True)
    for i in range(args.warmup):
        detect(inputs[(i + 1) % len(inputs)], "warmup", "idle", i)

    print(f"[4/4] hot idle {args.rounds}x{len(inputs)}", flush=True)
    for rnd in range(args.rounds):
        for i, p in enumerate(inputs):
            detect(p, "hot", "idle", rnd * len(inputs) + i)

    # 后台学习负载：提交反馈事件（TEMP 工作态，可安全修改），同时跑一轮热态
    n_ev = args.learning_events
    if n_ev > 0:
        print(f"[4/4b] hot with background learning x{n_ev} events", flush=True)
        learn_done = threading.Event()
        learn_log = []

        def learn():
            for i in range(n_ev):
                ev_path = inputs[(i * 7 + 3) % len(inputs)]
                done = threading.Event()
                out = {}
                t0 = time.perf_counter()
                engine.submit_feedback("gold_finger", ev_path, "correct", label=0,
                                       on_result=lambda res, err: (out.update(result=res, error=err),
                                                                   done.set()))
                while not done.wait(5):
                    print(f"  learning {i+1}/{n_ev} running {time.perf_counter()-t0:.1f}s", flush=True)
                learn_log.append({"event": i, "wall_ms": (time.perf_counter() - t0) * 1000,
                                  "error": str(out.get("error")) if out.get("error") else None})
                print(f"  learning {i+1}/{n_ev} done {(time.perf_counter()-t0)*1000:.1f}ms "
                      f"{out.get('error') or 'OK'}", flush=True)
            learn_done.set()

        th = threading.Thread(target=learn, daemon=True)
        th.start()
        time.sleep(0.2)  # 让学习先进入运行态
        i = 0
        while not learn_done.is_set() and i < len(inputs) * 2:
            detect(inputs[i % len(inputs)], "hot", "learning", i)
            i += 1
        th.join(timeout=600)
        (TASK / "raw" / f"learning_{args.mode}.json").write_text(
            json.dumps(learn_log, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = TASK / "raw" / f"per_image_{args.mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import numpy as np
    def pct(vals):
        if not vals:
            return {}
        return {"n": len(vals), "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99)),
                "max": float(max(vals)), "mean": float(np.mean(vals))}
    summary = {"mode": args.mode, "isolated_temp": str(outdir), "n_images": len(inputs),
               "rounds": args.rounds, "warmup": args.warmup, "seed": man["seed"],
               "cold_wall_ms": rows[0]["wall_ms"],
               "hot_idle_wall": pct([r["wall_ms"] for r in rows if r["phase"] == "hot" and r["load"] == "idle"]),
               "hot_learning_wall": pct([r["wall_ms"] for r in rows if r["phase"] == "hot" and r["load"] == "learning"]),
               "hot_idle_infer": pct([r["infer_ms"] for r in rows if r["phase"] == "hot" and r["load"] == "idle"]),
               "hot_idle_e2e": pct([r["e2e_ms"] for r in rows if r["phase"] == "hot" and r["load"] == "idle"]),
               "http_transport_included": args.mode == "http"}
    sp = TASK / "raw" / f"summary_{args.mode}.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if client:
        client.close()
    if server:
        server.should_exit = True
        server_thread.join(timeout=15)


if __name__ == "__main__":
    main()
