"""Isolated complete-service benchmark; never resize inputs or modify source assets.

Native input manifest: {"native": true, "images": [{"path": "...",
"sha256": "...", "source_size": [2500, 2500]}]}.
Run direct and HTTP in separate processes. --engineering-prior accepts the
existing TEMP screen only; its results can NEVER qualify as native acceptance.
Optional learning events: [{"path": "...", "verdict": "wrong", "label": 1}].
--learning-path feedback (default) calls actual submit_feedback on the isolated
working state; candidate calls learn_candidate per event (parent = current
active version) and then one consolidate(activate=False) after all events.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import requests
import torch
import yaml

from .hard01_service import ROOT, digest, sync


def stats(values):
    if not values:
        return {}
    return {"n": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": max(values),
            "mean": float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", type=Path)
    source.add_argument("--engineering-prior", type=Path)
    parser.add_argument("--snapshot-root", type=Path,
                        help="Snapshot root containing category/current.json; copied read-only")
    parser.add_argument("--category", default="gold_finger")
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--learning-events", type=Path)
    parser.add_argument("--learning-path", choices=("feedback", "candidate"), default="feedback")
    parser.add_argument("--learning-parallel", action="store_true",
                        help="learning runs concurrently with inference (default: serial, learning yields)")
    parser.add_argument("--candidate-upgrade-parent", action="store_true",
                        help="Untimed: re-save the isolated active legacy snapshot via "
                             "consolidate(activate=True) so learn_candidate can load it")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0:
        parser.error("rounds >= 1 and warmup >= 0 required")
    native = args.input_manifest is not None
    provenance_path = args.input_manifest or args.engineering_prior
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if native:
        if provenance.get("native") is not True or not args.snapshot_root:
            parser.error("Native manifest must declare native=true and provide --snapshot-root")
        records = provenance["images"]
        snapshot_root = args.snapshot_root.resolve()
    else:
        prior_dir = args.engineering_prior.parent.resolve()
        if not prior_dir.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            parser.error("Engineering prior must be in TEMP")
        records = [{"path": str(prior_dir / row["input"])} for row in provenance["rows"]]
        snapshot_root = prior_dir / "storage" / "snapshots"
        if not snapshot_root.is_dir():
            snapshot_root = prior_dir / "engine" / "snapshots"
    records = list({str(row["path"]): row for row in records}.values())
    if not records or len(records) > 8:
        parser.error("Engineering tuning is limited to 1..8 unique inputs; formal sample not frozen")
    for row in records:
        path = Path(row["path"]).resolve()
        with Image.open(path) as image:
            if image.size != (2500, 2500):
                raise ValueError(f"Not 2500x2500: {path}")
        if native and (row.get("source_size") != [2500, 2500]
                       or row.get("sha256") != digest(path)):
            raise ValueError(f"Native provenance/hash mismatch: {path}")
        row["path"] = str(path)
    outdir = Path(tempfile.mkdtemp(prefix="aoi_native_fullchain_"))
    print(f"[1/5] isolate inputs/snapshots/SQLite: {outdir}", flush=True)
    shutil.copytree(snapshot_root / args.category,
                    outdir / "engine" / "snapshots" / args.category)
    inputs = []
    for index, row in enumerate(records):
        dest = outdir / f"input_{index:02d}{Path(row['path']).suffix}"
        shutil.copy2(row["path"], dest)
        inputs.append(dest)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"),
                         engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                         host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    config_path = outdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(config_path)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    from backend.engine import get_engine
    from backend.pipeline.service import get_detection_service
    from backend.db import database
    from backend.db.models import Detection, StatsDaily, OperationLog, Model
    import cv2

    engine = get_engine()
    engine.learning_parallel = args.learning_parallel
    service = get_detection_service()
    rows, learning_rows = [], []
    learning_failures = []
    server = None
    server_thread = None
    client = requests.Session()
    with ExitStack() as stack:
        # predict_frame normally bypasses path cache. Assert/clear additionally at
        # the entry, under the same reentrant production category lock.
        original_predict = engine.predict_ndarray
        lock_waits = {}
        def uncached_predict(category, rgb, path_hint=None):
            wait_start = time.perf_counter()
            # 与生产 predict_ndarray 同序：先标记前台再等锁，串行模式下学习据此让行。
            with engine._foreground(), engine._pipe_lock(category):
                lock_waits[str(path_hint)] = (time.perf_counter() - wait_start) * 1000
                engine.get_pipeline(category)._item_cache.clear()
                return original_predict(category, rgb, path_hint=path_hint)
        stack.enter_context(patch.object(engine, "predict_ndarray", uncached_predict))
        if args.mode == "http":
            # Real startup governs orphan snapshots. Register the isolated copy
            # as an explicitly imported asset, exactly as snapshot import does.
            with database.session_scope() as session:
                for version in (outdir / "engine" / "snapshots" / args.category).glob("v*"):
                    if version.is_dir():
                        session.add(Model(name=f"bench-{args.category}-{version.name}",
                                          category=args.category, version=version.name,
                                          path=str(version), format="engine", origin="uploaded"))
            import uvicorn
            from backend.api.app import create_app
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            port = sock.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(create_app(), log_level="warning"))
            server_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            server_thread.start()
            deadline = time.monotonic() + 30
            while not server.started:
                if not server_thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("Isolated HTTP startup failed")
                time.sleep(0.1)
            base = f"http://127.0.0.1:{port}"
        print("[2/5] first complete request: cold model/first image, separately recorded", flush=True)
        def detect(path, phase, index):
            sync()
            start = time.perf_counter()
            if args.mode == "direct":
                response = service.detect_image(path, args.category, persist=True, with_heatmap=True)
                payload = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
            else:
                key = os.environ.get("AOI_API_KEY", "")
                request = client.post(base + "/api/detect/image",
                                      json={"path": str(path), "category": args.category,
                                            "with_heatmap": True},
                                      headers={"X-API-Key": key} if key else {}, timeout=180)
                if request.status_code != 200:
                    raise RuntimeError(f"HTTP {request.status_code}: {request.text[:1000]}")
                payload = request.content
                response = request.json()
            sync()
            end = time.perf_counter()
            row = {"phase": phase, "index": index, "input": path.name,
                   "start": start, "end": end, "wall_ms": (end - start) * 1000,
                   "json_bytes": len(payload), "score": response["final_score"],
                   "decision": response["decision"], "model_version": response["model_version"],
                   "reported_model_ms": response["latency_ms"],
                   "reported_partial_e2e_ms": response["latency_e2e_ms"],
                   "category_lock_wait_ms": lock_waits.get(str(path)),
                   "http_transport_ms": None}
            assert response["persisted"] and response["detection_id"]
            assert "alarm" in response
            for field in ("heatmap_path", "overlay_path"):
                assert Path(response[field]).is_file(), field
            rows.append(row)
            print(f"  {phase} {index}: {row['wall_ms']:.3f}ms {response['decision']}", flush=True)
        try:
            detect(inputs[0], "cold_first_request", 0)
            print(f"[3/5] warmup {args.warmup} complete requests (excluded)", flush=True)
            for index in range(args.warmup):
                detect(inputs[index % len(inputs)], "warmup", index)
            events = []
            if args.learning_events:
                events = json.loads(args.learning_events.read_text(encoding="utf-8"))
                if not events or len(events) > 16:
                    raise ValueError("Expected 1..16 explicitly labeled learning events")
                for index, event in enumerate(events):
                    if event.get("label") not in (0, 1) or event.get("verdict") not in ("wrong", "correct"):
                        raise ValueError("Explicit verdict and binary label required")
                    src = Path(event["path"])
                    dest = outdir / f"learning_{index:02d}{src.suffix}"
                    shutil.copy2(src, dest)
                    event["path"] = str(dest)
            def timed(kind, label, fn):
                # Synchronous call in a worker; heartbeat every 5s from here.
                start = time.perf_counter()
                done = threading.Event()
                outcome = {}
                def run():
                    try:
                        result = fn()
                        if isinstance(result, dict):  # entry may hold non-JSON values
                            result = {key: result.get(key) for key in
                                      ("version_num", "parent_version", "event_id",
                                       "candidate_state", "activated", "acceptance")}
                        outcome["result"] = result
                        outcome["error"] = None
                    except BaseException as exc:
                        outcome["result"] = None
                        outcome["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        done.set()
                print(f"  {label} start", flush=True)
                threading.Thread(target=run, daemon=True).start()
                while not done.wait(5):
                    print(f"  {label} still running ({time.perf_counter()-start:.1f}s)", flush=True)
                end = time.perf_counter()
                learning_rows.append({"kind": kind, "label": label, "start": start, "end": end,
                                      "wall_ms": (end - start) * 1000, **outcome})
                if outcome["error"]:
                    learning_failures.append(f"{label}: {outcome['error']}")
                print(f"  {label} done {(end-start)*1000:.1f}ms: {outcome['error'] or 'OK'}", flush=True)
            def learn_candidate_path():
                parent = engine.current_version(args.category)
                for index, event in enumerate(events):
                    timed("candidate", f"candidate {index + 1}/{len(events)}",
                          lambda event=event, index=index: engine.learn_candidate(
                              args.category, event["path"], verdict=event["verdict"],
                              label=event["label"], event_id=f"bench-{index}",
                              parent_version=parent))
                timed("consolidate", "consolidate(activate=False)",
                      lambda: engine.consolidate(args.category, note="bench candidate",
                                                 activate=False))
            def learn():
                if args.learning_path == "candidate":
                    return learn_candidate_path()
                for index, event in enumerate(events):
                    start = time.perf_counter()
                    completed = threading.Event()
                    outcome = {}
                    def callback(result, error):
                        outcome.update(result=result, error=str(error) if error else None)
                        completed.set()
                    print(f"  learning start {index + 1}/{len(events)}", flush=True)
                    engine.submit_feedback(args.category, event["path"], event["verdict"],
                                           label=event["label"], on_result=callback)
                    while not completed.wait(5):
                        print(f"  learning {index + 1} still running ({time.perf_counter()-start:.1f}s)", flush=True)
                    learning_rows.append({"start": start, "end": time.perf_counter(), **outcome})
                    if outcome["error"]:
                        learning_failures.append(outcome["error"])
                    print(f"  learning done {index + 1}: {outcome['error'] or 'OK'}", flush=True)
            parent_upgrade = None
            if events and args.learning_path == "candidate" and args.candidate_upgrade_parent:
                before = engine.current_version(args.category)
                upgrade_start = time.perf_counter()
                after = engine.consolidate(args.category, note="bench parent format upgrade",
                                           activate=True)
                parent_upgrade = {"from": before, "to": after,
                                  "wall_ms": (time.perf_counter() - upgrade_start) * 1000}
                print(f"  untimed parent upgrade v{before} -> v{after}", flush=True)
            print("[4/5] timed complete production service; feature cache disabled", flush=True)
            with ThreadPoolExecutor(max_workers=1) as pool:
                learning = pool.submit(learn) if events else None
                for rnd in range(args.rounds):
                    for index, path in enumerate(inputs):
                        detect(path, "hot", rnd * len(inputs) + index)
                if learning:
                    learning.result()
            with database.session_scope() as session:
                count = session.query(Detection).count()
                daily = session.query(StatsDaily).first()
                assert count == len(rows) and daily.n_inspected == count
                alarm_count = session.query(OperationLog).filter(OperationLog.action == "alarm").count()
            hot = [row for row in rows if row["phase"] == "hot"]
            for row in hot:
                row["learning_overlap"] = any(row["start"] < event["end"] and row["end"] > event["start"]
                                              for event in learning_rows)
            snapshot_hashes = {str(path.relative_to(outdir / "engine" / "snapshots")): digest(path)
                               for path in sorted((outdir / "engine" / "snapshots").rglob("*")) if path.is_file()}
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                  "--format=csv,noheader"], text=True, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                commit, driver = None, None
            report = {"manifest": {"native_2500": native, "formal_acceptance": False,
                       "reason": "Formal sample/precision gates not frozen; engineering inputs never qualify",
                       "mode": args.mode, "category": args.category, "seed": 42,
                       "boundary": "input path request through decode/inference/fusion/decision/heatmap/DB/archive/alarm/JSON return",
                       "http_transport_queue_included": args.mode == "http",
                       "cache": "feature cache disabled; OS page cache not flushed/unknown",
                       "percentile_method": "numpy linear interpolation; small sample exploratory only",
                       "gpu_sync": "outer request boundaries only; no nested profiling in wall measurement",
                       "python": platform.python_version(), "platform": platform.platform(),
                       "processor": platform.processor(), "torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       "driver": driver, "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
                       "commit": commit, "config_sha256": digest(config_path),
                       "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                           [Path(__file__), ROOT / "backend/pipeline/service.py", ROOT / "backend/engine/__init__.py",
                            ROOT / "algo/eval/offline.py"]},
                       "snapshot_hashes": snapshot_hashes, "provenance_sha256": digest(provenance_path),
                       "inputs": [{**record, "isolated_sha256": digest(path)} for record, path in zip(records, inputs)],
                       "config": {key: value for key, value in cfg.items() if key != "security"},
                       "warmup": args.warmup, "rounds": args.rounds,
                       "webhook": "disabled; synchronous production dispatch still called",
                       "learning_path": args.learning_path if args.learning_events else None,
                       "candidate_parent_upgrade": parent_upgrade,
                       "learning_parallel": args.learning_parallel,
                       "learning_events_sha256": digest(args.learning_events) if args.learning_events else None},
                      "temp": str(outdir), "rows": rows, "learning": learning_rows,
                      "learning_errors": learning_failures, "persisted_count": count, "alarm_audit_count": alarm_count,
                      "stats": {"hot": stats([row["wall_ms"] for row in hot]),
                                "learning_overlap": stats([row["wall_ms"] for row in hot if row["learning_overlap"]]),
                                "idle": stats([row["wall_ms"] for row in hot if not row["learning_overlap"]]),
                                "learning_duration_ms": {kind: stats([(event["end"] - event["start"]) * 1000
                                                                      for event in learning_rows
                                                                      if event.get("kind", "feedback") == kind])
                                                         for kind in sorted({event.get("kind", "feedback")
                                                                             for event in learning_rows})},
                                "category_lock_wait_ms": stats([row["category_lock_wait_ms"] for row in hot
                                                                if row["category_lock_wait_ms"] is not None])}}
            report["timing_gate_only"] = report["stats"]["hot"]["p95"] <= 200 and not learning_failures
            output = outdir / "result.json"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            print(f"[5/5] RESULT={output}\n{json.dumps(report['stats'], indent=2)}", flush=True)
            print("G1 NOT ACCEPTED (see manifest and native/accuracy/sample/competition gates)", flush=True)
        finally:
            client.close()
            if server:
                server.should_exit = True
                server_thread.join(timeout=15)


if __name__ == "__main__":
    main()
