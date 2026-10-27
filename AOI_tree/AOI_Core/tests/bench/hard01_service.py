"""HARD-01 isolated <=8-image engineering screen (not production acceptance).

python -m tests.bench.hard01_service --prior-result <TEMP/result.json>
Copies the prior TEMP snapshot and inputs; never writes source snapshots/DB.
No HTTP transport is measured. Nested times are inclusive and exclusive;
only exclusive times may be added. CUDA synchronization adds probe overhead.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import ExitStack
from concurrent.futures import Future
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Probe:
    def __init__(self):
        self.rows = {}
        self.stack = []
        self.thread = threading.get_ident()

    def reset(self):
        self.rows = {}

    def wrap(self, name, fn):
        def measured(*args, **kwargs):
            if threading.get_ident() != self.thread:
                return fn(*args, **kwargs)
            sync()
            start = time.perf_counter()
            frame = [name, 0.0]
            self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                sync()
                elapsed = (time.perf_counter() - start) * 1000
                self.stack.pop()
                key = "/".join([f[0] for f in self.stack] + [name])
                row = self.rows.setdefault(key, {"inclusive_ms": 0.0,
                                                 "exclusive_ms": 0.0, "calls": 0})
                row["inclusive_ms"] += elapsed
                row["exclusive_ms"] += elapsed - frame[1]
                row["calls"] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured

    def install(self, stack, obj, attr, name=None):
        if obj is not None and hasattr(obj, attr):
            stack.enter_context(patch.object(obj, attr,
                                self.wrap(name or attr, getattr(obj, attr))))


class TimedLock:
    def __init__(self, lock, probe):
        self.lock = lock
        self.probe = probe

    def __enter__(self):
        self.probe.wrap("category_lock_wait", self.lock.acquire)()
        return self

    def __exit__(self, *args):
        self.lock.release()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {"n": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": max(values),
            "mean": float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-result", required=True, type=Path)
    parser.add_argument("--legacy-display", action="store_true",
                        help="Measure the pre-optimization display channel order")
    parser.add_argument("--legacy-audit", action="store_true",
                        help="Measure original database audit entry, including cold Web import")
    args = parser.parse_args()
    prior = json.loads(args.prior_result.read_text(encoding="utf-8"))
    source = args.prior_result.parent.resolve()
    # Explicitly limited to the engineering inputs recorded by the prior probe.
    if not 1 <= len(prior["rows"]) <= 8:
        raise ValueError("Expected 1..8 prior inputs")
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not source.is_relative_to(temp_root):
        raise ValueError("Prior inputs/snapshots must be in TEMP")
    outdir = Path(tempfile.mkdtemp(prefix="aoi_hard01_service_"))
    print(f"[1/5] isolate inputs/snapshot/SQLite: {outdir}", flush=True)
    shutil.copytree(source / "storage" / "snapshots", outdir / "engine" / "snapshots")
    inputs = []
    for row in prior["rows"]:
        src = (source / row["input"]).resolve()
        if not src.is_relative_to(source):
            raise ValueError("Input outside source TEMP")
        dst = outdir / src.name
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            if im.size != (2500, 2500):
                raise ValueError("Input must be 2500x2500")
        inputs.append(dst)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"),
                          engine_storage=str(outdir / "engine"),
                          engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                          database_url="sqlite:///" + (outdir / "bench.db").as_posix())
    # Never contact a configured production webhook from an engineering screen.
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
    from backend.pipeline import service as svc_module
    from backend.db import database
    from backend.db.models import Detection, StatsDaily, OperationLog, Image as ImageRow
    from backend.core import alarm, datasets
    from sqlalchemy.orm import Session
    from algo import persist as persistence
    from algo.eval import offline
    from algo.decision import hierarchical
    import cv2

    print("[2/5] initialize isolated database; cold-load actual engine", flush=True)
    database.get_engine()
    engine = get_engine()
    probe = Probe()
    with ExitStack() as stack:
        if args.legacy_audit:
            stack.enter_context(patch.object(svc_module, "log_action", database.log_action))
        if args.legacy_display:
            from tests.smoke.test_display_order_equivalence import legacy_finish
            stack.enter_context(patch.object(svc_module.DetectionService,
                                             "_finish_detect", legacy_finish()))
        original_lock = engine._pipe_lock
        stack.enter_context(patch.object(engine, "_pipe_lock",
                            lambda cat: TimedLock(original_lock(cat), probe)))
        probe.install(stack, engine, "_ensure_loaded", "ensure_loaded")
        sync()
        t0 = time.perf_counter()
        pipe = engine.get_pipeline("gold_finger")
        sync()
        cold_ms = (time.perf_counter() - t0) * 1000
        cold_segments = {key: dict(value) for key, value in probe.rows.items()}
        print(f"  cold load {cold_ms:.3f}ms", flush=True)
        service = svc_module.get_detection_service()
        for attr in ("predict", "predict_frame", "_preprocess", "_get_item", "_frame_item",
                     "_record", "_finalize", "_hier", "_attribute_types", "_align_offset",
                     "_tta_refine", "_slot_heatmaps"):
            probe.install(stack, pipe, attr)
        probe.install(stack, offline, "load_image", "decode")
        probe.install(stack, engine, "_to_result")
        for attr in ("_finish_detect", "_persist", "_auto_archive", "_record_alarm"):
            probe.install(stack, service, attr)
        for attr in ("render_heatmap_overlay", "imwrite", "_best_heatmap",
                     "log_action", "notify_event", "get_alarm_monitor"):
            probe.install(stack, svc_module, attr)
        for attr in ("get_engine", "get_session", "_migrate_sqlite"):
            probe.install(stack, database, attr, "db_" + attr)
        for attr in ("flush", "commit", "get"):
            probe.install(stack, Session, attr, "session_" + attr)
        probe.install(stack, database.Base.metadata, "create_all", "schema_create_all")
        probe.install(stack, persistence, "load_snapshot", "load_snapshot")
        probe.install(stack, datasets, "get_or_create_dataset")
        probe.install(stack, alarm.AlarmMonitor, "record", "alarm_window_record")
        original_import = builtins.__import__
        def measured_import(name, *args, **kwargs):
            if name.endswith("security"):
                return probe.wrap("import_security", original_import)(name, *args, **kwargs)
            return original_import(name, *args, **kwargs)
        stack.enter_context(patch.object(builtins, "__import__", measured_import))
        for attr in ("build_heatmap_mask", "boxes_from_mask", "attribute_type"):
            probe.install(stack, hierarchical, attr)
        probe.install(stack, cv2, "resize", "resize")
        probe.install(stack, Future, "result", "future_wait")
        if pipe.handler is not None:
            probe.install(stack, pipe.handler.defect_bank, "intercept_score")
            probe.install(stack, pipe.handler.normal_bank, "_observe_decision")
        probe.install(stack, pipe.head_mgr, "boost", "head_boost")
        probe.install(stack, pipe.defect_clf, "predict", "classifier_predict")

        print("[3/5] warmup each actual entry (excluded; cache cleared)", flush=True)
        warmups = []
        for mode in ("engine_path", "service_heatmap_persist"):
            pipe._item_cache.clear()
            sync()
            t0 = time.perf_counter()
            if mode == "engine_path":
                engine.predict_image_path("gold_finger", str(inputs[0]))
            else:
                service.detect_image(inputs[0], "gold_finger", persist=True, with_heatmap=True)
            sync()
            warmups.append({"mode": mode, "wall_ms": (time.perf_counter() - t0) * 1000})
            print(f"  {mode}: {warmups[-1]['wall_ms']:.3f}ms", flush=True)

        print("[4/5] <=8 unique inputs, two actual entry measurements per input", flush=True)
        rows = []
        for index, path in enumerate(inputs):
            for mode in ("engine_path", "service_heatmap_persist"):
                print(f"  start {index + 1}/{len(inputs)} {mode} {path.name}", flush=True)
                pipe._item_cache.clear()
                probe.reset()
                sync()
                t0 = time.perf_counter()
                if mode == "engine_path":
                    result = engine.predict_image_path("gold_finger", str(path))
                    response = {"score": result.score, "decision": result.decision,
                                "model_version": result.model_version, "n_tiles": result.n_tiles}
                else:
                    response = service.detect_image(path, "gold_finger", persist=True,
                                                    with_heatmap=True)
                json_start = time.perf_counter()
                payload = json.dumps(response, ensure_ascii=False, allow_nan=False)
                json_ms = (time.perf_counter() - json_start) * 1000
                sync()
                total_ms = (time.perf_counter() - t0) * 1000
                rows.append({"input": path.name, "input_sha256": digest(path), "mode": mode,
                             "total_wall_ms": total_ms, "json_ms": json_ms,
                             "json_bytes": len(payload.encode("utf-8")),
                             "decision": response["decision"],
                             "score": response.get("score", response.get("final_score")),
                             "model_version": response["model_version"],
                             "reported_latency_ms": response.get("latency_ms"),
                             "reported_total_latency_ms": response.get("total_latency_ms"),
                             "reported_e2e_ms": response.get("latency_e2e_ms"),
                             "segments": probe.rows})
                if mode.startswith("service"):
                    assert response["persisted"] and response["detection_id"]
                    assert Path(response["heatmap_path"]).is_file()
                    assert Path(response["overlay_path"]).is_file()
                print(f"  done wall={total_ms:.3f}ms JSON={json_ms:.3f}ms", flush=True)
        with database.session_scope() as session:
            count = session.query(Detection).count()
            daily = session.query(StatsDaily).first()
            assert count == len(inputs) + 1
            assert daily.n_inspected == count
            audit_rows = [{"user": r.user, "action": r.action, "detail": r.detail,
                           "extra": r.extra} for r in session.query(OperationLog).all()]
            assert any(r["action"] == "alarm" for r in audit_rows)
        summary = {}
        for mode in ("engine_path", "service_heatmap_persist"):
            subset = [r for r in rows if r["mode"] == mode]
            summary[mode] = {"total_wall_ms": stats([r["total_wall_ms"] for r in subset]),
                             "json_ms": stats([r["json_ms"] for r in subset])}
            keys = sorted({k for r in subset for k in r["segments"]})
            summary[mode]["segments"] = {
                k: {field: stats([r["segments"].get(k, {}).get(field, 0.0) for r in subset])
                    for field in ("inclusive_ms", "exclusive_ms")} for k in keys}
        snapshot = outdir / "engine" / "snapshots" / "gold_finger" / "v1"
        report = {"manifest": {"display_order": "legacy" if args.legacy_display else "resize_before_color",
                  "audit_entry": "legacy_database" if args.legacy_audit else "lightweight_context",
                  "security_imported_at_end": "backend.core.security" in sys.modules,
                  "formal_acceptance": False, "native_2500_available": False,
                  "accuracy_gate": "still failed; not retested by positive-only screen",
                  "http_included": False, "concurrency_included": False,
                  "scope": "actual engine path + actual service heatmap/persist/archive/alarm + JSON; no HTTP",
                  "timing": "nested inclusive/exclusive; do not sum ancestors and descendants",
                  "cache": "cleared each call; OS page cache not flushed", "seed": 42,
                  "gpu_sync": "section boundaries, includes instrumentation overhead",
                  "python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
                  "overlay_max_dim": cfg["pipeline"]["overlay_max_dim"],
                  "archive_enabled": cfg.get("archive", {}).get("enabled", True),
                  "webhook": "disabled in isolated config",
                  "config_sha256": digest(config_path), "snapshot_cfg_sha256": digest(snapshot / "cfg.yaml"),
                  "prior_result_sha256": digest(args.prior_result)},
                  "temp": str(outdir), "cold_load_ms": cold_ms, "cold_segments": cold_segments,
                  "warmups": warmups, "rows": rows, "stats": summary,
                  "persisted_count_including_warmup": count, "audit_rows": audit_rows}
        output = outdir / "result.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[5/5] complete; verified SQLite rows and heatmap files", flush=True)
        for mode, data in summary.items():
            print(f"  {mode}: {data['total_wall_ms']}", flush=True)
        print(f"RESULT={output}", flush=True)


if __name__ == "__main__":
    main()
