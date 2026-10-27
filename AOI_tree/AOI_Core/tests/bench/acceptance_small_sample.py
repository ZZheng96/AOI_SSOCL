"""R01-R12/R18 小样本验收证据生成器。

本脚本只做可复现的机制级验收，并在存在真实快照/图片时报告候选链路是否可运行；
合成特征结果明确标为 mechanism，不宣称生产增益、GT 指标或真实泛化。
输出固定写入 d:/CGAIC，临时文件仅使用系统 TEMP。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from algo.ssocl.banks import DefectBank, NormalBank
from algo.ssocl.incubate import HeadManager

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("d:/CGAIC")
D, G = 16, 6


def _tile(rng, center, noise=0.03):
    return (center[None, :, None, None] + noise * rng.normal(size=(1, D, G, G))).astype(np.float32)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def normalbank_experiment():
    rng = np.random.default_rng(20261015)
    core_seed = rng.normal(size=(4, D, G, G)).astype(np.float32)
    bank = NormalBank(cluster_k=3, sim_thresh=0.75, defect_ratio_thresh=0.30)
    bank.seed_core(core_seed, rng=np.random.default_rng(1))
    core_before = bank.lineage()["core"]["hash"]

    for _ in range(7):
        bank._observe_decision(True)
    blocked_after_anomalies = bool(bank.fuse_blocked)
    far = rng.normal(size=(1, D, G, G)).astype(np.float32)
    rejected = bank.reflow_candidate(far, image_id="contaminated-defect")
    ext_before = len(bank.ext)

    bank.reset_stream_stats()
    near_results = []
    center = core_seed.mean(axis=(0, 2, 3))
    for i in range(3):
        near_results.append(bank.reflow_candidate(
            _tile(np.random.default_rng(100 + i), center), image_id=f"normal-{i}"
        ))
    core_after = bank.lineage()["core"]["hash"]
    return {
        "kind": "mechanism",
        "fuse_threshold": bank.defect_ratio_thresh,
        "defect_ratio_after_7_anomalies": 0.7,
        "fuse_blocked_after_anomalies": blocked_after_anomalies,
        "contaminated_far_candidate": rejected,
        "normal_cluster_results": near_results,
        "core_unchanged": core_before == core_after,
        "ext_unchanged_by_reflow": len(bank.ext) == ext_before,
        "lineage_has_protected_core": bool(bank.lineage()["core"]["protected"]),
        "pass": blocked_after_anomalies and rejected == "rejected" and core_before == core_after,
    }


def typed_head_experiment():
    rng = np.random.default_rng(42)
    core = rng.normal(size=(4, D, G, G)).astype(np.float32)
    centers = {"scratch": rng.normal(size=D) * 3, "dent": rng.normal(size=D) * 3}
    db = DefectBank()
    boxes = ([0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0])
    for dtype, seed0 in (("scratch", 100), ("dent", 200)):
        for i in range(4):
            for box in boxes:
                db.add_box(_tile(np.random.default_rng(seed0 + i), centers[dtype]), box,
                           image_id=f"{dtype}-{i}.png", defect_type=dtype)
    hm = HeadManager({"min_samples": 4, "holdout_images": 2,
                      "approve_min_sep": 0.2, "per_type": True})
    generic = hm.maybe_incubate(db, core)
    typed = hm.type_scores(_tile(np.random.default_rng(999), centers["scratch"]))
    status = hm.status(db)
    typed_logs = [e for e in hm.log if "defect_type" in e]
    pass_checks = {
        "generic_head_trained": generic is not None and hm.head is not None,
        "two_independent_type_heads": set(hm.heads) == {"scratch", "dent"},
        "typed_logs_approved": bool(typed_logs) and all(e["approved"] for e in typed_logs),
        "typed_scores_reported": set(typed) == {"scratch", "dent"},
        "type_heads_do_not_change_boost": hm.boost(_tile(np.random.default_rng(1000), centers["scratch"]))
        == hm.head.score(_tile(np.random.default_rng(1000), centers["scratch"])),
    }
    return {
        "kind": "mechanism",
        "sample_images_per_type": 4,
        "status": status,
        "typed_log_count": len(typed_logs),
        "typed_scores": typed,
        "checks": pass_checks,
        "pass": all(pass_checks.values()),
    }


def real_data_probe():
    candidates = [
        ROOT / "storage" / "archive" / "gold_finger" / "20261011",
        ROOT / "data_local",
        ROOT / "data",
    ]
    found = []
    for base in candidates:
        if base.is_dir():
            found.extend(str(p) for p in sorted(base.rglob("*.jpg"))[:8])
            found.extend(str(p) for p in sorted(base.rglob("*.png"))[:8])
            if found:
                break
    return {"kind": "probe", "real_images_found": len(found), "paths": found,
            "formal_real_image_experiment_ready": bool(found)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_ROOT / "acceptance_small_sample.json")
    args = parser.parse_args()
    start = time.perf_counter()
    print("[1/4] 运行 NormalBank 污染/熔断机制验收", flush=True)
    normal = normalbank_experiment()
    print(f"  pass={normal['pass']} rejected={normal['contaminated_far_candidate']}", flush=True)
    print("[2/4] 运行 SYNC-34 类型孵育头机制验收", flush=True)
    typed = typed_head_experiment()
    print(f"  pass={typed['pass']} heads={typed['status']['per_type']}", flush=True)
    print("[3/4] 探测真实图片（不生成、不伪造 GT）", flush=True)
    probe = real_data_probe()
    print(f"  real_images={probe['real_images_found']}", flush=True)
    print("[4/4] 写出证据 JSON", flush=True)
    report = {
        "schema": "acceptance-small-sample-v1",
        "generated_at": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "temp_root": tempfile.gettempdir(),
        "evidence_boundary": "mechanism results are not production gain, GT, recall, AUROC, or acceptance metrics",
        "R01": {"status": "blocked", "reason": "no validated native manifest/snapshot was found by this run", "real_probe": probe},
        "R02": {"status": "blocked", "reason": "no confirmed real batch/board grouping"},
        "R03": {"status": "blocked", "reason": "no confirmed real dataset and slot labels"},
        "R04": {"status": "blocked", "reason": "no validated template-pair dataset"},
        "R05": {"status": "blocked", "reason": "no pixel/box GT supplied"},
        "R06": {"status": "blocked", "reason": "fixed production threshold acceptance set not frozen"},
        "R07": {"status": "blocked", "reason": "native service run requires validated snapshot and inputs"},
        "R08": {"status": "blocked", "reason": "no independent cross-domain set confirmed"},
        "R09": {"status": "blocked", "reason": "no real evaluation set confirmed"},
        "R10": {"status": "partial", "evidence": normal, "reason": "mechanism-only pollution guard evidence"},
        "R11": {"status": "blocked", "reason": "no E_open set confirmed"},
        "R12": {"status": "partial", "evidence": normal, "reason": "mechanism-only NormalBank contamination/fuse evidence"},
        "R18": {"status": "partial", "evidence": typed, "reason": "SYNC-34 mechanism acceptance; not production benefit"},
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "repro_command": "python -m tests.bench.acceptance_small_sample",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{args.output}", flush=True)


if __name__ == "__main__":
    main()
