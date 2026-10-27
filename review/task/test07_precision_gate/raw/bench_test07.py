"""TEST-07 精度门禁小样本实测（AC-067）。

隔离 TEMP 环境：train(14 normal)+train_anomaly(30) prepare 新品类
gold_finger_eval，独立 test 切分（10 normal+10 anomaly）评估。
冻结阈值（用户 2026-10-13）：FPR≤5%、Recall≥95%、gray≤15%。
Wilson 95% CI；样本不足时如实标注无法证实。
"""
import csv, hashlib, json, math, os, random, shutil, sqlite3, sys, tempfile, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test07_precision_gate")
DB = ROOT / "database" / "aoi.db"
THRESH = {"fpr_max": 0.05, "recall_min": 0.95, "gray_max": 0.15}
CATEGORY = "gold_finger_eval"


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    d = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - d) / den, (c + d) / den)


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    db = sqlite3.connect("file:" + str(DB) + "?mode=ro", uri=True)
    splits = {}
    for split in ("train", "train_anomaly", "test"):
        rows = db.execute(
            "SELECT id,path,label FROM images WHERE category='gold_finger' AND split=? ORDER BY id",
            (split,)).fetchall()
        splits[split] = [{"id": r[0], "path": r[1], "label": r[2]} for r in rows]
    db.close()
    manifest = {s: [{"id": r["id"], "path": r["path"], "label": r["label"],
                     "sha256": digest(r["path"])} for r in rows]
                for s, rows in splits.items()}
    (TASK / "raw" / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print({s: len(v) for s, v in manifest.items()}, flush=True)

    outdir = Path(tempfile.mkdtemp(prefix="aoi_test07_"))
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
    engine = get_engine()
    engine.learning_parallel = False
    service = get_detection_service()

    # guard_bundle 要求 train 域：复制到 TEMP/train
    def staged(rows):
        tdir = outdir / "train"
        tdir.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rows:
            dst = tdir / f"{r['id']}{Path(r['path']).suffix}"
            shutil.copy2(r["path"], dst)
            out.append(str(dst))
        return out

    normals = staged(manifest["train"])
    defects = staged(manifest["train_anomaly"])
    t0 = time.perf_counter()
    ver = engine.prepare(CATEGORY,
                         {"init_normal": normals, "init_defect": defects, "val": [], "test": []},
                         activate=True, progress_cb=lambda f, m: None)
    print(f"[prepare] v{ver} {time.perf_counter()-t0:.1f}s "
          f"({len(normals)}n+{len(defects)}a)", flush=True)

    rows = []
    test = manifest["test"]
    for i, r in enumerate(test):
        t1 = time.perf_counter()
        res = service.detect_image(r["path"], CATEGORY, persist=False, with_heatmap=False)
        row = {"id": r["id"], "path": r["path"], "label": r["label"],
               "decision": res["decision"], "score": round(float(res["final_score"]), 6),
               "threshold": res.get("threshold"), "gray_threshold": res.get("gray_threshold"),
               "model_version": res["model_version"],
               "wall_ms": round((time.perf_counter() - t1) * 1000, 1)}
        rows.append(row)
        print(f"  [{i+1}/{len(test)}] {r['label']:8s} -> {res['decision']:8s} "
              f"score={row['score']}", flush=True)

    with open(TASK / "raw" / "per_image_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    norm = [r for r in rows if r["label"] == "normal"]
    anom = [r for r in rows if r["label"] == "anomaly"]
    fp = sum(1 for r in norm if r["decision"] == "anomaly")
    tp = sum(1 for r in anom if r["decision"] == "anomaly")
    tp_gray = sum(1 for r in anom if r["decision"] in ("anomaly", "gray"))
    gray = sum(1 for r in rows if r["decision"] == "gray")
    fpr = fp / len(norm) if norm else None
    rec = tp / len(anom) if anom else None
    rec_g = tp_gray / len(anom) if anom else None
    gray_rate = gray / len(rows) if rows else None
    metrics = {
        "n_normal": len(norm), "n_anomaly": len(anom),
        "fpr": {"value": fpr, "wilson95": wilson(fp, len(norm)), "threshold": THRESH["fpr_max"],
                "point_pass": fpr is not None and fpr <= THRESH["fpr_max"]},
        "recall": {"value": rec, "wilson95": wilson(tp, len(anom)),
                   "threshold": THRESH["recall_min"],
                   "point_pass": rec is not None and rec >= THRESH["recall_min"]},
        "recall_incl_gray": rec_g,
        "gray_rate": {"value": gray_rate, "wilson95": wilson(gray, len(rows)),
                      "threshold": THRESH["gray_max"],
                      "point_pass": gray_rate is not None and gray_rate <= THRESH["gray_max"]},
        "thresholds_frozen": THRESH,
        "threshold_source": "用户定值 2026-10-13（SYNC-27 business_budget）",
        "model_version": ver, "category": CATEGORY,
        "config_hash": digest(str(cfgp)),
        "data_manifest_sha256": digest(str(TASK / "raw" / "split_manifest.json")),
        "temp": str(outdir),
        "note": "n=10/类过小，Wilson 区间必然很宽；点估计通过不等于门禁证实",
    }
    (TASK / "raw" / "eval_run.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
