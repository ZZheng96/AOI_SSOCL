"""Round3 精度根因最小复现（开发版 AOI_tree，TEMP 隔离，小子集）。

用法: python repro_accuracy.py <mode> [--budget-s 900]
  mode=baseline : TEST-07 原切分（train=14 张 0598-0621 repaired.png；test normal=Image_2026* 批）
  mode=domain   : 训练正常覆盖 test 采集批（Image_2026 前 5 张 + 1289-1307 repaired），
                  test 留出同批的其余正常 + 1308-1334 缺陷（不含其 repaired 配对，防泄漏）
输出: raw/<mode>_per_image.json（逐图 raw/slot/fused/boost/decision）+ raw/<mode>_fit.json
"""
import json, os, shutil, sys, tempfile, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\round3_accuracy")
DATA = Path(r"D:\CGAIC\data_local\gold_finger")
MANIFEST = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test07_precision_gate\raw\split_manifest.json")
CATEGORY = "gold_finger_r3"


def splits(mode):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p = lambda rows: [r["path"] for r in rows]
    if mode == "baseline":
        return p(m["train"]), p(m["train_anomaly"]), \
            [(r["path"], 0 if r["label"] == "normal" else 1) for r in m["test"]]
    # domain：按采集批覆盖训练正常
    good = sorted((DATA / "train" / "good").iterdir())
    img26 = sorted((DATA / "test" / "good").glob("Image_2026*.jpg"))
    tfov = sorted((DATA / "test" / "good").glob("t_fov*.jpg"))
    train_img26 = [g for g in good if g.name.startswith("Image_2026")]
    rep_13xx = [g for g in good if g.name[:4].isdigit() and 1289 <= int(g.name[:4]) <= 1307]
    fov_train = [g for g in good if g.name.startswith("fov_")]
    normals = [str(x) for x in p(m["train"])] + [str(g) for g in train_img26 + rep_13xx + fov_train]
    test_def = [f for f in sorted((DATA / "test" / "defect").glob("*.jpg"))
                if f.stem.isdigit() and 1308 <= int(f.stem) <= 1334][:20]
    test = [(str(f), 0) for f in img26 + tfov] + [(str(f), 1) for f in test_def]
    return normals, p(m["train_anomaly"]), test


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    budget = float(sys.argv[sys.argv.index("--budget-s") + 1]) if "--budget-s" in sys.argv else 900.0
    T0 = time.perf_counter()
    normals, defects, test = splits(mode)
    print(f"[{mode}] train_normal={len(normals)} train_defect={len(defects)} "
          f"test={len(test)} (n={sum(1 for _, l in test if l == 0)}, a={sum(l for _, l in test)})",
          flush=True)

    outdir = Path(tempfile.mkdtemp(prefix=f"aoi_r3_{mode}_"))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                         host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    cfgp = outdir / "config.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(cfgp)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from backend.engine import get_engine
    engine = get_engine()
    engine.learning_parallel = False

    def staged(rows, sub):
        tdir = outdir / "train" / sub
        tdir.mkdir(parents=True, exist_ok=True)
        out = []
        for i, r in enumerate(rows):
            dst = tdir / f"{i:03d}_{Path(r).name}"
            shutil.copy2(r, dst)
            out.append(str(dst))
        return out

    n_st, d_st = staged(normals, "n"), staged(defects, "a")
    t0 = time.perf_counter()
    ver = engine.prepare(CATEGORY, {"init_normal": n_st, "init_defect": d_st, "val": [], "test": []},
                         activate=True, progress_cb=lambda f, m: print(f"  [prepare {f:.0%}] {m}", flush=True)
                         if int(f * 100) % 10 == 0 else None)
    print(f"[prepare] v{ver} {time.perf_counter() - t0:.1f}s", flush=True)
    pipe = engine.get_pipeline(CATEGORY)
    cal = {}
    for n, c in pipe.calibrators.items():
        e = getattr(c, "edges", None)
        cal[n] = {"type": type(c).__name__,
                  "lo": float(e[0]) if e is not None else None,
                  "hi": float(e[-1]) if e is not None else None}
    fit = {"mode": mode, "version": ver, "weights": dict(pipe.weights),
           "score_slots": list(pipe._score_slots),
           "train_auroc": getattr(pipe, "train_auroc", None),
           "tau_high": float(pipe.decider.tau_high), "tau_gray": float(pipe.decider.tau_gray),
           "calibrators": cal, "isolation": getattr(pipe, "calib_isolation", None),
           "n_train_normal": len(normals), "n_train_defect": len(defects)}
    print(json.dumps(fit, ensure_ascii=False, default=str), flush=True)

    rows = []
    for i, (path, lab) in enumerate(test):
        if time.perf_counter() - T0 > budget:
            print(f"[budget] 超 {budget}s，提前停止于 {i}/{len(test)}", flush=True)
            break
        t1 = time.perf_counter()
        rec = pipe.predict(path)
        rows.append({"path": path, "label": lab, "decision": rec["decision"],
                     "fused": float(rec["fused"]), "boost": float(rec.get("boost", 0.0)),
                     "slot_scores": {k: float(v) for k, v in rec.get("slot_scores", {}).items()},
                     "raw_scores": {k: float(v) for k, v in rec.get("raw_scores", {}).items()},
                     "ms": round((time.perf_counter() - t1) * 1000, 1)})
        r = rows[-1]
        print(f"  [{i + 1}/{len(test)}] y={lab} {r['decision']:7s} fused={r['fused']:.3f} "
              f"boost={r['boost']:.3f} " + " ".join(f"{k}={v:.2f}" for k, v in r["slot_scores"].items())
              + f" {r['ms']:.0f}ms", flush=True)
    raw = TASK / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{mode}_fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    (raw / f"{mode}_per_image.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"[done] {mode} total {time.perf_counter() - T0:.0f}s temp={outdir}", flush=True)


if __name__ == "__main__":
    main()
