"""Round4 诊断：sem 方向 / shead 置零根因（TEMP 隔离、小样本、seed 固定）。
用法: python diag.py <baseline|domain> [--equal]
  --equal : fusion.weight_mode=equal（对照：sanity 置零 vs 等权）
"""
import json, os, re, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
DATA = Path(r"D:\CGAIC\data_local\gold_finger")
sys.path.insert(0, r"d:\CGAIC\AOI_SSOCL\review\task\round3_accuracy")
from repro_accuracy import splits  # noqa: E402
from analyze import op_points  # noqa: E402

OUT = Path(r"C:\Users\imia\AppData\Local\Temp\r4_diag\out")


def stem_id(p):
    s = Path(p).stem
    return s[:-len("_repaired")] if s.endswith("_repaired") else s


def twin_of(defect_path):
    """缺陷原图 -> train/good 或 test/good 中的 *_repaired 配对图。"""
    sid = stem_id(defect_path)
    for sub in ("train/good", "test/good"):
        for ext in (".png", ".jpg"):
            p = DATA / sub / f"{sid}_repaired{ext}"
            if p.exists():
                return str(p)
    return None


def main():
    mode = sys.argv[1]
    equal = "--equal" in sys.argv
    tag = mode + ("_equal" if equal else "")
    T0 = time.perf_counter()
    normals, defects, test = splits(mode)
    outdir = Path(tempfile.mkdtemp(prefix=f"aoi_r4_{tag}_"))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base_cfg = ROOT / "configs" / "engine_fast.yaml"
    if equal:
        ec = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        ec.setdefault("fusion", {})["weight_mode"] = "equal"
        base_cfg = outdir / "engine_fast_equal.yaml"
        base_cfg.write_text(yaml.safe_dump(ec, allow_unicode=True), encoding="utf-8")
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(base_cfg),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                         host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    cfgp = outdir / "config.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(cfgp)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from backend.engine import get_engine
    from algo.eval.offline import resolve_isolation
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
    print(f"[{tag}] train_normal={len(normals)} train_defect={len(defects)} test={len(test)}", flush=True)
    engine.prepare("gf_r4", {"init_normal": n_st, "init_defect": d_st, "val": [], "test": []},
                   activate=True)
    pipe = engine.get_pipeline("gf_r4")
    names = list(pipe._score_slots)
    seed = int(pipe.cfg.get("seed", 42))
    _, cal_idx, fit_idx, info = resolve_isolation(pipe.cfg.get("calibration", {}) or {}, len(n_st), seed)
    print(f"[fit] weights={pipe.weights} sanity={pipe.slot_sanity} iso={info}", flush=True)

    # ---- 体检复算：校准正常（cal_idx） vs init_defect 的逐槽 raw 分 + 配对检验 ----
    def raw_scores(paths, label):
        items = pipe._tile_and_extract(paths)
        res = []
        for k, it in enumerate(items):
            res.append({n: float(pipe._slot_image_score(pipe.slots[n], it)[0]) for n in names})
            print(f"  [score-{label}] {k + 1}/{len(items)}", flush=True) if (k + 1) % 10 == 0 else None
        return res

    t_ns = raw_scores(n_st, "normal")
    t_ds = raw_scores(d_st, "defect")
    fit_twin_ids = {stem_id(normals[i]) for i in fit_idx}
    sanity = {}
    for n in names:
        cn = [t_ns[i][n] for i in cal_idx]
        cd = [s[n] for s in t_ds]
        sanity[n] = round(float(roc_auc_score([0] * len(cn) + [1] * len(cd), cn + cd)), 4)
    # init_defect 按"配对 repaired 是否在 fit 库"分组的平均 raw 分
    grp = {"twin_in_fit": [], "twin_not_in_fit": []}
    for p, s in zip(defects, t_ds):
        grp["twin_in_fit" if stem_id(p) in fit_twin_ids else "twin_not_in_fit"].append(s)
    grp_mean = {g: {n: round(float(np.mean([s[n] for s in v])), 4) for n in names} if v else None
                for g, v in grp.items()}
    grp_n = {g: len(v) for g, v in grp.items()}
    cal_mean = {n: round(float(np.mean([t_ns[i][n] for i in cal_idx])), 4) for n in names}
    print(f"[sanity-recalc] {sanity} cal_normal_mean={cal_mean} defect_grp_mean={grp_mean} n={grp_n}",
          flush=True)

    # ---- test 推理：fused AUROC / FPR≤5% recall / p95 ----
    for p, _ in test[:2]:   # 预热（不计时）
        pipe.predict(p)
    rows = []
    for i, (path, lab) in enumerate(test):
        t1 = time.perf_counter()
        rec = pipe.predict(path)
        ms = (time.perf_counter() - t1) * 1000
        rows.append({"path": path, "label": lab, "fused": float(rec["fused"]), "ms": ms,
                     "raw": {k: float(v) for k, v in rec.get("raw_scores", {}).items()},
                     "cal": {k: float(v) for k, v in rec.get("slot_scores", {}).items()}})
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            print(f"  [predict] {i + 1}/{len(test)} last={ms:.0f}ms", flush=True)
    y = [r["label"] for r in rows]
    fused = [r["fused"] for r in rows]
    ms = np.array([r["ms"] for r in rows])
    w_eq = {n: 1.0 / len(names) for n in names}
    from algo.fusion.fixed import fuse
    fused_eq = [fuse({n: r["cal"][n] for n in names}, w_eq) for r in rows]
    per = {n: {"auroc_raw": round(roc_auc_score(y, [r["raw"][n] for r in rows]), 4),
               "auroc_cal": round(roc_auc_score(y, [r["cal"][n] for r in rows]), 4)} for n in names}

    # ---- sem 配对符号检验：test 缺陷原图 vs 自身 repaired 配对图（同库、同批，只差缺陷区域）----
    pairs = []
    for path, lab in test:
        if lab != 1:
            continue
        tw = twin_of(path)
        if tw is None:
            continue
        rd = pipe.predict(path)["raw_scores"]
        rt = pipe.predict(tw)["raw_scores"]
        pairs.append({"id": stem_id(path), "twin_in_train": stem_id(path) in {stem_id(x) for x in normals},
                      **{f"{n}_d": float(rd[n]) for n in names}, **{f"{n}_t": float(rt[n]) for n in names}})
    pair_stat = {}
    for n in names:
        diff = np.array([q[f"{n}_d"] - q[f"{n}_t"] for q in pairs])
        pair_stat[n] = {"n": len(diff), "frac_defect_gt_twin": round(float((diff > 0).mean()), 3) if len(diff) else None,
                        "mean_diff": round(float(diff.mean()), 4) if len(diff) else None}
    out = {"mode": tag, "weights": pipe.weights, "slot_sanity_pipe": pipe.slot_sanity,
           "slot_sanity_recalc_raw": sanity, "isolation": info,
           "cal_normal_mean_raw": cal_mean, "init_defect_mean_raw_by_twin": grp_mean, "group_n": grp_n,
           "train_auroc": pipe.train_auroc, "n_test": len(rows),
           "fused_auroc": round(roc_auc_score(y, fused), 4),
           "recall_at_fpr5": op_points(y, fused)["max_recall_at_fpr5"][0],
           "fused_equal_auroc": round(roc_auc_score(y, fused_eq), 4),
           "fused_equal_recall_at_fpr5": op_points(y, fused_eq)["max_recall_at_fpr5"][0],
           "p50_ms": round(float(np.percentile(ms, 50)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1),
           "per_slot_test": per, "pair_test": pair_stat, "total_s": round(time.perf_counter() - T0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{tag}_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[RESULT] " + json.dumps(out, ensure_ascii=False), flush=True)
    shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    main()
