"""REMAIN-01（P0，AC-062/TEST-06）：AOI_feature vs AOI_Core Trad 全链路 parity 复核。

双协议：
  --tree stable  稳定版树（验收对象；候选=legacy_u27 口径，即历史 TEST-06 FAIL 时口径）
  --tree dev     开发版树（诊断参照；候选=explicit_rng42 口径，即修复后口径）

输入与 TEST-06 旧协议完全一致（同造法同种子，atol=1e-6）：
  train=6 张 64x64（seed 0-5），probe=3 张（300x280 seed100 / 20x40 seed101 /
  64x64 seed102），另加常量图 np.full((64,64,3),128) 做 dtype/非有限值专项。

输出（--outdir）：
  parity_result.json            harness 八项 checks 原始证据
  group_diffs.json              15 组 x 3 探针逐组 max|diff|
  neighbors_agreement.json      邻居一致率（train[:3]+probe 共 6 图）
  rng_snapshots.json            全局 RNG 前后状态快照（sha256 摘要）
  constant_image.json           常量图 dtype/非有限值专项
  env_snapshot.json             运行环境/版本快照
  01_extraction.json .. 08_report.json   任务八项逐项对比（每项一个文件）

用法（Windows，py -3.11）：
  py -3.11 run_remain01.py --tree stable --outdir <raw>
  py -3.11 run_remain01.py --tree dev    --outdir <raw>\\dev
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

TREES = {
    "stable": r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree",
    "dev": r"d:\CGAIC\AOI_SSOCL\AOI_tree",
}
MODES = {"stable": "legacy_u27", "dev": "explicit_rng42"}
MODE_NOTE = {
    "legacy_u27": "全局 RandomState + np.random.seed(0) 包裹（非 default_rng(42)）",
    "explicit_rng42": "显式 np.random.default_rng(42)（Generator，不碰全局 RNG）",
}
SEED = 0
ATOL = 1e-6


def _img(seed, h=64, w=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 40 * np.sin(xx / (3 + seed % 4)) + 30 * np.cos(yy / 5.0)
    g = np.clip(base + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([g, np.roll(g, 1, 0), np.roll(g, 1, 1)], -1)


def _rng_snap(tag):
    st = np.random.get_state()
    return {"tag": tag, "algorithm": st[0], "pos": int(st[2]),
            "state_sha256_16": hashlib.sha256(st[1].tobytes()).hexdigest()[:16]}


def _maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _finite(x):
    x = np.asarray(x)
    return bool(np.all(np.isfinite(x.astype(np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", choices=sorted(TREES), required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    tree, mode = Path(TREES[args.tree]), MODES[args.tree]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in (str(tree / "AOI_Core"), str(tree),
              str(Path(__file__).resolve().parent)):
        if p not in sys.path:
            sys.path.insert(0, p)

    t0 = time.time()
    print(f"[remain01] tree={args.tree} ({tree})", flush=True)
    print(f"[remain01] candidate_mode={mode}: {MODE_NOTE[mode]}", flush=True)

    print("[remain01] 导入 AOI_Core harness 与 AOI_feature 候选...", flush=True)
    from algo.feature_revision import PARITY_CHECKS, production_hashes
    from algo.features import registry as freg
    from algo.features.parity import ProductionCandidate, run_parity
    from remain01_candidate import AOIFeatureCandidate

    def dump(name, obj):
        (outdir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"[remain01] 写出 {name}", flush=True)

    # ---------- 环境快照 ----------
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "tree": str(tree), "candidate_mode": mode,
           "atol": ATOL, "global_seed": SEED}
    for mod in ("cv2", "skimage", "scipy", "pywt"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            env[mod] = f"unavailable: {e}"
    dump("env_snapshot.json", env)

    # ---------- 固定输入（与 TEST-06 同造法同种子）----------
    np.random.seed(SEED)
    snaps = [_rng_snap("t0_after_global_seed0")]
    print("[remain01] 构造固定输入：train=6x64x64(seed0-5) + probe=300x280(seed100)/"
          "20x40(seed101)/64x64(seed102) + 常量图128...", flush=True)
    train = [_img(s) for s in range(6)]
    probe = [_img(100, 300, 280), _img(101, 20, 40), _img(102)]
    const = np.full((64, 64, 3), 128, np.uint8)
    input_manifest = {
        "train": [{"seed": s, "shape": [64, 64, 3]} for s in range(6)],
        "probe": [{"seed": 100, "shape": [300, 280, 3]},
                  {"seed": 101, "shape": [20, 40, 3]},
                  {"seed": 102, "shape": [64, 64, 3]}],
        "constant_image": {"fill": 128, "shape": [64, 64, 3]},
        "recipe": "default_rng(seed) 正弦/余弦基 + N(0,8) 噪声，3 通道 roll（同 TEST-06）",
        "global_seed": SEED, "atol": ATOL}

    print("[remain01] 实例化候选适配器（FeatureSet trunk_200 + GroupKNN k=3）...",
          flush=True)
    cand = AOIFeatureCandidate(mode)

    # ---------- harness 全链路 ----------
    print("[remain01] run_parity 开始（参照=生产 TradSlot 全新实例）...", flush=True)
    ev = run_parity(cand, train, probe, atol=ATOL,
                    lineage={"revision_id": f"remain01-aoi-feature-{args.tree}",
                             "seed": SEED, "dataset_split": input_manifest,
                             "model_snapshot": {"category": "none",
                                                "note": "合成图现场拟合，无快照"}})
    snaps.append(_rng_snap("after_run_parity"))
    dump("parity_result.json", ev)
    print(f"[remain01] run_parity 完成：status={ev['status']} "
          f"failed={ev['failed_checks']} "
          f"max_abs_diff={ev['numeric']['max_abs_diff']:.3e}", flush=True)

    # ---------- 逐项明细（与 harness 同口径独立复算，供八项 JSON）----------
    print("[remain01] 逐项明细复算（生产参照新实例 + fit）...", flush=True)
    ref = ProductionCandidate()
    ref.fit(train)

    # 01 特征提取：3 探针逐图 + 全向量
    print("  [1/8] extraction ...", flush=True)
    ext_rows = []
    for i, im in enumerate(probe):
        vc, _ = cand.extract(im)
        vr, _ = ref.extract(im)
        ext_rows.append({"probe": i, "shape": list(im.shape),
                         "candidate_finite": _finite(vc),
                         "production_finite": _finite(vr),
                         "max_abs_diff": _maxdiff(vc, vr)})
    ext_max = max(r["max_abs_diff"] for r in ext_rows)

    # 02 预处理：spec 一致性 + 越界图数值
    print("  [2/8] preprocess ...", flush=True)
    ps_c, ps_r = cand.preprocess_spec(), ref.preprocess_spec()
    odd = [im for im in probe
           if max(im.shape[:2]) > ps_r["max_side"] or min(im.shape[:2]) < ps_r["min_side"]]
    pd = max([_maxdiff(cand.extract(im)[0], ref.extract(im)[0]) for im in odd] or [0.0])

    # 03 维度/顺序/slice：候选 trace vs registry.TRAD_GROUPS
    print("  [3/8] dimension_order_slice ...", flush=True)
    _, tc = cand.extract(probe[2])
    trace_errs = freg.check_trace(tc)
    cand_slices = [{"key": m["key"],
                    "slice": [int(m["slice"].start), int(m["slice"].stop)]}
                   for m in cand._meta]
    ref_slices = [{"key": g["key"], "slice": [int(g["slice"][0]),
                                              int(g["slice"][1])]}
                  for g in freg.TRAD_GROUPS]
    slice_equal = cand_slices == ref_slices

    # 04 rng：重复确定 + 全局泄漏（复刻 harness marker 法）+ 口径说明
    print("  [4/8] rng ...", flush=True)
    st = np.random.get_state()
    a1 = cand.extract(probe[0])[0]
    np.random.seed(12345)
    b1 = cand.extract(probe[0])[0]
    np.random.seed(12345)
    marker = np.random.random()
    np.random.seed(12345)
    cand.extract(probe[0])
    leak = bool(np.random.random() != marker)
    np.random.set_state(st)
    rng_repeat_d = _maxdiff(a1, b1)

    # 05 dtype/非有限值：常量图专项
    print("  [5/8] dtype/nonfinite（常量图 128）...", flush=True)
    vc_raw = np.asarray(cand.extract_raw(const))
    vr, _ = ref.extract(const)
    vr64 = np.asarray(vr, dtype=np.float64)
    vc64 = vc_raw.astype(np.float64)
    both_finite = np.isfinite(vc64) & np.isfinite(vr64)
    const_check = {
        "image": "np.full((64,64,3), 128, uint8)",
        "candidate_raw_dtype": str(vc_raw.dtype),
        "production_dtype": str(np.asarray(vr).dtype),
        "candidate_nan": int(np.isnan(vc64).sum()),
        "candidate_posinf": int((vc64 == np.inf).sum()),
        "candidate_neginf": int((vc64 == -np.inf).sum()),
        "production_nan": int(np.isnan(vr64).sum()),
        "production_inf": int(np.isinf(vr64).sum()),
        "nonfinite_position_match": bool(np.array_equal(~np.isfinite(vc64),
                                                        ~np.isfinite(vr64))),
        "max_abs_diff_on_both_finite": float(np.max(np.abs(vc64[both_finite]
                                                           - vr64[both_finite]))),
        "n_dims": int(vc64.size)}
    snaps.append(_rng_snap("after_constant_image"))

    # 06/07 kNN 邻居/自匹配 + 评分：train[:3]+probe 共 6 图
    print("  [6/8] knn_neighbors_self_match ...", flush=True)
    print("  [7/8] scoring ...", flush=True)
    nb_rows, sc_rows = [], []
    names = [f"train{i}" for i in range(3)] + [f"probe{i}" for i in range(3)]
    for name, im in zip(names, train[:3] + probe):
        sc_c, nb_c = cand.neighbors(im)
        sc_r, nb_r = ref.neighbors(im)
        inter = len(set(nb_c) & set(nb_r))
        nb_rows.append({"image": name, "candidate": nb_c, "production": nb_r,
                        "exact_match": nb_c == nb_r, "overlap": inter})
        sc_rows.append({"image": name, "candidate_score": sc_c,
                        "production_score": sc_r,
                        "abs_diff": abs(sc_c - sc_r)})
    n_exact = sum(r["exact_match"] for r in nb_rows)
    agree = {"n_images": len(nb_rows),
             "exact_match_rate": n_exact / len(nb_rows),
             "mean_overlap_rate": float(np.mean([r["overlap"] / 3 for r in nb_rows])),
             "rows": nb_rows}
    sc_max = max(r["abs_diff"] for r in sc_rows)

    # 08 报告：3 探针
    print("  [8/8] report ...", flush=True)
    rp_rows = []
    for i, im in enumerate(probe):
        rc, rr = cand.report(im), ref.report(im)
        rp_rows.append({"probe": i, "candidate": rc, "production": rr,
                        "equal": rc["neighbors"] == rr["neighbors"]
                        and rc["enabled_groups"] == rr["enabled_groups"]
                        and abs(rc["score"] - rr["score"]) <= ATOL})

    snaps.append(_rng_snap("t_end"))
    dump("rng_snapshots.json", {"tree": args.tree, "mode": mode, "snapshots": snaps})
    dump("neighbors_agreement.json", agree)
    dump("constant_image.json", const_check)

    # ---------- 15 组逐组归因 ----------
    print("[remain01] 逐组差异归因（15 组 x 3 探针）...", flush=True)
    group_diffs = {}
    for pi, im in enumerate(probe):
        vc, _ = cand.extract(im)
        vr, _ = ref.extract(im)
        per = {}
        for g in freg.TRAD_GROUPS:
            a, b = int(g["slice"][0]), int(g["slice"][1])
            per[g["key"]] = float(np.max(np.abs(
                vc[a:b].astype(np.float64) - vr[a:b].astype(np.float64))))
        group_diffs[f"probe{pi}_{im.shape[0]}x{im.shape[1]}"] = per
        worst = max(per, key=per.get)
        print(f"  probe{pi} {im.shape[0]}x{im.shape[1]}: worst={worst} "
              f"max|diff|={per[worst]:.3e}", flush=True)
    dump("group_diffs.json", group_diffs)

    # ---------- 任务八项逐项 JSON ----------
    print("[remain01] 拆分任务八项逐项 JSON...", flush=True)
    hk = ev["checks"]
    task_checks = [
        ("01_extraction", "特征提取", "extraction", hk["extraction"],
         {"per_probe": ext_rows, "all_vector_max_abs_diff": ext_max, "atol": ATOL}),
        ("02_preprocess", "预处理", "preprocess", hk["preprocess"],
         {"spec_equal": ps_c == ps_r, "candidate_spec": ps_c, "production_spec": ps_r,
          "resize_probe_max_abs_diff": pd, "n_resize_probes": len(odd), "atol": ATOL}),
        ("03_dimension_order_slice", "15组/200维顺序与slice", "dimension_order_slice",
         hk["dimension_order_slice"],
         {"trace_errors": trace_errs, "candidate_slices": cand_slices,
          "production_slices": ref_slices, "slices_equal": slice_equal,
          "dimension": int(sum(s["slice"][1] - s["slice"][0]
                               for s in cand_slices))}),
        ("04_rng", "显式default_rng(42)行为", "rng", hk["rng"],
         {"harness_repeat_max_abs_diff": rng_repeat_d, "harness_global_rng_mutated": leak,
          "candidate_rng_semantics": MODE_NOTE[mode],
          "production_rng_semantics": "extractor 显式 default_rng(42)",
          "semantics_match_production": mode == "explicit_rng42",
          "rng_snapshots_file": "rng_snapshots.json"}),
        ("05_dtype_nonfinite", "dtype/非有限值处理", None,
         const_check["nonfinite_position_match"]
         and const_check["max_abs_diff_on_both_finite"] <= ATOL
         and const_check["candidate_raw_dtype"] == const_check["production_dtype"],
         const_check),
        ("06_knn_neighbors_self_match", "kNN邻居/自匹配", "knn_neighbors_self_match",
         hk["knn_neighbors_self_match"],
         {"exact_match_rate": agree["exact_match_rate"],
          "mean_overlap_rate": agree["mean_overlap_rate"],
          "production_rule": "skip nearest (self) → sort(d)[1:4]",
          "rows": nb_rows}),
        ("07_scoring", "评分", "scoring", hk["scoring"],
         {"max_abs_diff": sc_max, "atol": ATOL, "rows": sc_rows}),
        ("08_report", "报告", "report", hk["report"],
         {"per_probe_equal": [r["equal"] for r in rp_rows], "rows": rp_rows}),
    ]
    results = []
    for fname, cname, hkey, ok, detail in task_checks:
        payload = {"task_check": cname, "tree": args.tree,
                   "candidate_mode": mode, "harness_check_key": hkey,
                   "pass": bool(ok), "evidence": detail}
        dump(f"{fname}.json", payload)
        results.append((fname, cname, bool(ok)))

    # ---------- 汇总 ----------
    print(f"[remain01] ==== 任务八项逐项结论（tree={args.tree}）====", flush=True)
    for fname, cname, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {fname}  {cname}", flush=True)
    print(f"[remain01] harness 八项: " +
          ", ".join(f"{k}={'T' if ev['checks'][k] else 'F'}" for k in PARITY_CHECKS),
          flush=True)
    print(f"[remain01] baseline_hashes={ev['baseline_implementation_hashes']}",
          flush=True)
    print(f"[remain01] production_hashes(now)={production_hashes()}", flush=True)
    print(f"[remain01] 全部完成，用时 {time.time() - t0:.1f}s", flush=True)
    return 0 if ev["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
