"""OPT-DECODE 开发自测（缩小版 TEST-01 同口径）——不能替代 TEST-01 复测。

口径与 raw/bench_test01.py 相同：decode=total_latency_ms-latency_ms，
infer=latency_ms，render=latency_e2e_ms-total_latency_ms，persist≈wall-e2e。
另以 monkeypatch 方式逐步计时 decode 窗口内的子步骤（不改被测代码）。

隔离：首次运行在 --env 下用 stable 归档 train 域正常图 prepare（seed=123，与
测试样本无交集）并缓存；每次运行把缓存 engine 复制到新 TEMP 目录，独立 SQLite，
AOI_CONFIG 指向 TEMP，不触碰开发/稳定版 DB 与快照。before/after 共用同一缓存模型，
保证 score/decision 可逐图对比。learning_parallel=False（生产默认）。
"""
import argparse, csv, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
OUT = Path(__file__).resolve().parent


def pct(vals):
    import numpy as np
    if not vals:
        return {}
    return {"n": len(vals), "p50": round(float(np.percentile(vals, 50)), 2),
            "p95": round(float(np.percentile(vals, 95)), 2),
            "max": round(float(max(vals)), 2), "mean": round(float(np.mean(vals)), 2)}


def make_cfg(outdir: Path, root: Path) -> Path:
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"),
                         engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(root / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(),
                         host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    p = outdir / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="before / after")
    ap.add_argument("--root", type=Path, default=DEV)
    ap.add_argument("--env", type=Path, default=Path(tempfile.gettempdir()) / "aoi_optdecode_env")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--prepare-n", type=int, default=40)
    args = ap.parse_args()

    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    test_paths = [r["path"] for r in man["images"]]
    uniq = list(dict.fromkeys(test_paths))
    random.Random(args.seed).shuffle(uniq)
    inputs = uniq[:args.n]

    env = args.env
    if not (env / "engine").exists():
        # 首次：在 env 内 prepare 并缓存（与 bench_test01 --prepare-n 同做法）
        env.mkdir(parents=True, exist_ok=True)
        os.environ["AOI_CONFIG"] = str(make_cfg(env, args.root))
        os.chdir(args.root); sys.path.insert(0, str(args.root))
        from backend.engine import get_engine
        from PIL import Image as _Img
        test_set = set(test_paths)
        pool = []
        for p in sorted((STABLE / "storage" / "archive" / "gold_finger" / "20261011").iterdir()):
            if p.suffix.lower() not in (".jpg", ".png") or str(p) in test_set:
                continue
            with _Img.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(str(p))
        random.seed(123); random.shuffle(pool)
        train = env / "train"; train.mkdir(exist_ok=True)
        normals = []
        for i, src in enumerate(pool[:args.prepare_n]):
            dst = train / f"normal_{i:03d}{Path(src).suffix}"
            shutil.copy2(src, dst); normals.append(str(dst))
        print(f"[prepare] {len(normals)} normals -> {env}", flush=True)
        eng = get_engine()
        eng.prepare("gold_finger", {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                    activate=True, progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}", flush=True))
        print("[prepare] done; re-run this script to benchmark", flush=True)
        return

    outdir = Path(tempfile.mkdtemp(prefix=f"aoi_optdecode_{args.tag}_"))
    shutil.copytree(env / "engine", outdir / "engine")
    os.environ["AOI_CONFIG"] = str(make_cfg(outdir, args.root))
    os.chdir(args.root); sys.path.insert(0, str(args.root))
    print(f"[1/3] tag={args.tag} root={args.root} temp={outdir} n={len(inputs)}", flush=True)

    import numpy as np
    from backend.pipeline import service as svc_mod
    from backend.pipeline.service import get_detection_service
    from backend.engine import get_engine
    service = get_detection_service()
    engine = get_engine()
    engine.learning_parallel = False

    # ---- 子步骤计时（monkeypatch 包装，不改变行为） ----
    stage = {}
    def wrap(obj, name, key):
        orig = getattr(obj, name)
        def f(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                stage[key] = stage.get(key, 0.0) + (time.perf_counter() - t) * 1000
        setattr(obj, name, f)
    for fn in ("_decode_rgb", "_decode_for_detect"):
        if hasattr(svc_mod, fn):
            wrap(svc_mod, fn, "decode_fn")
    wrap(svc_mod, "render_heatmap_overlay", "render_overlay_fn")
    wrap(engine, "_to_result", "to_result")
    engine._ensure_loaded("gold_finger")
    pipe = engine._pipes["gold_finger"][0]
    wrap(pipe, "predict_frame", "predict_frame")

    rows = []
    def detect(path, phase, idx):
        stage.clear()
        start = time.perf_counter()
        r = service.detect_image(path, "gold_finger", persist=True, with_heatmap=True)
        wall = (time.perf_counter() - start) * 1000
        infer = float(r["latency_ms"]); total = float(r["total_latency_ms"])
        e2e = float(r["latency_e2e_ms"])
        row = {"image": Path(path).name, "tag": args.tag, "phase": phase,
               "decode_ms": round(total - infer, 3), "infer_ms": round(infer, 3),
               "render_ms": round(e2e - total, 3), "persist_ms": round(max(wall - e2e, 0.0), 3),
               "e2e_ms": round(e2e, 3), "wall_ms": round(wall, 3),
               "sub_decode_fn_ms": round(stage.get("decode_fn", 0.0), 3),
               "sub_to_result_ms": round(stage.get("to_result", 0.0), 3),
               "sub_decode_other_ms": round(total - infer - stage.get("decode_fn", 0.0)
                                            - stage.get("to_result", 0.0), 3),
               "sub_render_overlay_ms": round(stage.get("render_overlay_fn", 0.0), 3),
               "final_score": repr(float(r["final_score"])), "decision": r["decision"],
               "has_overlay": bool(r.get("overlay_path")), "has_mask": bool(r.get("mask_path")),
               "has_heatmap": bool(r.get("heatmap_path")),
               "n_defect_boxes": len(r.get("defect_boxes") or []),
               "heatmap_boxes": len((r.get("localization") or {}).get("heatmap_boxes") or []),
               "model_version": r["model_version"]}
        rows.append(row)
        print(f"  {phase} {idx}: e2e={e2e:.1f} dec={row['decode_ms']:.1f} "
              f"(fn={row['sub_decode_fn_ms']:.1f}) inf={infer:.1f} rend={row['render_ms']:.1f} "
              f"{r['decision']}", flush=True)

    print(f"[2/3] cold + warmup x{args.warmup}", flush=True)
    detect(inputs[0], "cold", 0)
    for i in range(args.warmup):
        detect(inputs[(i + 1) % len(inputs)], "warmup", i)
    print(f"[3/3] hot {args.rounds}x{len(inputs)}", flush=True)
    for rnd in range(args.rounds):
        for i, p in enumerate(inputs):
            detect(p, "hot", rnd * len(inputs) + i)
            if (i + 1) % 10 == 0:
                print(f"  progress round {rnd+1}/{args.rounds} {i+1}/{len(inputs)}", flush=True)

    csv_path = OUT / f"per_image_{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    hot = [r for r in rows if r["phase"] == "hot"]
    summ = {"tag": args.tag, "note": "开发自测（缩小版同口径），不能替代 TEST-01 复测",
            "root": str(args.root), "temp": str(outdir), "n_images": len(inputs),
            "rounds": args.rounds, "warmup": args.warmup, "sample_seed": args.seed,
            "learning_parallel": False, "entry": "direct",
            **{k: pct([r[k] for r in hot]) for k in (
                "decode_ms", "infer_ms", "render_ms", "persist_ms", "e2e_ms", "wall_ms",
                "sub_decode_fn_ms", "sub_to_result_ms", "sub_decode_other_ms",
                "sub_render_overlay_ms")},
            "decisions": {d: sum(1 for r in hot if r["decision"] == d)
                          for d in sorted({r["decision"] for r in hot})}}
    (OUT / f"summary_{args.tag}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print("SUMMARY", json.dumps({k: summ[k] for k in ("decode_ms", "infer_ms", "render_ms",
                                                      "e2e_ms", "sub_decode_fn_ms",
                                                      "sub_to_result_ms")}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
