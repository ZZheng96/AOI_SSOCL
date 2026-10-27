"""OPT-DECODE 同进程 A/B：旧 _decode_rgb（PNG 走 PIL）vs 新（cv2 直出 RGB）。

隔离：TEMP storage/engine/SQLite；init_normal 从 stable 归档复制 14 张到 TEMP train
（与测试图无交集），测试图=TEST-01 清单去重 6 张，共 20 张。不触碰生产 DB/快照。
口径同 TEST-01：decode=total_latency_ms-latency_ms，e2e=latency_e2e_ms，wall 含 persist。
精度：同图 old/new 的 score/decision/slots/raw/boxes/fused_heatmap 逐项 maxdiff。
"""
import argparse, json, os, random, shutil, statistics, sys, tempfile, time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")


def old_decode_rgb(image_path):
    """OPT-DECODE 前的实现（逐字复刻）。"""
    import cv2
    from backend.core.imaging import imread
    with Image.open(str(image_path)) as pil:
        if pil.format in {"JPEG", "BMP"} and pil.mode in {"RGB", "L"}:
            bgr = imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.array(pil.convert("RGB"))


def pct(v):
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(np.mean(v)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prepare-n", type=int, default=14)
    args = ap.parse_args()
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); tests.append(r["path"])
    outdir = Path(tempfile.mkdtemp(prefix="aoi_optdec_"))
    print(f"[1/5] isolate -> {outdir}", flush=True)
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(outdir / "storage"), engine_storage=str(outdir / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (outdir / "bench.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(outdir / "config.yaml")
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))

    from backend.pipeline import service as svc
    from backend.engine import get_engine
    service = svc.get_detection_service()
    engine = get_engine()
    new_decode = svc._decode_rgb

    test_set = set(tests)
    pool = []
    for p in sorted((STABLE / "storage" / "archive" / "gold_finger" / "20261011").iterdir()):
        if p.suffix.lower() in (".jpg", ".png") and str(p) not in test_set:
            with Image.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(str(p))
    random.seed(123); random.shuffle(pool)
    train = outdir / "train"; train.mkdir(parents=True)
    normals = []
    for i, src in enumerate(pool[:args.prepare_n]):
        dst = train / f"normal_{i:03d}{Path(src).suffix}"; shutil.copy2(src, dst); normals.append(str(dst))
    print(f"[2/5] prepare {len(normals)} normals + {len(tests)} test images", flush=True)
    t0 = time.perf_counter()
    engine.prepare("gold_finger", {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}", flush=True))
    print(f"  prepare done {time.perf_counter()-t0:.1f}s", flush=True)

    def run(tag, path):
        svc._decode_rgb = old_decode_rgb if tag == "old" else new_decode
        st = time.perf_counter()
        r = service.detect_image(path, "gold_finger", persist=True, with_heatmap=True)
        wall = (time.perf_counter() - st) * 1000
        return r, wall

    print("[3/5] warmup", flush=True)
    for i in range(4):
        run("old" if i % 2 else "new", tests[i % len(tests)])

    print("[4/5] precision old vs new (engine.predict_ndarray, 同图)", flush=True)
    prec = []
    for p in tests:
        a = old_decode_rgb(p); b = new_decode(p)
        ra = engine.predict_ndarray("gold_finger", a, path_hint=p)
        rb = engine.predict_ndarray("gold_finger", b, path_hint=p)
        rc = engine.predict_ndarray("gold_finger", a, path_hint=p)  # 旧路径自重复（确定性基线）
        def hm(r):
            return r.fused_heatmap if isinstance(r.fused_heatmap, np.ndarray) else np.zeros(1)
        row = {"image": Path(p).name,
               "pixel_maxdiff": int(np.abs(a.astype(np.int16) - b).max()),
               "score_old": ra.score, "score_new": rb.score, "score_absdiff": abs(ra.score - rb.score),
               "decision_old": ra.decision, "decision_new": rb.decision,
               "slots_maxdiff": max([abs(ra.slot_scores[k] - rb.slot_scores.get(k, 1e9)) for k in ra.slot_scores] or [0]),
               "raw_maxdiff": max([abs(ra.raw_scores[k] - rb.raw_scores.get(k, 1e9)) for k in ra.raw_scores] or [0]),
               "boxes_equal": [b_["bbox"] for b_ in ra.defect_boxes] == [b_["bbox"] for b_ in rb.defect_boxes],
               "heatmap_maxdiff": float(np.abs(hm(ra) - hm(rb)).max()),
               "old_repeat_score_absdiff": abs(ra.score - rc.score),
               "old_repeat_heatmap_maxdiff": float(np.abs(hm(ra) - hm(rc)).max())}
        prec.append(row)
        print(f"  {row['image']} px={row['pixel_maxdiff']} dscore={row['score_absdiff']:.3g} "
              f"{row['decision_old']}/{row['decision_new']} hm={row['heatmap_maxdiff']:.3g} "
              f"boxes_eq={row['boxes_equal']} (old-repeat dscore={row['old_repeat_score_absdiff']:.3g})", flush=True)

    print(f"[5/5] latency A/B {args.rounds} rounds x {len(tests)} x 2 (交替)", flush=True)
    rows = []
    for rnd in range(args.rounds):
        for i, p in enumerate(tests):
            order = ("old", "new") if (rnd + i) % 2 == 0 else ("new", "old")
            for tag in order:
                r, wall = run(tag, p)
                infer = float(r["latency_ms"]); total = float(r["total_latency_ms"]); e2e = float(r["latency_e2e_ms"])
                rows.append({"tag": tag, "image": Path(p).name, "decode_ms": total - infer, "infer_ms": infer,
                             "render_ms": e2e - total, "e2e_ms": e2e, "wall_ms": wall, "decision": r["decision"],
                             "score": r["final_score"]})
            print(f"  round {rnd+1}/{args.rounds} img {i+1}/{len(tests)} "
                  f"old_e2e={rows[-2 if order[0]=='old' else -1]['e2e_ms']:.0f} "
                  f"new_e2e={rows[-1 if order[1]=='new' else -2]['e2e_ms']:.0f}", flush=True)
    svc._decode_rgb = new_decode
    summ = {}
    for tag in ("old", "new"):
        rr = [r for r in rows if r["tag"] == tag]
        summ[tag] = {k: pct([r[k] for r in rr]) for k in ("decode_ms", "infer_ms", "render_ms", "e2e_ms", "wall_ms")}
    dec_same = all(a["decision"] == b["decision"] and a["score"] == b["score"]
                   for a in rows if a["tag"] == "old"
                   for b in rows if b["tag"] == "new" and b["image"] == a["image"])
    out = {"isolated_temp": str(outdir), "n_test_images": len(tests), "n_prepare": len(normals),
           "rounds": args.rounds, "precision": prec, "latency_summary": summ,
           "ab_rows_decision_and_score_identical": dec_same, "rows": rows}
    (TASK / "dev_opt" / "e2e_ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False, indent=1), "\nAB_IDENTICAL", dec_same, flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)


if __name__ == "__main__":
    main()
