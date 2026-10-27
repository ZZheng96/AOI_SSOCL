"""流/视频帧内存直检 A/B：旧（imwrite 临时 jpg + detect_image）vs 新（detect_ndarray 内存直检）。

隔离：TEMP storage/engine/SQLite；init_normal 从 stable 归档复制 14 张。
帧源：TEST-01 去重测试图 cv2 读成 BGR ndarray（6 张）+ 合成帧 2 张（真实帧加高斯噪声/亮度偏移），共 8 帧。
口径：wall=单帧墙钟（旧路径含 imwrite+detect_image+remove；新路径仅 detect_ndarray），均 persist=True、
with_heatmap=False、target_type="stream"，与 RTSP 循环一致。
精度：同帧 old/new 的 decision/final_score 逐帧对照（旧路径经 JPEG 有损往返，新路径为原始像素）。
"""
import argparse, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test01_e2e_latency")
OUT = TASK / "dev_frame"


def pct(v):
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(np.mean(v)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prepare-n", type=int, default=14)
    args = ap.parse_args()
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); tests.append(r["path"])
    outdir = Path(tempfile.mkdtemp(prefix="aoi_frame_ab_"))
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

    frames = []
    for p in tests:
        arr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        frames.append((Path(p).name, arr))
    rng = np.random.default_rng(7)
    base = frames[0][1]
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    bright = np.clip(frames[-1][1].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    frames += [("synth_noise.png", noisy), ("synth_bright.png", bright)]
    print(f"[2/5] prepare {len(normals)} normals; frames={len(frames)}", flush=True)
    t0 = time.perf_counter()
    engine.prepare("gold_finger", {"init_normal": normals, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: print(f"  prepare {f*100:.0f}% {m}", flush=True))
    print(f"  prepare done {time.perf_counter()-t0:.1f}s", flush=True)

    tmpdir = outdir / "frames_tmp"; tmpdir.mkdir()

    def run(tag, idx):
        frame = frames[idx][1]
        st = time.perf_counter()
        if tag == "old":
            tmp = str(tmpdir / f"rtsp_{idx}_{time.time_ns()}.jpg")
            cv2.imwrite(tmp, frame)
            try:
                r = service.detect_image(tmp, "gold_finger", persist=True, with_heatmap=False,
                                         target_type="stream")
            finally:
                os.remove(tmp)
        else:
            r = service.detect_ndarray(frame, "gold_finger", persist=True, with_heatmap=False,
                                       target_type="stream")
        return r, (time.perf_counter() - st) * 1000

    print("[3/5] warmup", flush=True)
    for i in range(4):
        run("old" if i % 2 else "new", i % len(frames))

    print("[4/5] precision old vs new (逐帧)", flush=True)
    prec = []
    for i, (name, _) in enumerate(frames):
        ra, _ = run("old", i)
        rb, _ = run("new", i)
        row = {"frame": name, "decision_old": ra["decision"], "decision_new": rb["decision"],
               "score_old": ra["final_score"], "score_new": rb["final_score"],
               "score_absdiff": abs(float(ra["final_score"]) - float(rb["final_score"])),
               "decision_equal": ra["decision"] == rb["decision"]}
        prec.append(row)
        print(f"  [{i+1}/{len(frames)}] {name} {row['decision_old']}/{row['decision_new']} "
              f"score {row['score_old']:.6g}/{row['score_new']:.6g} |d|={row['score_absdiff']:.3g}", flush=True)

    # 归因对照：①同为无损 PNG 输入，path 入口(detect_image) vs 内存入口(detect_ndarray)；
    # ②同为 JPEG 有损像素，path 入口(旧) vs 内存入口(新，喂 jpg 解码帧)。
    print("[4b/5] attribution: route vs JPEG", flush=True)
    attrib = []
    for i, p in enumerate(tests):
        name, frame = frames[i]
        r_path_png = service.detect_image(p, "gold_finger", persist=False, with_heatmap=False)
        r_nd_png = service.detect_ndarray(frame, "gold_finger", persist=False, with_heatmap=False)
        tmp = str(tmpdir / f"attr_{i}.jpg"); cv2.imwrite(tmp, frame)
        r_path_jpg = service.detect_image(tmp, "gold_finger", persist=False, with_heatmap=False)
        r_nd_jpg = service.detect_ndarray(cv2.imread(tmp), "gold_finger", persist=False, with_heatmap=False)
        os.remove(tmp)
        row = {"frame": name,
               "png_path": [r_path_png["decision"], r_path_png["final_score"]],
               "png_ndarray": [r_nd_png["decision"], r_nd_png["final_score"]],
               "jpg_path": [r_path_jpg["decision"], r_path_jpg["final_score"]],
               "jpg_ndarray": [r_nd_jpg["decision"], r_nd_jpg["final_score"]]}
        row["route_equal_png"] = row["png_path"] == row["png_ndarray"]
        row["route_equal_jpg"] = row["jpg_path"] == row["jpg_ndarray"]
        attrib.append(row)
        print(f"  [{i+1}/{len(tests)}] {name} png path/nd={row['png_path']}/{row['png_ndarray']} "
              f"jpg path/nd={row['jpg_path']}/{row['jpg_ndarray']}", flush=True)

    print(f"[5/5] latency A/B {args.rounds} rounds x {len(frames)} x 2 (交替)", flush=True)
    rows = []
    for rnd in range(args.rounds):
        for i, (name, _) in enumerate(frames):
            order = ("old", "new") if (rnd + i) % 2 == 0 else ("new", "old")
            cur = {}
            for tag in order:
                r, wall = run(tag, i)
                cur[tag] = {"tag": tag, "frame": name, "wall_ms": wall, "infer_ms": float(r["latency_ms"]),
                            "decision": r["decision"], "score": r["final_score"]}
                rows.append(cur[tag])
            print(f"  round {rnd+1}/{args.rounds} frame {i+1}/{len(frames)} "
                  f"old wall={cur['old']['wall_ms']:.0f} | new wall={cur['new']['wall_ms']:.0f}", flush=True)
    summ = {tag: {k: pct([r[k] for r in rows if r["tag"] == tag]) for k in ("wall_ms", "infer_ms")}
            for tag in ("old", "new")}
    out = {"isolated_temp": str(outdir), "n_frames": len(frames), "n_prepare": len(normals),
           "rounds": args.rounds, "precision": prec,
           "decision_all_equal": all(r["decision_equal"] for r in prec),
           "score_absdiff_max": max(r["score_absdiff"] for r in prec),
           "attribution": attrib,
           "route_equal_all": all(r["route_equal_png"] and r["route_equal_jpg"] for r in attrib),
           "latency_summary": summ, "rows": rows}
    (OUT / "ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ, ensure_ascii=False, indent=1),
          "\nDECISION_ALL_EQUAL", out["decision_all_equal"], "\nSCORE_ABSDIFF_MAX", out["score_absdiff_max"],
          "\nROUTE_EQUAL_ALL", out["route_equal_all"],
          flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)


if __name__ == "__main__":
    main()
