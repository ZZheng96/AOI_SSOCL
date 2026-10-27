"""下采样解码 精度 + e2e A/B（AC-066 / §69）。

隔离：TEMP storage/engine/SQLite（engine_fast.yaml，同 TEST-01 口径），不触碰生产 DB/快照。
品类：
  gf_full   = 稳定版归档 2500² PNG 正常图 prepare（排除 TEST-01 测试图），用于 PNG 延迟
  dl_full   = data_local train/good 全分辨率 prepare
  dl_ds1024 = 同一批 train/good 先 INTER_AREA 到 max 1024 存 PNG 再 prepare（“下采样后重新 prepare”）
精度图：data_local test 10 good + 24 defect（有 GT 掩码），延迟图：TEST-01 PNG 6 张 + 精度集中 6 张 JPG。
评估图合计 40 张。框按缩放比映射回原图后与 full 结果做 IoU 贪心匹配，并用 GT 掩码判命中。
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
OUT = TASK / "dev_downsample"
DL = Path(r"D:\CGAIC\data_local\gold_finger")
RGB = getattr(cv2, "IMREAD_COLOR_RGB", None)
IGN = cv2.IMREAD_IGNORE_ORIENTATION


def rd(p, flags):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flags)


def dec_full(p):
    return rd(p, IGN | RGB)


def area(img, side):
    h, w = img.shape[:2]
    if max(h, w) <= side:
        return img
    s = side / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


VARIANTS = {
    "full": dec_full,
    "area1024": lambda p: area(dec_full(p), 1024),
    "area512": lambda p: area(dec_full(p), 512),
    "reduced4": lambda p: cv2.cvtColor(rd(p, IGN | cv2.IMREAD_REDUCED_COLOR_4), cv2.COLOR_BGR2RGB),
}


def pct(v):
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 1),
            "p95": round(float(np.percentile(v, 95)), 1), "mean": round(float(np.mean(v)), 1)}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def greedy_iou(ref, cand):
    """ref 每框与 cand 贪心配对，返回各 ref 框的最佳 IoU 列表。"""
    used, out = set(), []
    for r in ref:
        best, bj = 0.0, -1
        for j, c in enumerate(cand):
            if j in used:
                continue
            v = iou(r, c)
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
        out.append(best)
    return out


def boxes_orig(res, sx, sy):
    return [[b["bbox"][0] * sx, b["bbox"][1] * sy, b["bbox"][2] * sx, b["bbox"][3] * sy]
            for b in (res.defect_boxes or [])]


def gt_hit(boxes, mask):
    if mask is None or not boxes:
        return False
    for x0, y0, x1, y1 in boxes:
        if mask[int(max(0, y0)):int(np.ceil(y1)), int(max(0, x0)):int(np.ceil(x1))].any():
            return True
    return False


def hm_corr(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    a = cv2.resize(a.astype(np.float32), (128, 128)); b = cv2.resize(b.astype(np.float32), (128, 128))
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--prepare-n", type=int, default=20)
    args = ap.parse_args()
    t_start = time.perf_counter()
    outdir = Path(tempfile.mkdtemp(prefix="aoi_ds_"))
    print(f"[1/7] isolate -> {outdir}", flush=True)
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
    orig_decode = svc._decode_rgb

    # ---- 图集 ----
    man = json.loads((TASK / "raw" / "images.json").read_text(encoding="utf-8"))
    seen, png_tests = set(), []
    for r in man["images"]:
        if r["sha256"] not in seen:
            seen.add(r["sha256"]); png_tests.append(r["path"])
    png_tests = png_tests[:6]
    pool = []
    for p in sorted((STABLE / "storage" / "archive" / "gold_finger" / "20261011").iterdir()):
        if p.suffix.lower() == ".png" and str(p) not in set(png_tests):
            with Image.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(str(p))
    random.seed(123); random.shuffle(pool)

    def big_jpgs(d):
        out = []
        for p in sorted(d.glob("*.jpg")):
            with Image.open(p) as im:
                out.append((str(p), im.size))
        return out

    tr = big_jpgs(DL / "train" / "good")
    random.seed(7); random.shuffle(tr)
    tr = [p for p, s in tr if s in ((4096, 3000), (5120, 5120))][:args.prepare_n]
    gmasks = {p.name[:-len("_mask.png")]: p for p in (DL / "ground_truth" / "defect").glob("*_mask.png")}
    good = big_jpgs(DL / "test" / "good")
    good = [p for p, s in good if s == (4096, 3000)][:5] + [p for p, s in good if s == (5120, 5120)][:5]
    dfx = [(p, s) for p, s in big_jpgs(DL / "test" / "defect") if Path(p).stem in gmasks]
    dfx = [p for p, s in dfx if s == (4096, 3000)][:18] + [p for p, s in dfx if s == (5120, 5120)][:6]
    acc_imgs = [(p, "good") for p in good] + [(p, "defect") for p in dfx]
    print(f"  png_tests={len(png_tests)} acc good={len(good)} defect={len(dfx)} dl_train={len(tr)} "
          f"gf_train={min(args.prepare_n, len(pool))}", flush=True)

    # ---- prepare ----
    def prep(cat, paths):
        t0 = time.perf_counter()
        engine.prepare(cat, {"init_normal": paths, "init_defect": [], "val": [], "test": []}, activate=True,
                       progress_cb=lambda f, m: print(f"  [{cat}] prepare {f*100:.0f}% {m}", flush=True))
        dt = time.perf_counter() - t0
        print(f"  [{cat}] prepare done {dt:.1f}s", flush=True)
        return round(dt, 1)

    print("[2/7] prepare gf_full / dl_full / dl_ds1024", flush=True)
    prep_t = {}
    d = outdir / "train" / "gf"; d.mkdir(parents=True)  # 隔离红线：路径须含 train 段
    gf = []
    for i, s in enumerate(pool[:args.prepare_n]):
        dst = d / f"n_{i:03d}.png"; shutil.copy2(s, dst); gf.append(str(dst))
    prep_t["gf_full"] = prep("gf_full", gf)
    d = outdir / "train" / "dl"; d.mkdir(); d2 = outdir / "train" / "dl_ds"; d2.mkdir()
    dl, dlds = [], []
    for i, s in enumerate(tr):
        dst = d / f"n_{i:03d}.jpg"; shutil.copy2(s, dst); dl.append(str(dst))
        img = area(dec_full(s), 1024)
        dst2 = d2 / f"n_{i:03d}.png"
        cv2.imwrite(str(dst2), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); dlds.append(str(dst2))
    prep_t["dl_full"] = prep("dl_full", dl)
    prep_t["dl_ds1024"] = prep("dl_ds1024", dlds)

    print("[3/7] warmup", flush=True)
    for cat, p in (("gf_full", png_tests[0]), ("dl_full", acc_imgs[0][0]), ("dl_ds1024", acc_imgs[0][0])):
        for _ in range(2):
            engine.predict_ndarray(cat, dec_full(p), path_hint=p)

    # ---- 精度 ----
    print(f"[4/7] precision {len(acc_imgs)} imgs x variants", flush=True)
    prec = []
    for k, (p, lab) in enumerate(acc_imgs):
        full = dec_full(p); H, W = full.shape[:2]
        mask = None
        if lab == "defect":
            m = cv2.imdecode(np.fromfile(str(gmasks[Path(p).stem]), np.uint8), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0 if m is not None else None
        ref = engine.predict_ndarray("dl_full", full, path_hint=p)
        rep = engine.predict_ndarray("dl_full", full, path_hint=p)
        ref_boxes = boxes_orig(ref, 1, 1)
        row = {"image": Path(p).name, "label": lab, "size": [W, H], "full": {
            "score": ref.score, "decision": ref.decision, "slots": ref.slot_scores, "n_boxes": len(ref_boxes),
            "gt_hit": gt_hit(ref_boxes, mask), "repeat_score_absdiff": abs(ref.score - rep.score)}}
        runs = [("dl_full", v) for v in ("area1024", "area512", "reduced4")] + [("dl_ds1024", "area1024")]
        for cat, v in runs:
            img = VARIANTS[v](p); h, w = img.shape[:2]
            r = engine.predict_ndarray(cat, img, path_hint=p)
            bx = boxes_orig(r, W / w, H / h)
            ious = greedy_iou(ref_boxes, bx)
            row[f"{cat}:{v}"] = {
                "shape": [w, h], "score": r.score, "score_absdiff": abs(r.score - ref.score),
                "decision": r.decision, "decision_same": r.decision == ref.decision,
                "slots_maxdiff": max([abs(ref.slot_scores[s] - r.slot_scores.get(s, 1e9)) for s in ref.slot_scores] or [0]),
                "slots": r.slot_scores, "n_boxes": len(bx),
                "box_iou_mean": float(np.mean(ious)) if ious else None,
                "box_iou_min": float(np.min(ious)) if ious else None,
                "gt_hit": gt_hit(bx, mask), "heatmap_corr": hm_corr(ref.fused_heatmap, r.fused_heatmap)}
        prec.append(row)
        print(f"  {k+1}/{len(acc_imgs)} {row['image']} {lab} full={ref.score:.3f}/{ref.decision} " + " ".join(
            f"{key.split(':')[0][3:]}:{key.split(':')[1]}={row[key]['score']:.3f}/{row[key]['decision'][:4]}"
            for key in row if ":" in key), flush=True)

    # ---- 精度汇总 ----
    def summ_var(key):
        rr = [r for r in prec]
        g = lambda r: r["full"] if key == "full" else r[key]
        out = {"n": len(rr),
               "good_flagged(non-normal)": sum(g(r)["decision"] != "normal" for r in rr if r["label"] == "good"),
               "good_anomaly": sum(g(r)["decision"] == "anomaly" for r in rr if r["label"] == "good"),
               "defect_detected(non-normal)": sum(g(r)["decision"] != "normal" for r in rr if r["label"] == "defect"),
               "defect_anomaly": sum(g(r)["decision"] == "anomaly" for r in rr if r["label"] == "defect"),
               "defect_gt_hit": sum(bool(g(r)["gt_hit"]) for r in rr if r["label"] == "defect"),
               "n_good": sum(r["label"] == "good" for r in rr), "n_defect": sum(r["label"] == "defect" for r in rr)}
        if key != "full":
            out["decision_flips"] = sum(not r[key]["decision_same"] for r in rr)
            out["score_absdiff"] = pct([r[key]["score_absdiff"] for r in rr])
            out["slots_maxdiff"] = pct([r[key]["slots_maxdiff"] for r in rr])
            ious = [r[key]["box_iou_mean"] for r in rr if r[key]["box_iou_mean"] is not None]
            out["box_iou_mean"] = pct(ious) if ious else None
            hc = [r[key]["heatmap_corr"] for r in rr if r[key]["heatmap_corr"] is not None]
            out["heatmap_corr"] = pct(hc) if hc else None
            out["gt_hit_lost_vs_full"] = sum(bool(r["full"]["gt_hit"]) and not r[key]["gt_hit"] for r in rr)
            out["detected_lost_vs_full"] = sum(r["label"] == "defect" and r["full"]["decision"] != "normal"
                                               and r[key]["decision"] == "normal" for r in rr)
        else:
            out["repeat_score_absdiff_max"] = max(r["full"]["repeat_score_absdiff"] for r in rr)
        return out

    prec_summary = {k: summ_var(k) for k in ("full", "dl_full:area1024", "dl_full:area512",
                                             "dl_full:reduced4", "dl_ds1024:area1024")}
    print("PREC_SUMMARY", json.dumps(prec_summary, ensure_ascii=False, indent=1), flush=True)

    # ---- 延迟 ----
    jpg_lat = [p for p, _ in acc_imgs if Image.open(p).size == (4096, 3000)][:6]
    lat_sets = [("png2500", "gf_full", png_tests, ("full", "area1024")),
                ("jpg4096x3000", "dl_full", jpg_lat, ("full", "area1024", "reduced4"))]
    print(f"[5/7] latency A/B {args.rounds} rounds (service.detect_image, persist=True, with_heatmap=True)", flush=True)
    rows = []
    for gname, cat, imgs, vs in lat_sets:
        for p in imgs[:2]:  # 预热该组
            for v in vs:
                svc._decode_rgb = VARIANTS[v]; service.detect_image(p, cat, persist=True, with_heatmap=True)
        for rnd in range(args.rounds):
            for i, p in enumerate(imgs):
                order = list(vs) if (rnd + i) % 2 == 0 else list(reversed(vs))
                msg = []
                for v in order:
                    svc._decode_rgb = VARIANTS[v]
                    st = time.perf_counter()
                    r = service.detect_image(p, cat, persist=True, with_heatmap=True)
                    wall = (time.perf_counter() - st) * 1000
                    infer = float(r["latency_ms"]); total = float(r["total_latency_ms"]); e2e = float(r["latency_e2e_ms"])
                    rows.append({"group": gname, "variant": v, "image": Path(p).name, "decode_ms": total - infer,
                                 "infer_ms": infer, "render_ms": e2e - total, "e2e_ms": e2e, "wall_ms": wall,
                                 "decision": r["decision"], "score": r["final_score"]})
                    msg.append(f"{v}={e2e:.0f}")
                print(f"  [{gname}] round {rnd+1}/{args.rounds} img {i+1}/{len(imgs)} " + " ".join(msg), flush=True)
    svc._decode_rgb = orig_decode
    lat = {}
    for gname, _, _, vs in lat_sets:
        lat[gname] = {}
        for v in vs:
            rr = [r for r in rows if r["group"] == gname and r["variant"] == v]
            lat[gname][v] = {k: pct([r[k] for r in rr]) for k in ("decode_ms", "infer_ms", "render_ms", "e2e_ms", "wall_ms")}
    print("LAT_SUMMARY", json.dumps(lat, ensure_ascii=False, indent=1), flush=True)

    print("[6/7] write ds_ab.json", flush=True)
    out = {"isolated_temp": str(outdir), "cv2": cv2.__version__, "prepare_s": prep_t,
           "n_eval_images": len(acc_imgs) + len(png_tests), "n_prepare_each": args.prepare_n,
           "rounds": args.rounds, "precision_summary": prec_summary, "latency_summary": lat,
           "precision": prec, "latency_rows": rows, "elapsed_s": round(time.perf_counter() - t_start, 1)}
    (OUT / "ds_ab.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("[7/7] dispose + cleanup", flush=True)
    try:
        from backend.db import database
        database.get_engine().dispose()
    except Exception as e:  # noqa: BLE001
        print("dispose skipped:", e)
    shutil.rmtree(outdir, ignore_errors=True)
    print(f"DONE elapsed={time.perf_counter()-t_start:.0f}s temp_removed={not outdir.exists()}", flush=True)


if __name__ == "__main__":
    main()
