"""round3_speed：单图纯推理分阶段 profiler + 逐图精度对照。

口径与 final_perf/bench_decode.py 一致：开发版代码 + 稳定版快照 TEMP 只读副本，
latency_ms = engine.predict_ndarray 的 dt_ms（锁外预提取 + 锁内 predict，不含解码/渲染/落库）。
用法：python bench_profile.py <tag> [--cats gold_finger,smt_solder] [--reps 1]
输出：bench_<tag>.json（逐图 score/decision/slot 分 + 分阶段耗时 + 汇总）
"""
import argparse, json, os, random, shutil, statistics, sys, tempfile, threading, time
from pathlib import Path

import yaml

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STABLE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
GF_ARCHIVE = STABLE / "storage" / "archive" / "gold_finger" / "20261011"
DATA = Path(r"d:\CGAIC\data_local")
HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--cats", default="gold_finger,smt_solder")
ap.add_argument("--reps", type=int, default=1)
args = ap.parse_args()
CATS = args.cats.split(",")


def sample_images():
    random.seed(20261013)
    out = {}
    pngs = sorted(p for p in GF_ARCHIVE.iterdir() if p.suffix.lower() == ".png")
    gf = random.sample(pngs, 20)                       # 与 final_perf 同 20 张
    rr = random.Random(7)
    gt = DATA / "gold_finger" / "test"
    if gt.exists():
        defects = sorted(p for d in gt.iterdir() if d.is_dir() and d.name != "good"
                         for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".bmp"))
        gf += rr.sample(defects, min(6, len(defects)))
    out["gold_finger"] = gf
    st = DATA / "solder_smt" / "test"
    smt = sorted((st / "good").iterdir())[:6]
    for d in ("bridge", "insufficient", "cold_solder", "excess"):
        ps = sorted(p for p in (st / d).iterdir() if p.suffix.lower() in (".png", ".jpg"))
        smt += rr.sample(ps, min(3, len(ps)))
    out["smt_solder"] = smt
    return out


tmp = Path(tempfile.mkdtemp(prefix="aoi_r3speed_"))
for c in CATS:
    shutil.copytree(STABLE / "storage" / "engine" / "snapshots" / c,
                    tmp / "engine" / "snapshots" / c)
cfg = yaml.safe_load((DEV / "configs" / "default.yaml").read_text(encoding="utf-8"))
cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                     engine_base_cfg=str(DEV / "configs" / "engine_fast.yaml"),
                     database_url="sqlite:///" + (tmp / "bench.db").as_posix(), host="127.0.0.1")
cfg.setdefault("notify", {})["webhook_url"] = ""
(tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
os.chdir(DEV)
sys.path.insert(0, str(DEV))

import numpy as np  # noqa: E402
from backend.pipeline import service as svc_mod  # noqa: E402
from backend.engine import get_engine  # noqa: E402

_tls = threading.local()
EVENTS = []


def instrument(eng, pipe):
    """猴补丁计时：CPU 线程池任务（排队/运行）、各槽位打分、_hier、_to_result。"""
    ex = pipe._cpu_exec
    orig_submit = ex.submit

    def submit(fn, *a, **k):
        name = getattr(fn, "__name__", "task")
        if name in ("_slot_image_score", "_score_trad", "_timed_slot", "slot_score") and a:
            name = "slot:" + getattr(a[0], "name", "?")
        ts = time.perf_counter()
        ev = EVENTS

        def run():
            t1 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                ev.append((name, (t1 - ts) * 1000, (time.perf_counter() - t1) * 1000))
        return orig_submit(run)
    ex.submit = submit

    orig_slot = pipe._slot_image_score

    def slot_score(slot, item):
        t = time.perf_counter()
        try:
            return orig_slot(slot, item)
        finally:
            if threading.current_thread() is threading.main_thread():
                EVENTS.append(("slot:" + slot.name, 0.0, (time.perf_counter() - t) * 1000))
    pipe._slot_image_score = slot_score

    orig_tr = eng._to_result

    def to_result(*a, **k):
        t = time.perf_counter()
        rec = a[1] if len(a) > 1 else k.get("rec")
        _tls.timing = dict(rec.get("timing") or {}) if isinstance(rec, dict) else None
        r = orig_tr(*a, **k)
        _tls.ser_ms = (time.perf_counter() - t) * 1000
        return r
    eng._to_result = to_result


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1)


def summ(xs):
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95),
            "mean": round(statistics.mean(xs), 1), "max": round(max(xs), 1)}


eng = get_engine()
samples = sample_images()
res = {"tag": args.tag, "env": {k: v for k, v in os.environ.items() if k.startswith("AOI_R3")},
       "cats": {}}
for cat in CATS:
    imgs = samples[cat]
    print(f"[{cat}] {len(imgs)} 张，装载+预热", flush=True)
    t = time.perf_counter()
    first = svc_mod._decode_rgb(imgs[0])
    wu = eng.warmup(cat, first)
    print(f"    warmup {wu} ({(time.perf_counter() - t):.1f}s)", flush=True)
    pipe, _ = eng._pipe_entry(cat)
    instrument(eng, pipe)
    rows = []
    for rep in range(args.reps):
        for i, p in enumerate(imgs):
            t = time.perf_counter()
            arr = svc_mod._decode_rgb(p)
            dec_ms = (time.perf_counter() - t) * 1000
            EVENTS.clear()
            _tls.ser_ms = 0.0
            t = time.perf_counter()
            r = eng.predict_ndarray(cat, arr)
            wall = (time.perf_counter() - t) * 1000
            time.sleep(0.02)                       # 让尾部线程池事件落地
            ev = list(EVENTS)
            row = {"image": p.name, "shape": list(arr.shape[:2]), "rep": rep,
                   "decision": r.decision, "score": r.score,
                   "slot_scores": r.slot_scores, "raw_scores": r.raw_scores,
                   "n_boxes": len(r.defect_boxes), "latency_ms": round(r.latency_ms, 2),
                   "wall_ms": round(wall, 2), "decode_ms": round(dec_ms, 2),
                   "serialize_ms": round(_tls.ser_ms, 2),
                   "tasks": {n: {"wait": round(w, 1), "run": round(x, 1)} for n, w, x in ev}}
            row["timing"] = getattr(_tls, "timing", None)
            rows.append(row)
            tk = " ".join(f"{n}={v['run']:.0f}(+{v['wait']:.0f})" for n, v in row["tasks"].items())
            print(f"    [{cat} r{rep} {i + 1}/{len(imgs)}] {arr.shape[1]}x{arr.shape[0]} "
                  f"lat={r.latency_ms:.1f} dec={dec_ms:.0f} {r.decision} {r.score:.4f} "
                  f"tm={row['timing']} {tk}", flush=True)
    lat = [x["latency_ms"] for x in rows]
    agg = {"latency_ms": summ(lat), "decode_ms": summ([x["decode_ms"] for x in rows]),
           "serialize_ms": summ([x["serialize_ms"] for x in rows])}
    for k in ("io", "tile", "forward", "score", "local", "hier", "decide", "total"):
        vs = [x["timing"][k] for x in rows if x.get("timing") and k in x["timing"]]
        if vs:
            agg["t_" + k] = summ(vs)
    names = sorted({n for x in rows for n in x["tasks"]})
    for n in names:
        vs = [x["tasks"][n]["run"] for x in rows if n in x["tasks"]]
        ws = [x["tasks"][n]["wait"] for x in rows if n in x["tasks"]]
        agg["run:" + n] = summ(vs)
        agg["wait:" + n] = summ(ws)
    res["cats"][cat] = {"rows": rows, "agg": agg}
    print(f"[{cat}] latency {agg['latency_ms']}", flush=True)
    for k, v in agg.items():
        print(f"    {k:28s} p50={v['p50']:8.1f} p95={v['p95']:8.1f} max={v['max']:8.1f}", flush=True)

out = HERE / f"bench_{args.tag}.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("saved", out, flush=True)
shutil.rmtree(tmp, ignore_errors=True)
