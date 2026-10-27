"""探针：gate.check/recenter 内单锚 predict 各段（槽位打分/finalize/tta）耗时分解。"""
import functools, json, os, random, shutil, sys, tempfile, threading, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test02_learn_lock\dev_split")
ARCHIVE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="aoi_probe_gs_"))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                         engine_base_cfg=str(ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (tmp / "p.db").as_posix(), host="127.0.0.1")
    cfg.setdefault("notify", {})["webhook_url"] = ""
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from PIL import Image
    pool = []
    for p in sorted(ARCHIVE.iterdir()):
        if p.suffix.lower() in (".jpg", ".png"):
            with Image.open(p) as im:
                if im.size == (2500, 2500):
                    pool.append(p)
    random.seed(777)
    random.shuffle(pool)
    from backend.engine import get_engine
    engine = get_engine()
    engine.learning_parallel = False
    td = tmp / "train"
    td.mkdir(parents=True)
    paths = []
    for i, s in enumerate(pool[:40]):
        d = td / f"n{i:03d}{s.suffix}"
        shutil.copy2(s, d)
        paths.append(str(d))
    engine.prepare("gf", {"init_normal": paths, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: None)
    pipe, _ = engine._pipe_entry("gf")
    h = pipe.handler
    prof, in_gate = {}, threading.local()

    def rec(name, ms):
        if getattr(in_gate, "on", False):
            prof.setdefault(name, []).append(ms)

    orig_sis = pipe._slot_image_score

    def sis(slot, item):
        t = time.perf_counter()
        try:
            return orig_sis(slot, item)
        finally:
            nm = next((k for k, v in pipe.slots.items() if v is slot), "?")
            prof.setdefault("slot:" + nm, []).append((time.perf_counter() - t) * 1000)
    pipe._slot_image_score = sis
    for n in ("_finalize", "_tta_refine", "_record", "predict"):
        f = getattr(pipe, n)

        def mk(f, n):
            @functools.wraps(f)
            def g(*a, **k):
                t = time.perf_counter()
                try:
                    return f(*a, **k)
                finally:
                    rec(n, (time.perf_counter() - t) * 1000)
            return g
        setattr(pipe, n, mk(f, n))
    def wrap_obj(obj, n, key):
        f = getattr(obj, n)

        @functools.wraps(f)
        def g(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                rec(key, (time.perf_counter() - t) * 1000)
        setattr(obj, n, g)
    for n in ("_hier", "_apply_local_branch", "_prefetch_cpu", "_prefetch_clf_imgfeat"):
        wrap_obj(pipe, n, n)
    wrap_obj(h.defect_bank, "intercept_score", "intercept_score")
    if pipe.head_mgr is not None:
        wrap_obj(pipe.head_mgr, "boost", "head_boost")
    if pipe.trace_api is not None:
        wrap_obj(pipe.trace_api, "record", "trace_record")
    fin = pipe._finalize

    def fin2(item, r):
        out_ = fin(item, r)
        rec("dec:" + str(out_.get("decision")), 0.0)
        return out_
    pipe._finalize = fin2
    raws = {}
    rc = pipe._record

    def rc2(item, cpu_futures=None):
        r = rc(item, cpu_futures=cpu_futures)
        if getattr(in_gate, "on", False):
            raws.setdefault(item["path"], []).append(dict(r["raw_scores"]))
        return r
    pipe._record = rc2
    for n in ("check", "recenter"):
        f = getattr(h.gate, n)

        def mk2(f, n):
            @functools.wraps(f)
            def g(*a, **k):
                in_gate.on = True
                t = time.perf_counter()
                try:
                    return f(*a, **k)
                finally:
                    in_gate.on = False
                    prof.setdefault("gate." + n, []).append((time.perf_counter() - t) * 1000)
            return g
        setattr(h.gate, n, mk2(f, n))
    out = []
    print(f"slots={list(pipe.slots)} score_slots={sorted(pipe._score_slots)} "
          f"needs_dino={ {k: v.needs_dino for k, v in pipe.slots.items()} } "
          f"tta={pipe.tta_enabled} n_anchor={len(h.gate.anchor_paths)}", flush=True)
    for i, p in enumerate(pool[40:46]):
        engine.predict_image_path("gf", str(p))
        prof.clear()
        r = engine.submit_feedback("gf", str(p), "correct", label=0)
        e = {k: [round(sum(v), 1), len(v)] for k, v in prof.items()}
        e["hold"] = r["lock"]["lock_hold_ms"]
        out.append(e)
        print(f"  [{i+1}/6] {e}", flush=True)
    inv = {}
    for pth, lst in raws.items():
        for k in lst[0]:
            vals = {v[k] for v in lst}
            inv.setdefault(k, []).append(len(vals) == 1)
    print("raw 不变比例:", {k: f"{sum(v)}/{len(v)}" for k, v in inv.items()}, flush=True)
    meta = {"slots": list(pipe.slots), "score_slots": sorted(pipe._score_slots),
            "needs_dino": {k: bool(v.needs_dino) for k, v in pipe.slots.items()},
            "tta": bool(pipe.tta_enabled), "n_anchor": len(h.gate.anchor_paths)}
    (OUT / "probe_gate_slots.json").write_text(json.dumps({"meta": meta, "events": out},
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
