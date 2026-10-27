"""探针：量化 submit_feedback 锁内 pipe.feedback 各段耗时 + 影子 deepcopy 成本（少量本地样本）。"""
import copy, functools, json, os, random, shutil, sys, tempfile, time
from pathlib import Path

import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test02_learn_lock\dev_split")
ARCHIVE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="aoi_probe_split_"))
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
    t0 = time.perf_counter()
    engine.prepare("gf", {"init_normal": paths, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: None)
    print(f"[prepare] {time.perf_counter()-t0:.1f}s", flush=True)
    pipe, _ = engine._pipe_entry("gf")
    h = pipe.handler
    prof = {}

    def wrap(obj, name):
        f = getattr(obj, name)

        @functools.wraps(f)
        def g(*a, **k):
            t = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                prof.setdefault(name, []).append((time.perf_counter() - t) * 1000)
        setattr(obj, name, g)

    for n in ("_record", "_reflow_labeled", "_add_labeled", "_maybe_thresh_recal",
              "_maybe_pair_thresh", "_maybe_head_ft", "_maybe_disc_ft", "_maybe_apply_weights",
              "_labeled_defect_recall"):
        if hasattr(h, n):
            wrap(h, n)
    for n in ("_record", "_recalibrate_on_reflow", "_sem_add"):
        if hasattr(pipe, n):
            wrap(pipe, n)
    for n in ("check", "recenter"):
        wrap(h.gate, n)
    inputs = [str(p) for p in pool[40:52]]
    events = []
    for i, p in enumerate(inputs):
        engine.predict_image_path("gf", p)
        # deepcopy 成本（排除 backbone/线程池/锁/缓存）
        memo = {id(pipe.backbone): pipe.backbone, id(pipe._cpu_exec): pipe._cpu_exec,
                id(pipe._trad_lock): pipe._trad_lock, id(pipe._item_cache): {}}
        if pipe.trace_api is not None:
            memo[id(pipe.trace_api)] = pipe.trace_api
        tc = time.perf_counter()
        try:
            shadow = copy.deepcopy(pipe, memo)
            dc_ms = (time.perf_counter() - tc) * 1000
            del shadow
            dc_err = None
        except Exception as e:  # noqa: BLE001
            dc_ms, dc_err = None, repr(e)
        prof.clear()
        r = engine.submit_feedback("gf", p, "correct", label=0)
        events.append({"i": i, "lock": r.get("lock"), "feedback_ms": r.get("feedback_ms"),
                       "action": r.get("action"), "deepcopy_ms": dc_ms, "deepcopy_err": dc_err,
                       "parts": {k: round(sum(v), 1) for k, v in prof.items()},
                       "calls": {k: len(v) for k, v in prof.items()}})
        print(f"  [{i+1}/{len(inputs)}] hold={r['lock']['lock_hold_ms']} fb={r.get('feedback_ms')} "
              f"dc={dc_ms if dc_ms is None else round(dc_ms,1)} {events[-1]['parts']}", flush=True)
    (OUT / "probe_breakdown.json").write_text(json.dumps(events, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
