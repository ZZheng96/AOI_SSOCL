"""§78 探针：_gate_state_token 与 _gate_anchor_pass 相对逐张 _predict_final_fused 的额外开销（TEMP 引擎）。"""
import sys, time
sys.argv = [sys.argv[0], "after"]
import bench_learnlock as B
import numpy as np

_orig_main = B.main


def run():
    import os, shutil, tempfile, yaml, random
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp(prefix="aoi_probe78_"))
    cfg = yaml.safe_load((B.ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["system"].update(storage_dir=str(tmp / "storage"), engine_storage=str(tmp / "engine"),
                         engine_base_cfg=str(B.ROOT / "configs" / "engine_fast.yaml"),
                         database_url="sqlite:///" + (tmp / "p.db").as_posix(), host="127.0.0.1")
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["AOI_CONFIG"] = str(tmp / "config.yaml")
    os.chdir(B.ROOT)
    sys.path.insert(0, str(B.ROOT))
    from backend.engine import get_engine
    pool = sorted(p for p in B.ARCHIVE.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    random.seed(777)
    random.shuffle(pool)
    engine = get_engine()
    td = tmp / "train"
    td.mkdir(parents=True)
    train = []
    for i, s in enumerate(pool[:B.N_TRAIN]):
        d = td / f"n{i:03d}{s.suffix}"
        shutil.copy2(s, d)
        train.append(str(d))
    engine.prepare("gf", {"init_normal": train, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: None)
    pipe, _ = engine._pipe_entry("gf")
    paths = list(pipe.handler.gate.anchor_paths)
    print("anchors", len(paths), "trace_api", pipe.trace_api is not None, flush=True)
    for _ in range(2):
        [pipe._predict_final_fused(p) for p in paths]
    for r in range(3):
        t = time.perf_counter(); a = [pipe._predict_final_fused(p) for p in paths]; t1 = time.perf_counter()
        b = pipe._gate_anchor_pass(paths); t2 = time.perf_counter()
        tok = pipe._gate_state_token(); t3 = time.perf_counter()
        print(f"round{r}: per_image={1e3*(t1-t):.1f}ms anchor_pass={1e3*(t2-t1):.1f}ms "
              f"token={1e3*(t3-t2):.2f}ms equal={a == [x[0] for x in b]}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


run()
