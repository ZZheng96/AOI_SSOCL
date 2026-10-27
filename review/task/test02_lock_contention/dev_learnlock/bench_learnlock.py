"""§78 bench：学习进行中前台推理 wall P50/P95 + 学习锁内持有 ms，before/after 同进程状态序列对比。
用法：python bench_learnlock.py before|after|ab（ab=同进程逐事件交替关/开；BENCH_NFB 改反馈数）
  before = 关闭 §78 复用（Pipeline._gate_anchor_pass=None → RegressGate 逐张重算，等价改动前）
  after  = 默认（check→recenter 锚定分复用）
TEMP 引擎/TEMP sqlite，图片取 stable archive gold_finger 少量样本；不碰生产 DB/快照。"""
import json, os, random, shutil, sys, tempfile, threading, time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
OUT = Path(__file__).resolve().parent
ARCHIVE = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\storage\archive\gold_finger\20261011")
N_TRAIN, N_FB, N_FG, N_EVAL = 40, int(os.environ.get("BENCH_NFB", 12)), 6, 8


def pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def main(mode):
    tmp = Path(tempfile.mkdtemp(prefix=f"aoi_bench78_{mode}_"))
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
    from algo.eval.offline import Pipeline
    if mode == "before":
        Pipeline._gate_anchor_pass = None
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
    train = []
    for i, s in enumerate(pool[:N_TRAIN]):
        d = td / f"n{i:03d}{s.suffix}"
        shutil.copy2(s, d)
        train.append(str(d))
    t0 = time.perf_counter()
    engine.prepare("gf", {"init_normal": train, "init_defect": [], "val": [], "test": []},
                   activate=True, progress_cb=lambda f, m: None)
    print(f"[{mode}] prepare {time.perf_counter()-t0:.1f}s", flush=True)
    a, b = N_TRAIN, N_TRAIN + N_FB
    fb_paths = [str(p) for p in pool[a:b]]
    fg_paths = [str(p) for p in pool[b:b + N_FG]]
    ev_paths = [str(p) for p in pool[b + N_FG:b + N_FG + N_EVAL]]
    for p in fg_paths:                       # 预热：前台测的是缓存命中推理的锁竞争
        engine.predict_image_path("gf", p)
    for p in fb_paths:                       # 反馈前先推理（同探针流程）
        engine.predict_image_path("gf", p)

    stop, fg = threading.Event(), []

    def fg_loop():
        i = 0
        while not stop.is_set():
            t = time.perf_counter()
            engine.predict_image_path("gf", fg_paths[i % N_FG])
            fg.append((time.perf_counter() - t) * 1000)
            i += 1
            time.sleep(0.02)

    # 无学习时的前台基线
    base = []
    for i in range(3 * N_FG):
        t = time.perf_counter()
        engine.predict_image_path("gf", fg_paths[i % N_FG])
        base.append((time.perf_counter() - t) * 1000)
    th = threading.Thread(target=fg_loop, daemon=True)
    th.start()
    events = []
    orig_pass = Pipeline._gate_anchor_pass
    for i, p in enumerate(fb_paths):
        if mode == "ab":                     # 同进程逐事件交替关/开复用，消除跨进程噪声
            Pipeline._gate_anchor_pass = None if i % 2 == 0 else orig_pass
        r = engine.submit_feedback("gf", p, "correct", label=0)
        lk = r.get("lock") or {}
        events.append({"i": i, "action": r.get("action"), "lock": lk,
                       "reuse_on": Pipeline._gate_anchor_pass is not None})
        print(f"  [{mode} {i+1}/{N_FB}] action={r.get('action')} hold={lk.get('lock_hold_ms')} "
              f"fg_n={len(fg)}", flush=True)
    stop.set()
    th.join(5)
    pipe, _ = engine._pipe_entry("gf")
    gate = pipe.handler.gate
    ev_raw = [float(pipe._predict_final_fused(p)) for p in ev_paths]
    holds = [e["lock"].get("lock_hold_ms") for e in events if e["lock"].get("lock_hold_ms") is not None]
    acc = [e["lock"]["lock_hold_ms"] for e in events if "accepted" in str(e["action"])]
    rb = [e["lock"]["lock_hold_ms"] for e in events if "accepted" not in str(e["action"])]
    res = {"mode": mode, "n_feedback": N_FB, "n_fg_during_learning": len(fg),
           "fg_idle_ms": {"p50": pct(base, 50), "p95": pct(base, 95)},
           "fg_during_learning_ms": {"p50": pct(fg, 50), "p95": pct(fg, 95), "max": pct(fg, 100)},
           "lock_hold_ms": {"mean": round(float(np.mean(holds)), 1), "p50": pct(holds, 50),
                            "p95": pct(holds, 95), "max": pct(holds, 100)},
           "lock_hold_accepted_ms": acc, "lock_hold_rolled_back_ms": rb,
           "actions": [e["action"] for e in events],
           "gate_reuse_stats": getattr(gate, "reuse_stats", None),
           "gate_base_mean": gate.base_mean, "gate_base_scores": gate.base_scores,
           "eval_fused_after_learning": ev_raw, "events": events, "fg_raw_ms": fg}
    (OUT / f"bench_{mode}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                       default=str), encoding="utf-8")
    print(f"[{mode}] hold mean={res['lock_hold_ms']['mean']} p95={res['lock_hold_ms']['p95']} "
          f"fg p50={res['fg_during_learning_ms']['p50']} p95={res['fg_during_learning_ms']['p95']} "
          f"reuse={res['gate_reuse_stats']}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
