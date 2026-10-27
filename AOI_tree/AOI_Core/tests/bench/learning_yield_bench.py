"""AC-069 真实图小测：前台推理 P95 在 idle / 后台即学并发下的对比（≤20 张图）。

- 开发版 gold_finger 快照复制到 TEMP 使用，不写生产快照/DB。
- learn_before：模拟改动前（不让行 + 特征提取在品类锁内）；
  learn_after：当前实现（有上界让行 + 锁外预提取）；
  learn_after_cached：反馈图已被前台检测过（特征缓存命中，预提取跳过）。
- learning_parallel 显式 False（生产默认）。

用法（AOI_Core 下）：python -m tests.bench.learning_yield_bench [--fg 10 --fb 10 --reqs 30]
"""
import argparse, json, os, shutil, sys, tempfile, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "storage" / "engine" / "snapshots" / "gold_finger"
ARCHIVE = ROOT / "storage" / "archive" / "gold_finger" / "20261011"
CAT = "gold_finger"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg", type=int, default=10, help="前台图数")
    ap.add_argument("--fb", type=int, default=10, help="反馈图数（fg+fb≤20）")
    ap.add_argument("--reqs", type=int, default=30, help="每档前台请求数")
    ap.add_argument("--version", type=int, default=1, help="TEMP 副本使用的快照版本")
    ap.add_argument("--out", default="", help="产物目录（summary.json/per_request.json）")
    args = ap.parse_args()
    assert args.fg + args.fb <= 20
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from backend.engine import DetectionEngine

    tmp = Path(tempfile.mkdtemp(prefix="aoi_ac069_bench_"))
    print(f"[isolate] {tmp}", flush=True)
    shutil.copytree(SNAP, tmp / "engine" / "snapshots" / CAT)
    # 生产 current=v2 为无主干降配快照（feats=None，即学在 add_ext 处失败），
    # 仅在 TEMP 副本上把指针切到带 DINO 主干的版本
    (tmp / "engine" / "snapshots" / CAT / "current.json").write_text(
        json.dumps({"version": args.version, "set_at": "bench"}), encoding="utf-8")
    imgs = sorted(p for p in ARCHIVE.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    imgs = imgs[:args.fg + args.fb]
    fg_imgs = [str(p) for p in imgs[:args.fg]]
    fb_src = imgs[args.fg:]

    eng = DetectionEngine(str(tmp / "engine"), str(ROOT / "configs" / "engine_fast.yaml"))
    eng.learning_parallel = False
    t0 = time.perf_counter()
    eng._ensure_loaded(CAT)
    print(f"[load] v{eng.current_version(CAT)} {time.perf_counter() - t0:.1f}s device={eng.device}",
          flush=True)
    pipe = eng._pipes[CAT][0]
    real_build = type(pipe).build_item

    def fb_copies(tag):
        d = tmp / "fb" / tag
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for i, p in enumerate(fb_src):
            dst = d / f"{tag}_{i:02d}{p.suffix}"
            shutil.copy2(p, dst)
            out.append(str(dst))
        return out

    for p in fg_imgs[:3]:   # 热身
        eng.predict_image_path(CAT, p)
    print("[warmup] done", flush=True)

    summary, per_request, learn_all = {}, [], {}
    for mode in ("idle", "learn_before", "learn_after", "learn_after_cached"):
        fb_paths = fb_copies(mode) if mode != "idle" else []
        if mode == "learn_after_cached":
            for p in fb_paths:          # 模拟“先检测、后反馈”：特征已在缓存
                eng.predict_image_path(CAT, p)
        if mode == "learn_before":
            eng.feedback_yield_max_ms = 0
            lock = eng._pipe_lock(CAT)

            def only_in_lock(self, *a, **k):   # 锁外预提取失败 → 回落锁内（=改动前）
                if not lock._is_owned():
                    raise RuntimeError("bench: emulate legacy in-lock extract")
                return real_build(self, *a, **k)
            type(pipe).build_item = only_in_lock
        else:
            eng.feedback_yield_max_ms = DetectionEngine.feedback_yield_max_ms
            type(pipe).build_item = real_build

        stop = threading.Event()
        learn_log = []

        def learn_loop():
            i = 0
            while not stop.is_set() and i < len(fb_paths):
                ev = threading.Event()
                box = {}
                ts = time.perf_counter()
                try:
                    eng.submit_feedback(CAT, fb_paths[i], "correct", label=0,
                                        on_result=lambda r, e: (box.update(r=r, e=e), ev.set()))
                except Exception as e:  # noqa: BLE001 记录后继续（on_result 已带回 e）
                    box.setdefault("e", e)
                    ev.set()
                ev.wait(30)
                lk = (box.get("r") or {}).get("lock") or {}
                learn_log.append({"wall_ms": (time.perf_counter() - ts) * 1000, **lk,
                                  "error": repr(box.get("e")) if box.get("e") else None})
                i += 1

        th = None
        if fb_paths:
            th = threading.Thread(target=learn_loop, daemon=True)
            th.start()
            time.sleep(0.1)
        walls, waits = [], []
        for k in range(args.reqs):
            ts = time.perf_counter()
            r = eng.predict_image_path(CAT, fg_imgs[k % len(fg_imgs)])
            walls.append((time.perf_counter() - ts) * 1000)
            waits.append(r.extra.get("lock_wait_ms") or 0.0)
            per_request.append({"mode": mode, "k": k, "wall_ms": round(walls[-1], 3),
                                "infer_ms": round(float(r.latency_ms), 3),
                                "lock_wait_ms": waits[-1]})
            if (k + 1) % 10 == 0:
                print(f"  [{mode}] {k + 1}/{args.reqs} learned={len(learn_log)}", flush=True)
        stop.set()
        if th:
            th.join(120)
        holds = [x["lock_hold_ms"] for x in learn_log if x.get("lock_hold_ms") is not None]
        ylds = [x["yield_ms"] for x in learn_log if x.get("yield_ms") is not None]
        learn_all[mode] = learn_log
        s = {"n": len(walls), "p50": float(np.percentile(walls, 50)),
             "p95": float(np.percentile(walls, 95)), "max": float(max(walls)),
             "lock_wait_p50": float(np.percentile(waits, 50)),
             "lock_wait_p95": float(np.percentile(waits, 95)), "lock_wait_max": float(max(waits)),
             "lock_wait_gt0": sum(w > 1.0 for w in waits),
             "yield_p50": float(np.percentile(ylds, 50)) if ylds else None,
             "yield_max": float(max(ylds)) if ylds else None,
             "learn_wall_p50": (float(np.percentile([x["wall_ms"] for x in learn_log], 50))
                                if learn_log else None),
             "learned": len(learn_log),
             "hold_p50": float(np.percentile(holds, 50)) if holds else None,
             "hold_max": float(max(holds)) if holds else None,
             "prefetched": sum(bool(x.get("prefetched")) for x in learn_log),
             "errors": [x["error"] for x in learn_log if x.get("error")]}
        summary[mode] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()}
        print(f"[{mode}] " + json.dumps(summary[mode], ensure_ascii=False), flush=True)

    type(pipe).build_item = real_build
    summary["lock_stats"] = eng.lock_stats()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        summary["args"] = vars(args)
        summary["device"] = eng.device
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        (out / "per_request.json").write_text(json.dumps(
            {"foreground": per_request, "learning": learn_all}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[out] {out}", flush=True)
    del eng, pipe
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
