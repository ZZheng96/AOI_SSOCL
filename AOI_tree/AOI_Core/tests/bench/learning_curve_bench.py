"""在线学习曲线协议化实测（生产引擎路径，2026-10-11）。

数据协议与 tests.bench.static_accuracy 相同（EXPs 加载器只读引用）；
划分镜像 EXPs m5_learning_curve：
  single：eval_items = test[-eval_n:]，stream = test[:-eval_n]（不重叠）
  rounds：eval_items = test[:eval_n]（固定锚定集，从不反馈），
          pool = test[eval_n : eval_n + rounds*round_size]
学习/打分全部经生产引擎 DetectionEngine.simulate_learning(_rounds)
（独立 pipe 副本回放，不影响线上态）。

口径声明：生产 SSOCL 为 2026-10 根因修复后版本（LOO 诚实打分、双侧阈值重估、
门控拒写回等），结果属**新诚实口径**；与项目文档旧口径（含已修复的自匹配
泄漏）双轨登记，方向性对照，不逐位对账。

用法：
    python -m tests.bench.learning_curve_bench --dataset mvtec --category screw --mode rounds --rounds 3
    python -m tests.bench.learning_curve_bench --dataset datalocal --category gold_finger --mode single --eval-n 40 --max-feedback 30
    python -m tests.bench.learning_curve_bench --dataset gyudet --mode rounds --rounds 10 --max-test 380
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.bench.static_accuracy import load_bundle  # noqa: E402  复用同一数据协议


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["mvtec", "btad", "mpdd", "datalocal", "gyudet"])
    ap.add_argument("--category", default=None, help="gyudet 可省略")
    ap.add_argument("--mode", default="rounds", choices=["single", "rounds"])
    ap.add_argument("--profile", default="accuracy")
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-size", type=int, default=30)
    ap.add_argument("--feedback-ratio", type=float, default=1.0)
    ap.add_argument("--max-feedback", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--storage", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat = args.category or "gyudet"
    storage = args.storage or tempfile.mkdtemp(prefix="aoi_learn_bench_")
    print(f"[learning_bench] {args.dataset}/{cat} mode={args.mode} storage={storage}", flush=True)

    bundle = load_bundle(args.dataset, args.category or "gyudet", args.seed, args.max_test)
    test = list(bundle["test"])
    print(f"[data] init_normal={len(bundle['init_normal'])} init_defect={len(bundle['init_defect'])} "
          f"test={len(test)}", flush=True)

    from backend.engine import DetectionEngine  # noqa: E402
    engine = DetectionEngine(storage, str(REPO_ROOT / "configs" / "engine_fast.yaml"))
    print(f"[engine] device={engine.device}", flush=True)
    t0 = time.time()
    v = engine.prepare(cat, bundle, scenario="L1a", profile=args.profile,
                       trigger="learning_bench", note=f"learning_curve_bench {args.dataset}/{cat}")
    print(f"[prepare] v{v} ({time.time() - t0:.0f}s)", flush=True)

    if args.mode == "single":
        eval_items = test[-args.eval_n:]
        stream = test[:len(test) - args.eval_n]
        print(f"[protocol] single: stream={len(stream)} eval={len(eval_items)} "
              f"feedback_ratio={args.feedback_ratio} max_feedback={args.max_feedback}", flush=True)
        g = engine.simulate_learning(cat, stream, eval_items,
                                     feedback_ratio=args.feedback_ratio,
                                     eval_every=max(1, (args.max_feedback or 30) // 6),
                                     max_feedback=args.max_feedback)
    else:
        eval_items = test[:args.eval_n]
        pool = test[args.eval_n: args.eval_n + args.rounds * args.round_size]
        assert len(pool) >= args.rounds * args.round_size * 0.5, \
            f"pool 不足: {len(pool)}（eval_n+rounds*size 超 test={len(test)}，请调 --max-test）"
        print(f"[protocol] rounds: pool={len(pool)} eval={len(eval_items)} "
              f"rounds={args.rounds}x{args.round_size}", flush=True)
        g = engine.simulate_learning_rounds(cat, pool, eval_items,
                                            round_size=args.round_size,
                                            n_rounds=args.rounds)

    out = {"dataset": args.dataset, "category": cat, "mode": args.mode,
           "profile": args.profile, "seed": args.seed,
           "protocol": {"eval_n": len(eval_items),
                        "rounds": args.rounds if args.mode == "rounds" else None,
                        "round_size": args.round_size if args.mode == "rounds" else None},
           "caliber": "生产引擎 SSOCL（2026-10 LOO 诚实口径）",
           "initial_auroc": g.get("initial_auroc"), "final_auroc": g.get("final_auroc"),
           "auroc_gain": g.get("auroc_gain"), "peak_auroc": g.get("peak_auroc"),
           "curve": g.get("curve"), "gate": g.get("gate") or g.get("gate_stats"),
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[result] initial={out['initial_auroc']} final={out['final_auroc']} "
          f"gain={out['auroc_gain']} peak={out['peak_auroc']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
