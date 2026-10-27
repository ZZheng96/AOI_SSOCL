"""静态精度协议化实测（生产引擎路径，2026-10-11）。

目的：把"文档静态精度指标可复现"从研究脚本（EXPs m0_baseline）扩展到
生产引擎——数据协议（100+30 init、n_eval_good、max_test、seed=42、
init_defect 从评测集剔除）只读引用 EXPs/src/data 加载器（与产生 U99/U55
快照完全同协议，保证可比性），全部 fit/打分经生产引擎
backend.engine.DetectionEngine + algo Pipeline.evaluate。

用法：
    python -m tests.bench.static_accuracy --dataset mvtec --profile accuracy
    python -m tests.bench.static_accuracy --dataset mvtec --categories bottle,screw --profile fast
    python -m tests.bench.static_accuracy --dataset btad --profile accuracy
    python -m tests.bench.static_accuracy --dataset datalocal --profile accuracy
    python -m tests.bench.static_accuracy --dataset gyudet --profile accuracy
报告写 --out 指定 JSON；--storage 可复用已准备品类（默认临时目录，跑完即弃）。
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
# 数据协议加载器（只读）：保证与随包快照同分布、同抽样序列
EXPS_SRC = REPO_ROOT.parent / "EXPs" / "src"
if str(EXPS_SRC) not in sys.path:
    sys.path.insert(0, str(EXPS_SRC))

DATA_ROOTS = {
    "mvtec": os.environ.get("AOI_DATA_MVTEC", r"D:\CGAIC\data_origin\mvtec"),
    "btad": os.environ.get("AOI_DATA_BTAD", r"D:\CGAIC\data_origin\BTAD\BTech_Dataset_transformed"),
    "mpdd": os.environ.get("AOI_DATA_MPDD", r"D:\CGAIC\data_origin\MPDD"),
    "datalocal": os.environ.get("AOI_DATA_DATALOCAL", r"D:\CGAIC\data_local"),
    "gyudet": os.environ.get("AOI_DATA_GYUDET", r"D:\CGAIC\data_origin\GYU-DET"),
}
DATALOCAL_CATEGORIES = ["component", "extra_part", "gold_finger", "solder_smt"]


def load_bundle(dataset: str, category: str | None, seed: int, max_test: int):
    """与 EXPs m0_baseline 相同的加载语义（n_eval_good: mvtec=25，其余 None）。"""
    from data import mvtec_like, datalocal, gyudet  # noqa: E402  (EXPs 协议加载器，只读)
    root = DATA_ROOTS[dataset]
    if dataset == "mvtec":
        return mvtec_like.load_category(root, category, seed=seed,
                                        n_eval_good=25, max_test=max_test)
    if dataset == "btad":
        return mvtec_like.load_btad(root, category, seed=seed, max_test=max_test)
    if dataset == "mpdd":
        return mvtec_like.load_category(root, category, seed=seed, max_test=max_test)
    if dataset == "datalocal":
        return datalocal.load_category(root, category, seed=seed, max_test=max_test)
    if dataset == "gyudet":
        return gyudet.load_gyudet(root, seed=seed, max_test=max_test)
    raise ValueError(dataset)


def default_categories(dataset: str) -> list[str]:
    from data import mvtec_like  # noqa: E402
    return {
        "mvtec": mvtec_like.MVTEC_CATEGORIES,
        "btad": mvtec_like.BTAD_CATEGORIES,
        "mpdd": mvtec_like.MPDD_CATEGORIES,
        "datalocal": DATALOCAL_CATEGORIES,
        "gyudet": ["gyudet"],
    }[dataset]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["mvtec", "btad", "mpdd", "datalocal", "gyudet"])
    ap.add_argument("--categories", default=None, help="逗号分隔；缺省全品类")
    ap.add_argument("--profile", default="accuracy", choices=["fast", "accuracy", "cpu"])
    ap.add_argument("--scenario", default="L1a")
    ap.add_argument("--base-cfg", default=None,
                    help="缺省 configs/engine_fast.yaml（9 槽全定义；profile/scenario 决定启用子集）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=150)
    ap.add_argument("--storage", default=None, help="快照存储目录（缺省临时目录）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cats = args.categories.split(",") if args.categories else default_categories(args.dataset)
    base_cfg = args.base_cfg or str(REPO_ROOT / "configs" / "engine_fast.yaml")
    storage = args.storage or tempfile.mkdtemp(prefix="aoi_static_bench_")
    print(f"[static_accuracy] dataset={args.dataset} profile={args.profile} "
          f"base_cfg={os.path.basename(base_cfg)} storage={storage}", flush=True)

    from backend.engine import DetectionEngine  # noqa: E402
    engine = DetectionEngine(storage, base_cfg)
    print(f"[engine] device={engine.device}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    report = {"dataset": args.dataset, "profile": args.profile, "scenario": args.scenario,
              "base_cfg": os.path.basename(base_cfg), "seed": args.seed,
              "max_test": args.max_test, "device": engine.device,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "categories": {}}
    for ci, cat in enumerate(cats, 1):
        key = cat if args.dataset != "gyudet" else "gyudet"
        t0 = time.time()
        print(f"\n=== [{ci}/{len(cats)}] {args.dataset}/{cat} ===", flush=True)
        bundle = load_bundle(args.dataset, cat, args.seed, args.max_test)
        print(f"[data] init_normal={len(bundle['init_normal'])} "
              f"init_defect={len(bundle['init_defect'])} test={len(bundle['test'])}", flush=True)
        v = engine.prepare(key, bundle, scenario=args.scenario, profile=args.profile,
                           trigger="static_bench",
                           note=f"static_accuracy {args.dataset}/{cat} {args.profile}")
        res = engine.evaluate(key, bundle["test"])
        entry = {"version": v, "n_test": len(bundle["test"]),
                 "fused_auroc": res["fused"]["auroc"], "fused_ap": res["fused"].get("ap"),
                 "ms_per_image": res["ms_per_image"],
                 "slots": {n: m["auroc"] for n, m in res.items()
                           if isinstance(m, dict) and "auroc" in m and n not in ("fused", "fused_max")},
                 "fused_max_auroc": res.get("fused_max", {}).get("auroc"),
                 "elapsed_s": round(time.time() - t0, 1)}
        report["categories"][cat] = entry
        print(f"[result] {cat}: fused={entry['fused_auroc']:.4f} "
              f"slots={ {n: round(a, 4) for n, a in entry['slots'].items()} } "
              f"({entry['elapsed_s']}s)", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:      # 每品类落盘，长跑可续查
            json.dump(report, f, ensure_ascii=False, indent=2)

    aucs = [e["fused_auroc"] for e in report["categories"].values()]
    report["summary"] = {"n": len(aucs), "avg_auroc": round(sum(aucs) / max(len(aucs), 1), 4),
                         "ge09": sum(1 for a in aucs if a >= 0.9),
                         "min": round(min(aucs), 4) if aucs else None}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[summary] avg={report['summary']['avg_auroc']} "
          f">=0.9: {report['summary']['ge09']}/{len(aucs)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
