# -*- coding: utf-8 -*-
"""demo4 诚实回退基线验证（demo5 vs demo4 对比，最差情况回退论证）

背景：demo5 的改进必须能证明"最差情况能回退到 demo4 水平"。
本脚本用 demo4 的 Pipeline 在 demo5 的诚实数据协议下跑零样本基线。

红线（必须遵守，见设计文档 U27）：
1. r 分支（参考图配对差分）在 data_local 上靠 test 域文件名配对拿到 0.9983，
   属已确认的数据泄露——本脚本一律传 pairs={}，任何数据集不提供配对参考，
   r 分支在 predict 中自动退化为 g 分支（pipeline.py: scores["r"]=scores["g"]）。
2. 只用少量随机图：每个品类 test 流随机抽样 <=150（demo5 加载器 max_test=150）。
3. 评估协议：fit 只用 init_normal(train/good) + init_defect(协议内缺陷)，
   test 域绝不进 fit；predict 一律 update=False（零样本基线，不做在线学习）。

输出：<PROJECT_ROOT>\\demo5\\outputs\\m0\\demo4_fallback.json
"""
import os
import sys
import json
import time
import traceback

import numpy as np
import torch
import yaml

CGAIC = r"d:\CGAIC"
DEMO4_ROOT = os.path.join(CGAIC, "demo4")
DEMO5_ROOT = os.path.join(CGAIC, "demo5")
if CGAIC not in sys.path:
    sys.path.insert(0, CGAIC)

from demo4.src.pipeline import Pipeline
from demo4.src.evaluation.metrics import compute_metrics, compute_auroc
from demo5.src.data.mvtec_like import load_category, load_btad
from demo5.src.data.gyudet import load_gyudet
from demo5.src.data.datalocal import load_category as load_datalocal_category

OUT_PATH = os.path.join(DEMO5_ROOT, "outputs", "m0", "demo4_fallback.json")
MAX_TEST = 150
N_INIT_NORMAL = 100
N_INIT_DEFECT = 30
SEED = 42
BRANCH_KEYS = ["g", "p", "r", "s", "t"]


def _loader(fn, root, cat):
    return fn(root, cat, n_init_normal=N_INIT_NORMAL, n_init_defect=N_INIT_DEFECT,
              seed=SEED, max_test=MAX_TEST)


def _loader_gyudet(fn, root, cat):
    return fn(root, n_init_normal=N_INIT_NORMAL, n_init_defect=N_INIT_DEFECT,
              seed=SEED, max_test=MAX_TEST)


DATASETS = [
    ("mvtec", _loader, load_category, r"D:\CGAIC\data_origin\mvtec",
     ["bottle", "screw", "transistor", "capsule", "cable", "solder", "grid"]),
    ("btad", _loader, load_btad, r"D:\CGAIC\data_origin\BTAD\BTech_Dataset_transformed",
     ["01", "02", "03"]),
    ("mpdd", _loader, load_category, r"D:\CGAIC\data_origin\MPDD",
     ["bracket_white", "connector", "metal_plate"]),
    ("gyudet", _loader_gyudet, load_gyudet, r"D:\CGAIC\data_origin\GYU-DET",
     ["gyudet"]),
    ("data_local", _loader, load_datalocal_category, r"D:\CGAIC\data_local",
     ["component", "extra_part", "gold_finger", "solder_smt"]),
]


def run_category(ds_name, cat, loader, fn, root, config):
    """单个品类：load -> fit(pairs={}) -> predict(update=False) -> metrics"""
    t0 = time.time()
    print(f"=== {ds_name}/{cat} 开始 ===", flush=True)

    if ds_name == "gyudet":
        cat_data = fn(root, n_init_normal=N_INIT_NORMAL, n_init_defect=N_INIT_DEFECT,
                      seed=SEED, max_test=MAX_TEST)
    else:
        cat_data = fn(root, cat, n_init_normal=N_INIT_NORMAL,
                      n_init_defect=N_INIT_DEFECT, seed=SEED, max_test=MAX_TEST)

    # 诚实红线 1：pairs={} —— 不给任何配对参考，r 分支自动退化为 g 分支
    pipeline = Pipeline(config=config)
    try:
        pipeline.fit_initial(cat_data["init_normal"], cat_data["init_defect"],
                             pairs={})
    except Exception:
        print(f"  [WARN] fit_initial 对 pairs={{}} 抛异常，改传 None 重试", flush=True)
        traceback.print_exc()
        pipeline = Pipeline(config=config)
        pipeline.fit_initial(cat_data["init_normal"], cat_data["init_defect"],
                             pairs=None)

    # 诚实红线 3：零样本基线，predict 一律 update=False
    test = cat_data["test"]
    n = len(test)
    scores_all, labels_all = [], []
    branch_scores = {k: [] for k in BRANCH_KEYS}
    n_fail = 0
    for i, (path, label) in enumerate(test):
        try:
            result = pipeline.predict(path, update=False)
        except Exception as e:
            n_fail += 1
            print(f"  [WARN] predict failed {path}: {e}", flush=True)
            continue
        scores_all.append(result["score"])
        labels_all.append(label)
        for k in BRANCH_KEYS:
            s = result.get("scores", {}).get(k)
            branch_scores[k].append(float(s) if s is not None else 0.0)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [eval] {i+1}/{n}", flush=True)

    m = compute_metrics(labels_all, scores_all)
    branches = {}
    for k in BRANCH_KEYS:
        if len(branch_scores[k]) > 1:
            branches[k] = float(compute_auroc(labels_all, branch_scores[k]))
    out = {
        "auroc": float(m["auroc"]),
        "ap": float(m["ap"]),
        "f1": float(m["f1"]),
        "n": len(scores_all),
        "n_fail": n_fail,
        "branches": branches,
        "n_init_normal": len(cat_data["init_normal"]),
        "n_init_defect": len(cat_data["init_defect"]),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  [cat] {ds_name}/{cat} AUROC={out['auroc']:.4f} AP={out['ap']:.4f} "
          f"F1={out['f1']:.4f} n={out['n']}", flush=True)
    print(f"=== {ds_name}/{cat} 结束 ({out['seconds']}s) ===", flush=True)
    return out


def main():
    with open(os.path.join(DEMO4_ROOT, "configs", "default.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退 CPU（会很慢）", flush=True)
        config["device"] = "cpu"

    results = {}
    failed = []
    for ds_name, loader, fn, root, cats in DATASETS:
        results[ds_name] = {}
        for cat in cats:
            try:
                results[ds_name][cat] = run_category(ds_name, cat, loader, fn, root, config)
            except Exception as e:
                failed.append({"dataset": ds_name, "category": cat,
                               "error": f"{type(e).__name__}: {e}"})
                print(f"  [FAIL] {ds_name}/{cat}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()
            # 增量保存，避免中途崩溃丢失结果
            payload = {"results": results, "failed": failed,
                       "meta": {"protocol": "demo4 pipeline, pairs={} (r 分支禁用)",
                                "max_test": MAX_TEST, "seed": SEED,
                                "config": os.path.join(DEMO4_ROOT, "configs", "default.yaml")}}
            os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    # 汇总表
    print("\n===== 汇总表 =====", flush=True)
    print(f"{'数据集':<11}{'品类':<16}{'AUROC':>8}{'AP':>8}{'F1':>8}{'n':>6}", flush=True)
    for ds_name, cats in results.items():
        for cat, r in cats.items():
            print(f"{ds_name:<11}{cat:<16}{r['auroc']:>8.4f}{r['ap']:>8.4f}"
                  f"{r['f1']:>8.4f}{r['n']:>6}", flush=True)
    if failed:
        print("\n----- 失败品类 -----", flush=True)
        for f_ in failed:
            print(f"  {f_['dataset']}/{f_['category']}: {f_['error']}", flush=True)
    print(f"\n结果已写入: {OUT_PATH}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
