"""M4 裁剪泛化验证（诚实流程）：GYU-DET 独立 val 选子集 → test 验证。

U15 教训：子集选择用 eval 侧标签会过拟合（exp_prune 的 test 上选子集是研究用，
生产必须走"验证集选参、测试集验证"。GYU-DET 有独立 val/test，本脚本完成该流程：
  1. fit（100 正常+30 缺陷，train 域）
  2. evaluate(val) → 单槽 AUROC → 选子集（drop_lt_0.5 / drop_lt_0.6 / keep_top2）
  3. evaluate(test) → 用 val 选的子集重算 fused AUROC（诚实：test 只验不选）
  4. 对比全槽 vs 裁剪（val 选）在 test 上的增益

输出：outputs/m0/m4_prune_gyudet.json
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import numpy as np
import torch
from src.data import gyudet
from src.eval.metrics import image_metrics
from src.fusion.fixed import fuse
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def fused_auroc(per_slot, labels, keep):
    w = {n: 1.0 / len(keep) for n in keep}
    fused = [fuse({n: float(per_slot[n][i]) for n in keep}, w)
             for i in range(len(labels))]
    return image_metrics(labels, fused)["auroc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=0, help="test 侧子采样（0=全量）")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"], seed=cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    # 1) val 上单槽 AUROC（选参）
    pipe.evaluate(bundle["val"])
    le = pipe._last_eval
    labels = le["labels"]; per_slot = le["per_slot"]
    names = list(per_slot)
    single = {n: round(image_metrics(labels, per_slot[n])["auroc"], 4) for n in names}
    print(f"[val] 单槽 AUROC: {single}")
    full = fused_auroc(per_slot, labels, names)
    strategies = {"all_equal": (full, names)}
    for th in (0.5, 0.6):
        keep = [n for n in names if single[n] >= th]
        strategies[f"drop_lt_{th}"] = (fused_auroc(per_slot, labels, keep) if keep else 0.0, keep)
    top2 = sorted(names, key=lambda n: -single[n])[:2]
    strategies["keep_top2"] = (fused_auroc(per_slot, labels, top2), top2)
    print("[val] 策略（val 上）:")
    for k, (au, keep) in strategies.items():
        print(f"  {k}: AUROC={au:.4f} keep={keep}")
    # 2) test 上验证（val 选的子集，只验不选）
    test = bundle["test"][:args.max_eval] if args.max_eval else bundle["test"]
    pipe.evaluate(test)
    le_t = pipe._last_eval
    labels_t = le_t["labels"]; per_slot_t = le_t["per_slot"]
    out = {"val_single": single, "strategies": {}, "test": {}}
    for k, (_, keep) in strategies.items():
        au_t = fused_auroc(per_slot_t, labels_t, keep)
        out["strategies"][k] = {"val_auroc": round(strategies[k][0], 4),
                                "test_auroc": round(au_t, 4), "keep": keep}
        print(f"[test] {k}: AUROC={au_t:.4f} keep={keep}")
    # 最佳策略（val 上选）在 test 上的增益
    best_k = max(strategies, key=lambda k: strategies[k][0])
    out["best_on_val"] = best_k
    out["full_test"] = out["strategies"]["all_equal"]["test_auroc"]
    out["prune_test"] = out["strategies"][best_k]["test_auroc"]
    out["test_gain"] = round(out["prune_test"] - out["full_test"], 4)
    print(f"[结论] val 选 best={best_k}，test 上 全槽 {out['full_test']} -> "
          f"裁剪 {out['prune_test']}（增益 {out['test_gain']:+}）")
    out_path = os.path.join(cfg["output_dir"], "m4_prune_gyudet.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
