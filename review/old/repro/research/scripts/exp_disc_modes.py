"""三模式对比：disc 槽位预训练初始化 vs 从零（§9.1，U11 落地验证）

模式：
  A. from_scratch : 每品类从头训练（epochs=30）
  B. pretrain_ft  : 预训练头 + 少量微调（init_from + finetune_epochs=5）
  C. pretrain_fz  : 预训练头冻结（init_from + freeze=true，零训练）

评估：data_local gold_finger，单槽位 AUROC/AP（CDF 校准后，红线 3 只用 train/good）
结论回答：跨品类预训练 + 少量微调 是否比 从零/冻结 更好。

用法：
  python scripts/exp_disc_modes.py --category gold_finger [--max-eval 60]
"""
import os
import sys
import copy
import argparse
import time
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"))
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = cfg["datasets"]["data_local"]
    bundle = datalocal.load_category(root, args.category, 100, 30, cfg["seed"])
    test = bundle["test"]
    if args.max_eval:
        test = test[: args.max_eval]
    print(f"[{args.category}] init_normal={len(bundle['init_normal'])} "
          f"init_defect={len(bundle['init_defect'])} test={len(test)}")

    pretrain = os.path.join(cfg["output_dir"], "disc_pretrain.pt")
    base = {"enabled": True, "hidden": 128, "epochs": 30, "lr": 1e-3,
            "pseudo_per_image": 8, "max_base_tiles": 96}
    modes = {
        "A_from_scratch": dict(base),
        "B_pretrain_ft": {**base, "init_from": pretrain, "finetune_epochs": 5},
        "C_pretrain_fz": {**base, "init_from": pretrain, "freeze": True},
    }
    results = {}
    for name, disc_cfg in modes.items():
        cfg_m = copy.deepcopy(cfg)
        # 只启用 disc，其余槽位关掉（build() 要求 key 齐全）
        for slot_name in cfg_m["slots"]:
            cfg_m["slots"][slot_name]["enabled"] = (slot_name == "disc")
        cfg_m["slots"]["disc"] = disc_cfg
        backbone, slots = build(cfg_m, device)
        pipe = Pipeline(cfg_m, backbone, slots)
        t0 = time.time()
        pipe.fit(bundle)
        r = pipe.evaluate(test)
        results[name] = {"auroc": r["disc"]["auroc"], "ap": r["disc"]["ap"],
                         "ms_per_image": round(r["ms_per_image"], 1),
                         "fit_seconds": round(pipe.fit_seconds, 1)}
        print(f"[{name}] fit={results[name]['fit_seconds']}s "
              f"AUROC={results[name]['auroc']:.4f} AP={results[name]['ap']:.4f} "
              f"{results[name]['ms_per_image']}ms/img", flush=True)

    print("\n===== 三模式对比（U11）=====")
    print(f"{'mode':<16}{'AUROC':>8}{'AP':>8}{'fit_s':>8}{'ms/img':>8}")
    for name, r in results.items():
        print(f"{name:<16}{r['auroc']:.4f}  {r['ap']:.4f}  {r['fit_seconds']:>7.1f}  {r['ms_per_image']:>7.1f}")
    best = max(results, key=lambda k: results[k]["auroc"])
    print(f"\n最佳模式: {best} (AUROC={results[best]['auroc']:.4f})")


if __name__ == "__main__":
    main()
