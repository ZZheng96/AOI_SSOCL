"""diag_shead_gyudet.py：shead 槽位 v2 gyudet 跨域诊断（2026-08-16）

目标：把 demo5 shead 槽 gyudet 跨域判别能力从 v1 的 0.359 提升到 ≥0.6
（demo4 s 分支同概念实测 0.79）。

用法：
  python scripts/diag_shead_gyudet.py [--max-test 150] [--no-higres]

跑 4 组（等权融合，关闭共识保 equal；B/C 显式启用 shead——m0.yaml 默认关闭）：
  A  shead-only                 —— shead v2 槽位单独 AUROC（448px）
  B  全槽位（含 shead 新增）      —— 融合 AUROC（对照基线 0.5533）
  C  全槽位除 disc（shead 替换）  —— 融合 AUROC 对照
  D  shead-only + backbone.grid=64（896px 提取，不改架构）—— 分辨率对照（默认必跑）
结果统一存 outputs/m0/shead_v2_gyudet.json（含 shead 训练 loss 曲线关键点）
"""
import os
import sys
import json
import argparse
import time
import copy
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(line_buffering=True)

from src.backbone.dino import FrozenDINO
from src.slots.sem import SemSlot
from src.slots.disc import DiscSlot
from src.slots.shead import SheadSlot
from src.slots.blob import BlobSlot
from src.slots.trad import TradSlot
from src.slots.layout import LayoutSlot
from src.slots.inp import InpSlot
from src.data import gyudet
from src.eval.offline import Pipeline


ALL_SLOTS = ("sem", "disc", "shead", "blob", "trad", "layout", "inp")


def build(cfg, device, slot_names):
    """按 slot_names 子集构建槽位（镜像 m0_baseline.build 的构造约定）。"""
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bcfg = cfg["backbone"]
    backbone = FrozenDINO(output_layer=bcfg["output_layer"], grid=bcfg["grid"],
                          device=device)
    s = cfg["slots"]
    slots = []
    if "sem" in slot_names and s["sem"]["enabled"]:
        slots.append(SemSlot(s["sem"], device))
    if "disc" in slot_names and s["disc"]["enabled"]:
        slots.append(DiscSlot({**s["disc"], "seed": cfg["seed"]}, backbone, device))
    if "shead" in slot_names and s["shead"]["enabled"]:
        slots.append(SheadSlot({**s["shead"], "seed": cfg["seed"]}, backbone, device))
    if "blob" in slot_names and s["blob"]["enabled"]:
        slots.append(BlobSlot(s["blob"], device))
    if "trad" in slot_names and s["trad"]["enabled"]:
        slots.append(TradSlot(s["trad"]))
    if "layout" in slot_names and s["layout"]["enabled"]:
        slots.append(LayoutSlot(s["layout"]))
    if "inp" in slot_names and s["inp"]["enabled"]:
        slots.append(InpSlot(s["inp"], device))
    return backbone, slots


def run(name, cfg, bundle, device, slot_names, note=""):
    """跑一次 fit+test 评测，返回 (rep, pipe)。等权融合：关共识、weight_mode=equal。"""
    print(f"\n==== [run {name}] {note} start ====", flush=True)
    t0 = time.time()
    backbone, slots = build(cfg, device, slot_names)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    rep = {"name": name, "note": note, "slots": list(pipe.slots),
           "fit_seconds": round(pipe.fit_seconds, 2),
           "grid": cfg["backbone"]["grid"],
           "n_init_normal": len(bundle["init_normal"]),
           "n_init_defect": len(bundle["init_defect"]),
           "slot_sanity": pipe.slot_sanity}
    # shead v2 训练 loss 曲线关键点（正常/异常损失分量）
    if "shead" in pipe.slots:
        lh = getattr(pipe.slots["shead"], "loss_history", None)
        if lh:
            rep["shead_loss_curve"] = [
                {"epoch": e, "loss_normal": round(n, 4), "loss_anomaly": round(a, 4)}
                for e, n, a in lh]
    if bundle["test"]:
        rep["test"] = pipe.evaluate(bundle["test"])
    rep["eval_seconds"] = round(time.time() - t0, 2)
    for sn in rep["test"]:
        if isinstance(rep["test"][sn], dict) and "auroc" in rep["test"][sn]:
            print(f"  [run {name}] {sn} AUROC={rep['test'][sn]['auroc']:.4f}", flush=True)
    print(f"==== [run {name}] end ({rep['eval_seconds']}s) ====", flush=True)
    return rep, pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=150,
                    help="test 分层抽样上限（0=全量）")
    ap.add_argument("--no-higres", action="store_true",
                    help="跳过分辨率对照实验")
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    # 等权融合保底：关共识（共识会改权重），weight_mode=equal
    base_cfg["consensus"]["enabled"] = False
    base_cfg["fusion"]["weight_mode"] = "equal"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = base_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    bundle = gyudet.load_gyudet(base_cfg["datasets"]["gyu_det"],
                                base_cfg["protocol"]["n_init_normal"],
                                base_cfg["protocol"]["n_init_defect"],
                                base_cfg["seed"], max_test=args.max_test)
    n_test = len(bundle["test"])
    print(f"[diag] gyudet: init_normal={len(bundle['init_normal'])} "
          f"init_defect={len(bundle['init_defect'])} test={n_test} "
          f"(grid={base_cfg['backbone']['grid']}, 448px)", flush=True)

    report = {"protocol": "gyudet 跨域 train->test", "max_test": args.max_test,
              "n_test": n_test, "date": "2026-08-16", "version": "shead v2",
              "baselines": {"shead_v1_448px": 0.359, "shead_v1_896px": 0.390,
                            "demo4_s": 0.79, "disc": 0.378,
                            "demo5_fused_equal": 0.5533,
                            "demo5_online_learned": 0.6599},
              "runs": {}}

    # ---- A: shead-only ----
    cfgA = copy.deepcopy(base_cfg)
    for k in ALL_SLOTS:
        cfgA["slots"][k]["enabled"] = (k == "shead")
    repA, _ = run("A_shead_only", cfgA, bundle, device, ("shead",),
                  "shead 槽位单独（等权融合=单槽）")
    report["runs"]["A_shead_only"] = repA
    shead_auroc = repA["test"]["shead"]["auroc"]

    # ---- B: 全槽位含 shead（新增） ----
    cfgB = copy.deepcopy(base_cfg)
    cfgB["slots"]["shead"]["enabled"] = True   # 显式启用（m0.yaml 默认关闭）
    repB, _ = run("B_all_with_shead", cfgB, bundle, device, ALL_SLOTS,
                  "全槽位（shead 新增，等权）")
    report["runs"]["B_all_with_shead"] = repB

    # ---- C: 全槽位除 disc（shead 替换 disc） ----
    cfgC = copy.deepcopy(base_cfg)
    cfgC["slots"]["disc"]["enabled"] = False
    cfgC["slots"]["shead"]["enabled"] = True   # 显式启用（m0.yaml 默认关闭）
    repC, _ = run("C_shead_replaces_disc", cfgC, bundle, device,
                  tuple(k for k in ALL_SLOTS if k != "disc"),
                  "全槽位除 disc（shead 替换，等权）")
    report["runs"]["C_shead_replaces_disc"] = repC

    # ---- D: 分辨率对照（v2 任务要求，默认必跑；--no-higres 跳过） ----
    if not args.no_higres:
        print(f"\n[shead] shead AUROC={shead_auroc:.4f}，追加 896px 分辨率对照", flush=True)
        cfgD = copy.deepcopy(base_cfg)
        for k in ALL_SLOTS:
            cfgD["slots"][k]["enabled"] = (k == "shead")
        cfgD["backbone"]["grid"] = 64          # 896px（不动架构，只改输入分辨率）
        cfgD["slots"]["shead"]["extract_batch"] = 1   # 896px 显存安全
        repD, _ = run("D_shead_896px", cfgD, bundle, device, ("shead",),
                      "shead-only grid=64（896px 提取）")
        report["runs"]["D_shead_896px"] = repD

    # ---- 汇总对照 ----
    fused_b = repB["test"]["fused"]["auroc"]
    fused_c = repC["test"]["fused"]["auroc"]
    summary = {
        "shead_alone_auroc": round(float(shead_auroc), 4),
        "fused_equal_with_shead": round(float(fused_b), 4),
        "fused_equal_shead_replaces_disc": round(float(fused_c), 4),
        "vs_shead_v1_0.359": round(float(shead_auroc - 0.359), 4),
        "vs_demo4_s_0.79": round(float(shead_auroc - 0.79), 4),
        "vs_baseline_fused_0.5533": round(float(fused_b - 0.5533), 4),
    }
    if "D_shead_896px" in report["runs"]:
        summary["shead_896px_auroc"] = round(
            float(report["runs"]["D_shead_896px"]["test"]["shead"]["auroc"]), 4)
    report["summary"] = summary
    out_path = os.path.join(out_dir, "shead_v2_gyudet.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[diag] 结果已存 {out_path}", flush=True)
    print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
