"""U86 集成（2026-08-20）：pipeline 端到端使用学习后的 shead head（u86v8_head.pt）。

流程：build pipeline（ablB 配置）-> fit（CDF/权重/槽位）-> 覆盖 shead 为
U86 学习后 best head（crop_size 699 + chan/patch stats）-> 评估 val/test。

配置改动（U86 速度红线）：
- shead.score_overlap: 0.5 -> 0.3（37 窗 853ms/张 达标，U86v7 实测）
- fusion.skip_zero_weight: true（只跑 only_slots=[shead]，省其他槽位耗时）

评估口径：与 m0_baseline.py 一致（fused AUROC = shead 校准分 AUROC，CDF 单调
不改变 AUROC）。--max-test 控制抽样（默认 150；0=全量）。
用法：
  python scripts/m0_u86_integ.py --max-test 150
  python scripts/m0_u86_integ.py --max-test 0
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m0_baseline import build
from src.data import gyudet
from src.eval.offline import Pipeline, save_report

U86_HEAD = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                        "u86v8_head.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=150,
                    help="test 随机抽样上限（0=全量；默认 150 用户规则调优少跑）")
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "..", "configs",
                                                     "m0_gyudet_cropv4_ablB.yaml"))
    ap.add_argument("--out-suffix", default="_u86")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # U86 集成配置：速度红线（ov0.3 37 窗）+ 只跑白名单槽位
    cfg["slots"]["shead"]["score_overlap"] = 0.1   # U86v16：无重叠（窗最少，大图 35 窗）
    cfg["slots"]["shead"]["score_max_crops"] = 200
    cfg["slots"]["shead"]["extract_batch"] = 16   # U86v16：crop 打分 batch 8->16 减批数
    cfg["slots"]["shead"]["crop_size"] = 800      # U86v16：800 = 735ms 达标（699 的 972ms 贴线）
    cfg["slots"]["shead"]["adaptive_score"] = False   # ov0.1 固定无重叠即可达标
    cfg["slots"]["open"]["enabled"] = False       # U86v16：关 open 哨兵（省 feats 打分）
    cfg["fusion"]["skip_zero_weight"] = True
    cfg["fusion"]["only_slots"] = ["shead"]
    print(f"[u86-integ] score_overlap=0.3 skip_zero_weight=True only_slots=[shead]",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], cfg["seed"],
                                max_test=args.max_test)
    if args.max_eval:
        bundle = dict(bundle)
        bundle["val"] = bundle["val"][:args.max_eval]
        bundle["test"] = bundle["test"][:args.max_eval]

    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    pipe.hier = False        # U86v16：评估速度优先，跳过 L2/L3 分层输出（AUROC 不受影响）

    # U86：覆盖 shead 为在线学习后的 best head（valid 门控选出）
    ck = torch.load(U86_HEAD, map_location=device)
    shead = pipe.slots["shead"]
    shead.head.load_state_dict(ck["head"])
    shead.crop_size = 800   # U86v16：速度红线（699 打分 972ms 贴线，800 = 735ms 达标）
    shead.chan_stats = ck["chan_stats"]
    shead.patch_stats = ck["patch_stats"]
    print(f"[u86-integ] shead 已覆盖为 U86 head "
          f"(crop_size={shead.crop_size}, train_test_final={ck['test_final']:.4f})",
          flush=True)

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    rep = {"name": "gyudet", "mode": cfg["tiling"]["mode"],
           "fit_seconds": round(pipe.fit_seconds, 2),
           "u86_head": os.path.basename(U86_HEAD),
           "u86_test_final_60": round(ck["test_final"], 4),
           "n_init_normal": len(bundle["init_normal"]),
           "n_init_defect": len(bundle["init_defect"]),
           "slot_sanity": pipe.slot_sanity, "weights": pipe.weights}
    if bundle["val"]:
        rep["val"] = pipe.evaluate(bundle["val"])
    if bundle["test"]:
        rep["test"] = pipe.evaluate(bundle["test"])
    save_report(rep, os.path.join(out_dir, f"m0_gyudet_single{args.out_suffix}.json"))
    for split in ("val", "test"):
        if split in rep:
            r = rep[split]
            print(f"  {split}: fused AUROC={r['fused']['auroc']:.4f} "
                  f"AP={r['fused']['ap']:.4f} | shead={r['shead']['auroc']:.4f} "
                  f"| {r['ms_per_image']:.0f}ms/img", flush=True)


if __name__ == "__main__":
    main()
