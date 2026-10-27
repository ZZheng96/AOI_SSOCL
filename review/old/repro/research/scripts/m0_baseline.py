"""M0 主入口：诚实基线（sem+disc+blob，CDF 校准，等权保底融合）

用法：
  python scripts/m0_baseline.py --dataset gyudet --mode single
  python scripts/m0_baseline.py --dataset datalocal --category gold_finger --mode tiles36
  python scripts/m0_baseline.py --dataset datalocal --mode single|tiles9|tiles36  (全品类)
  python scripts/m0_baseline.py --speed-sweep    # 粒度-速度工作点实测（§6.1）
"""
import os
import sys
import argparse
import time
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone.dino import FrozenDINO
from src.slots.sem import SemSlot
from src.slots.disc import DiscSlot
from src.slots.blob import BlobSlot
from src.slots.trad import TradSlot
from src.slots.layout import LayoutSlot
from src.slots.inp import InpSlot
from src.slots.shead import SheadSlot
from src.data import gyudet, datalocal
from src.eval.offline import Pipeline, save_report
from src.eval.metrics import image_metrics


def build(cfg, device):
    # U25（2026-08-16）：槽位模型（disc/inp）在构建时初始化即消费全局 torch RNG，
    # 而 seed 重置原在 Pipeline.fit() 里（晚于构建）——导致同 seed 下 fit 不可复现
    #（同进程两次 fit 的 initial AUROC 0.3457 vs 0.4074）。这里提前固定。
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
    # U25：TF32 卷积/矩阵乘在 GPU 上非确定（图灵 1660S），关闭保证数值可复现
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bcfg = cfg["backbone"]
    backbone = None
    if bcfg.get("enabled", True):        # M4 CPU 降配：可整体关闭主干（纯手工槽位）
        if bcfg.get("name") == "pdn":    # M4/U9：PDN 蒸馏学生（CPU/速度路径）
            from src.backbone.pdn import PDNBackbone
            backbone = PDNBackbone(grid=bcfg["grid"], device=device,
                                   checkpoint=bcfg.get("checkpoint"))
        else:
            backbone = FrozenDINO(output_layer=bcfg["output_layer"],
                                  grid=bcfg["grid"], device=device)
    slots = []
    s = cfg["slots"]
    if s["sem"]["enabled"] and backbone is not None:
        slots.append(SemSlot(s["sem"], device))
    if s["disc"]["enabled"] and backbone is not None:
        slots.append(DiscSlot({**s["disc"], "seed": cfg["seed"]}, backbone, device))
    if s["shead"]["enabled"] and backbone is not None:
        slots.append(SheadSlot({**s["shead"], "seed": cfg["seed"]}, backbone, device))
    if s["blob"]["enabled"]:
        slots.append(BlobSlot(s["blob"], device))
    if s["trad"]["enabled"]:
        slots.append(TradSlot(s["trad"]))
    if s["layout"]["enabled"]:
        slots.append(LayoutSlot(s["layout"]))
    if s.get("color", {}).get("enabled"):
        from src.slots.color import ColorSlot
        slots.append(ColorSlot(s["color"], device))
    if s["inp"]["enabled"] and backbone is not None:
        slots.append(InpSlot(s["inp"], device))
    if s.get("tpl", {}).get("enabled") and backbone is not None:
        from src.slots.tpl import TplSlot
        slots.append(TplSlot(s["tpl"], backbone, device))
    return backbone, slots


def run_one(name, bundle, cfg, device, out_dir, max_eval=None, max_train=None,
            contribution=False, suffix=""):
    if max_train:  # 冒烟迭代用：子采样训练集（正式数字必须全量）
        rng = np.random.default_rng(cfg["seed"])
        bundle = dict(bundle)
        bundle["init_normal"] = sorted(rng.choice(
            bundle["init_normal"], min(max_train, len(bundle["init_normal"])), replace=False).tolist())
    if max_eval:   # 用户指示：迭代期减少测试图片
        bundle = dict(bundle)
        bundle["val"] = bundle["val"][:max_eval]
        bundle["test"] = bundle["test"][:max_eval]
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    trace_val = os.path.join(out_dir, f"trace_{name}_val.jsonl")
    rep = {"name": name, "mode": cfg["tiling"]["mode"],
           "fit_seconds": round(pipe.fit_seconds, 2),
           "n_init_normal": len(bundle["init_normal"]),
           "n_init_defect": len(bundle["init_defect"]),
           "slot_sanity": pipe.slot_sanity, "weights": pipe.weights}
    if bundle["val"]:
        rep["val"] = pipe.evaluate(bundle["val"], trace_val)
    if bundle["test"]:
        rep["test"] = pipe.evaluate(bundle["test"])
        # U31：每样本槽位校准分落盘（离线融合方案分析零成本复用；不参与任何训练）
        if hasattr(pipe, "_per_sample") and pipe._per_sample:
            rep["test"]["_per_sample"] = pipe._per_sample
            rep["test"]["_per_sample_labels"] = pipe._last_eval["labels"]
    if contribution and bundle["test"]:
        from src.eval.contribution import save_profile
        rep["contribution"] = pipe.contribution_report(
            bundle["test"], os.path.join(out_dir, f"contribution_{name}.json"))
    save_report(rep, os.path.join(out_dir, f"m0_{name}_{cfg['tiling']['mode']}{suffix}.json"))
    print(f"[{name}] mode={rep['mode']} fit={rep['fit_seconds']}s")
    for split in ("val", "test"):
        if split in rep:
            r = rep[split]
            slot_str = " ".join(f"{n}={r[n]['auroc']:.4f}" for n in cfg["slots"] if n in r)
            extra = f" | max_fuse={r['fused_max']['auroc']:.4f}" if "fused_max" in r else ""
            print(f"  {split}: fused AUROC={r['fused']['auroc']:.4f} AP={r['fused']['ap']:.4f} "
                  f"| {slot_str} | {r['ms_per_image']:.0f}ms/img{extra}")
    if contribution and "contribution" in rep:
        abl = rep["contribution"]["ablation"]
        print(f"  contribution: full={abl['full']:.4f} leave-one-out=" +
              " ".join(f"{n}:{v:.4f}" for n, v in abl["leave_one_out"].items()))
    return rep


def speed_sweep(cfg, device, out_dir):
    """粒度-速度工作点（§6.1）：single/tiles9/tiles36 × {ms/img, val AUROC}"""
    root = cfg["datasets"]["data_local"]
    bundle = datalocal.load_category(root, "gold_finger", 100, 30, cfg["seed"])
    results = {}
    for mode in ("single", "tiles9", "tiles36"):
        cfg_m = yaml.safe_load(yaml.safe_dump(cfg))
        cfg_m["tiling"]["mode"] = mode
        backbone, slots = build(cfg_m, device)
        pipe = Pipeline(cfg_m, backbone, slots)
        pipe.fit(bundle)
        subset = bundle["val"][:120] if bundle["val"] else bundle["test"][:120]
        r = pipe.evaluate(subset)
        results[mode] = {"ms_per_image": round(r["ms_per_image"], 1),
                         "fused": r["fused"]}
        print(f"[sweep:{mode}] {r['ms_per_image']:.0f}ms/img AUROC={r['fused']['auroc']:.4f}")
    save_report({"sweep": results, "note": "gold_finger 子集 n=120，1660 SUPER 实测"},
                os.path.join(out_dir, "m0_speed_sweep.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gyudet",
                    choices=["gyudet", "datalocal", "mvtec", "btad", "mpdd"])
    ap.add_argument("--category", default=None,
                    help="单品类；mvtec/btad/mpdd 也接受逗号分隔多品类")
    ap.add_argument("--mode", default=None, choices=["single", "tiles9", "tiles36"])
    ap.add_argument("--speed-sweep", action="store_true")
    ap.add_argument("--contribution", action="store_true", help="评测后输出三层评价贡献档案（§3.2）")
    ap.add_argument("--max-eval", type=int, default=None, help="迭代期子采样评测集")
    ap.add_argument("--max-train", type=int, default=None, help="冒烟用子采样 train/good")
    ap.add_argument("--max-eval-good", type=int, default=None,
                    help="mvtec 类大数据集 test/good 抽样上限（用户规则：测试数据多时只用少量；"
                         "None=mvtec 默认 25，其余数据集全量）")
    ap.add_argument("--max-test", type=int, default=150,
                    help="test 随机抽样上限（用户规则 2026-08-16：调优只跑少量图；"
                         "0=关闭抽样全量评测）")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"))
    ap.add_argument("--slots", default=None,
                    help="品类特化槽位子集（§9.4 裁剪杠杆）：逗号分隔，如 --slots disc,blob；未列槽位关闭")
    ap.add_argument("--weight-mode", default=None, choices=["equal", "sanity"],
                    help="融合权重模式（U40）：equal=等权保底；sanity=init_defect 槽位 AUROC 加权剔除反向槽")
    ap.add_argument("--focus-topk", type=float, default=None,
                    help="U70：shead 聚焦模式——按 patch 异常度（与全图均值 cosine 距离）取 "
                         "top-k 比例（0.01/0.05/0.10；None=全图统计）")
    ap.add_argument("--focus-score-only", action="store_true", default=False,
                    help="U70：shead 混合口径——训练用全图特征、推理用聚焦（默认 False=fit/score 同口径）")
    ap.add_argument("--out-suffix", default="",
                    help="输出文件名后缀（默认 ''；如 --out-suffix _focus01 存为 "
                         "m0_gyudet_single_focus01.json）")
    ap.add_argument("--tpl", action="store_true",
                    help="启用 tpl 槽位（L3 模板差分，需 --templates 提供模板）")
    ap.add_argument("--templates", default=None,
                    help="L3 用户显式模板：目录（取全部图）或逗号分隔文件列表（红线："
                         "任何文件名隐式配对=泄漏，模板必须显式提供）")
    ap.add_argument("--tpl-max", type=int, default=3,
                    help="模板目录时随机抽取的模板张数（品类级模板 3-5 张足够；"
                         "tpl 逐位置差分是 O(模板数) 的 GPU 循环，模板过多拖慢 eval）")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.weight_mode:
        cfg.setdefault("fusion", {})["weight_mode"] = args.weight_mode
        print(f"[fusion] weight_mode={args.weight_mode}")
    if args.focus_topk is not None:
        cfg["slots"]["shead"]["focus_topk"] = args.focus_topk
        if args.focus_score_only:
            cfg["slots"]["shead"]["focus_score_only"] = True
        print(f"[shead] U70 focus_topk={args.focus_topk} "
              f"focus_score_only={args.focus_score_only}")
    if args.mode:
        cfg["tiling"]["mode"] = args.mode
    if args.slots:
        keep = set(x.strip() for x in args.slots.split(",") if x.strip())
        for k in cfg["slots"]:
            cfg["slots"][k]["enabled"] = k in keep
        print(f"[slots] 品类特化子集: {sorted(keep)}")
    if args.tpl:
        cfg["slots"]["tpl"]["enabled"] = True
        print("[tpl] L3 模板差分槽位启用（用户显式模板，非隐式配对）")

    def inject_templates(bundle):
        """L3：用户显式模板注入 bundle["templates"]（offline.py fit 已消费）。"""
        if not args.templates:
            return bundle
        import os as _os
        from src.data.datalocal import _list_images
        if _os.path.isdir(args.templates):
            tpl_paths = _list_images(args.templates)
            if len(tpl_paths) > args.tpl_max:      # 品类级模板：随机抽 N 张（固定 seed 可复现）
                rng = np.random.default_rng(0)
                tpl_paths = sorted(rng.choice(
                    tpl_paths, args.tpl_max, replace=False).tolist())
        else:
            tpl_paths = [p.strip() for p in args.templates.split(",") if p.strip()]
        if tpl_paths:
            bundle["templates"] = tpl_paths
            print(f"[L3] 模板差分: {len(tpl_paths)} 张显式模板", flush=True)
        return bundle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if args.speed_sweep:
        speed_sweep(cfg, device, out_dir)
        return
    if args.dataset == "gyudet":
        bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                    cfg["protocol"]["n_init_normal"],
                                    cfg["protocol"]["n_init_defect"], cfg["seed"],
                                    max_test=args.max_test)
        bundle = inject_templates(bundle)
        run_one("gyudet", bundle, cfg, device, out_dir, args.max_eval, args.max_train,
                suffix=args.out_suffix)
    elif args.dataset in ("mvtec", "btad", "mpdd"):
        from src.data import mvtec_like
        root = cfg["datasets"][args.dataset]
        default_cats = {"mvtec": mvtec_like.MVTEC_CATEGORIES,
                        "btad": mvtec_like.BTAD_CATEGORIES,
                        "mpdd": mvtec_like.MPDD_CATEGORIES}[args.dataset]
        cats = ([c.strip() for c in args.category.split(",") if c.strip()]
                if args.category else default_cats)
        loader = (mvtec_like.load_btad if args.dataset == "btad"
                  else mvtec_like.load_category)
        # 用户规则：测试数据较多时只用少量（mvtec 每类 test/good ~50，抽样默认 25）
        n_eval_good = (args.max_eval_good if args.max_eval_good is not None
                       else (25 if args.dataset == "mvtec" else None))
        if n_eval_good:
            print(f"[mvtec_like] test/good 抽样上限 n_eval_good={n_eval_good}")
        for cat in cats:
            bundle = loader(root, cat,
                            n_init_normal=cfg["protocol"]["n_init_normal"],
                            n_init_defect=cfg["protocol"]["n_init_defect"],
                            seed=cfg["seed"], n_eval_good=n_eval_good,
                            max_test=args.max_test)
            bundle = inject_templates(bundle)
            run_one(f"{args.dataset}_{cat}", bundle, cfg, device, out_dir,
                    args.max_eval, args.max_train, args.contribution,
                    suffix=args.out_suffix)
    else:
        bundles = datalocal.load_all(cfg["datasets"]["data_local"], cfg,
                                     max_test=args.max_test)
        for name, bundle in bundles.items():
            if args.category and name != args.category:
                continue
            bundle = inject_templates(bundle)
            run_one(name, bundle, cfg, device, out_dir, args.max_eval, args.max_train,
                    args.contribution, suffix=args.out_suffix)


if __name__ == "__main__":
    main()
