"""G5 落地（§9 3b/3c）：五类缺陷覆盖矩阵 + 类型归因 top-1 准确率实测（2026-08-24）

评审自检 G5 指出"五类缺陷×场景覆盖矩阵、类型归因 top-1 准确率，设计文档写了评测计划
（§9 3b/3c）但未执行"。本脚本兑现：

1. 覆盖矩阵：五类缺陷（尺寸偏差/缺件少件/逻辑错误/色彩变化/常见外观缺陷）× 检出率
   ——用 MVTec（文件夹名=缺陷类型）+ data_local（类名）构造测试集，逐格报检出率。
   覆盖不了的格子如实标灰（当前无 color 槽位 → 色彩变化归因缺能力；无尺寸/逻辑数据）。
2. 类型归因 top-1 准确率：L3 attribute_type 输出 vs 路径缺陷类型标签（映射到五类）。

诚实边界：归因是规则推断（SLOT_TYPE_MAP），当前只能粗分"布局类(3细分)/外观类"，
色彩变化无对应槽位会被归为"常见外观缺陷"——如实报告。

用法: python scripts/exp_coverage_matrix.py [--per-defect 8] [--config configs/m0.yaml]
"""
import os
import sys
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like, datalocal

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0", "coverage_matrix.json")

# 赛题五类
FIVE = ["尺寸偏差", "缺件少件", "逻辑错误", "色彩变化", "常见外观缺陷"]


def classify(defect_type, category, dataset):
    """具体缺陷类型/类名 → 赛题五类。U96：统一用 defect_classifier.type_from_path
    （文件夹名 → FIVE_MAP，含尺寸偏差/逻辑错误细分），保证 ground-truth 与
    有监督判别器训练标签一致。"""
    from src.decision.defect_classifier import FIVE_MAP
    if defect_type in FIVE_MAP:
        return FIVE_MAP[defect_type]
    return "常见外观缺陷"


def collect_mvtec(root, category, per_defect):
    """返回 [(path, defect_type_dir)]，每缺陷类型抽 per_defect 张。"""
    cat_dir = os.path.join(root, category)
    test_dir = os.path.join(cat_dir, "test")
    out = []
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good" or not os.path.isdir(os.path.join(test_dir, sub)):
            continue
        imgs = sorted(f for f in os.listdir(os.path.join(test_dir, sub))
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for f in imgs[:per_defect]:
            out.append((os.path.join(test_dir, sub, f), sub))
    return out


def collect_datalocal(root, category, per_defect):
    d, _ = datalocal._split_good_defect(os.path.join(root, category), "test")
    # U96：缺陷类型取路径文件夹名（missing/shift/extra/bridge 等），供 classify 细分
    return [(p, os.path.basename(os.path.dirname(p))) for p in d[:per_defect]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-defect", type=int, default=8)
    ap.add_argument("--config", default=CFG)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 覆盖矩阵 + 归因累积
    det = {c: {"tp": 0, "n": 0} for c in FIVE}     # 检出（decision=anomaly）
    attr = {c: {"hit": 0, "n": 0} for c in FIVE}   # 归因 top-1 正确

    # ---- MVTec（全品类，每缺陷类型 per_defect 张）----
    mvtec_root = cfg["datasets"]["mvtec"]
    for cat in mvtec_like.MVTEC_CATEGORIES:
        bundle = mvtec_like.load_category(mvtec_root, cat,
                                          cfg["protocol"]["n_init_normal"],
                                          cfg["protocol"]["n_init_defect"],
                                          cfg["seed"], n_eval_good=0, max_test=0)
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        samples = collect_mvtec(mvtec_root, cat, args.per_defect)
        for p, dt in samples:
            cls = classify(dt, cat, "mvtec")
            r = pipe.predict(p)
            det[cls]["n"] += 1
            if r["decision"] != "normal":
                det[cls]["tp"] += 1
            attr[cls]["n"] += 1
            top = r["types"][0]["type"] if r["types"] else "未分类异常"
            if top == cls:
                attr[cls]["hit"] += 1
        print(f"[G5] mvtec/{cat} 完成（抽 {len(samples)} 缺陷）", flush=True)

    # ---- data_local（4 品类）----
    dl_root = cfg["datasets"]["data_local"]
    for cat in datalocal.CATEGORIES:
        bundle = datalocal.load_category(dl_root, cat,
                                         cfg["protocol"]["n_init_normal"],
                                         cfg["protocol"]["n_init_defect"],
                                         cfg["seed"], max_test=0)
        backbone, slots = build(cfg, device)
        pipe = Pipeline(cfg, backbone, slots)
        pipe.fit(bundle)
        samples = collect_datalocal(dl_root, cat, args.per_defect * 4)
        for p, dt in samples:
            cls = classify(dt, cat, "data_local")
            r = pipe.predict(p)
            det[cls]["n"] += 1
            if r["decision"] != "normal":
                det[cls]["tp"] += 1
            attr[cls]["n"] += 1
            top = r["types"][0]["type"] if r["types"] else "未分类异常"
            if top == cls:
                attr[cls]["hit"] += 1
        print(f"[G5] data_local/{cat} 完成（抽 {len(samples)} 缺陷）", flush=True)

    # ---- 汇总 ----
    result = {"coverage": {}, "attribution": {}, "note": ""}
    print("\n=== 五类缺陷覆盖矩阵（检出率 = decision==anomaly 比例）===")
    for c in FIVE:
        d = det[c]
        rate = d["tp"] / d["n"] if d["n"] else None
        a = attr[c]
        acc = a["hit"] / a["n"] if a["n"] else None
        result["coverage"][c] = {"detected": d["tp"], "n": d["n"],
                                 "rate": round(rate, 4) if rate is not None else None}
        result["attribution"][c] = {"hit": a["hit"], "n": a["n"],
                                    "top1_acc": round(acc, 4) if acc is not None else None}
        ds = f"{d['tp']}/{d['n']}" if d["n"] else "无数据(灰)"
        ar = f"{acc:.3f}" if acc is not None else "-"
        rt = f"{rate:.3f}" if rate is not None else "-"
        print(f"  {c:8s} 检出 {ds:12s} (rate={rt}) | 归因top1 {ar} "
              f"({a['hit']}/{a['n']})", flush=True)
    result["note"] = ("尺寸偏差/逻辑错误当前数据集无对应缺陷类型标签（标灰）；"
                      "色彩变化无 color 槽位，检出靠外观类槽位、归因会误归为外观缺陷")
    import json
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[G5] 结果落盘 {OUT}")


if __name__ == "__main__":
    main()
