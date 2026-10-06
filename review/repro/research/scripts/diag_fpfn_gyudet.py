"""GYU-DET 误报/漏检主因诊断（U71，2026-08-18）

目的：搞清 GYU-DET 达不到 0.9 的主因是误报(FP)还是漏检(FN)。
两种输入模式：
  1) 基线 m0_*_single.json（test._per_sample 各槽位校准分 + labels + weights）
  2) 学习后 m5_learning_*_learned.json（rounds_on.eval_per_sample scores/labels/decisions）
用法：
  python scripts/diag_fpfn_gyudet.py --json outputs/m0/m0_gyudet_single.json
  python scripts/diag_fpfn_gyudet.py --json outputs/m0/m5_learning_gyudet_gyudet_rounds10_learned.json
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def fpfn_report(fused, y, tag=""):
    """对一组 fused 分数 + 标签做 FP/FN 全阈值分析并打印。"""
    N = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    print(f"\n=== FP/FN 分析 {tag}===  N={N} 缺陷={n_pos} 正常={n_neg}")
    a = auroc(y, fused)
    print(f"  AUROC={a:.4f}")

    thr = np.linspace(fused.min(), fused.max(), 61)
    best_f1, best_thr = -1, 0
    rows = []
    for t in thr:
        pred = (fused >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        tpr = rec
        fpr = fp / n_neg if n_neg > 0 else 0
        rows.append((t, tp, fp, fn, tn, prec, rec, f1, tpr, fpr))
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    br = [r for r in rows if r[7] == best_f1][0]
    t, tp, fp, fn, tn, prec, rec, f1, tpr, fpr = br
    print(f"  最佳F1: thr={t:.4f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"    误报率(FP/正常)={fp}/{n_neg}={fpr:.3f}  漏检率(FN/缺陷)={fn}/{n_pos}={fn/n_pos:.3f}")
    youden = max(rows, key=lambda r: r[8] - r[9])
    t, tp, fp, fn, tn, prec, rec, f1, tpr, fpr = youden
    print(f"  Youden: thr={t:.4f}  TPR={tpr:.3f} FPR={fpr:.3f}  F1={f1:.3f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"    误报率={fp}/{n_neg}={fpr:.3f}  漏检率={fn}/{n_pos}={fn/n_pos:.3f}")
    print(f"  阈值扫描表:")
    print(f"    {'thr':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6} {'FNR':>6}")
    for i in range(0, len(rows), max(1, len(rows) // 12)):
        t, tp, fp, fn, tn, prec, rec, f1, tpr, fpr = rows[i]
        fnr = fn / n_pos if n_pos > 0 else 0
        print(f"    {t:7.4f} {tp:4d} {fp:4d} {fn:4d} {tn:4d} {prec:6.3f} {rec:6.3f} {f1:6.3f} {fpr:6.3f} {fnr:6.3f}")
    # 得分分布
    fp_ = fused[neg]; tp_ = fused[pos]
    print(f"  正常样本: mean={fp_.mean():.4f} std={fp_.std():.4f} 中位={np.median(fp_):.4f}")
    print(f"  缺陷样本: mean={tp_.mean():.4f} std={tp_.std():.4f} 中位={np.median(tp_):.4f}")
    overlap = np.mean(fp_ >= np.percentile(tp_, 25))
    miss_low = np.mean(tp_ <= np.percentile(fp_, 75))
    print(f"  正常>=缺陷Q25: {overlap:.3f}(误报倾向)  缺陷<=正常Q75: {miss_low:.3f}(漏检倾向)")
    # 主因判定
    t, tp, fp, fn, tn, prec, rec, f1, tpr, fpr = br
    fp_ratio = fp / n_neg if n_neg > 0 else 0
    fn_ratio = fn / n_pos if n_pos > 0 else 0
    print(f"  主因判定(@最佳F1): FP率={fp_ratio:.1%} FN率={fn_ratio:.1%}", end=" ")
    if fn_ratio > fp_ratio + 0.05:
        print(f"-> 漏检为主（差{fn_ratio-fp_ratio:.1%}）-> 提灵敏度/缩patch/增强缺陷槽位")
    elif fp_ratio > fn_ratio + 0.05:
        print(f"-> 误报为主（差{fp_ratio-fn_ratio:.1%}）-> 改进校准/降权虚高槽位")
    else:
        print("-> 误报漏检并重，两侧均需改善")
    return fp_ratio, fn_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/m0/m0_gyudet_single.json")
    args = ap.parse_args()
    d = json.load(open(args.json, encoding="utf-8"))

    # 模式判定
    if "rounds_on" in d and "eval_per_sample" in d.get("rounds_on", {}):
        # 学习后模式
        ps = d["rounds_on"]["eval_per_sample"]
        fused = np.asarray(ps["scores"], dtype=float)
        y = np.asarray(ps["labels"])
        ld = d["rounds_on"].get("learned", {})
        print(f"[学习后诊断] {os.path.basename(args.json)}")
        print(f"  final_auroc={d['rounds_on']['final_auroc']} "
              f"n_feedback={d['rounds_on']['n_feedback']}")
        if ld:
            print(f"  learned_acc={ld.get('final_learned_acc')} "
                  f"flip={ld.get('final_flip_rate')} "
                  f"defect_recall={ld.get('final_defect_recall')}")
        fpfn_report(fused, y, tag="(学习后 eval锚定集)")
        # 决策级 FP/FN（用 decision != normal 判定）
        decisions = ps.get("decisions", [])
        if decisions:
            print(f"\n=== 决策级混淆矩阵（decision != normal 即判缺陷）===")
            preds = np.array([0 if dec == "normal" else 1 for dec in decisions])
            tp = int(((preds == 1) & (y == 1)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
            print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
            print(f"  误报率(FP/正常)={fp}/{n_neg}={fp/n_neg:.3f}")
            print(f"  漏检率(FN/缺陷)={fn}/{n_pos}={fn/n_pos:.3f}")
    else:
        # 基线模式
        per = d["test"]["_per_sample"]
        labels = np.asarray(d["test"]["_per_sample_labels"])
        weights = d["weights"]
        slots = sorted(per.keys())
        X = np.stack([per[n] for n in slots], axis=1)
        y = labels
        N = len(y)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        print(f"[基线诊断] {os.path.basename(args.json)}  N={N} 缺陷={len(pos)} 正常={len(neg)}")
        print(f"  槽位AUROC: " + " ".join(f"{n}={auroc(y,X[:,j]):.3f}" for j, n in enumerate(slots)))
        w = np.array([weights.get(n, 0.0) for n in slots])
        fused = X @ w
        a_recon = auroc(y, fused)
        print(f"  重建fused.auroc={a_recon:.4f} (JSON={d['test']['fused']['auroc']:.4f})")
        fpfn_report(fused, y, tag="(基线 fused)")
        # 各槽位独立诊断
        print(f"\n=== 各槽位独立诊断（最佳F1）===")
        print(f"  {'slot':>7} {'auroc':>7} {'FP/正常':>8} {'FN/缺陷':>8}")
        for j, n in enumerate(slots):
            s = X[:, j]
            a = auroc(y, s)
            bf, bt, b_fp, b_fn = -1, 0, 0, 0
            for t in np.linspace(s.min(), s.max(), 101):
                pred = (s >= t).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                if f1 > bf:
                    bf, bt, b_fp, b_fn = f1, t, fp, fn
            print(f"  {n:>7} {a:7.4f} {b_fp/len(neg):8.3f} {b_fn/len(pos):8.3f}")
        # oracle 融合
        aucs = np.array([auroc(y, X[:, j]) for j in range(X.shape[1])])
        w_or = np.clip(aucs - 0.5, 0, None)
        if w_or.sum() > 0:
            w_or = w_or / w_or.sum()
            print(f"\n  oracle融合: AUROC={auroc(X @ w_or):.4f} "
                  f"权重=" + " ".join(f"{n}={w_or[j]:.3f}" for j, n in enumerate(slots)))


if __name__ == "__main__":
    main()
