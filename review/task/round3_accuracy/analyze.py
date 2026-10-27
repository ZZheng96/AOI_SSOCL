"""离线分析 repro 输出：逐槽 raw/校准 AUROC、饱和率、最优工作点（不重跑推理）。
用法: python analyze.py <mode>
"""
import json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RAW = Path(r"d:\CGAIC\AOI_SSOCL\review\task\round3_accuracy\raw")


def op_points(y, s, fpr_max=0.05, rec_min=0.95):
    """遍历阈值：返回 FPR≤fpr_max 下最大 recall，及 recall≥rec_min 下最小 FPR。"""
    y, s = np.asarray(y), np.asarray(s)
    best_r, best_f = (0.0, None), (1.0, None)
    for t in np.unique(s):
        pred = s >= t
        fpr = float(pred[y == 0].mean())
        rec = float(pred[y == 1].mean())
        if fpr <= fpr_max and rec > best_r[0]:
            best_r = (rec, float(t))
        if rec >= rec_min and fpr < best_f[0]:
            best_f = (fpr, float(t))
    return {"max_recall_at_fpr5": best_r, "min_fpr_at_recall95": best_f}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    rows = json.loads((RAW / f"{mode}_per_image.json").read_text(encoding="utf-8"))
    fit = json.loads((RAW / f"{mode}_fit.json").read_text(encoding="utf-8"))
    y = [r["label"] for r in rows]
    out = {"mode": mode, "n": len(rows), "tau_high": fit["tau_high"], "weights": fit["weights"]}
    fused = [r["fused"] for r in rows]
    dec = [r["decision"] for r in rows]
    yn = np.asarray(y)
    an = np.asarray([d == "anomaly" for d in dec])
    out["system"] = {"auroc_fused": roc_auc_score(y, fused),
                     "fpr": float(an[yn == 0].mean()), "recall": float(an[yn == 1].mean()),
                     "op": op_points(y, fused)}
    per = {}
    for n in fit["score_slots"]:
        raw = [r["raw_scores"][n] for r in rows]
        cal = [r["slot_scores"][n] for r in rows]
        per[n] = {"auroc_raw": round(roc_auc_score(y, raw), 4),
                  "auroc_cal": round(roc_auc_score(y, cal), 4),
                  "sat_rate": round(float(np.mean(np.asarray(cal) >= 2.0 - 1e-9)), 3),
                  "n_ties_top": int(np.sum(np.asarray(cal) == np.max(cal))),
                  "train_hi": fit["calibrators"][n]["hi"],
                  "test_normal_raw_min": round(min(r for r, l in zip(raw, y) if l == 0), 4),
                  "test_defect_raw_min": round(min(r for r, l in zip(raw, y) if l == 1), 4),
                  "op_raw": op_points(y, raw)}
    out["slots"] = per
    # 无钳位重构融合：raw→按 train 上沿线性外推（不截 2.0），检验钳位是否丢排序信息
    names = fit["score_slots"]
    w = fit["weights"]
    unc = []
    for r in rows:
        s = 0.0
        for n in names:
            c = fit["calibrators"][n]
            hi, lo = c["hi"], c["lo"]
            s += w[n] * (r["raw_scores"][n] - lo) / max(hi - lo, 1e-12)
        unc.append(s)
    out["fused_unclipped_auroc"] = roc_auc_score(y, unc)
    out["fused_unclipped_op"] = op_points(y, unc)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (RAW / f"{mode}_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
