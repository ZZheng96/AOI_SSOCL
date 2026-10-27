"""E_open 哨兵验证（§3.3）：open_score 分辨力 + open_alert 行为 + 无回归确认。

用法：
  python scripts/diag_open.py --category gold_finger
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
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--max-eval", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    assert pipe.open_det is not None, "E_open 未启用"
    test = bundle["test"][:args.max_eval] if args.max_eval else bundle["test"]

    rows = []
    for p, y in test:
        r = pipe.predict(p)
        rows.append({"y": y, "decision": r["decision"], "open": r["open_score"],
                     "open_alert": r.get("open_alert", False), "fused": r["fused"]})
    # open 单槽 AUROC（哨兵分辨力）
    from sklearn.metrics import roc_auc_score
    ys = np.array([x["y"] for x in rows])
    os_ = np.array([x["open"] for x in rows])
    try:
        au = roc_auc_score(ys, os_)
    except Exception:
        au = float("nan")
    normal_open = np.mean([x["open"] for x in rows if x["y"] == 0])
    defect_open = np.mean([x["open"] for x in rows if x["y"] == 1])
    n_alert = sum(1 for x in rows if x["open_alert"])
    n_normal_decision = sum(1 for x in rows if x["decision"] == "normal")
    # open_alert 例
    alerts = [x for x in rows if x["open_alert"]][:3]
    print(f"[open] 单槽 AUROC={au:.4f} | 正常图 open 均值={normal_open:.3f} "
          f"缺陷图 open 均值={defect_open:.3f}")
    print(f"[open] open_alert={n_alert}/{len(test)}（open 高但判定 normal=疑似未解释信号）")
    for x in alerts:
        print(f"   alert: y={x['y']} decision={x['decision']} open={x['open']:.3f} "
              f"fused={x['fused']:.3f}")
    out = {"category": args.category, "n": len(test),
           "open_auroc": round(float(au), 4) if au == au else None,
           "normal_open_mean": round(float(normal_open), 4),
           "defect_open_mean": round(float(defect_open), 4),
           "open_alerts": n_alert, "n_normal_decision": n_normal_decision}
    out_path = os.path.join(cfg["output_dir"], f"m4_open_{args.category}.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
