"""M4 追溯 API 演示（§9.2 交付物）：fit -> attach_trace_api -> 推理 -> 查询/导出。

用法：
  python scripts/m4_trace_demo.py --category gold_finger --max-eval 20
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from src.data import datalocal
from scripts.m0_baseline import build
from src.eval.offline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--max-eval", type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    # 挂追溯 API（§9.2）
    api = pipe.attach_trace_api(os.path.join(cfg["output_dir"], "trace_api"))
    for p, _ in bundle["test"][:args.max_eval]:
        pipe.predict(p)
    # 查询接口演示
    print("== recent_inferences（最近 3 条）==")
    for r in api.recent_inferences(3):
        print(" ", r["path"], "fused=", round(r["fused"], 4), r["decision"])
    print("== system_state ==")
    st = api.system_state()
    print("  weights:", {k: round(v, 3) for k, v in st["weights"].items()})
    print("  decider:", st["decider"], "  n_records:", st["n_records"])
    print("== update_log / export ==")
    print("  ", api.update_log())
    out = api.export(os.path.join(cfg["output_dir"], "trace_api", "audit.json"))
    print(f"  [save] {out}")
    api.close()


if __name__ == "__main__":
    main()
