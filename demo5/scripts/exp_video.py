"""M2/M4 视频模态验证（§6）：合成视频（正常段+缺陷段）→ VideoInspector 端到端。

无真实视频环境：用 data_local 图片合成 mp4（good 段 + defect 段 + good 段），
验证关键帧管线 + 时序平滑 + 短脉冲降级 + 缺陷段检出。

用法：
  python scripts/exp_video.py --category gold_finger --max-eval 20
"""
import os
import sys
import json
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
import cv2
import numpy as np
from src.data import datalocal
from src.common.io import load_image
from scripts.m0_baseline import build
from src.eval.offline import Pipeline
from src.video.video_pipeline import VideoInspector


def synth_video(good_paths, defect_paths, out_path, fps=25, secs_per_img=0.4):
    """正常段 + 缺陷段 + 正常段合成 mp4（统一 resize 512²）。"""
    frames_per_img = max(1, int(fps * secs_per_img))
    writer = None
    for group in (good_paths, defect_paths, good_paths):   # 正常-缺陷-正常
        for p in group:
            img = cv2.cvtColor(load_image(p), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (512, 512))
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (512, 512))
            for _ in range(frames_per_img):
                writer.write(img)
    writer.release()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="gold_finger")
    ap.add_argument("--max-eval", type=int, default=20)
    ap.add_argument("--step", type=int, default=3, help="关键帧采样步长")
    ap.add_argument("--alpha", type=float, default=0.5, help="EMA 平滑系数")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")))
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    bundle = datalocal.load_category(cfg["datasets"]["data_local"], args.category,
                                     cfg["protocol"]["n_init_normal"],
                                     cfg["protocol"]["n_init_defect"], cfg["seed"])
    # 合成视频（正常-缺陷-正常三段）
    goods = [p for p, y in bundle["test"] if y == 0][:5]
    defects = [p for p, y in bundle["test"] if y == 1][:8]
    video_path = os.path.join(cfg["output_dir"], f"synth_{args.category}.mp4")
    synth_video(goods, defects, video_path)
    print(f"[video] 合成 {video_path}（good {len(goods)} + defect {len(defects)} + good）")

    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    inspector = VideoInspector(pipe, step=args.step, alpha=args.alpha)
    out = inspector.inspect(video_path)
    # 打印逐帧判定序列（简短）
    seq = "".join({"normal": "N", "gray": "?", "anomaly": "A"}[r["decision_smooth"]]
                  for r in out["frames"])
    print(f"[video] 帧判定序列({len(out['frames'])}帧): {seq}")
    s = out["summary"]
    print(f"[video] summary: anomaly={s['anomaly_frames']} gray={s['gray_frames']} "
          f"total={s['total_frames']} ratio={s['anomaly_ratio']} "
          f"mean={s['mean_fused']} max={s['max_fused']} "
          f"dominant_type={s['dominant_type']} final={s['final_decision']}")
    rep = {"category": args.category, "video": video_path, "summary": s,
           "sequence": seq}
    out_json = os.path.join(cfg["output_dir"], f"m4_video_{args.category}.json")
    json.dump(rep, open(out_json, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
