"""视频异常检测协议化实验（P0-2 证据工具，2026-10-02）。

把"视频模态可用"从机制存在变成可复跑协议：用当前品类快照副本 +
数据集图片**合成已知真值的视频**（不动线上模型、不依赖外部视频素材）：

  V1 注入视频：正常帧 + 中段连续注入缺陷帧（>= pulse_win，真值区间已知）
     → 断言 final_decision == "anomaly"，且注入段帧级召回 >= min_recall。
  V0 对照视频：纯正常帧
     → 断言 final_decision != "anomaly"（允许 gray，报告实际值）。

指标：视频级判定、注入段帧级召回（decision_smooth=="anomaly" 的注入帧
比例，脉冲降级后口径）、异常帧率、dominant_type、source/dropped 帧统计。
环境 manifest 与 fullchain 同构（commit/python/torch/cuda/config_hash）。

诚实边界：合成视频的缺陷帧取自真实缺陷图（逐帧静止呈现），时序纹理/
运动模糊不在本协议覆盖范围；产线 RTSP 流为在线逐帧路径（脉冲降级仅
离线 VideoInspector 汇总语义，已在文档声明）。赛题答辩若给官方视频，
用同一管线复跑归档本报告 JSON 即可。
报告写 storage/logs/video_protocol_{时间戳}.json。

用法：
    python -m tests.bench.video_protocol                 # 标准档
    python -m tests.bench.video_protocol --quick         # 快速自检
    python -m tests.bench.video_protocol --inject 8 --n-normal 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.testdata import pick_category, list_imgs  # noqa: E402

MIN_INJECT_RECALL = 0.5   # 注入段帧级召回下限（脉冲降级后口径）


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _torch_manifest() -> dict:
    try:
        import torch
        return {"version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None)}
    except Exception:  # noqa: BLE001
        return {"version": "unavailable"}


def _file_hash(path: str) -> str:
    if not path or not os.path.isfile(path):
        return "unavailable"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_bgr(path: str, size) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读图失败 {path}")
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size)
    return img


def _make_video(path: str, normal_imgs, defect_imgs, inject_at, inject_len,
                n_frames, size, fps=10):
    """合成视频：defect_imgs 为空则纯正常对照。返回注入段真值区间列表。
    size 取源图原生分辨率（缩放会造成分布漂移、正常帧被误判）。"""
    # FFV1 无损写盘：有损编码（mp4v/MJPG）的块效应足以让引擎把正常帧
    # 判成异常（实测 fused 0.09→1.88），协议视频必须走无损链路——
    # 评的是检测能力，不是编码鲁棒性。
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    if not vw.isOpened():
        raise IOError(f"无法写视频 {path}")
    segments = []
    for i in range(n_frames):
        in_seg = (defect_imgs and inject_at <= i < inject_at + inject_len)
        if in_seg:
            frame = _load_bgr(defect_imgs[(i - inject_at) % len(defect_imgs)],
                              size)
            if i == inject_at:
                segments.append([inject_at, inject_at + inject_len - 1])
        else:
            frame = _load_bgr(normal_imgs[i % len(normal_imgs)], size)
        vw.write(frame)
    vw.release()
    return segments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-normal", type=int, default=24, help="V1 总帧数")
    ap.add_argument("--inject", type=int, default=8, help="注入段长度（帧）")
    ap.add_argument("--control-frames", type=int, default=16,
                    help="V0 对照视频帧数")
    ap.add_argument("--pulse-win", type=int, default=3)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs"
                                            / "engine_fast.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="快速自检（V1=12帧注入4，V0=8帧）")
    args = ap.parse_args()
    if args.quick:
        args.n_normal, args.inject, args.control_frames = 12, 4, 8
    if args.inject < args.pulse_win:
        raise SystemExit(f"--inject ({args.inject}) 必须 >= --pulse-win "
                         f"({args.pulse_win})，否则脉冲降级必然吞掉注入段")

    cat_dir, cat_name, synthetic = pick_category()
    normals = list_imgs(os.path.join(cat_dir, "train", "good"), 12)
    defects = []
    test_dir = os.path.join(cat_dir, "test")
    for name in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, name)
        if os.path.isdir(d) and name != "good":
            defects += list_imgs(d, 4)
    if len(normals) < 6 or len(defects) < 2:
        raise SystemExit("样本不足：需要 >=6 正常图、>=2 缺陷图")
    print(f"[video] 品类={cat_name} synthetic={synthetic} "
          f"normal={len(normals)} defect={len(defects)}", flush=True)

    # 准备引擎快照副本（独立临时存储，不动线上）
    tmp_storage = tempfile.mkdtemp(prefix="aoi_vp_")
    from backend.engine import DetectionEngine
    engine = DetectionEngine(tmp_storage, args.config)
    test_good = list_imgs(os.path.join(cat_dir, "test", "good"), 2)
    test_item = (test_good[0], 0) if test_good else (normals[-1], 0)
    init_n = [p for p in normals[:10] if p != test_item[0]]
    bundle = {"init_normal": init_n, "init_defect": defects[:3],
              "val": [], "test": [test_item]}
    cat = f"{cat_name}_video"
    v = engine.prepare(cat, bundle, scenario="L1a", profile="fast",
                       trigger="bench", note="video protocol")
    from algo.persist import current_version, load_snapshot
    cv = current_version(engine._cat_dir(cat))
    pipe = load_snapshot(engine._v_dir(cat, cv), engine.device)
    print(f"[video] prepare v{v}，快照副本已加载", flush=True)

    from algo.video.video_pipeline import VideoInspector
    insp = VideoInspector(pipe, step=1, pulse_win=args.pulse_win)

    tmp_dir = tempfile.mkdtemp(prefix="aoi_vp_v_")
    probe = cv2.imread(normals[0], cv2.IMREAD_COLOR)
    size = (probe.shape[1], probe.shape[0])   # 源图原生分辨率
    inject_at = max(2, args.n_normal // 3)
    v1 = os.path.join(tmp_dir, "v1_inject.avi")
    segments = _make_video(v1, normals[4:], defects, inject_at, args.inject,
                           args.n_normal, size)
    v0 = os.path.join(tmp_dir, "v0_control.avi")
    _make_video(v0, normals[4:], [], 0, 0, args.control_frames, size)

    r1 = insp.inspect(v1)
    r0 = insp.inspect(v0)

    # 注入段帧级召回（脉冲降级后 decision_smooth 口径）
    s0, s1 = segments[0]
    seg_frames = [f for f in r1["frames"] if s0 <= f["index"] <= s1]
    seg_hit = sum(1 for f in seg_frames
                  if f["decision_smooth"] == "anomaly")
    inject_recall = round(seg_hit / max(len(seg_frames), 1), 4)

    failures = []
    d1 = r1["summary"]["final_decision"]
    d0 = r0["summary"]["final_decision"]
    if d1 != "anomaly":
        failures.append(f"V1 注入视频 final_decision={d1}（期望 anomaly）")
    if inject_recall < MIN_INJECT_RECALL:
        failures.append(f"V1 注入段帧级召回 {inject_recall} < "
                        f"{MIN_INJECT_RECALL}")
    if d0 == "anomaly":
        failures.append(f"V0 纯正常对照 final_decision=anomaly（误报）")

    torch_m = _torch_manifest()
    report = {
        "manifest": {
            "commit": _git_commit(), "python": sys.version,
            "platform": platform.platform(), "torch": torch_m,
            "device": ("gpu" if torch_m.get("cuda_available") else "cpu"),
            "config": args.config, "config_hash": _file_hash(args.config),
            "dataset": cat_dir, "category": cat_name, "synthetic": synthetic,
            "pulse_win": args.pulse_win, "min_inject_recall": MIN_INJECT_RECALL,
            "note": "合成视频（真实缺陷图静止注入）；时序纹理/运动模糊"
                    "不在覆盖范围；官方视频用同一管线复跑归档本报告",
        },
        "V1_inject": {"truth_segments": segments,
                      "final_decision": d1,
                      "inject_recall": inject_recall,
                      "summary": r1["summary"]},
        "V0_control": {"final_decision": d0, "summary": r0["summary"]},
        "failures": failures,
    }
    log_dir = REPO_ROOT / "storage" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"video_protocol_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[video] V1 final={d1} 注入段召回={inject_recall} "
          f"异常帧={r1['summary']['anomaly_frames']}/{r1['summary']['total_frames']}",
          flush=True)
    print(f"[video] V0 final={d0} 异常帧={r0['summary']['anomaly_frames']}",
          flush=True)
    print(f"[video] 报告已写 {out}", flush=True)
    ok = not failures
    print(f"[video] 结论: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f"（{failures}）"), flush=True)
    import shutil
    shutil.rmtree(tmp_storage, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
