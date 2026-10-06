"""视频模态（§6）：关键帧 → 同一管线 → 时序平滑

设计（§6）：视频不做独立检测路径，而是：
  1. 关键帧采样（默认每 N 帧取一帧，控制吞吐）
  2. 每帧走 Pipeline.predict_frame（与图片同一槽位管线，帧级判定+框+类型）
  3. 时序平滑：滑动 EMA 抑制单帧抖动；短脉冲帧（连续 <2 帧异常）降级为灰区

边界声明（U7）："顺序错误"在视频流中是时序模式，关键帧+平滑可能漏检——
已声明为已知边界，若赛题视频权重高，评估加轻量时序槽位。

输出：逐帧结果（fused/decision/boxes/types/类型归因）+ 汇总（异常帧率、主导缺陷类型）。
"""
import os
import numpy as np
import cv2


def keyframes_from_video(video_path, step=5, max_frames=None):
    """视频取样；max_frames=None 时完整读到 EOF，返回帧、FPS 和采样统计。"""
    step = max(1, int(step))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频 {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    idx = 0
    decoded = 0
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1
        decoded += 1
    cap.release()
    if not frames:
        raise IOError(f"视频 {video_path} 无有效帧")
    return frames, fps, decoded, idx


def smooth_ema(scores, alpha=0.5):
    """时序平滑（§6）：EMA，首帧保底。scores: list[float] -> list[float]"""
    out = []
    prev = scores[0] if scores else 0.0
    for s in scores:
        prev = alpha * s + (1 - alpha) * prev
        out.append(float(prev))
    return out


class VideoInspector:
    """视频质检器：包装 Pipeline，逐关键帧检测 + 时序平滑 + 短脉冲降级。

    输出（逐帧 + 汇总）：
      frames: list[{"index", "fused", "fused_smooth", "decision", "decision_smooth",
                    "boxes", "types"}]
      summary: {"anomaly_frames", "gray_frames", "total_frames", "fps",
                "anomaly_ratio", "mean_fused", "max_fused",
                "dominant_type", "final_decision"}
    """

    def __init__(self, pipeline, step=1, alpha=0.5, max_frames=None,
                 pulse_win=2):
        self.pipe = pipeline
        self.step = step
        self.alpha = alpha
        self.max_frames = max_frames
        self.pulse_win = pulse_win        # 短脉冲判定窗口（连续 <N 帧异常 → 灰区）

    def inspect(self, video_path):
        frames, fps, decoded_frames, source_frames = keyframes_from_video(
            video_path, self.step, self.max_frames)
        results = []
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = self.pipe.predict_frame(rgb, path=f"{video_path}#f{i}")
            r["index"] = i
            results.append(r)
        # 时序平滑（§6）
        smoothed = smooth_ema([r["fused"] for r in results], self.alpha)
        for r, s in zip(results, smoothed):
            r["fused_smooth"] = s
            r["decision_smooth"] = self.pipe.decider.decide(s)
        # 短脉冲降级：连续异常 < pulse_win 帧 → 灰区（防单帧误报，§6）
        self._downgrade_pulses(results)
        anomaly = sum(1 for r in results if r["decision_smooth"] == "anomaly")
        gray = sum(1 for r in results if r["decision_smooth"] == "gray")
        final = "anomaly" if anomaly > 0 else ("gray" if gray > 0 else "normal")
        # 缺陷类型归因聚合（§6.2 L3）：跨帧统计 dominant_type
        type_counter = {}
        for r in results:
            for t in (r.get("types") or []):
                if isinstance(t, dict):
                    type_counter[t.get("type", "unknown")] = \
                        type_counter.get(t.get("type", "unknown"), 0) + 1
                else:
                    type_counter[str(t)] = type_counter.get(str(t), 0) + 1
        dom = max(type_counter, key=type_counter.get) if type_counter else None
        summary = {"anomaly_frames": anomaly, "gray_frames": gray,
                   "total_frames": len(results), "source_frames": source_frames,
                   "decoded_frames": decoded_frames,
                   "sample_step": self.step,
                   "dropped_frames": max(0, source_frames - decoded_frames),
                   "fps": float(fps),
                   "anomaly_ratio": round(anomaly / max(len(results), 1), 4),
                   "mean_fused": round(float(np.mean(smoothed)), 4),
                   "max_fused": round(float(np.max(smoothed)), 4),
                   "dominant_type": dom,
                   "type_counter": type_counter,
                   "final_decision": final}
        return {"frames": results, "summary": summary}

    def _downgrade_pulses(self, results):
        """短脉冲降级（§6）：连续 < pulse_win 帧的异常判定 → 灰区。"""
        n = len(results)
        i = 0
        while i < n:
            if results[i]["decision_smooth"] == "anomaly":
                j = i
                while j + 1 < n and results[j + 1]["decision_smooth"] == "anomaly":
                    j += 1
                if (j - i + 1) < self.pulse_win:     # 孤立/短脉冲 → 降级灰区
                    for k in range(i, j + 1):
                        results[k]["decision_smooth"] = "gray"
                        results[k]["pulse_downgraded"] = True
                i = j + 1
            else:
                i += 1
