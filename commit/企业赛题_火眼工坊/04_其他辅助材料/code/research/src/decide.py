"""决策层：双阈值 + 灰区 + 检测框 + 追溯记录（§6）。

红线 3：阈值只由 train/good 分位数定。
红线 5：每次推理落盘 {slot_scores, fused, boxes}。
"""
import json
import numpy as np


class Decider:
    def __init__(self, tau_high, tau_gray):
        self.tau_high = float(tau_high)
        self.tau_gray = float(tau_gray)

    @classmethod
    def from_normal_scores(cls, fused_normal_scores, q_high=0.99, q_gray=0.95):
        s = np.asarray(fused_normal_scores, dtype=np.float64)
        return cls(np.quantile(s, q_high), np.quantile(s, q_gray))

    def decide(self, score):
        if score >= self.tau_high:
            return "anomaly"
        if score >= self.tau_gray:
            return "gray"       # 灰区：缓判/请人工（§7.1 三级反馈的入口）
        return "normal"


def heatmap_to_boxes(heatmap, tiles, image_size, thresh):
    """热力图 -> 检测框（原图坐标）。heatmap: HxW float（已按 tile 拼合 max）。"""
    import cv2
    mask = (heatmap >= thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= 4:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


class TraceLogger:
    """红线 5：推理记录落盘，可追溯（§6 L4 输出层）。"""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def log(self, record):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
