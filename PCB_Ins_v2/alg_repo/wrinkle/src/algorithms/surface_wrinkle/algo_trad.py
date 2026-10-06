"""PCB 表皮起皱检测核心：Laplacian 纹理 + PCA 弯曲筛选。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_CONFIG: Dict[str, Any] = {
    "resize_width": 1400,
    "pre_blur_sigma": 1.2,
    "lap_thresh": 5.0,
    "min_blob_area": 30,
    "min_blob_diag": 40.0,
    "curvature_ratio_min": 0.05,
    "wrinkle_score_thresh": 0.25,
    "dilate_px": 40,
    "merge_gap_px": 50,
    "max_defect_boxes": 4,
    # region=零散区域框（闭运算+邻近合并，默认）；line=每一条皱纹单独框
    "box_display_mode": "region",
}

_BOX_MODE_LINE = frozenset({"line", "each", "strip", "blob"})


def _normalize_box_display_mode(value: Any) -> str:
    """region=零散区域；line=每一条。非法值回退 region。"""
    mode = str(value or "region").strip().lower()
    if mode in _BOX_MODE_LINE:
        return "line"
    return "region"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _straightness_ratio(xs: np.ndarray, ys: np.ndarray, diag: float) -> float:
    if len(xs) < 3:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = (centered.T @ centered) / len(xs)
    _, eigvecs = np.linalg.eigh(cov)
    minor_axis = eigvecs[:, 0]
    proj_minor = centered @ minor_axis
    return float(np.abs(proj_minor).max() / max(1.0, diag))


def _merge_nearby_boxes(
    cand: List[Tuple[float, int, int, int, int, float]],
    gap: int,
) -> List[Tuple[float, int, int, int, int, float]]:
    if not cand:
        return []
    boxes = [[x, y, x + w, y + h, conf, area] for area, x, y, w, h, conf in cand]
    changed = True
    while changed:
        changed = False
        out: List[List[float]] = []
        used = [False] * len(boxes)
        for i, box in enumerate(boxes):
            if used[i]:
                continue
            x1, y1, x2, y2, conf, area = box
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2, oconf, oarea = boxes[j]
                if (
                    x2 + gap < ox1 - gap
                    or ox2 + gap < x1 - gap
                    or y2 + gap < oy1 - gap
                    or oy2 + gap < y1 - gap
                ):
                    continue
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
                conf = max(conf, oconf)
                area += oarea
                used[j] = True
                changed = True
            out.append([x1, y1, x2, y2, conf, area])
        boxes = out
    return [
        (float(area), int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(conf))
        for x1, y1, x2, y2, conf, area in boxes
    ]


def run(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    gray = _to_gray(image)
    h, w = gray.shape[:2]
    resize_width = max(200, int(cfg["resize_width"]))
    scale = min(1.0, resize_width / float(max(h, w)))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if scale < 1.0:
        small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        small = gray.astype(np.float32)

    sm = cv2.GaussianBlur(small, (0, 0), float(cfg["pre_blur_sigma"]))
    lap = np.abs(cv2.Laplacian(sm, cv2.CV_32F, ksize=3))
    edge = (lap > float(cfg["lap_thresh"])).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
    min_area = int(cfg["min_blob_area"])
    min_diag = float(cfg["min_blob_diag"])
    curv_min = float(cfg["curvature_ratio_min"])

    curvy_mask = np.zeros_like(edge)
    n_curvy_blobs = 0
    # 每条候选皱纹连通域框：(area, x, y, w, h)，供 line 模式直接出框
    line_blobs: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        diag = float((bw ** 2 + bh ** 2) ** 0.5)
        if diag <= min_diag:
            continue
        label_crop = labels[by:by + bh, bx:bx + bw]
        local_ys, local_xs = np.nonzero(label_crop == i)
        ratio = _straightness_ratio(local_xs, local_ys, diag)
        if ratio <= curv_min:
            continue
        curvy_mask[by:by + bh, bx:bx + bw][label_crop == i] = 1
        n_curvy_blobs += 1
        line_blobs.append((area, bx, by, bw, bh))

    total_px = int(edge.size)
    curvy_px = int(curvy_mask.sum())
    score = curvy_px / float(total_px) if total_px else 0.0
    status = "NG" if score > float(cfg["wrinkle_score_thresh"]) else "OK"
    box_mode = _normalize_box_display_mode(cfg.get("box_display_mode"))

    boxes_out: List[Dict[str, Any]] = []
    if status == "NG" and curvy_px > 0:
        inv_scale = 1.0 / scale
        cand: List[Tuple[float, int, int, int, int, float]] = []

        if box_mode == "line":
            # 具体每一条：每个通过弯曲筛选的连通域单独出框，不做闭运算/邻近合并
            for area_c, x, y, bw, bh in line_blobs:
                fill = float(area_c) / float(max(1, bw * bh))
                conf = max(0.0, min(1.0, fill * 2.0 + score))
                cand.append((float(area_c), x, y, bw, bh, conf))
        else:
            # 零散区域：闭运算聚合成片，再按间隙合并邻近框（原逻辑）
            dilate_px = max(1, int(cfg["dilate_px"]))
            close_ds = 4 if min(curvy_mask.shape) >= 4 * dilate_px else 1
            if close_ds > 1:
                cm_h, cm_w = curvy_mask.shape
                ds_w = max(1, cm_w // close_ds)
                ds_h = max(1, cm_h // close_ds)
                mask_small = cv2.resize(curvy_mask, (ds_w, ds_h), interpolation=cv2.INTER_NEAREST)
                k_sz = max(3, dilate_px // close_ds)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
                closed_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)
                closed = cv2.resize(closed_small, (cm_w, cm_h), interpolation=cv2.INTER_NEAREST)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
                closed = cv2.morphologyEx(curvy_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area_c = float(cv2.contourArea(c))
                if area_c < min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                sub = curvy_mask[y:y + bh, x:x + bw]
                fill = float(sub.mean()) if sub.size else 0.0
                conf = max(0.0, min(1.0, fill * 2.0 + score))
                cand.append((area_c, x, y, bw, bh, conf))

            merge_gap = max(0, int(cfg["merge_gap_px"]))
            cand = _merge_nearby_boxes(cand, merge_gap)

        cand.sort(key=lambda t: t[0], reverse=True)
        if cand:
            keep_min = cand[0][0] * 0.05
            cand = [c for c in cand if c[0] >= keep_min]
        for area_c, x, y, bw, bh, conf in cand[: int(cfg["max_defect_boxes"])]:
            boxes_out.append({
                "x": int(round(x * inv_scale)),
                "y": int(round(y * inv_scale)),
                "width": int(round(bw * inv_scale)),
                "height": int(round(bh * inv_scale)),
                "confidence": conf,
                "area_px": int(area_c),
            })

    return {
        "status": status,
        "score": score,
        "boxes": boxes_out,
        "cost_time": time.perf_counter() - t0,
        "extra": {
            "curvy_px": curvy_px,
            "total_px": total_px,
            "curvy_blobs": n_curvy_blobs,
            "analysis_scale": scale,
            "box_display_mode": box_mode,
        },
    }
