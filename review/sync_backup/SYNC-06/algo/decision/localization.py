"""定位后处理与区域复核（AC-063~065 / LOC-1~5）。

- 后处理参数版本化：阈值/形态学/最小面积/工作网格/top_k 绑定 version_id，
  定位评估与区域记录都引用同一版本，避免“改了参数旧指标还挂着”。
- 区域记录：坐标统一为原图像素 [x0,y0,x1,y1]（左闭右开，已裁剪到图内），
  标明来源 engine_box/local_branch/heatmap_cc/human_box；默认 status=candidate，
  未经人工确认不进入框级训练（trainable_region_boxes）。
- 定位评估：实例召回/精度/误定位/平均 IoU；后处理选择只允许 dev 集。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime

import numpy as np

POSTPROC_KEYS = ("mask_thresh", "min_area", "morph", "work_side", "top_k")
SOURCES = ("engine_box", "local_branch", "heatmap_cc", "human_box")
REVIEW_OPS = ("confirm", "reject", "add", "revise")


def postproc_params(cfg=None):
    """由 cfg.hierarchical 生成版本化后处理参数（同参数 → 同 version_id）。"""
    h = ((cfg or {}).get("hierarchical", {}) or {})
    p = {"mask_thresh": float(h.get("mask_thresh", 0.5)),
         "min_area": int(h.get("min_area", 16)),
         "morph": "ellipse5_open_close",
         "work_side": int(h.get("work_side", 512)),
         "top_k": (int(h["top_k"]) if h.get("top_k") else None)}
    digest = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
    return {**p, "version_id": f"pp-{digest}"}


def coord_space(H, W):
    return {"space": "original", "format": "xyxy", "size": [int(W), int(H)]}


def finalize_regions(boxes, fused_work, scale, image_hw, params, source="engine_box"):
    """工作网格框 → 原图区域记录（就地补字段并返回排序/截断后的列表）。

    score 在工作网格融合热图上取框内峰值/均值；坐标 floor/ceil 放大并裁剪。"""
    H, W = int(image_hw[0]), int(image_hw[1])
    inv = 1.0 / scale if scale > 0 else 1.0
    out = []
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        peak = mean = None
        if fused_work is not None:
            roi = fused_work[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            if roi.size:
                peak, mean = float(roi.max()), float(roi.mean())
        X0 = min(max(int(math.floor(x0 * inv)), 0), W)
        Y0 = min(max(int(math.floor(y0 * inv)), 0), H)
        X1 = min(max(int(math.ceil(x1 * inv)), X0), W)
        Y1 = min(max(int(math.ceil(y1 * inv)), Y0), H)
        b.update(bbox=[X0, Y0, X1, Y1], area_work=int(b.get("area", 0)),
                 area=int(round(b.get("area", 0) * inv * inv)),
                 score_peak=peak, score_mean=mean,
                 source=b.get("source") or source,
                 method="heatmap_mask_cc", postproc=params["version_id"],
                 status="candidate", confirmed=False)
        out.append(b)
    out.sort(key=lambda r: -(r["score_peak"] if r["score_peak"] is not None else -1.0))
    if params.get("top_k"):
        out = out[:params["top_k"]]
    for i, r in enumerate(out):
        r["region_id"] = f"r{i}"
    return out


def localization_summary(regions, params, image_hw, status=None, reason=None):
    """检测结果中的定位容器（透出给 API/UI/落库）。"""
    st = status or ("ok" if regions else "no_region")
    d = {"status": st, "postproc": params, "coord_space": coord_space(*image_hw),
         "n_regions": len(regions)}
    if reason:
        d["reason"] = reason
    return d


# ---------------- 区域复核（AC-064） ----------------

def _region_view(r):
    return {k: r.get(k) for k in ("region_id", "bbox", "area", "source", "score_peak",
                                  "score_mean", "dominant_slot", "postproc")}


def new_review(regions, source_version=None):
    return {"revision": 0, "source_version": source_version,
            "regions": [{**_region_view(r), "status": "candidate", "confirmed": False}
                        for r in regions or []],
            "events": []}


def _valid_bbox(b, size=None):
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return False
    try:
        x0, y0, x1, y1 = (float(v) for v in b)
    except (TypeError, ValueError):
        return False
    if not (x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0):
        return False
    if size:
        W, H = size
        return x1 <= W and y1 <= H
    return True


def apply_op(review, op, size=None):
    """纯函数：返回新 review；op={op, region_id?, bbox?, operator, event_id,
    expected_revision, reason?}。幂等/冲突由调用方按 event_id/revision 判。"""
    kind = op.get("op")
    if kind not in REVIEW_OPS:
        raise ValueError(f"未知区域操作: {kind}")
    regions = [dict(r) for r in review.get("regions", [])]
    now = datetime.now().isoformat()
    meta = {"operator": op["operator"], "at": now, "event_id": op["event_id"]}
    if kind == "add":
        if not _valid_bbox(op.get("bbox"), size):
            raise ValueError("bbox 非法（需原图像素 xyxy 且在图内）")
        x0, y0, x1, y1 = (int(round(v)) for v in op["bbox"])
        rid = f"h{sum(1 for r in regions if r['region_id'].startswith('h'))}"
        regions.append({"region_id": rid, "bbox": [x0, y0, x1, y1],
                        "area": (x1 - x0) * (y1 - y0), "source": "human_box",
                        "score_peak": None, "score_mean": None, "dominant_slot": None,
                        "postproc": None, "status": "confirmed", "confirmed": True, **meta})
        target = rid
    else:
        target = op.get("region_id")
        hit = next((r for r in regions if r["region_id"] == target), None)
        if hit is None:
            raise KeyError(target)
        if kind == "confirm":
            hit.update(status="confirmed", confirmed=True, **meta)
        elif kind == "reject":
            hit.update(status="rejected", confirmed=False, **meta)
        else:                                  # revise：人工修正坐标 → 视为人工框
            if not _valid_bbox(op.get("bbox"), size):
                raise ValueError("bbox 非法（需原图像素 xyxy 且在图内）")
            x0, y0, x1, y1 = (int(round(v)) for v in op["bbox"])
            hit.update(revised_from={"bbox": hit["bbox"], "source": hit["source"]},
                       bbox=[x0, y0, x1, y1], area=(x1 - x0) * (y1 - y0),
                       source="human_box", status="confirmed", confirmed=True, **meta)
    event = {"event_id": op["event_id"], "op": kind, "region_id": target,
             "payload": {k: op.get(k) for k in ("op", "region_id", "bbox", "operator",
                                                "expected_revision", "reason")},
             "at": now}
    return {**review, "revision": int(review.get("revision", 0)) + 1,
            "regions": regions, "events": [*review.get("events", []), event]}


def trainable_region_boxes(review):
    """仅已确认 / 人工框可进入框级训练；candidate/rejected 隔离。"""
    return [list(r["bbox"]) for r in (review or {}).get("regions", [])
            if r.get("confirmed") and r.get("status") == "confirmed"]


# ---------------- 定位评估（AC-065） ----------------

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def eval_localization(samples, iou_thr=0.5, postproc=None):
    """samples=[{"pred":[xyxy...], "gt":[xyxy...]}]；贪心按 IoU 匹配。

    mislocalization：有 GT 的图预测了框但无一命中（图级可能判对，框却错位）。"""
    tp = n_pred = n_gt = 0
    ious, misloc, imgs_with_gt = [], 0, 0
    for s in samples:
        preds, gts = list(s.get("pred") or []), list(s.get("gt") or [])
        n_pred += len(preds)
        n_gt += len(gts)
        pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(preds)
                        for j, g in enumerate(gts)), reverse=True)
        used_p, used_g, hit = set(), set(), 0
        for v, i, j in pairs:
            if v < iou_thr:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            ious.append(v)
            hit += 1
        tp += hit
        if gts:
            imgs_with_gt += 1
            if preds and hit == 0:
                misloc += 1
    return {"iou_thr": iou_thr, "postproc": postproc,
            "n_images": len(samples), "n_gt": n_gt, "n_pred": n_pred, "tp": tp,
            "instance_recall": (tp / n_gt) if n_gt else None,
            "precision": (tp / n_pred) if n_pred else None,
            "mislocalization_rate": (misloc / imgs_with_gt) if imgs_with_gt else None,
            "mean_iou": float(np.mean(ious)) if ious else None}


def select_postproc(candidates, split, metric="instance_recall"):
    """candidates=[(params, eval_result)]；只允许 dev 集，禁止测试集调参或用图级 AUC 代替。"""
    if split != "dev":
        raise ValueError("定位后处理只能在 dev 集选择（禁止测试集调参）")
    if metric not in ("instance_recall", "precision", "mean_iou"):
        raise ValueError("定位选择须用定位指标，不得以图级 AUC 代替")
    scored = [(c, r) for c, r in candidates if r.get(metric) is not None]
    if not scored:
        raise ValueError("无可用定位指标")
    best = max(scored, key=lambda cr: (cr[1][metric], cr[1].get("precision") or 0.0))
    return {"params": best[0], "metric": metric, "split": split, "result": best[1]}
