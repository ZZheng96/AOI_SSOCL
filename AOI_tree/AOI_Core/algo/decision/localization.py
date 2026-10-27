"""定位后处理与区域复核（AC-063~065 / LOC-1~5）。

- 后处理参数版本化：阈值/形态学/最小面积/工作网格/top_k 绑定 version_id，
  定位评估与区域记录都引用同一版本，避免“改了参数旧指标还挂着”。
- 区域记录：坐标统一为原图像素 [x0,y0,x1,y1]（左闭右开，已裁剪到图内），
  标明来源 engine_box/local_branch/heatmap_cc/human_box；默认 status=candidate，
  未经人工确认不进入框级训练（trainable_region_boxes）。
- 定位评估：实例召回/精度/误定位/平均 IoU；后处理选择只允许 dev 集。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime

import numpy as np

POSTPROC_KEYS = ("mask_thresh", "min_area", "morph", "work_side", "top_k")
SOURCES = ("engine_box", "local_branch", "heatmap_cc", "human_box")
REVIEW_OPS = ("confirm", "reject", "add", "revise", "dispute", "delete", "undo", "redo")
INACTIVE_STATUS = ("deleted", "retracted")   # 软删除/撤销新增：保留修订链，不训练、不可再操作


class ReviewConflict(ValueError):
    """撤销目标已被他人修改（修订链不一致）→ API 映射 409。"""


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
    if kind == "undo":
        return _apply_undo(review, op)
    if kind == "redo":
        return _apply_redo(review, op)
    regions = [dict(r) for r in review.get("regions", [])]
    now = datetime.now().isoformat()
    meta = {"operator": op["operator"], "at": now, "event_id": op["event_id"]}
    before = None                              # 操作前区域快照（撤销用）；add 无前态
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
        if not target:
            raise ValueError(f"{kind} 需要 region_id")
        hit = next((r for r in regions if r["region_id"] == target), None)
        if hit is None:
            raise KeyError(target)
        before = copy.deepcopy(hit)
        prev_st, prev_op = hit.get("status"), hit.get("operator")
        reason = (op.get("reason") or "").strip()
        if prev_st in INACTIVE_STATUS:
            raise ValueError(f"区域已{'删除' if prev_st == 'deleted' else '撤销'}，不可再操作")
        if kind == "delete":
            # 仅人工框可删（原始来源为 human_box）；模型候选框只能拒绝
            if ((hit.get("revised_from") or {}).get("source") or hit.get("source")) != "human_box":
                raise ValueError("模型候选框不可删除，请使用拒绝")
            if not reason:
                raise ValueError("删除人工框须填写 reason")
        elif prev_st == "disputed" and kind != "dispute" and not reason:
            raise ValueError("争议区域仲裁须填写 reason")
        # 噪声隔离：他人推翻已确认/已拒绝结论且无理由 → 转 disputed，不进训练
        flip = (prev_st in ("confirmed", "rejected") and prev_op
                and prev_op != op["operator"] and not reason
                and ((kind == "confirm" and prev_st == "rejected")
                     or (kind == "reject" and prev_st == "confirmed")))
        if kind == "delete":                   # 软删除：保留 bbox 与修订链，不训练
            hit.update(status="deleted", confirmed=False,
                       deleted_from={"status": prev_st, "operator": prev_op}, **meta)
        elif kind == "dispute" or flip:
            hit.update(status="disputed", confirmed=False,
                       dispute={"prev_status": prev_st, "prev_operator": prev_op,
                                "requested": kind}, **meta)
        elif kind == "confirm":
            hit.update(status="confirmed", confirmed=True, **meta)
        elif kind == "reject":
            hit.update(status="rejected", confirmed=False, **meta)
        else:                                  # revise：人工修正坐标 → 视为人工框
            if not _valid_bbox(op.get("bbox"), size):
                raise ValueError("bbox 非法（需原图像素 xyxy 且在图内）")
            x0, y0, x1, y1 = (int(round(v)) for v in op["bbox"])
            # 多次修订保留最初来源（否则模型框二次修订后 origin 变 human_box，可被误删/误判）
            origin = (hit.get("revised_from") or {}).get("source") or hit["source"]
            hit.update(revised_from={"bbox": hit["bbox"], "source": origin},
                       bbox=[x0, y0, x1, y1], area=(x1 - x0) * (y1 - y0),
                       source="human_box", status="confirmed", confirmed=True, **meta)
    event = {"event_id": op["event_id"], "op": kind, "region_id": target,
             "payload": {k: op.get(k) for k in ("op", "region_id", "bbox", "operator",
                                                "expected_revision", "reason")},
             "at": now, "before": before}
    return {**review, "revision": int(review.get("revision", 0)) + 1,
            "regions": regions, "events": [*review.get("events", []), event]}


def _apply_undo(review, op):
    """撤销本人最近一次未撤销的操作：追加反向修订事件（不改写历史）。

    - add → 目标区域 status=retracted（保留记录，不训练、region_id 不复用）；
    - 其余 → 恢复事件记录的操作前快照（before）；
    - 目标区域在该操作之后被他人改过 → ReviewConflict（409）。"""
    events = review.get("events") or []
    undone = _undone_ids(events)
    live = [(i, e) for i, e in enumerate(events)
            if e.get("op") not in ("undo", "redo") and e.get("event_id") not in undone]
    mine = [(i, e) for i, e in live if (e.get("payload") or {}).get("operator") == op["operator"]]
    if not mine:
        raise ValueError("没有可撤销的本人操作")
    idx, tgt = mine[-1]
    if tgt.get("op") != "add" and not isinstance(tgt.get("before"), dict):
        raise ValueError("该操作缺少操作前快照，无法撤销")
    rid = tgt.get("region_id")
    if any(e.get("region_id") == rid for i, e in live if i > idx):
        raise ReviewConflict("撤销目标已被他人修改，请刷新后重试")
    regions = [dict(r) for r in review.get("regions", [])]
    pos = next((k for k, r in enumerate(regions) if r.get("region_id") == rid), None)
    if pos is None:
        raise KeyError(rid)
    now = datetime.now().isoformat()
    before = copy.deepcopy(regions[pos])
    mark = {"undone_by": {"event_id": op["event_id"], "operator": op["operator"],
                          "at": now, "undo_of": tgt["event_id"]}}
    if tgt.get("op") == "add":
        regions[pos] = {**regions[pos], "status": "retracted", "confirmed": False, **mark}
    else:
        regions[pos] = {**copy.deepcopy(tgt["before"]), **mark}
    event = {"event_id": op["event_id"], "op": "undo", "region_id": rid,
             "undo_of": tgt["event_id"], "undo_op": tgt.get("op"),
             "payload": {k: op.get(k) for k in ("op", "region_id", "bbox", "operator",
                                                "expected_revision", "reason")},
             "at": now, "before": before}
    return {**review, "revision": int(review.get("revision", 0)) + 1,
            "regions": regions, "events": [*events, event]}


def _undone_ids(events):
    """当前仍生效的撤销所指向的原事件 event_id（已被 redo 的撤销不算）。"""
    redone = {e.get("redo_of") for e in events if e.get("op") == "redo"}
    return {e.get("undo_of") for e in events
            if e.get("op") == "undo" and e.get("event_id") not in redone}


def _apply_redo(review, op):
    """重做本人最近一次未被重做的撤销：恢复撤销前快照（追加事件，不改写历史）。

    - 撤销之后本人又做了新操作 → 重做链失效（ValueError）；
    - 目标区域在撤销之后被他人改过 → ReviewConflict（409）。"""
    events = review.get("events") or []
    redone = {e.get("redo_of") for e in events if e.get("op") == "redo"}
    me = op["operator"]
    mine = [(i, e) for i, e in enumerate(events)
            if (e.get("payload") or {}).get("operator") == me]
    undos = [(i, e) for i, e in mine if e.get("op") == "undo" and e.get("event_id") not in redone]
    if not undos:
        raise ValueError("没有可重做的本人撤销")
    idx, tgt = undos[-1]
    if any(e.get("op") not in ("undo", "redo") for i, e in mine if i > idx):
        raise ValueError("撤销后已有新操作，重做链已失效")
    rid = tgt.get("region_id")
    if any(e.get("region_id") == rid and (e.get("payload") or {}).get("operator") != me
           for e in events[idx + 1:]):
        raise ReviewConflict("重做目标已被他人修改，请刷新后重试")
    if not isinstance(tgt.get("before"), dict):
        raise ValueError("该撤销缺少快照，无法重做")
    regions = [dict(r) for r in review.get("regions", [])]
    pos = next((k for k, r in enumerate(regions) if r.get("region_id") == rid), None)
    if pos is None:
        raise KeyError(rid)
    now = datetime.now().isoformat()
    before = copy.deepcopy(regions[pos])
    restored = {k: v for k, v in copy.deepcopy(tgt["before"]).items() if k != "undone_by"}
    regions[pos] = {**restored, "redone_by": {"event_id": op["event_id"], "operator": me,
                                              "at": now, "redo_of": tgt["event_id"]}}
    event = {"event_id": op["event_id"], "op": "redo", "region_id": rid,
             "redo_of": tgt["event_id"], "undo_of": tgt.get("undo_of"),
             "payload": {k: op.get(k) for k in ("op", "region_id", "bbox", "operator",
                                                "expected_revision", "reason")},
             "at": now, "before": before}
    return {**review, "revision": int(review.get("revision", 0)) + 1,
            "regions": regions, "events": [*events, event]}


TRAIN_STATUS = {"candidate": "excluded_candidate", "rejected": "excluded_rejected",
                "disputed": "excluded_disputed", "visual_only": "excluded_visual",
                "deleted": "excluded_deleted", "retracted": "excluded_retracted"}


def train_status(r):
    """AC-063 区域训练状态（派生，不落库）：仅 confirmed 且 confirmed=True 为 eligible；
    candidate/rejected/disputed/visual_only 一律 excluded_*（未知状态按候选处理）。"""
    st = r.get("status")
    if st == "confirmed" and r.get("confirmed") is True:
        return "eligible"
    return TRAIN_STATUS.get(st, "excluded_candidate")


def annotate_review(review):
    """AC-063 区域契约视图（纯函数，不改存储）：由修订链派生
    origin_source / source / reviewer / review_revision / train_status。

    人工确认/新增/修订的框对外 source=human_box；未经人工确认的候选、被拒或争议框
    保留原始来源，候选框与人工真值以 source + train_status 区分。"""
    rv = review or {}
    events = rv.get("events") or []
    base = int(rv.get("revision", 0)) - len(events)   # 每事件修订号 +1
    last, geom_human = {}, set()
    undone = _undone_ids(events)  # 已撤销（且未重做）的几何修改不算人工
    for i, ev in enumerate(events):
        last[ev.get("region_id")] = base + i + 1
        if ev.get("op") in ("add", "revise") and ev.get("event_id") not in undone:
            geom_human.add(ev.get("region_id"))
    regions = []
    for r in rv.get("regions") or []:
        rid = r.get("region_id")
        # 原始来源：revise 前的来源；add 的框本身即 human_box
        origin = (r.get("revised_from") or {}).get("source") or r.get("source")
        ts = train_status(r)
        human = rid in geom_human or ts == "eligible"
        regions.append({**r, "origin_source": origin,
                        "source": "human_box" if human else origin,
                        "reviewer": r.get("operator") if rid in last else None,
                        "review_revision": last.get(rid),
                        "confirmed": ts == "eligible", "train_status": ts})
    # AC-063 训练消费追溯（派生）：used / used_by / stale_after_use / stale_reason
    stale = {s["region_id"]: s["reason"] for s in stale_regions(rv, consumed_snapshot(rv))}
    by = {}
    for c in rv.get("consumption") or []:
        by.setdefault(c.get("region_id"), []).append(c)
    for r in regions:
        rid = r.get("region_id")
        r.update(used=rid in by, used_by=by.get(rid, []),
                 stale_after_use=rid in stale, stale_reason=stale.get(rid))
    return {**rv, "regions": regions,
            "stale_consumption": [{"region_id": k, "reason": v} for k, v in stale.items()]}


def trainable_region_boxes(review):
    """仅 train_status=eligible（人工确认/新增/修订且无争议）可进入框级训练；
    candidate/rejected/disputed 隔离。"""
    return [list(r["bbox"]) for r in (review or {}).get("regions", [])
            if train_status(r) == "eligible"]


# ---------------- 区域训练消费追溯（AC-063/064） ----------------
def _region_revisions(review):
    """region_id → 最近一次涉及它的事件修订号（与 annotate_review.review_revision 一致）。"""
    events = (review or {}).get("events") or []
    base = int((review or {}).get("revision", 0)) - len(events)
    return {ev.get("region_id"): base + i + 1 for i, ev in enumerate(events)}


def consumable_regions(review):
    """训练消费快照：eligible 区域的 {region_id, region_revision, bbox}。
    region_revision 无事件（仅初始）时为 0。"""
    revs = _region_revisions(review)
    return [{"region_id": r["region_id"], "region_revision": revs.get(r["region_id"], 0),
             "bbox": list(r["bbox"])}
            for r in (review or {}).get("regions", []) if train_status(r) == "eligible"]


def record_consumption(review, records):
    """纯函数：向 review['consumption'] 追加消费记录（按 candidate_id+region_id 去重）；
    不改 revision/events（消费不是人工修订）。"""
    old = list((review or {}).get("consumption") or [])
    seen = {(c.get("candidate_id"), c.get("region_id")) for c in old}
    add = [dict(c) for c in records
           if (c.get("candidate_id"), c.get("region_id")) not in seen]
    return {**(review or {}), "consumption": old + add}


def consumed_snapshot(review):
    """每个 region 最近一次消费记录的快照（用于派生 stale_after_use）。"""
    latest = {}
    for c in (review or {}).get("consumption") or []:
        latest[c.get("region_id")] = c
    return list(latest.values())


def stale_regions(review, snapshot):
    """对比消费快照与当前 review：消费后被删除/作废/修订的区域。
    reason：deleted（区域不存在或软删除/撤销）> excluded（不再 eligible）> revised
    （修订号或坐标变化）。返回 [{region_id, reason, train_status?}]。"""
    cur = {r.get("region_id"): r for r in (review or {}).get("regions") or []}
    revs = _region_revisions(review)
    out = []
    for s in snapshot or []:
        rid = s.get("region_id")
        r = cur.get(rid)
        if r is None or r.get("status") in INACTIVE_STATUS:
            out.append({"region_id": rid, "reason": "deleted"})
            continue
        ts = train_status(r)
        if ts != "eligible":
            out.append({"region_id": rid, "reason": "excluded", "train_status": ts})
        elif (revs.get(rid, 0) != int(s.get("region_revision") or 0)
              or list(r.get("bbox") or []) != list(s.get("bbox") or [])):
            out.append({"region_id": rid, "reason": "revised"})
    return out


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
