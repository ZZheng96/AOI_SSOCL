"""双检融合：传统算法（模板差分）+ 特征学习（品类模型）结果汇总判定。

engine_mode（模板配置）：
- traditional：仅传统算法
- feature：仅特征学习（AOI 品类模型）
- dual：双检并行，任一 NG/anomaly → NG，均 OK → OK

设计原则：两个引擎独立执行、独立可解释；融合只做"汇总判定 + 分栏展示"，
不做跨引擎分数混合（避免不可解释），灰区（gray）单独标记供人工复判。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.detect.types import DetectSummary
from app.engines.feature import DetectionResult

ENGINE_MODES = ("traditional", "feature", "dual")


def result_interpretation(overall: str | None) -> dict[str, Any]:
    status = {"OK": "PASS", "NG": "ANOMALY", "GRAY": "REVIEW",
              "ERROR": "ERROR"}.get(overall, "REVIEW")
    return {"final_status": status, "is_defect": status == "ANOMALY",
            "requires_review": status == "REVIEW",
            "is_system_error": status == "ERROR"}


def feature_to_dict(r: DetectionResult) -> dict[str, Any]:
    """DetectionResult → 可 JSON 序列化字典（供 API/UI）。"""
    return {
        "score": round(float(r.score), 4),
        "decision": r.decision,
        "is_anomaly": bool(r.is_anomaly),
        **result_interpretation({"normal": "OK", "anomaly": "NG",
                                 "gray": "GRAY", "error": "ERROR"}.get(r.decision)),
        "threshold": round(float(r.threshold), 4),
        "gray_threshold": round(float(r.gray_threshold), 4),
        "slot_scores": {k: round(float(v), 4) for k, v in r.slot_scores.items()},
        "weights": {k: round(float(v), 4) for k, v in r.weights.items()},
        "triggered_slot": r.triggered_slot,
        "open_alert": bool(r.open_alert),
        "latency_ms": round(float(r.latency_ms), 2),
        "n_tiles": int(r.n_tiles),
        "boxes": r.defect_boxes,
        "types": r.types,
        "heatmap_paths": dict(r.heatmap_paths or {}),
        "detection_id": (r.extra or {}).get("detection_id"),
        "backend": (r.extra or {}).get("backend", "local"),
    }


@dataclass
class DualSummary:
    """单图双检汇总结果。"""
    overall: str = "OK"                     # OK / NG / GRAY / ERROR（GRAY=灰区待复判，不算通过）
    overall_ok: bool = True
    engine_mode: str = "dual"
    traditional: DetectSummary | None = None
    feature: dict[str, Any] | None = None
    feature_error: str = ""                 # 品类模型未准备等
    gray: bool = False                      # 任一引擎灰区
    boxes: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0
    template_id: str | None = None
    template_version: int | None = None
    source: str = "dual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            **result_interpretation(self.overall),
            "overall_ok": self.overall_ok,
            "engine_mode": self.engine_mode,
            "gray": self.gray,
            "feature_error": self.feature_error,
            "feature": self.feature,
            "traditional": {
                "overall": self.traditional.overall if self.traditional else None,
                "ng_count": self.traditional.ng_count if self.traditional else 0,
                "elapsed_ms": self.traditional.elapsed_ms if self.traditional else 0,
                "results": [
                    {"status": r.status, "display_name": r.display_name,
                     "message": r.message, "defect_count": r.defect_count,
                     "elapsed_ms": r.elapsed_ms, "metadata": r.metadata,
                     "boxes": [
                        {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "label": b.label}
                        for b in r.boxes]}
                    for r in (self.traditional.results if self.traditional else [])
                ],
            } if self.traditional else None,
            "boxes": self.boxes,
            "elapsed_ms": self.elapsed_ms,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "source": self.source,
        }


def _box_iou(a: dict, b: dict) -> float:
    ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
    ax2, ay2 = ax1 + float(a.get("w", 0)), ay1 + float(a.get("h", 0))
    bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
    bx2, by2 = bx1 + float(b.get("w", 0)), by1 + float(b.get("h", 0))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
             + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


def _dedupe_boxes(boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    # 固定排序保证传统/特征返回顺序变化时合并结果与 sources 顺序稳定。
    ordered = sorted(boxes, key=lambda b: (
        str(b.get("engine", "")), str(b.get("label", "")),
        float(b.get("x", 0)), float(b.get("y", 0)),
        float(b.get("w", 0)), float(b.get("h", 0))))
    merged: list[dict] = []
    for box in ordered:
        match = next((item for item in merged
                      if item.get("engine") != box.get("engine")
                      and _box_iou(item, box) >= iou_threshold), None)
        if match is None:
            item = dict(box)
            item["sources"] = [box.get("engine", "unknown")]
            merged.append(item)
            continue
        x1 = min(float(match["x"]), float(box["x"]))
        y1 = min(float(match["y"]), float(box["y"]))
        x2 = max(float(match["x"]) + float(match["w"]),
                 float(box["x"]) + float(box["w"]))
        y2 = max(float(match["y"]) + float(match["h"]),
                 float(box["y"]) + float(box["h"]))
        match.update(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1),
                     engine="dual")
        match.setdefault("sources", []).append(box.get("engine", "unknown"))
    return merged


def _collect_boxes(trad: DetectSummary | None, feat: dict[str, Any] | None) -> list[dict]:
    boxes: list[dict] = []
    if trad is not None:
        for r in trad.results:
            if r.ok or r.skipped or r.status == "SKIP":
                continue
            for b in r.boxes:
                boxes.append({"x": b.x, "y": b.y, "w": b.w, "h": b.h,
                              "label": f"传统·{b.label}", "engine": "traditional"})
    if feat is not None:
        for b in feat.get("boxes", []):
            bbox = b.get("bbox") or [0, 0, 0, 0]
            boxes.append({"x": int(bbox[0]), "y": int(bbox[1]),
                          "w": int(bbox[2] - bbox[0]), "h": int(bbox[3] - bbox[1]),
                          "label": f"AOI·{b.get('dominant_slot')}", "engine": "feature"})
    return _dedupe_boxes(boxes)


def fuse_dual(trad: DetectSummary | None, feat: dict[str, Any] | None,
              feat_err: str, mode: str, elapsed_ms: int) -> DualSummary:
    gray = False
    overall = "OK"
    if mode in ("dual", "feature") and feat is not None:
        gray = feat["decision"] == "gray"

    required_missing = (
        mode == "traditional" and trad is None
        or mode == "feature" and feat is None
        or mode == "dual" and (trad is None or feat is None)
    )
    execution_error = (
        (trad is not None and (trad.gate_blocked or trad.overall == "ERROR"))
        or (feat is not None and feat.get("decision") == "error")
    )
    if required_missing or execution_error:
        overall = "ERROR"
    elif mode == "traditional":
        overall = "NG" if trad.overall == "NG" else "OK"
    elif mode == "feature":
        if feat["is_anomaly"]:
            overall = "NG"
        elif gray:
            # 灰区不得静默放行：透出 GRAY 供人工复判
            overall = "GRAY"
    else:  # dual
        trad_ng = trad.overall == "NG"
        feat_ng = bool(feat["is_anomaly"])
        if trad_ng or feat_ng:
            overall = "NG"
        elif gray:
            # 任一引擎灰区且无 NG：整体 GRAY（需复判），不得判 OK
            overall = "GRAY"
        else:
            overall = "OK"

    return DualSummary(
        overall=overall,
        overall_ok=(overall == "OK"),
        engine_mode=mode,
        traditional=trad,
        feature=feat,
        feature_error=feat_err,
        gray=gray,
        boxes=_collect_boxes(trad, feat),
        elapsed_ms=elapsed_ms,
        template_id=(trad.template_id if trad else None),
        template_version=(trad.template_version if trad else None),
        source="dual",
    )
