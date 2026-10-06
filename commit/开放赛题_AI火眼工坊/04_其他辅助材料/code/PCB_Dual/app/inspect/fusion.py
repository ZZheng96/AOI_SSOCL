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


def feature_to_dict(r: DetectionResult) -> dict[str, Any]:
    """DetectionResult → 可 JSON 序列化字典（供 API/UI）。"""
    return {
        "score": round(float(r.score), 4),
        "decision": r.decision,
        "is_anomaly": bool(r.is_anomaly),
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
                     "elapsed_ms": r.elapsed_ms, "boxes": [
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
    return _dedup_boxes(boxes)


def _dedup_boxes(boxes: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """按引擎固定顺序合并重叠框，保持传统框作为主框。"""
    merged: list[dict] = []
    for box in boxes:
        host = next((item for item in merged if _box_iou(item, box) >= iou_thresh), None)
        if host is None:
            merged.append(dict(box))
            continue
        labels = host.setdefault("labels", [host.get("label", "")])
        if box.get("label") not in labels:
            labels.append(box.get("label", ""))
        host["label"] = " · ".join(labels)
        x1 = min(host["x"], box["x"])
        y1 = min(host["y"], box["y"])
        x2 = max(host["x"] + host["w"], box["x"] + box["w"])
        y2 = max(host["y"] + host["h"], box["y"] + box["h"])
        host.update(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
    return merged


def _box_iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def fuse_dual(trad: DetectSummary | None, feat: dict[str, Any] | None,
              feat_err: str, mode: str) -> DualSummary:
    gray = False
    overall = "OK"
    if mode in ("dual", "feature") and feat is not None:
        if feat["decision"] == "gray":
            gray = True
    if mode in ("dual", "traditional") and trad is not None:
        # 传统引擎 gate_blocked / ERROR → 整体 ERROR（但特征侧仍可参考）
        if trad.gate_blocked or trad.overall == "ERROR":
            overall = "ERROR"

    if mode == "traditional":
        if trad is not None and (trad.overall == "NG"):
            overall = "NG"
        elif trad is not None and trad.overall == "ERROR":
            overall = "ERROR"
    elif mode == "feature":
        if feat is None:
            overall = "ERROR"
        elif feat["is_anomaly"]:
            overall = "NG"
        elif feat["decision"] == "gray":
            # 灰区不得静默放行：透出 GRAY 供人工复判
            overall = "GRAY"
    else:  # dual
        trad_ng = bool(trad is not None and trad.overall == "NG")
        feat_ng = bool(feat is not None and feat["is_anomaly"])
        if feat is None:
            # 双检承诺两引擎都出结果；特征引擎缺失/失败不能静默判 OK
            overall = "ERROR"
        elif trad_ng or feat_ng:
            overall = "NG"
        elif overall == "ERROR":
            pass  # 传统引擎 gate/执行失败优先，保持 ERROR
        elif gray:
            # 任一引擎灰区且无 NG：整体 GRAY（需复判），不得判 OK
            overall = "GRAY"
        else:
            overall = "OK"

    elapsed = (trad.elapsed_ms if trad else 0) + (feat.get("latency_ms", 0) if feat else 0)
    return DualSummary(
        overall=overall,
        overall_ok=(overall == "OK"),
        engine_mode=mode,
        traditional=trad,
        feature=feat,
        feature_error=feat_err,
        gray=gray,
        boxes=_collect_boxes(trad, feat),
        elapsed_ms=int(elapsed),
        template_id=(trad.template_id if trad else None),
        template_version=(trad.template_version if trad else None),
        source="dual",
    )
