"""AC-049 配对点估计工具：仅消费离线预测，不读写 DB 或快照。"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


_METRICS = ("auroc", "fpr", "recall", "gray")


def paired_prediction_delta(
    baseline: Sequence[Mapping], candidate: Sequence[Mapping], *,
    seed_a=None, seed_b=None, id_key: str = "image_id",
) -> dict:
    """同 seed、完整同 ID 集的 candidate - baseline 指标差（不按行号配对）。

    label 为 0/1；score（或 fused）用于 AUROC；decision 支持
    normal/anomaly/gray，gray 按 learning_protocol 口径视作异常。
    无 decision 时可用 pred/is_anomaly 计算 FPR/recall，但不能推断 gray。
    gray 是全样本灰区比例；不重新选择阈值，不提供虚构 CI 或显著性。
    path 审计记录可显式使用 id_key="path"。
    """
    out = {"status": "unavailable", "direction": "candidate - baseline",
           "n_pairs": 0, "gaps": [], "metrics": {
               k: {"status": "unavailable", "baseline": None,
                   "candidate": None, "delta": None, "reason": "pairing_unavailable"}
               for k in _METRICS},
           "uncertainty": {"status": "unavailable", "reason": "not_estimated"}}

    def index(rows, side):
        indexed = {}
        for row in rows:
            identity = row.get(id_key)
            if isinstance(identity, bool) or not isinstance(identity, (str, int)) or identity == "":
                out["gaps"].append({"reason": "invalid_sample_id", "side": side})
                continue
            if identity in indexed:
                out["gaps"].append({"reason": "duplicate_sample_id", "side": side,
                                    "sample_id": identity})
            indexed[identity] = row
        return indexed

    if (isinstance(seed_a, bool) or isinstance(seed_b, bool)
            or not isinstance(seed_a, int) or not isinstance(seed_b, int)):
        out["gaps"].append({"reason": "missing_or_invalid_seed"})
    elif seed_a != seed_b:
        out["gaps"].append({"reason": "seed_mismatch"})
    a, b = index(baseline, "baseline"), index(candidate, "candidate")
    missing_a, missing_b = [k for k in b if k not in a], [k for k in a if k not in b]
    if missing_a or missing_b:
        out["gaps"].append({"reason": "sample_set_mismatch",
                            "missing_baseline": missing_a, "missing_candidate": missing_b})
    if not a or not b:
        out["gaps"].append({"reason": "no_samples"})
    if out["gaps"]:
        return out
    ids = list(a)
    labels = [a[k].get("label") for k in ids]
    for k, label in zip(ids, labels):
        if label not in (0, 1) or b[k].get("label") not in (0, 1):
            out["gaps"].append({"reason": "invalid_label", "sample_id": k})
        elif label != b[k]["label"]:
            out["gaps"].append({"reason": "label_mismatch", "sample_id": k})
    if out["gaps"]:
        return out
    out["n_pairs"] = len(ids)

    def metrics(rows):
        result = {}
        scores = [rows[k].get("score", rows[k].get("fused")) for k in ids]
        finite_scores = all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores)
        if not finite_scores:
            result["auroc"] = (None, "missing_or_nonfinite_score")
        elif len(set(labels)) < 2:
            result["auroc"] = (None, "requires_both_classes")
        else:
            from sklearn.metrics import roc_auc_score
            result["auroc"] = (float(roc_auc_score(labels, scores)), None)
        decisions = [rows[k].get("decision") for k in ids]
        preds = []
        for k, decision in zip(ids, decisions):
            if decision in ("normal", "anomaly", "gray"):
                preds.append(int(decision != "normal"))
            elif decision is None:
                pred = rows[k].get("pred", rows[k].get("is_anomaly"))
                preds.append(pred if pred in (0, 1) else None)
            else:
                preds.append(None)
        for name, label in (("fpr", 0), ("recall", 1)):
            selected = [p for p, y in zip(preds, labels) if y == label]
            if not selected:
                result[name] = (None, "requires_normal_samples" if label == 0 else "requires_defect_samples")
            elif any(p is None for p in selected):
                result[name] = (None, "missing_or_invalid_prediction")
            else:
                result[name] = (float(sum(selected) / len(selected)), None)
        if all(d in ("normal", "anomaly", "gray") for d in decisions):
            result["gray"] = (sum(d == "gray" for d in decisions) / len(ids), None)
        else:
            result["gray"] = (None, "missing_or_invalid_decision")
        return result

    ma, mb = metrics(a), metrics(b)
    for key in _METRICS:
        va, ra = ma[key]
        vb, rb = mb[key]
        available = va is not None and vb is not None
        out["metrics"][key] = {"status": "available" if available else "unavailable",
                                "baseline": va, "candidate": vb,
                                "delta": vb - va if available else None}
        if not available:
            reasons = {side: reason for side, reason in (("baseline", ra), ("candidate", rb)) if reason}
            out["metrics"][key]["reason"] = reasons
            out["gaps"].append({"metric": key, "reason": reasons})
    count = sum(m["status"] == "available" for m in out["metrics"].values())
    out["status"] = "available" if count == len(_METRICS) else "partial" if count else "unavailable"
    return out
