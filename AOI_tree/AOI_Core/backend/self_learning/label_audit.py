"""AC-042 标签时序审计 / AC-044 标签身份（纯函数，不读写 DB、不触发训练）。

- audit_label_order：按检测时间检查反馈标签的释放时序（早于检测=时钟异常、迟到、乱序），
  并把检测分为 新样首次 / 已反馈再检，供分报与学习前过滤。
- LabelRecord + resolve_labels：人工与伪标签分存；人工优先替换伪标签，同源取最新修订；
  伪标签过期或 teacher 版本不符不再继续使用；人工内部冲突（同修订号标签不同）标争议。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional


def audit_label_order(events: Iterable[Dict], tolerance_days: float) -> Dict:
    """events: {"feedback_id","detection_id","image_key","detected_at","label_at"}（datetime）。

    返回每条 {"status": ok|late|before_detection|out_of_order|unknown_time, "seen": first|re_detect}
    与汇总计数。out_of_order：标签释放顺序与检测顺序倒置（后检测者先得标签）。"""
    rows = sorted(list(events), key=lambda e: (e.get("detected_at") or datetime.min,
                                              e.get("feedback_id") or 0))
    seen_keys: set = set()
    max_label_at: Optional[datetime] = None
    out: List[Dict] = []
    for e in rows:
        det_at, lab_at = e.get("detected_at"), e.get("label_at")
        key = e.get("image_key")
        seen = "re_detect" if key is not None and key in seen_keys else "first"
        if key is not None:
            seen_keys.add(key)
        if det_at is None or lab_at is None:
            status, lag = "unknown_time", None
        else:
            lag = (lab_at - det_at).total_seconds() / 86400.0
            if lag < 0:
                status = "before_detection"
            elif lag > tolerance_days:
                status = "late"
            elif max_label_at is not None and lab_at < max_label_at:
                status = "out_of_order"
            else:
                status = "ok"
            if status != "before_detection":
                max_label_at = lab_at if max_label_at is None else max(max_label_at, lab_at)
        out.append({"feedback_id": e.get("feedback_id"), "detection_id": e.get("detection_id"),
                    "status": status, "seen": seen,
                    "lag_days": None if lag is None else round(lag, 3),
                    "trainable": status in ("ok", "out_of_order")})
    counts: Dict[str, int] = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    by_seen = {s: sum(1 for r in out if r["seen"] == s) for s in ("first", "re_detect")}
    return {"items": out, "counts": counts, "by_seen": by_seen,
            "tolerance_days": tolerance_days,
            "note": "before_detection/late/unknown_time 不入训；out_of_order 仅审计不排除"}


@dataclass
class LabelRecord:
    sample_id: str
    label: int
    kind: str                      # "human" | "pseudo"
    revision: int = 0
    created_at: Optional[datetime] = None
    operator: Optional[str] = None
    teacher_version: Optional[str] = None
    predicted_at: Optional[datetime] = None
    selection_reason: Optional[str] = None
    confidence: Optional[float] = None
    valid_until: Optional[datetime] = None
    provenance: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("human", "pseudo"):
            raise ValueError(f"未知标签类型 {self.kind}")
        if self.kind == "pseudo" and not self.teacher_version:
            raise ValueError("伪标签必须记录 teacher_version")


def resolve_labels(records: Iterable[LabelRecord], now: datetime,
                   teacher_version: Optional[str] = None) -> Dict[str, Dict]:
    """按 sample_id 解析生效标签：human > pseudo；human 取最高 revision（同修订不同标签=disputed）；
    pseudo 过期或 teacher_version 与当前不符 → stale，不再使用。"""
    by_id: Dict[str, List[LabelRecord]] = {}
    for r in records:
        by_id.setdefault(r.sample_id, []).append(r)
    out: Dict[str, Dict] = {}
    for sid, rs in by_id.items():
        human = sorted([r for r in rs if r.kind == "human"], key=lambda r: r.revision)
        pseudo = [r for r in rs if r.kind == "pseudo"]
        if human:
            top = human[-1]
            same = {r.label for r in human if r.revision == top.revision}
            if len(same) > 1:
                out[sid] = {"status": "disputed", "label": None, "source": "human",
                            "usable": False, "replaced_pseudo": len(pseudo)}
            else:
                out[sid] = {"status": "active", "label": top.label, "source": "human",
                            "usable": True, "revision": top.revision,
                            "replaced_pseudo": len(pseudo), "record": asdict(top)}
            continue
        fresh = [r for r in pseudo
                 if (r.valid_until is None or r.valid_until > now)
                 and (teacher_version is None or r.teacher_version == teacher_version)]
        if not fresh:
            out[sid] = {"status": "stale", "label": None, "source": "pseudo", "usable": False,
                        "n_stale": len(pseudo)}
            continue
        best = max(fresh, key=lambda r: (r.confidence or 0.0, r.predicted_at or datetime.min))
        out[sid] = {"status": "active", "label": best.label, "source": "pseudo", "usable": True,
                    "record": asdict(best), "n_stale": len(pseudo) - len(fresh)}
    return out


# 保守默认：分数 ≤ 阈值×0.5 → pseudo OK；≥ 阈值×1.5 → pseudo NG；中间不标
PSEUDO_DEFAULTS = {"enabled": True, "ok_ratio": 0.5, "ng_ratio": 1.5,
                   "valid_days": 7, "max_items": 500}


def generate_pseudo_labels(samples: Iterable[Dict], teacher_version, *,
                           ok_ratio: float = PSEUDO_DEFAULTS["ok_ratio"],
                           ng_ratio: float = PSEUDO_DEFAULTS["ng_ratio"],
                           now: Optional[datetime] = None,
                           valid_until: Optional[datetime] = None) -> List[LabelRecord]:
    """teacher 高置信伪标签（只为远离阈值的样本出标签）。

    samples: {"sample_id","score","threshold"}；score ≤ threshold*ok_ratio → 0，
    score ≥ threshold*ng_ratio → 1，中间不确定区不标。threshold 缺失/非正 → 跳过。"""
    if not teacher_version:
        raise ValueError("伪标签必须记录 teacher_version")
    if not (0.0 < ok_ratio < 1.0 < ng_ratio):
        raise ValueError("要求 0 < ok_ratio < 1 < ng_ratio（保守区间）")
    now = now or datetime.now()
    out: List[LabelRecord] = []
    for s in samples:
        score, thr = s.get("score"), s.get("threshold")
        if score is None or thr is None or float(thr) <= 0:
            continue
        score, thr = float(score), float(thr)
        if score <= thr * ok_ratio:
            label, reason = 0, "teacher_far_below_threshold"
        elif score >= thr * ng_ratio:
            label, reason = 1, "teacher_far_above_threshold"
        else:
            continue
        # 置信度：距阈值的相对距离（0..1 截断）
        conf = min(1.0, abs(score - thr) / thr)
        out.append(LabelRecord(
            sample_id=str(s["sample_id"]), label=label, kind="pseudo",
            teacher_version=str(teacher_version), predicted_at=now,
            selection_reason=reason, confidence=round(conf, 4),
            valid_until=valid_until,
            provenance={"score": score, "threshold": thr,
                        "ok_ratio": ok_ratio, "ng_ratio": ng_ratio}))
    return out


def merge_pseudo_with_human(pseudo: Iterable[LabelRecord], human: Iterable[LabelRecord],
                            now: datetime, teacher_version) -> Dict[str, Dict]:
    """伪标签与人工标签合并（人工优先），返回 resolve_labels 结果。"""
    tv = str(teacher_version) if teacher_version is not None else None
    return resolve_labels(list(human) + list(pseudo), now, teacher_version=tv)
