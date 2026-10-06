"""不良过滤规则应用（P2：借鉴 Java AOI ng_filter「全局库+模板绑定」）。

检测落库后调用：命中规则的缺陷明细被"拦截"（从 defect_records 剔除，
原始判定仍保留在 Detection.dual_json 供追溯）。
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import DefectRecord, NgFilter


def _content_value(defect: DefectRecord, content: str) -> Optional[float]:
    if content == "area":
        return (defect.width or 0) * (defect.height or 0)
    if content == "width":
        return defect.width
    if content == "height":
        return defect.height
    if content == "score":
        return defect.score
    if content == "x":
        return defect.x
    if content == "y":
        return defect.y
    return None


def _match(cond: str, value: Optional[float], vmin: Optional[float],
           vmax: Optional[float]) -> bool:
    if value is None:
        return False
    if cond == "GT":
        return vmin is not None and value > vmin
    if cond == "LT":
        return vmax is not None and value < vmax
    if cond == "EQ":
        return vmin is not None and abs(value - vmin) < 1e-6
    if cond == "BETWEEN":
        return vmin is not None and vmax is not None and vmin <= value <= vmax
    return False


def _defect_matches_rule(defect: DefectRecord, rule: NgFilter) -> bool:
    """缺陷类型匹配：规则 defect_type 支持子串包含。"""
    dt = (defect.defect_type or "").lower()
    rt = (rule.defect_type or "").lower()
    if rt in dt:
        val = _content_value(defect, rule.filter_content or "area")
        return _match(rule.filter_condition, val, rule.value_min, rule.value_max)
    return False


def apply_ng_filters(session: Session, inspect_id: Optional[int],
                     template_id: Optional[str]) -> int:
    """对指定板/模板的缺陷明细应用过滤规则，返回拦截条数。"""
    q = session.query(NgFilter).filter(NgFilter.enabled.is_(True))
    rules = q.all()
    if not rules or inspect_id is None:
        return 0
    defects = (session.query(DefectRecord)
               .filter(DefectRecord.inspect_id == inspect_id).all())
    filtered = 0
    for d in defects:
        for r in rules:
            if r.template_ref and template_id and r.template_ref != template_id:
                continue  # 模板绑定规则只对该模板生效
            if _defect_matches_rule(d, r):
                session.delete(d)
                filtered += 1
                break
    return filtered
