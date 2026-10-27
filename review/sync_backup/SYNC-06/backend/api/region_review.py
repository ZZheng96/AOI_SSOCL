"""AC-064 区域级人工复核：确认/拒绝/新增/修正候选框，修订链 + 事件幂等 + 乐观锁。

候选框（engine_box/local_branch）默认 status=candidate，不进入框级训练；
仅 confirmed（含人工 human_box）可被 trainable_region_boxes 取用。
"""
from typing import List, Literal, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from algo.decision.localization import apply_op, new_review

from ..core.security import require_role
from ..db.database import session_scope
from ..db.models import Detection


class RegionOp(BaseModel):
    op: Literal["confirm", "reject", "add", "revise"]
    region_id: Optional[str] = Field(default=None, max_length=32)
    bbox: Optional[List[float]] = None
    operator: str = Field(min_length=1, max_length=64)
    event_id: str = Field(min_length=1, max_length=100)
    expected_revision: int = Field(ge=0)
    reason: Optional[str] = Field(default=None, max_length=2000)


def review_of(det):
    nt = det.n_tiles or {}
    loc = nt.get("localization") or {}
    rv = nt.get("region_review")
    if rv is None:
        pp = loc.get("postproc")
        rv = new_review(loc.get("regions") or [],
                        pp.get("version_id") if isinstance(pp, dict) else pp)
    return rv, (loc.get("coord_space") or {}).get("size")


def register(router):
    @router.get("/detections/{detection_id}/regions")
    def get_regions(detection_id: int):
        with session_scope() as s:
            det = s.get(Detection, detection_id)
            if det is None:
                raise HTTPException(404, "检测记录不存在")
            rv, size = review_of(det)
            return {"detection_id": detection_id, "coord_space": size, **rv}

    @router.post("/detections/{detection_id}/regions",
                 dependencies=[Depends(require_role("engineer"))])
    def post_region_op(detection_id: int, req: RegionOp):
        if not req.operator.strip() or not req.event_id.strip():
            raise HTTPException(422, "操作人和事件ID不能为空")
        op = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        payload = {k: op.get(k) for k in ("op", "region_id", "bbox", "operator",
                                          "expected_revision", "reason")}
        with session_scope() as s:
            det = s.get(Detection, detection_id)
            if det is None:
                raise HTTPException(404, "检测记录不存在")
            old = det.n_tiles or {}
            rv, size = review_of(det)
            for ev in rv.get("events", []):
                if ev["event_id"] == req.event_id:
                    if ev["payload"] != payload:
                        raise HTTPException(409, "事件ID已对应其他操作")
                    return {"review": rv, "event": ev, "idempotent": True}
            if int(rv.get("revision", 0)) != req.expected_revision:
                raise HTTPException(409, "区域复核已被修改，请刷新后重试")
            try:
                new = apply_op(rv, op, size=size)
            except ValueError as e:
                raise HTTPException(422, str(e))
            except KeyError:
                raise HTTPException(404, f"区域不存在: {req.region_id}")
            updated = {**old, "region_review": new}
            changed = (s.query(Detection).filter(Detection.id == detection_id,
                       Detection.n_tiles == old)
                       .update({Detection.n_tiles: updated}, synchronize_session=False))
            if changed != 1:
                raise HTTPException(409, "并发操作冲突，请刷新后重试")
            return {"review": new, "event": new["events"][-1], "idempotent": False}
