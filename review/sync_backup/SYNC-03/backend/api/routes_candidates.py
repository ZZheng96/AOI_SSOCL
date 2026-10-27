"""候选专用API：工程师验收、管理员决策/发布；不接受预计算指标。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from ..core.security import require_role, current_actor
from ..core.tasks import task_manager
from ..db.database import session_scope
from ..db.models import Image
from ..engine import get_engine
from ..self_learning.candidate_release import CandidateRelease, CandidateReleaseError

router = APIRouter()


class AcceptanceRequest(BaseModel):
    old_image_ids: list[int] = Field(default_factory=list, max_length=64)
    new_image_ids: list[int] = Field(default_factory=list, max_length=64)
    policy: dict[str, float] = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    approved: bool
    report_hash: str
    reason: str = Field(min_length=1, max_length=512)


def release():
    return CandidateRelease(get_engine())


def call(fn):
    try:
        return fn()
    except CandidateReleaseError as exc:
        raise HTTPException(status_code=409,
                            detail={"message": str(exc), "reason_code": exc.reason_code,
                                    "violations": exc.violations,
                                    "retryable": exc.retryable})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/candidates/recovery/prepared", dependencies=[Depends(require_role("engineer"))])
def prepared_releases(category: str | None = None):
    return call(lambda: release().discover_prepared(category))


@router.get("/candidates/{category}/{version}", dependencies=[Depends(require_role("engineer"))])
def status(category: str, version: int):
    return call(lambda: release().status(category, version))


@router.get("/candidates/{category}/recovery/prepared", dependencies=[Depends(require_role("engineer"))])
def prepared_category_releases(category: str):
    return call(lambda: release().discover_prepared(category))


@router.post("/candidates/{category}/{version}/acceptance", dependencies=[Depends(require_role("engineer"))])
def acceptance(category: str, version: int, req: AcceptanceRequest):
    def work(progress_cb):
        domains = {}
        ids = req.old_image_ids + req.new_image_ids
        if len(ids) != len(set(ids)):
            raise ValueError("新旧域图像ID重叠")
        with session_scope() as s:
            rows = s.query(Image).filter(Image.id.in_(ids)).all()
            if len(rows) != len(ids) or any(r.category != category or r.split != "test" or r.label not in ("normal", "anomaly") for r in rows):
                raise ValueError("审计图像须为已登记同品类test有效标注")
            info = {r.id: {"id": r.id, "path": r.path, "label": int(r.label == "anomaly"), "dataset_id": r.dataset_id} for r in rows}
            train = s.query(Image).filter(Image.category == category, Image.split.in_(("train", "train_anomaly", "feedback", "template"))).all()
            if {r.path for r in train} & {r.path for r in rows}:
                raise ValueError("审计资产与训练资产重叠")
            cutoff = max((r.id for r in s.query(Image.id).filter(Image.category == category)), default=0)
        domains["old"] = [info[i] for i in req.old_image_ids]
        domains["new"] = [info[i] for i in req.new_image_ids]
        progress_cb(0, 1, "冻结登记数据和独立验收报告")
        return release().evaluate(category, version, domains, req.policy, cutoff, progress_cb)
    return {"task_id": task_manager.submit("candidate_acceptance", req.model_dump(), work)}


@router.post("/candidates/{category}/{version}/decision")
def decision(category: str, version: int, req: DecisionRequest, actor=Depends(require_role("admin"))):
    return call(lambda: release().decide(category, version, req.approved, req.report_hash, current_actor.get(), req.reason))


@router.post("/candidates/{category}/{version}/activate", dependencies=[Depends(require_role("admin"))])
def activate(category: str, version: int):
    return call(lambda: release().switch(category, version))


@router.post("/candidates/{category}/{version}/rollback", dependencies=[Depends(require_role("admin"))])
def rollback(category: str, version: int):
    return call(lambda: release().switch(category, version, rollback=True))


@router.post("/candidates/{category}/recover", dependencies=[Depends(require_role("admin"))])
def recover(category: str):
    return call(lambda: release().recover(category))
