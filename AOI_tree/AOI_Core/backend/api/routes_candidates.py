"""候选专用API：工程师验收、管理员决策/发布；不接受预计算指标。"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from ..core.security import require_role, current_actor
from ..core.tasks import task_manager
from ..db.database import session_scope, log_action
from ..db.models import Image
from ..engine import get_engine
from ..self_learning.candidate_release import CandidateRelease, CandidateReleaseError
from ..core.external_anchor import export_anchor

router = APIRouter()
logger = logging.getLogger(__name__)


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


def audited(action: str, extra: dict, fn):
    """AC-074：候选写操作结构化审计；成功/409 失败均落 OperationLog，审计失败不掩盖业务结果。"""
    try:
        out = call(fn)
    except HTTPException as exc:
        d = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        _log(action + "_rejected", {**extra, "ok": False,
                                    "reason_code": d.get("reason_code"),
                                    "retryable": d.get("retryable")})
        raise
    res = out if isinstance(out, dict) else {}
    _log(action, {**extra, "ok": True, "state": res.get("state"),
                  "idempotent": bool(res.get("idempotent", False))})
    return out


def _log(action: str, extra: dict):
    try:
        log_action(action, " ".join(f"{k}={extra[k]}" for k in ("category", "version") if k in extra),
                   extra=extra)
    except Exception:
        logger.warning("候选审计日志写入失败 action=%s", action, exc_info=True)


@router.post("/candidates/{category}/{version}/decision")
def decision(category: str, version: int, req: DecisionRequest, actor=Depends(require_role("admin"))):
    out = audited("candidate_decision",
                  {"category": category, "version": version, "approved": req.approved,
                   "report_hash": req.report_hash, "reason": req.reason},
                  lambda: release().decide(category, version, req.approved, req.report_hash,
                                           current_actor.get(), req.reason))
    # 第三轮决策项：候选发布策略 auto_on_approve → 批准成功后自动激活。
    # 批准已提交，自动激活失败（含依赖锁阻断）不回滚批准，只回传失败原因。
    if (req.approved and isinstance(out, dict) and out.get("state") == "approved"
            and _governance()["candidate_release_mode"] == "auto_on_approve"):
        try:
            out = {**out, "auto_activate": {"ok": True, **activate(category, version)}}
        except HTTPException as exc:
            out = {**out, "auto_activate": {"ok": False, "detail": exc.detail}}
    return out


def _governance() -> dict:
    from ..core.config import get_settings
    return get_settings().governance_decisions()


def _guarded_switch(category: str, version: int):
    """第三轮决策项：激活前依赖锁检查（block 阻断 409，warn 附告警）。"""
    from ..core.deplock import dependency_guard
    st = dependency_guard(_governance()["dependency_lock_policy"])
    out = release().switch(category, version)
    if not st["ok"] and isinstance(out, dict):
        out = {**out, "dependency_lock_warning": {
            "mismatches": st["mismatches"], "missing": st["missing"]}}
    return out


@router.post("/candidates/{category}/{version}/activate", dependencies=[Depends(require_role("admin"))])
def activate(category: str, version: int):
    return audited("candidate_activate", {"category": category, "version": version},
                   lambda: _guarded_switch(category, version))


@router.post("/candidates/{category}/{version}/rollback", dependencies=[Depends(require_role("admin"))])
def rollback(category: str, version: int):
    return audited("candidate_rollback", {"category": category, "version": version},
                   lambda: release().switch(category, version, rollback=True))


@router.post("/candidates/{category}/recover", dependencies=[Depends(require_role("admin"))])
def recover(category: str):
    return audited("candidate_recover", {"category": category},
                   lambda: release().recover(category))


@router.post("/candidates/audit/anchor", dependencies=[Depends(require_role("admin"))])
def audit_anchor(body: dict | None = None):
    """导出审计链锚定；没有运维密钥和外部路径时明确返回未启用。"""
    body = body or {}
    category = body.get("category")
    chain = release().audit_chain(category)
    result = export_anchor(chain, body.get("output_path"))
    _log("audit_anchor", {"category": category or "*", "enabled": result["enabled"],
                           "chain_hash": result["chain_hash"]})
    return result
