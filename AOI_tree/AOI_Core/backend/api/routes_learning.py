"""学习曲线 / 槽位贡献档案接口（M2b：检测引擎评估能力对外暴露）。

学习曲线为**离线回放**（engine.simulate_learning 内部加载快照副本），
不影响线上激活模型；结果写 EvalRun（run_type=learning_curve/contribution），
latest 端点供 M3 UI 画曲线。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.tasks import task_manager
from ..db.database import session_scope
from ..db.models import EvalRun
from ..engine import get_engine
from ..evaluation import service as eval_service
from .schemas import ContributionRequest, LearningCurveRequest, to_dict

router = APIRouter()


def _latest_run_dict(category: str, run_type: str) -> dict:
    with session_scope() as s:
        row = (s.query(EvalRun)
               .filter(EvalRun.category == category,
                       EvalRun.run_type == run_type)
               .order_by(EvalRun.id.desc()).first())
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"品类 {category} 暂无 {run_type} 评估记录")
        return to_dict(row)


@router.post("/learning/curve")
def api_learning_curve(req: LearningCurveRequest):
    """后台任务：学习曲线离线回放。

    说明：回放在独立 pipe 副本上进行（加载当前激活快照的拷贝），
    反馈即学只作用于副本，**不影响线上激活模型**。
    """
    def fn(progress_cb):
        return eval_service.evaluate_learning_curve(
            req.category, n_stream=req.n_stream, n_eval=req.n_eval,
            feedback_ratio=req.feedback_ratio, eval_every=req.eval_every,
            max_feedback=req.max_feedback, mode=req.mode,
            round_size=req.round_size, n_rounds=req.n_rounds,
            eval_per_sample=req.eval_per_sample, progress_cb=progress_cb)

    task_id = task_manager.submit("learning_curve", req.model_dump(), fn)
    return {"task_id": task_id,
            "description": "离线回放（独立 pipe 副本），与产线并行，无需暂停产线"}


@router.get("/learning/curve/{category}/latest")
def api_learning_curve_latest(category: str):
    """最近一次学习曲线 EvalRun。

    2026-08-29 走查修复：补全 algo simulate_learning 早已返回但被端点丢弃的
    学习证据——A 再检（"学过就会"）、缺陷样例库拦截统计、学习杠杆留痕
    （权重更新/阈值重估/判别头微调/鉴别器微调）。"""
    row = _latest_run_dict(category, "learning_curve")
    m = row["metrics"] or {}
    learned = m.get("learned") or {}
    return {
        "eval_run_id": row["id"],
        "category": category,
        "created_at": row["created_at"],
        "dataset_desc": row["dataset_desc"],
        "mode": m.get("mode", "stream"),
        "steps": m.get("steps", []),
        "auroc": m.get("auroc", []),
        "f1": m.get("f1", []),
        # 多轮模式：轮次轴 + 已反馈重测指标（学过就会）
        "rounds": m.get("rounds", []),
        "round_size": m.get("round_size"),
        "n_rounds": m.get("n_rounds"),
        "learned_accs": learned.get("learned_accs", []),
        "flip_rates": learned.get("flip_rates", []),
        "defect_recalls": learned.get("defect_recalls", []),
        "n_fed": learned.get("n_fed", []),
        "final_learned_acc": learned.get("final_learned_acc"),
        "final_flip_rate": learned.get("final_flip_rate"),
        "final_defect_recall": learned.get("final_defect_recall"),
        "initial_auroc": m.get("initial_auroc"),
        "final_auroc": m.get("final_auroc"),
        "auroc_gain": m.get("auroc_gain"),
        "efficiency": m.get("efficiency"),
        "n_feedback": m.get("n_feedback"),
        # ── 学习证据（algo 实际返回，此前被丢弃）──
        "a_recheck_auroc": m.get("a_recheck_auroc"),
        "a_recheck_n_defect": m.get("a_recheck_n_defect"),
        "a_recheck_n_normal": m.get("a_recheck_n_normal"),
        "box_supervised": m.get("box_supervised"),
        "intercept": m.get("intercept"),
        "weight_learn": m.get("weight_learn"),
        "thresh_recal": m.get("thresh_recal"),
        "head_ft": m.get("head_ft"),
        "disc_ft": m.get("disc_ft"),
        # 末轮锚定集混淆推导指标（2026-08-30，algo 复用末次评估零额外前向）
        "final_metrics": m.get("final_metrics"),
    }


@router.get("/learning/weight_history/{category}")
def api_weight_history(category: str):
    """融合权重演化历史（学习效果页「权重随反馈演化」曲线）。

    algo 侧 FeedbackHandler 在权重写回 apply / 门控否决 reject 时落点
    （initial 点为首条带真值反馈时的学习前权重），随快照持久化。
    纯内存读取，不跑模型；品类未准备返回 enabled=False。"""
    return get_engine().weight_history(category)


@router.post("/learning/contribution")
def api_learning_contribution(req: ContributionRequest):
    """后台任务：槽位贡献档案（单变量/冗余度/消融三层评价）。"""
    def fn(progress_cb):
        return eval_service.evaluate_contribution(req.category,
                                                  progress_cb=progress_cb)

    task_id = task_manager.submit("contribution", req.model_dump(), fn)
    return {"task_id": task_id}


@router.get("/learning/contribution/{category}/latest")
def api_learning_contribution_latest(category: str):
    """最近一次贡献档案 EvalRun（含 ablation.leave_one_out 每槽位 AUROC 等）。"""
    row = _latest_run_dict(category, "contribution")
    m = row["metrics"] or {}
    return {
        "eval_run_id": row["id"],
        "category": category,
        "created_at": row["created_at"],
        "dataset_desc": row["dataset_desc"],
        "univariate": m.get("univariate"),
        "redundancy": m.get("redundancy"),
        "ablation": m.get("ablation"),
    }


# ── M5a：在线学习曲线（真实反馈流，纯 DB 计算）────────────────
@router.get("/learning/online_curve/{category}")
def api_online_curve(category: str, window: int = 10):
    """本机真实反馈以来的判对率曲线（区别于 /learning/curve 的离线回放）。

    横轴=反馈序号 k，纵轴=滚动窗口判对率；附累计反馈统计与当前融合权重快照。
    判对映射：false_positive/false_negative/new_defect→错；confirmed→对；
    review→人工 label 与系统二分类一致为对。纯 DB 计算，不跑模型。
    """
    from ..self_learning import service as sl_service
    return sl_service.online_learning_curve(category, window=window)


# ── M7a：翻案曲线（反馈前判定 pre vs 当前引擎重打分 post）──────
@router.get("/learning/flip_curve/{category}")
def api_flip_curve(category: str):
    """翻案曲线：被反馈图的"学习前判定 → 学习后判定"对比。

    取该品类有 pre 记录的反馈按时间升序，逐条用**当前引擎**重打分 → post；
    flipped = post 与人工 label 一致 且 pre 与人工不一致（即反馈即学把
    该图"翻案"）。有 pre 的反馈数 >50 时转后台任务（返回 task_id），
    否则同步返回。引擎未加载该品类 → 409。
    """
    from ..self_learning import service as sl_service
    engine = get_engine()
    if engine.current_version(category) is None:
        raise HTTPException(status_code=409,
                            detail=f"品类 '{category}' 未准备模型，无法重打分")

    n = sl_service.flip_curve_feedback_count(category)
    if n > sl_service.FLIP_CURVE_SYNC_MAX:
        def fn(progress_cb):
            return sl_service.flip_curve(category, progress_cb=progress_cb)

        task_id = task_manager.submit("flip_curve", {"category": category}, fn)
        return {"task_id": task_id, "n_feedback": n,
                "description": "反馈数较多，转后台任务逐条重打分"}
    try:
        return sl_service.flip_curve(category)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
