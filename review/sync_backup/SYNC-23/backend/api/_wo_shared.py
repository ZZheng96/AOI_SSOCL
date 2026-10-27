"""工单制接口共享层：常量 + 内部辅助函数（自 routes_workorder 拆分，A18）。

仅供 routes_wo_datasource / routes_wo_order / routes_wo_stats 三个
子路由模块 import，本身不注册任何路由，避免子模块间循环依赖。
"""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import Optional

from sqlalchemy import func

from ..core.groups import compute_groups
from ..db.models import (DataSource, Dataset, Detection, Feedback,
                         Image as ImageRow, Video, WorkOrder, WorkOrderSource)

# 标注档位（递进）与开关（可叠加）的中文名 -- UI 不露 L 代号
TIER_CN = {"L0": "仅正常图", "L1a": "图像级标注", "L1b": "缺陷位置标注"}
TIER_RANK = {"L0": 0, "L1a": 1, "L1b": 2}
GENERIC_CATEGORY = "通用"   # 无品类说明的数据兜底品类


def _int_or(default: int, v, lo: int = 0) -> int:
    """配置值解析：None → 默认；否则 clamp 下界 lo。0 是合法值
    （0 正常图 + 0 异常图 = 全部进检测组，不进预训练组）。"""
    if v is None:
        return default
    return max(int(v), lo)


def condition_text(cond) -> str:
    """数据条件 -> 中文串（三档 + 两开关，不露代号；吃 dict 或 ORM）。"""
    tier = cond.get("label_tier") if isinstance(cond, dict) \
        else getattr(cond, "label_tier", None)
    per = bool(cond.get("per_category")) if isinstance(cond, dict) \
        else bool(getattr(cond, "per_category", False))
    tpl = bool(cond.get("has_template")) if isinstance(cond, dict) \
        else bool(getattr(cond, "has_template", False))
    parts = [TIER_CN.get(str(tier or ""), str(tier or ""))]
    if per:
        parts.append("品类独立")
    if tpl:
        parts.append("模板比对")
    return "＋".join(p for p in parts if p)


def _workorder_sources(s, workorder_id: int) -> list:
    """工单挂接的数据源列表。"""
    return (s.query(DataSource)
            .join(WorkOrderSource, WorkOrderSource.datasource_id == DataSource.id)
            .filter(WorkOrderSource.workorder_id == workorder_id)
            .order_by(DataSource.id).all())


def _aggregate_conditions(s, sources: list) -> dict:
    """工单数据条件 = 所挂数据源属性自动聚合（前端反馈 v4：工单不再手填）：
    - 标注档位取源声明最高档；品类并集（有品类即「品类独立」）；
    - 模板比对（任一源有模板图）。"""
    tiers = [str(src.label_tier or "") for src in sources
             if str(src.label_tier or "") in TIER_RANK]
    label_tier = max(tiers, key=lambda t: TIER_RANK[t]) if tiers else "L0"
    cats: list = []
    has_template = False
    for src in sources:
        cap = _source_capability(s, src.id)
        cats.extend(cap.get("categories") or [])
        if (cap.get("n_templates") or 0) > 0:
            has_template = True
    cats = sorted(set(cats))
    # 前端反馈 v6：品类独立/模板比对 = 数据源声明 或 数据实际支撑
    per_category = bool(cats) or any(bool(src.per_category) for src in sources)
    has_template = has_template or any(bool(src.has_template) for src in sources)
    return {"label_tier": label_tier, "per_category": per_category,
            "has_template": has_template, "categories": cats}


def _wo_conditions(wo) -> dict:
    # 兼容旧调用：ORM 工单的三档两开关（已废弃，聚合取代）
    return {"label_tier": wo.label_tier, "per_category": bool(wo.per_category),
            "has_template": bool(wo.has_template)}


def check_datasource(s, src: DataSource) -> dict:
    """数据源体检：声明（档位/分品类/模板）vs 数据实际支撑。"""
    cap = _source_capability(s, src.id)
    declared = {"label_tier": src.label_tier,
                "per_category": bool(src.per_category),
                "has_template": bool(src.has_template)}
    sugg_tier = cap["tier"] or "L0"
    matched = TIER_RANK.get(src.label_tier, 0) <= TIER_RANK.get(sugg_tier, -1)
    warnings = []
    if not matched:
        warnings.append(
            f"「{src.name}」声明档位「{TIER_CN.get(src.label_tier, src.label_tier)}」"
            f"高于数据实际支撑，建议调整为「{TIER_CN.get(sugg_tier, sugg_tier)}」")
    if src.has_template and (cap.get("n_templates") or 0) == 0:
        warnings.append(
            f"「{src.name}」声明提供模板图，但尚无模板图：可把模板图放入 "
            f"template 子目录或按命名规则 A_tpl.png 重新导入")
    if src.per_category and not (cap.get("categories") or []):
        warnings.append(
            f"「{src.name}」声明已分品类，但数据尚无品类说明（将归入「通用」）；"
            "按 {数据源根}/{品类}/ 结构导入即可自动分品类")
    return {"capability": cap, "declared": declared,
            "suggestion": {"label_tier": sugg_tier,
                           "per_category": bool(cap.get("categories")),
                           "has_template": (cap.get("n_templates") or 0) > 0},
            "warnings": warnings, "matched": matched}


def _source_capability(s, datasource_id: int) -> dict:
    """单数据源数据支撑能力（体检原料，纯 DB 统计）。

    - tier：L1b（有掩码+缺陷标注）/ L1a（有缺陷标注）/ L0（有正常图）/ None（空）
    - categories：数据源内品类集合（仅"通用"= 无品类说明）
    - n_templates：模板图数（split=template）
    """
    ds_ids = [r[0] for r in s.query(Dataset.id).filter(
        Dataset.datasource_id == datasource_id).all()]
    cap = {"n_normal": 0, "n_anomaly": 0, "n_masks": 0, "n_images": 0,
           "n_videos": 0, "categories": [], "n_templates": 0, "tier": None}
    if not ds_ids:
        return cap
    rows = (s.query(ImageRow.label, func.count(ImageRow.id))
            .filter(ImageRow.dataset_id.in_(ds_ids)).group_by(ImageRow.label).all())
    for lbl, n in rows:
        if lbl == "normal":
            cap["n_normal"] = int(n)
        elif lbl == "anomaly":
            cap["n_anomaly"] = int(n)
    cap["n_images"] = cap["n_normal"] + cap["n_anomaly"] + sum(
        n for lbl, n in rows if lbl not in ("normal", "anomaly"))
    for (params,) in s.query(Dataset.params).filter(
            Dataset.id.in_(ds_ids)).all():
        cap["n_masks"] += int((params or {}).get("n_masks", 0) or 0)
    cats = [r[0] for r in s.query(ImageRow.category).filter(
        ImageRow.dataset_id.in_(ds_ids)).distinct().all()]
    cap["categories"] = sorted(cats)
    cap["n_templates"] = (s.query(func.count(ImageRow.id))
                          .filter(ImageRow.dataset_id.in_(ds_ids),
                                  ImageRow.split == "template").scalar() or 0)
    cap["n_videos"] = (s.query(func.count(Video.id))
                       .filter(Video.dataset_id.in_(ds_ids)).scalar() or 0)
    if cap["n_masks"] > 0 and cap["n_anomaly"] > 0:
        cap["tier"] = "L1b"
    elif cap["n_anomaly"] > 0:
        cap["tier"] = "L1a"
    elif cap["n_normal"] > 0:
        cap["tier"] = "L0"
    return cap


def _attach_source(s, workorder_id: int, datasource_id: int) -> None:
    exists = (s.query(WorkOrderSource)
              .filter(WorkOrderSource.workorder_id == workorder_id,
                      WorkOrderSource.datasource_id == datasource_id).first())
    if exists is None:
        s.add(WorkOrderSource(workorder_id=workorder_id,
                              datasource_id=datasource_id))


def archived_workorder_ids(s) -> list[int]:
    """已存档工单 id 列表（2026-10-09 存档功能）。

    全局统计/复核/反馈口径用它排除存档工单数据。注意 SQL 侧排除条件
    须写 `workorder_id.is_(None) | ~workorder_id.in_(ids)`——
    单用 `~in_` 会把 workorder_id 为 NULL 的孤儿记录一并排除。
    """
    return [r[0] for r in s.query(WorkOrder.id)
            .filter(WorkOrder.archived.is_(True)).all()]


def _n_undetected_by_ids(s, image_ids: list) -> int:
    """图片 id 列表内尚无任何检测记录的图数（普通检测批次的积压口径）。"""
    if not image_ids:
        return 0
    det_exists = (s.query(Detection.id)
                  .filter(Detection.image_id == ImageRow.id).exists())
    return int(s.query(func.count(ImageRow.id))
               .filter(ImageRow.id.in_(image_ids), ~det_exists).scalar() or 0)


def _n_pending_requeue_by_ids(s, image_ids: list, t_req) -> int:
    """回队检测批次待重检图数：无检测 或 最近一次检测早于回队时刻。

    与 pipeline_service._consume_one 的挑选口径一致。
    """
    if not image_ids:
        return 0
    latest = dict(s.query(Detection.image_id, func.max(Detection.created_at))
                  .filter(Detection.image_id.in_(image_ids))
                  .group_by(Detection.image_id).all())
    n = 0
    for img_id in image_ids:
        t = latest.get(img_id)
        if t is None or (t_req is not None and t < t_req):
            n += 1
    return n


def _batch_stats_by_ids(s, image_ids: list) -> dict:
    """检测批次统计（按图片 id 列表，而非导入批次 Dataset）。"""
    if not image_ids:
        return {"n_images": 0, "n_detected": 0, "n_anomaly": 0,
                "n_feedback": 0, "bad_rate": 0.0}
    dets = (s.query(Detection)
            .filter(Detection.image_id.in_(image_ids)).all())
    det_ids = [d.id for d in dets]
    n_fb = 0
    if det_ids:
        n_fb = int(s.query(func.count(Feedback.id))
                   .filter(Feedback.detection_id.in_(det_ids)).scalar() or 0)
    n_an = sum(1 for d in dets if d.is_anomaly)
    n_det = len(dets)
    return {"n_images": len(image_ids), "n_detected": n_det,
            "n_anomaly": n_an, "n_feedback": n_fb,
            "bad_rate": round(n_an / n_det, 3) if n_det else 0.0}


def _workorder_detect_batches(s, wo: WorkOrder) -> list:
    """工单全部检测批次（30 图/批，来自各数据源 compute_groups）。

    返回 [{batch_key, datasource_id, source, category, batch_index, image_ids}]。
    batch_key = "{datasource_id}:{category}:{batch_index}"（1 起始），
    与 compute_groups 的 detect_batches 顺序一致。
    """
    out: list = []
    for src in _workorder_sources(s, wo.id):
        pj = src.plan_json or {}
        groups = compute_groups(
            s, src.id, _int_or(100, src.pretrain_normal),
            _int_or(30, src.pretrain_anomaly), _int_or(30, src.batch_size, lo=1),
            pj.get("pretrain_ids"))
        for cat, g in (groups or {}).items():
            for bi, ids in enumerate(g.get("detect_batches") or [], 1):
                out.append({"batch_key": f"{src.id}:{cat}:{bi}",
                            "datasource_id": src.id, "source": src.name,
                            "category": cat, "batch_index": bi,
                            "image_ids": list(ids)})
    return out


def _pipeline_summary(s, wo: WorkOrder) -> dict:
    """产线空转体检（UI 走查 2026-08-29）：待检测图数 + 未准备品类。

    监控页用它提示「产线运行中但无可检数据」并引导去准备剩余品类：
    - n_pending：已准备品类批次的可消费积压（普通=未检测；回队=待重检）；
    - unprepared_categories：挂接数据源下无引擎快照的品类（整批跳过送检）；
    - n_images_blocked：未准备品类名下的图数。
    仅用于 GET /workorders/{id}（详情），列表接口不加（避免逐工单读快照）。
    """
    from ..engine import get_engine
    engine = get_engine()
    out = {"n_pending": 0, "unprepared_categories": [], "n_images_blocked": 0}
    batches = _workorder_detect_batches(s, wo)
    if not batches:
        return out
    unprepared = sorted({b["category"] for b in batches
                         if b["category"]
                         and engine.current_version(b["category"]) is None})
    out["unprepared_categories"] = unprepared
    qcfg = wo.queue_json or {}
    requeued_ids = {k: list(v) for k, v in
                    (qcfg.get("requeued_ids") or {}).items()}
    requeued_at = {k: v for k, v in (qcfg.get("requeued_at") or {}).items()}
    n_pending = 0
    for b in batches:
        if not b["category"]:
            continue
        if b["category"] in unprepared:
            out["n_images_blocked"] += len(b["image_ids"])
            continue
        key = b["batch_key"]
        if key in requeued_ids:
            try:
                t_req = datetime.fromisoformat(str(requeued_at.get(key)))
            except (TypeError, ValueError):
                t_req = None
            n_pending += _n_pending_requeue_by_ids(b["image_ids"], t_req)
        else:
            n_pending += _n_undetected_by_ids(s, b["image_ids"])
    out["n_pending"] = n_pending
    return out


def _n_pending_review(s, workorder_id: int) -> int:
    """该工单名下待复核检测数（产线路径 + 无有效反馈）。

    与 pipeline_service._apply_review_backpressure 口径一致，
    供监控页实时显示复判积压。
    """
    fb_ids = {r[0] for r in s.query(Feedback.detection_id)
              .filter(Feedback.invalidated == False).all()}  # noqa: E712
    rows = (s.query(Detection.id)
            .filter(Detection.workorder_id == workorder_id,
                    Detection.target_type.in_(("pipeline", "stream", "plc")))
            .all())
    return sum(1 for (did,) in rows if did not in fb_ids)


def _workorder_item(s, wo: WorkOrder, range_days: Optional[int] = None) -> dict:
    """单工单聚合 dict（挂接源/聚合条件/品类/统计/体检）。
    range_days=None 全量，0 今日，7 近7天。"""
    sources = _workorder_sources(s, wo.id)
    cats = []
    n_images = 0
    for src in sources:
        cap = _source_capability(s, src.id)
        cats.extend(cap["categories"])
        n_images += cap["n_images"]
    cats = sorted(set(cats))
    # 数据条件 = 数据源属性自动聚合（前端反馈 v4，工单不再手填）
    conditions = _aggregate_conditions(s, sources)
    det_q = s.query(func.count(Detection.id))
    an_q = (s.query(func.count(Detection.id))
            .filter(Detection.is_anomaly.is_(True)))
    if wo.review_enabled:
        # 人工复判：统计只认「已复核」的检测（复核提交自动生成 feedback）
        det_q = det_q.filter(
            s.query(Feedback.id).filter(
                Feedback.detection_id == Detection.id).exists())
        an_q = an_q.filter(
            s.query(Feedback.id).filter(
                Feedback.detection_id == Detection.id).exists())
    if cats:
        det_q = det_q.filter(Detection.category.in_(cats))
        an_q = an_q.filter(Detection.category.in_(cats))
    else:
        det_q = det_q.filter(Detection.id < 0)   # 无品类工单恒为 0
        an_q = an_q.filter(Detection.id < 0)
    if range_days == 0:
        day_start = datetime.combine(date.today(), dtime.min)
        det_q = det_q.filter(Detection.created_at >= day_start)
        an_q = an_q.filter(Detection.created_at >= day_start)
    elif range_days:
        since = datetime.combine(date.today() - timedelta(days=range_days - 1),
                                 dtime.min)
        det_q = det_q.filter(Detection.created_at >= since)
        an_q = an_q.filter(Detection.created_at >= since)
    n_ins = int(det_q.scalar() or 0)
    n_an = int(an_q.scalar() or 0)
    check = check_workorder(s, wo)
    return {
        "id": wo.id, "name": wo.name, "note": wo.note, "status": wo.status,
        "archived": bool(getattr(wo, "archived", False)),
        "review_enabled": bool(wo.review_enabled),
        "pipeline_status": wo.pipeline_status,
        "auto_resume": bool(wo.auto_resume),
        "conditions": conditions,
        "conditions_cn": condition_text(conditions),
        "datasources": [{"id": x.id, "name": x.name, "modality": x.modality}
                        for x in sources],
        "categories": cats, "n_images": n_images,
        "stats": {"inspected": n_ins, "anomaly": n_an,
                  "rate": round(n_an / n_ins, 4) if n_ins else 0.0},
        "check": check,
        "template_id": wo.template_id,
        "created_at": wo.created_at,
    }


def check_workorder(s, wo: WorkOrder) -> dict:
    """体检：工单聚合条件 vs 数据实际支撑（v4：条件由数据源属性聚合）。

    declared = 数据源聚合条件；capability = 全源支撑合并；
    suggestion = 按支撑建议；warnings 含「数据源层体检」警告。
    """
    sources = _workorder_sources(s, wo.id)
    cap = {"n_normal": 0, "n_anomaly": 0, "n_masks": 0, "n_images": 0,
           "n_videos": 0, "categories": [], "n_templates": 0, "tier": None}
    for src in sources:
        c = _source_capability(s, src.id)
        for k in ("n_normal", "n_anomaly", "n_masks", "n_images",
                  "n_videos", "n_templates"):
            cap[k] += c[k]
        cap["categories"] = sorted(set(cap["categories"]) | set(c["categories"]))
    if cap["n_masks"] > 0 and cap["n_anomaly"] > 0:
        cap["tier"] = "L1b"
    elif cap["n_anomaly"] > 0:
        cap["tier"] = "L1a"
    elif cap["n_normal"] > 0:
        cap["tier"] = "L0"

    cats = cap["categories"]
    no_category_info = (not cats) or cats == [GENERIC_CATEGORY]
    sup_pc = not no_category_info          # 数据已分品类
    sup_tpl = cap["n_templates"] > 0       # 有模板图
    declared = _aggregate_conditions(s, sources)
    warnings: list[str] = []
    if not sources:
        warnings.append("尚未挂接数据源：可导入新数据源或选择已有数据源")
    for src in sources:
        dchk = check_datasource(s, src)
        warnings.extend(dchk["warnings"])
    if TIER_RANK.get(declared["label_tier"], 0) > TIER_RANK.get(
            cap["tier"] or "", -1):
        warnings.append(
            f"聚合条件「{TIER_CN.get(declared['label_tier'], declared['label_tier'])}」"
            f"超出数据支撑（当前支撑：{TIER_CN.get(cap['tier'], '空')}）；"
            "可在数据源体检中自动校正")
    if declared["has_template"] and not sup_tpl:
        warnings.append("模板比对：数据源声明了模板能力但尚无模板图，"
                        "可把模板图放入 template 子目录重新导入")
    suggestion = {
        "label_tier": cap["tier"] or "L0",
        "per_category": sup_pc,
        "has_template": sup_tpl,
    }
    matched = not warnings
    return {"capability": cap, "declared": declared,
            "suggestion": suggestion, "warnings": warnings,
            "matched": matched}
