"""系统接口：健康检查 / 系统信息 / 本地文件读取（开发用）/ 系统自检。"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import func

from ..core.config import get_settings
from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import (ConsolidationBatch, ConsolidationFeedback, DataSource,
                         Dataset, Detection, Feedback, Image, StatsDaily,
                         Video, WorkOrder, WorkOrderSource)
from .schemas import device_info

router = APIRouter()


@router.get("/health")
def api_health():
    """健康检查（仅探测设备，不加载模型）。"""
    info = device_info()
    return {"status": "ok", "device": info["device"],
            "gpu_name": info["gpu_name"]}


@router.get("/system/info")
def api_system_info(request: Request):
    """系统概览：设备 / 数据规模 / 可用品类 / 延迟预算 / 当前角色（M15c）。"""
    info = device_info()
    with session_scope() as s:
        n_images = s.query(func.count(Image.id)).scalar() or 0
        n_videos = s.query(func.count(Video.id)).scalar() or 0
        n_detections = s.query(func.count(Detection.id)).scalar() or 0
        n_feedback_pending = (s.query(func.count(Feedback.id))
                              .filter(Feedback.consumed == False,  # noqa: E712
                                      Feedback.invalidated == False)  # noqa: E712
                              .scalar() or 0)
    return {
        "device": info["device"],
        "gpu_name": info["gpu_name"],
        "torch_version": info["torch_version"],
        "n_images": n_images,
        "n_videos": n_videos,
        "n_detections": n_detections,
        "n_feedback_pending": n_feedback_pending,
        "categories": _all_categories(),
        "latency_budget_ms": int(get_settings().get("pipeline",
                                                    "latency_budget_ms", 200)),
        # M15c：当前 API Key 对应角色（鉴权关闭时中间件默认 admin），
        # UI 按角色裁剪导航（operator 只见监控/反馈/统计）
        "role": getattr(request.state, "role", None) or "admin",
        # M16f（review R9）：能力开关状态透出——运维自检"我以为配了"类事故
        "features": _feature_flags(),
    }


def _feature_flags() -> dict:
    """能力开关状态（M16f，review R9）：从配置聚合各产线能力是否启用。"""
    cfg = get_settings()
    notify = cfg.section("notify") or {}
    plc = cfg.section("plc") or {}
    archive = cfg.section("archive") or {}
    sec = cfg.section("security") or {}
    return {
        "webhook": bool(notify.get("webhook_url")),
        # 实际控制口径：trigger_dir 非空即启用（routes_detect.py 的 PLC
        # 触发接口同口径）；yaml 没有 enabled 键，读它恒为关（前端反馈 v7-6）
        "plc_trigger": bool((plc.get("trigger_dir") or "").strip()),
        "auto_archive": bool(archive.get("enabled", True)),
        "auth": bool((sec.get("api_key") or "").strip() or sec.get("keys")),
    }


# ── M16d 设置中心（review R3）：白名单配置读/写 ─────────────
_CONFIG_SCHEMA = [
    # (section, key, 类型, 中文名, 说明)
    ("pipeline", "latency_budget_ms", "int", "延迟预算 ms",
     "检测看板/压测的红线参考值（2060 GPU 2500² 预算 200ms）"),
    ("data", "default_root", "str", "数据集默认根目录",
     "UI 导入标准数据集时的默认根目录（环境变量 AOI_DATA_ROOT 优先）"),
    ("pipeline", "review_backlog_high", "int", "复判积压高水位",
     "人工复判工单的待复核检测数达到该值时自动暂停送检（默认 30）"),
    ("pipeline", "review_backlog_low", "int", "复判积压低水位",
     "待复核消化到该值时自动恢复送检（默认 10），须小于高水位"),
    ("self_learning", "trigger_min_feedback", "int", "巩固提醒阈值",
     "未巩固反馈达到该数量时状态表提示（即学始终即时生效）"),
    ("self_learning", "auto_activate", "bool", "巩固后自动激活",
     "开启后巩固生成的新版本自动切换为当前版本（默认关，人工 A/B 后再激活）"),
    ("evaluation", "benchmark_n_images", "int", "延迟基准样本数", ""),
    ("notify", "webhook_url", "str", "Webhook 地址",
     "空=禁用；检测/告警事件推送到 MES/上位机"),
    ("notify", "events", "list", "订阅事件",
     "anomaly/gray/feedback/consolidate/alarm/open_set/align_warn"),
    ("notify", "alarm_window", "int", "告警滑窗帧数", "连续异常告警滑窗长度"),
    ("notify", "alarm_k", "int", "告警触发帧数", "窗口内异常帧数 ≥ k 触发告警"),
    ("notify", "alarm_cooldown_s", "float", "告警冷却秒", "防刷屏"),
    ("archive", "enabled", "bool", "缺陷自动归档", "命中决策的帧自动归档到 archive/品类/日期"),
    ("archive", "decisions", "list", "归档决策", "anomaly/gray"),
    ("plc", "trigger_dir", "str", "PLC 触发目录", "空=禁用；光电信号落图目录（FIFO 取最早图检测）"),
    ("logging", "max_mb", "int", "日志单文件 MB", "超出自动轮转（重启后生效）"),
    ("logging", "backups", "int", "日志保留份数", "重启后生效"),
]
_CONFIG_KEYS = {(s, k) for s, k, *_ in _CONFIG_SCHEMA}


@router.get("/system/config")
def api_get_config():
    """读取白名单配置项（security 段不暴露——密钥只写不读）。"""
    cfg = get_settings()
    items = []
    for sec, key, typ, name, desc in _CONFIG_SCHEMA:
        items.append({"section": sec, "key": key, "type": typ,
                      "name": name, "desc": desc,
                      "value": cfg.get(sec, key)})
    return {"items": items, "config_path": str(cfg.config_path)}


@router.post("/system/config",
             dependencies=[Depends(require_role("admin"))])
def api_update_config(updates: dict):
    """批量更新白名单配置项并写回 yaml（热生效，无需重启）。

    非白名单键一律拒绝（引擎槽位参数/安全密钥属高危项，仍走 yaml 手改）。
    """
    cfg = get_settings()
    applied, rejected = [], []
    for sec_key, value in (updates or {}).items():
        try:
            sec, key = sec_key.split(".", 1)
        except ValueError:
            rejected.append(sec_key)
            continue
        if (sec, key) not in _CONFIG_KEYS:
            rejected.append(sec_key)
            continue
        cfg.update(sec, key, value)
        applied.append(sec_key)
    log_action("config_update",
               f"applied={len(applied)} rejected={rejected}")
    return {"applied": applied, "rejected": rejected}


@router.get("/categories")
def api_categories():
    """可用品类列表：引擎快照已准备品类 + DB 中有图片的品类。

    W-workorder：品类是检测/建模的最小单元；工单（含数据条件）见
    /api/workorders，数据按数据源管理见 /api/datasources。
    """
    return _all_categories()


def _all_categories() -> list:
    """合并引擎快照目录与 DB 图片品类，标记 prepared/versions。"""
    prepared = {}
    try:
        from ..engine import get_engine
        engine = get_engine()
        root = engine.snap_root
        if os.path.isdir(root):
            for cat in sorted(os.listdir(root)):
                cat_dir = os.path.join(root, cat)
                if not os.path.isdir(cat_dir):
                    continue
                vs = engine.list_versions(cat)
                if vs:
                    prepared[cat] = {
                        "category": cat, "engine": "builtin", "prepared": True,
                        "versions": vs,
                        "current_version": engine.current_version(cat)}
    except Exception:  # noqa: BLE001 引擎配置缺失时仅列 DB 品类
        pass
    merged = dict(prepared)
    with session_scope() as s:
        rows = (s.query(Image.category, func.count(Image.id))
                .group_by(Image.category).all())
    for cat, n in rows:
        item = merged.setdefault(cat, {
            "category": cat, "engine": "builtin", "prepared": False,
            "versions": [], "current_version": None})
        item["n_images"] = int(n)
    return sorted(merged.values(), key=lambda x: x["category"])


@router.get("/file")
def api_file(path: str, thumb: int = 0):
    """按绝对路径读取本地文件（M10a 起加白名单：仅 storage 目录内或
    DB 已登记路径放行，其余 403——堵任意路径读取漏洞）。

    thumb=N：返回最长边 ≤N 的 JPEG 缩略图（缩略图列表不再整图下载原图，
    大幅降带宽/延迟——数据管理页"一直在加载/卡"的根因之一）。
    注意：不用 Optional[int]（future annotations 下 pydantic 解析失败）。
    """
    from ..core.security import is_path_allowed
    p = Path(path)
    # 2026-10-03 A10 修复：先权限后存在性——原顺序（先 404 后 403）会让
    # 未授权调用方通过响应码差异探测服务器上任意路径是否存在（路径探测）。
    if not is_path_allowed(path):
        raise HTTPException(status_code=403,
                            detail="路径不在白名单（仅 storage 目录或已登记图片）")
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    if thumb > 0:
        return _make_thumb(p, thumb)
    return FileResponse(str(p))


def _make_thumb(p: Path, size: int):
    """PIL 等比缩放生成 JPEG 缩略图（RGBA/P 垫白底，透明转白）。"""
    import io

    from fastapi.responses import Response
    from PIL import Image

    img = Image.open(p)
    img.thumbnail((size, size), Image.LANCZOS)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        mask = img.split()[-1]
        bg.paste(img.convert("RGB"), mask=mask)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ── 系统自检（2026-10-09）：数据残留 / 引用悬空 / 页面一致性 ──
#
# 背景：历史版本删除工单时把 Detection.workorder_id 置 NULL 保留记录，
# 导致工单删光后工作台 KPI、标注反馈页仍显示残留数字（用户反馈）。
# 自检把这类问题逐条量化，fixable=true 的项可由 cleanup 一键修复。

_FILE_CHECK_CAP = 5000  # 图片文件存在性抽查上限，避免全量 isfile 拖慢


def _check_item(cid: str, name: str, count: int, detail: str,
                fixable: bool = False, fail: bool = False) -> dict:
    return {"id": cid, "name": name, "count": int(count),
            "status": ("ok" if count == 0 else ("fail" if fail else "warn")),
            "detail": detail, "fixable": bool(fixable and count > 0)}


def _dangling_detection_ids(s, only_null_wo: bool) -> list[int]:
    """无工单归属（NULL）或指向已删工单的检测 id。"""
    q = s.query(Detection.id)
    if only_null_wo:
        q = q.filter(Detection.workorder_id.is_(None))
    else:
        q = (q.filter(Detection.workorder_id.isnot(None))
             .filter(~Detection.workorder_id.in_(
                 s.query(WorkOrder.id))))
    return [r[0] for r in q.all()]


def _dangling_feedback_ids(s) -> list[int]:
    """指向已不存在检测的反馈 id。"""
    return [r[0] for r in
            s.query(Feedback.id)
            .filter(~Feedback.detection_id.in_(s.query(Detection.id))).all()]


def _dangling_cf_rows(s) -> int:
    """合并反馈血缘中指向已删反馈/批次的行数。"""
    n_fb = (s.query(func.count(ConsolidationFeedback.feedback_id))
            .filter(~ConsolidationFeedback.feedback_id.in_(
                s.query(Feedback.id))).scalar() or 0)
    n_cb = (s.query(func.count(ConsolidationFeedback.consolidation_id))
            .filter(~ConsolidationFeedback.consolidation_id.in_(
                s.query(ConsolidationBatch.id))).scalar() or 0)
    return int(n_fb + n_cb)


def _missing_image_file_ids(s) -> list[int]:
    """DB 已登记但磁盘文件不存在的图片 id（抽查封顶 _FILE_CHECK_CAP）。"""
    rows = s.query(Image.id, Image.path).limit(_FILE_CHECK_CAP).all()
    return [i for i, p in rows if not os.path.isfile(p)]


def _stuck_claim_ids(s) -> list[int]:
    """认领指向已删工单的图片（卡死，永不消费）。"""
    return [r[0] for r in
            s.query(Image.id)
            .filter(Image.claim_workorder_id.isnot(None))
            .filter(~Image.claim_workorder_id.in_(s.query(WorkOrder.id)))
            .all()]


def _dangling_wosource_ids(s) -> list[int]:
    """工单-数据源关联中指向已删工单/数据源的行。"""
    bad_wo = (s.query(WorkOrderSource.id)
              .filter(~WorkOrderSource.workorder_id.in_(
                  s.query(WorkOrder.id))))
    bad_ds = (s.query(WorkOrderSource.id)
              .filter(~WorkOrderSource.datasource_id.in_(
                  s.query(DataSource.id))))
    return [r[0] for r in bad_wo.union(bad_ds).all()]


def _empty_datasource_names(s) -> list[str]:
    """名下无批次或无图片的数据源（数据管理页空壳）。"""
    names = []
    for ds in s.query(DataSource).all():
        n_ds = (s.query(func.count(Dataset.id))
                .filter(Dataset.datasource_id == ds.id).scalar() or 0)
        if n_ds == 0:
            names.append(ds.name)
    return names


def _orphan_dataset_ids(s) -> list[int]:
    """无数据源归属的批次 id：datasource_id 为空或指向已删数据源。
    自动归档/反馈回流/审计/历史导入批次天然不挂数据源——数据源删光后
    成为孤儿，其图片仍在数据管理/数据增强页显示（用户反馈 2026-10-09）。"""
    ds_ids = {r[0] for r in s.query(DataSource.id).all()}
    return [did for did, dsrc in s.query(Dataset.id, Dataset.datasource_id)
            if dsrc is None or dsrc not in ds_ids]


def _today_stats_mismatch(s) -> int:
    """今日 StatsDaily 聚合行与 Detection 表实时计数的偏差。"""
    today = date.today()
    row = (s.query(StatsDaily).filter(StatsDaily.date == today).first())
    n_det = (s.query(func.count(Detection.id))
             .filter(func.date(Detection.created_at) == today)
             .scalar() or 0)
    n_db = row.n_inspected if row else 0
    return abs(int(n_det) - int(n_db))


def _cascade_delete_detections(s, det_ids: list[int]) -> int:
    """级联删除检测及其反馈/合并反馈血缘（与删工单同口径）。"""
    if not det_ids:
        return 0
    fb_ids = [r[0] for r in s.query(Feedback.id)
              .filter(Feedback.detection_id.in_(det_ids)).all()]
    if fb_ids:
        s.query(ConsolidationFeedback).filter(
            ConsolidationFeedback.feedback_id.in_(fb_ids)
        ).delete(synchronize_session=False)
        s.query(Feedback).filter(Feedback.id.in_(fb_ids)
                                 ).delete(synchronize_session=False)
    s.query(Detection).filter(Detection.id.in_(det_ids)
                              ).delete(synchronize_session=False)
    return len(det_ids)


def _run_selfcheck(s) -> list[dict]:
    """逐项体检（只读，不改数据）。session 由调用方持有。"""
    items = []

    n = len(_dangling_detection_ids(s, only_null_wo=True))
    items.append(_check_item(
        "orphan_detections", "无工单归属的检测记录", n,
        "历史版本删工单时遗留的检测记录（工单归属为空）。会导致工作台 KPI、"
        "标注反馈、学习效果页在工单删光后仍显示残留数字。",
        fixable=True))

    n = len(_dangling_detection_ids(s, only_null_wo=False))
    items.append(_check_item(
        "dangling_detections", "检测指向已删工单", n,
        "检测记录的工单 id 在工单表中不存在（外键悬空）。",
        fixable=True, fail=True))

    n = len(_dangling_feedback_ids(s))
    items.append(_check_item(
        "dangling_feedback", "反馈指向已删检测", n,
        "反馈记录挂在已不存在的检测上，标注反馈页可能显示空图。",
        fixable=True, fail=True))

    n = _dangling_cf_rows(s)
    items.append(_check_item(
        "dangling_consolidation", "合并反馈血缘悬空", n,
        "巩固批次的反馈血缘指向已删除的反馈或批次。",
        fixable=True, fail=True))

    n = len(_missing_image_file_ids(s))
    items.append(_check_item(
        "missing_image_files", "图片文件缺失", n,
        f"数据库已登记但磁盘文件不存在（抽查上限 {_FILE_CHECK_CAP} 张）。"
        "数据管理/数据增强页会显示裂图。",
        fixable=True))

    ds_ids = _orphan_dataset_ids(s)
    n_img = ((s.query(func.count(Image.id))
              .filter(Image.dataset_id.in_(ds_ids)).scalar() or 0)
             if ds_ids else 0)
    items.append(_check_item(
        "orphan_datasets", "无数据源归属的批次与图片", len(ds_ids),
        f"批次未挂接任何数据源（自动归档/反馈回流/审计/历史导入遗留），"
        f"名下共 {n_img} 张图片。数据源删光后，数据管理缩略图、数据增强"
        f"基底图仍显示的就是它们。清理只删应用生成的文件（storage 目录内），"
        f"外部原始数据只删登记行、不动文件。",
        fixable=True))

    n = len(_stuck_claim_ids(s))
    items.append(_check_item(
        "stuck_claims", "图片认领卡死", n,
        "图片被已删除的工单认领未释放，永远不会被消费。",
        fixable=True, fail=True))

    n = len(_dangling_wosource_ids(s))
    items.append(_check_item(
        "dangling_workorder_sources", "工单-数据源关联悬空", n,
        "关联行指向已删除的工单或数据源。", fixable=True, fail=True))

    names = _empty_datasource_names(s)
    items.append(_check_item(
        "empty_datasources", "空数据源", len(names),
        ("名下没有任何批次，数据管理页显示为空壳。"
         + ("：" + "、".join(names[:5]) if names else "")),
        fixable=False))

    n = _today_stats_mismatch(s)
    items.append(_check_item(
        "stats_daily_mismatch", "今日统计聚合偏差", n,
        "stats_daily 今日行与检测表实时计数不一致，存档统计页趋势图可能"
        "与 KPI 对不上。", fixable=True))

    return items


@router.get("/system/selfcheck")
def api_system_selfcheck():
    """系统自检：数据残留 / 引用悬空 / 页面一致性（只读）。"""
    with session_scope() as s:
        items = _run_selfcheck(s)
    n_bad = sum(1 for i in items if i["status"] != "ok")
    return {"status": "ok" if n_bad == 0 else "issues",
            "n_issues": n_bad, "items": items}


@router.post("/system/selfcheck/cleanup",
             dependencies=[Depends(require_role("admin"))])
def api_system_selfcheck_cleanup():
    """一键修复自检中 fixable 的项。返回各项清理数量。"""
    fixed = {}
    with session_scope() as s:
        det_ids = (_dangling_detection_ids(s, only_null_wo=True)
                   + _dangling_detection_ids(s, only_null_wo=False))
        fixed["detections_removed"] = _cascade_delete_detections(s, det_ids)

        fb_ids = _dangling_feedback_ids(s)
        if fb_ids:
            s.query(ConsolidationFeedback).filter(
                ConsolidationFeedback.feedback_id.in_(fb_ids)
            ).delete(synchronize_session=False)
            s.query(Feedback).filter(Feedback.id.in_(fb_ids)
                                     ).delete(synchronize_session=False)
        fixed["feedback_removed"] = len(fb_ids)

        # 合并反馈血缘悬空（指向上一步已删之外的孤儿行）
        n_cf = (s.query(ConsolidationFeedback)
                .filter(~ConsolidationFeedback.feedback_id.in_(
                    s.query(Feedback.id))
                    | ~ConsolidationFeedback.consolidation_id.in_(
                        s.query(ConsolidationBatch.id)))
                .delete(synchronize_session=False))
        fixed["consolidation_rows_removed"] = int(n_cf or 0)

        # 无数据源归属的批次：删批次 + 图片登记行；文件只删应用生成的
        # （storage 目录内，如 uploads/archive），外部原始数据不动文件。
        ds_ids = _orphan_dataset_ids(s)
        fixed["orphan_datasets_removed"] = len(ds_ids)
        n_img_removed = 0
        n_files_removed = 0
        if ds_ids:
            img_rows = (s.query(Image.id, Image.path)
                        .filter(Image.dataset_id.in_(ds_ids)).all())
            img_ids = [i for i, _p in img_rows]
            if img_ids:
                # 检测记录的图片引用置空，避免悬空
                s.query(Detection).filter(Detection.image_id.in_(img_ids)).update(
                    {Detection.image_id: None}, synchronize_session=False)
                s.query(Image).filter(Image.id.in_(img_ids)
                                      ).delete(synchronize_session=False)
            n_img_removed = len(img_ids)
            try:
                root = get_settings().storage_dir.resolve()
            except Exception:  # noqa: BLE001
                root = None
            if root is not None:
                for _i, p in img_rows:
                    try:
                        fp = Path(p).resolve()
                        if fp.is_file() and root in fp.parents:
                            fp.unlink()
                            n_files_removed += 1
                    except Exception:  # noqa: BLE001 单文件失败不阻断整体清理
                        pass
            s.query(Dataset).filter(Dataset.id.in_(ds_ids)
                                    ).delete(synchronize_session=False)
        fixed["orphan_images_removed"] = n_img_removed
        fixed["orphan_files_removed"] = n_files_removed

        img_ids = _missing_image_file_ids(s)
        if img_ids:
            s.query(Image).filter(Image.id.in_(img_ids)
                                  ).delete(synchronize_session=False)
        fixed["missing_images_removed"] = len(img_ids)

        claim_ids = _stuck_claim_ids(s)
        if claim_ids:
            s.query(Image).filter(Image.id.in_(claim_ids)).update(
                {Image.claim_workorder_id: None, Image.claim_token: None,
                 Image.claimed_at: None, Image.processing_status: "pending"},
                synchronize_session=False)
        fixed["stuck_claims_released"] = len(claim_ids)

        wos_ids = _dangling_wosource_ids(s)
        if wos_ids:
            s.query(WorkOrderSource).filter(
                WorkOrderSource.id.in_(wos_ids)
            ).delete(synchronize_session=False)
        fixed["workorder_sources_removed"] = len(wos_ids)

        # 重算今日聚合行
        today = date.today()
        row = s.query(StatsDaily).filter(StatsDaily.date == today).first()
        if row is not None:
            n_det = (s.query(func.count(Detection.id))
                     .filter(func.date(Detection.created_at) == today)
                     .scalar() or 0)
            n_an = (s.query(func.count(Detection.id))
                    .filter(func.date(Detection.created_at) == today,
                            Detection.is_anomaly.is_(True)).scalar() or 0)
            fixed["stats_daily_recomputed"] = (
                0 if row.n_inspected == n_det and row.n_anomaly == n_an else 1)
            row.n_inspected, row.n_anomaly = int(n_det), int(n_an)
        else:
            fixed["stats_daily_recomputed"] = 0

    log_action("selfcheck_cleanup",
               " ".join(f"{k}={v}" for k, v in fixed.items()))
    return {"fixed": fixed}

