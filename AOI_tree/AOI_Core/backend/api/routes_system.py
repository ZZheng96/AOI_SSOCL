"""系统接口：健康检查 / 系统信息 / 本地文件读取（开发用）。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import func

from ..core.config import get_settings
from ..core.security import require_role
from ..db.database import log_action, session_scope
from ..db.models import Detection, Feedback, Image, Video
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
