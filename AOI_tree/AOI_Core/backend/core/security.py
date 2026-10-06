"""产线安全（M10a，评审自检 §9 P0）：API Key 鉴权 + /file 路径白名单。

设计口径：
- security.api_key 为空 = 开发模式关闭鉴权；生产容器通过 AOI_API_KEY 强制启用；
  产线部署设置 api_key 后，除 /api/health（负载探活）外所有 API、文档和
  WebSocket 请求必须带 X-API-Key 头。
- /api/file 白名单：仅放行 storage_dir / engine_storage 目录内文件，
  以及 DB 已登记路径（images.path / detections.image_path /
  heatmap_path / overlay_path）；其余一律 403。

2026-09-01 安全走查修正：
- 删除 _EXEMPT_PREFIXES 常量。它是死代码——dispatch 的判定条件是
  `path.startswith("/api")`，非 /api 路径本就不进校验，该"豁免清单"从未
  被读取。保留它会让人误以为 /files 静态目录是"经评估后豁免"，实际是
  无鉴权全量暴露；相应挂载已在 backend/api/app.py 移除。
- 鉴权口径：/api、/ws、/docs、/redoc、/openapi.json 统一进入校验；仅
  /api/health 保持探活豁免。
"""
from __future__ import annotations

import contextvars
import hmac
import logging
import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings

logger = logging.getLogger(__name__)

# A13（2026-10-03）：请求级操作者上下文——ApiKeyMiddleware 解析角色后
# 写入，审计日志（db.database.log_action）缺省读取，业务调用点零改动。
# contextvars 经 anyio 任务树传播到同步端点线程；后台 worker 线程
# （threading 直接 spawn）不继承，读默认值 "operator"（系统任务口径）。
current_actor: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aoi_current_actor", default="operator")

_EXEMPT_PATHS = ("/api/health",)          # 探活豁免（文档兼容常量）

# M11c 多角色（评审 §11.1 安全维度）：operator(1) < engineer(2) < admin(3)
ROLE_RANK = {"operator": 1, "engineer": 2, "admin": 3}


def get_configured_api_key(sec: dict) -> str:
    """管理密钥来源（2026-10-03 A10：yaml 不再只支持明文）。

    优先级：AOI_API_KEY 环境变量 > security.api_key_env 指向的自定义环境
    变量 > yaml 明文 security.api_key（仅开发兜底，生产勿用——明文入库
    会随配置备份/截图外泄）。
    """
    key = (os.environ.get("AOI_API_KEY") or "").strip()
    if key:
        return key
    env_name = str(sec.get("api_key_env") or "").strip()
    if env_name:
        return (os.environ.get(env_name) or "").strip()
    return str(sec.get("api_key") or "").strip()


def _resolve_role(key_header: str) -> str | None:
    """请求头 Key → 角色。鉴权关闭时返回 admin（开发透传）；
    开启时：keys 表命中按其角色，单 api_key 命中=admin，否则 None（未授权）。"""
    sec = get_settings().section("security") or {}
    single = get_configured_api_key(sec)
    keys = sec.get("keys") or {}
    if not single and not keys:
        return "admin"                    # 鉴权未启用
    for configured_key, configured_role in keys.items():
        if hmac.compare_digest(str(key_header), str(configured_key)):
            role = str(configured_role).strip().lower()
            return role if role in ROLE_RANK else None
    if single and hmac.compare_digest(key_header, single):
        return "admin"
    return None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """X-API-Key 校验 + 角色解析（request.state.role）。"""

    async def dispatch(self, request: Request, call_next):
        sec = get_settings().section("security") or {}
        auth_on = bool(get_configured_api_key(sec) or sec.get("keys"))
        path = request.url.path
        protected_path = path.startswith("/api") or path.startswith("/ws") or path in ("/docs", "/redoc", "/openapi.json")
        if auth_on and protected_path and path not in _EXEMPT_PATHS:
            role = _resolve_role(request.headers.get("X-API-Key", ""))
            if role is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "未授权：缺少或错误的 X-API-Key 请求头"})
            request.state.role = role
        else:
            request.state.role = "admin"
        # A13：角色写入请求上下文，审计日志经 current_actor 读取
        token = current_actor.set(request.state.role)
        try:
            return await call_next(request)
        finally:
            current_actor.reset(token)


def require_role(min_role: str):
    """FastAPI 依赖：要求角色 >= min_role（中间件已解析 request.state.role）。
    用法：@router.post(..., dependencies=[Depends(require_role("engineer"))])
    注意：注解必须用模块级 Request（__future__ annotations 下局部别名无法被
    FastAPI get_type_hints 解析，会把 request 误判为查询参数 → 422）。"""

    def _dep(request: Request):
        role = getattr(request.state, "role", "admin")
        if ROLE_RANK.get(role, 0) < ROLE_RANK[min_role]:
            raise HTTPException(
                status_code=403,
                detail=f"权限不足：该操作需要 {min_role} 及以上角色（当前 {role}）")
    return _dep


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    """上传体积硬顶（2026-10-03 A10）：此前上传端点（图片/快照 zip）无大小
    限制，单请求即可耗尽磁盘/内存（DoS）。

    按 Content-Length 请求头快速拒绝（413），上限取 security.max_upload_mb
    （默认 200MB，覆盖快照 zip 导入场景）。无 Content-Length 的 chunked
    请求交由端点流式处理，不在此处拦截。
    """

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            sec = get_settings().section("security") or {}
            try:
                max_mb = float(sec.get("max_upload_mb", 200))
            except (TypeError, ValueError):
                max_mb = 200.0
            cl = request.headers.get("content-length")
            if cl is not None:
                try:
                    if int(cl) > max_mb * 1024 * 1024:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": f"请求体超过上传上限 "
                                               f"{max_mb:g}MB"})
                except ValueError:
                    return JSONResponse(status_code=400,
                                        content={"detail": "非法 Content-Length"})
        return await call_next(request)


def is_path_allowed(path: str) -> bool:
    """/api/file 白名单判定：storage 目录内 或 DB 已登记路径。

    2026-08-31 走查改进：Windows 路径大小写不敏感。此前 resolve() 返回
    盘符大写（D:\\...）而 DB 登记路径为小写（d:\\...），SQLite 字符串
    比较大小写敏感 → 已登记图片被 403 误拒（传统 CV 复判接口复现）。
    统一 lower() 后再比较。
    """
    if not path:
        return False
    try:
        p = Path(path).resolve()
    except Exception:  # noqa: BLE001 非法路径
        return False
    p_lower = str(p).lower()
    settings = get_settings()
    # 1) 存储目录内（storage_dir / engine_storage）
    for root in (settings.storage_dir, settings.engine_storage):
        try:
            root_lower = str(Path(root).resolve()).lower()
            if os.path.commonpath([p_lower, root_lower]) == root_lower:
                return True
        except ValueError:  # 跨盘符
            continue
    # 2) DB 已登记路径（图片/检测/热力图/叠加图）——lower 匹配兼容大小写差异
    try:
        from ..db.database import session_scope
        from ..db.models import Detection, Image
        from sqlalchemy import func
        with session_scope() as sess:
            if sess.query(Image.id).filter(func.lower(Image.path) == p_lower).first():
                return True
            q = sess.query(Detection.id).filter(
                (func.lower(Detection.image_path) == p_lower)
                | (func.lower(Detection.heatmap_path) == p_lower)
                | (func.lower(Detection.overlay_path) == p_lower))
            if q.first():
                return True
    except Exception:  # noqa: BLE001 DB 异常时保守拒绝
        logger.warning("/file 白名单 DB 查询失败，按拒绝处理: %s", path)
    return False
