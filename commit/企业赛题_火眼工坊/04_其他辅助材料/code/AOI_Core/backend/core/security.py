"""产线安全（M10a，评审自检 §9 P0）：API Key 鉴权 + /file 路径白名单。

设计口径：
- security.api_key 为空 = 关闭鉴权（开发默认，零配置不破坏现有流程）；
  产线部署设置 api_key 后，除 /api/health（负载探活）外所有 /api/* REST
  请求必须带 X-API-Key 头；WebSocket 推流（/ws/*）豁免（同源桌面端，
  浏览器 WS 不便带自定义头，且 WS 不暴露写操作之外的额外面）。
- /api/file 白名单：仅放行 storage_dir / demo5_storage 目录内文件，
  以及 DB 已登记路径（images.path / detections.image_path /
  heatmap_path / overlay_path）；其余一律 403。

2026-09-01 安全走查修正：
- 删除 _EXEMPT_PREFIXES 常量。它是死代码——dispatch 的判定条件是
  `path.startswith("/api")`，非 /api 路径本就不进校验，该"豁免清单"从未
  被读取。保留它会让人误以为 /files 静态目录是"经评估后豁免"，实际是
  无鉴权全量暴露；相应挂载已在 backend/api/app.py 移除。
- 鉴权口径：所有非 /api 路径（含 /ws、/docs）均不校验 Key。因此产线
  必须靠回环绑定 + 反向代理限制 /docs 等入口，不能只依赖本模块。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = ("/api/health",)          # 探活豁免

# M11c 多角色（评审 §11.1 安全维度）：operator(1) < engineer(2) < admin(3)
ROLE_RANK = {"operator": 1, "engineer": 2, "admin": 3}


def _resolve_role(key_header: str) -> str | None:
    """请求头 Key → 角色。鉴权关闭时返回 admin（开发透传）；
    开启时：keys 表命中按其角色，单 api_key 命中=admin，否则 None（未授权）。"""
    sec = get_settings().section("security") or {}
    single = (sec.get("api_key") or "").strip()
    keys = sec.get("keys") or {}
    if not single and not keys:
        return "admin"                    # 鉴权未启用
    if key_header in keys:
        role = str(keys[key_header]).strip().lower()
        return role if role in ROLE_RANK else None
    if single and key_header == single:
        return "admin"
    return None


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """X-API-Key 校验 + 角色解析（request.state.role）。"""

    async def dispatch(self, request: Request, call_next):
        sec = get_settings().section("security") or {}
        auth_on = bool((sec.get("api_key") or "").strip() or sec.get("keys"))
        path = request.url.path
        if auth_on and path.startswith("/api") and path not in _EXEMPT_PATHS:
            role = _resolve_role(request.headers.get("X-API-Key", ""))
            if role is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "未授权：缺少或错误的 X-API-Key 请求头"})
            request.state.role = role
        else:
            request.state.role = "admin"
        return await call_next(request)


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
    # 1) 存储目录内（storage_dir / demo5_storage）
    for root in (settings.storage_dir, settings.demo5_storage):
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
