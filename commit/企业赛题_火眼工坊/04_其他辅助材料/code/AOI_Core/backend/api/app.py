"""FastAPI 应用装配：CORS / 静态文件 / REST 路由 / WebSocket / 启动初始化。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_settings
from . import routes_data, routes_detect, routes_feedback, routes_learning
from . import routes_models, routes_review, routes_stats, routes_system, ws
from . import routes_workorder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时：扫描 Demo5 快照目录并幂等登记到 Model 表；
    M8a：把 dataset_id 为空的存量图片按品类归入"历史未分组"批次（幂等）。"""
    try:
        n = routes_models.register_demo5_snapshots()
        logger.info("demo5 快照登记完成（新增 %d 行）", n)
    except Exception:  # noqa: BLE001 快照目录缺失不阻断服务启动
        logger.exception("启动初始化失败（demo5 快照扫描/登记）")
    try:
        from ..core.datasets import assign_legacy_datasets
        n = assign_legacy_datasets()
        if n:
            logger.info("历史未分组批次归组 %d 张存量图片", n)
    except Exception:  # noqa: BLE001 归组失败不阻断服务启动
        logger.exception("启动初始化失败（存量图片批次归组）")
    try:
        # 前端反馈 v5：产线服务（工单自动消费数据流检测）
        from ..pipeline.pipeline_service import PipelineService
        PipelineService.get().start()
    except Exception:  # noqa: BLE001 产线服务失败不阻断启动
        logger.exception("产线服务启动失败")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AOI 实时在线 AI 质检系统", lifespan=lifespan)

    # CORS 全开（桌面端/前端联调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # M10a：API Key 鉴权（security.api_key 非空时启用；/api/health 豁免）
    from ..core.security import ApiKeyMiddleware
    app.add_middleware(ApiKeyMiddleware)

    # 2026-09-01 安全走查：移除 /files 静态挂载。
    # 原挂载把整个 storage_dir 无鉴权暴露（ApiKeyMiddleware 只校验 /api 前缀），
    # 等于绕过 API Key 与 /api/file 白名单；全仓无调用方（UI 统一走
    # /api/file?path=，见 ui/api_client.py），故直接删除而非豁免。

    # REST 路由（统一 /api 前缀）
    for mod in (routes_system, routes_data, routes_detect, routes_feedback,
                routes_models, routes_stats, routes_learning, routes_review,
                routes_workorder):
        app.include_router(mod.router, prefix="/api")

    # WebSocket 路由（不带前缀）
    app.include_router(ws.router)
    return app


app = create_app()
