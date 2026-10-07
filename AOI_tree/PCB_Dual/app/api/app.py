"""PCB_Ins v2 REST API 装配。

对外接口（AOI_sys / 上位机接入点）：
- /api/templates/*      模板 CRUD / 发布 / 绑定品类模型
- /api/detect/*         检测（dual / traditional / feature / trigger）
- /api/detections/*     检测记录
- /api/models/*         品类模型准备 / 版本管理 / 学习状态
- /api/feedback         复判反馈即学
- /api/workorders/*     工单（产线任务实例）
- /api/datasources/*    数据源（数据资产）
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api import (routes_async, routes_cad, routes_detect,
                     routes_feedback, routes_filter, routes_models,
                     routes_stats, routes_templates, routes_workorder)

app = FastAPI(title="PCB_Ins v2", version="0.5.0")

app.include_router(routes_templates.router, prefix="/api", tags=["templates"])
app.include_router(routes_detect.router, prefix="/api", tags=["detect"])
app.include_router(routes_models.router, prefix="/api", tags=["models"])
app.include_router(routes_feedback.router, prefix="/api", tags=["feedback"])
app.include_router(routes_workorder.router, prefix="/api", tags=["workorder"])
app.include_router(routes_stats.router, prefix="/api", tags=["stats"])
app.include_router(routes_async.router, prefix="/api", tags=["async-detect"])
app.include_router(routes_filter.router, prefix="/api", tags=["filter-alarm"])
app.include_router(routes_cad.router, prefix="/api", tags=["cad"])


@app.middleware("http")
async def _api_key_auth(request, call_next):
    """简单权限（P2）：配置 auth.api_key 后，除 /api/health 外需 X-API-Key。"""
    from starlette.responses import JSONResponse
    from app.config import get_settings
    key = get_settings().get("auth", "api_key", "")
    if key and request.url.path != "/api/health":
        if request.headers.get("X-API-Key") != key:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "PCB_Ins v2"}


@app.on_event("startup")
def _startup() -> None:
    """启动产线服务与检测任务队列（反馈直达树干 AOI_Core，无需本地补偿 worker）。"""
    from app.config import get_settings
    from app.core.task_queue import get_task_queue
    get_task_queue().start()
    if get_settings().get("system", "pipeline_enabled", False):
        from app.core.pipeline_service import PipelineService
        PipelineService.get().start()
