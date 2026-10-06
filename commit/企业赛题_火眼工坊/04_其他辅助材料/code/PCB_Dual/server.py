"""PCB_Ins v2 服务端启动入口。

用法：
    python server.py            # 独立启动 uvicorn（host/port 取自 configs/default.yaml）
    start_server_background()   # 守护线程内嵌启动（桌面端 UI 进程内复用）
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

# 保证以脚本方式运行时能解析 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    from app.config import get_settings
    root = logging.getLogger()
    if any(getattr(h, "_pcbins_rotating", False) for h in root.handlers):
        return
    try:
        settings = get_settings()
        log_dir = settings.logs_dir
        max_mb = int(settings.get("logging", "max_mb", 10))
        backups = int(settings.get("logging", "backups", 5))
        fh = RotatingFileHandler(log_dir / "server.log", maxBytes=max_mb * 1024 * 1024,
                                 backupCount=backups, encoding="utf-8")
        fh._pcbins_rotating = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    except Exception:  # noqa: BLE001
        logger.exception("日志轮转初始化失败（继续仅控制台输出）")


def _make_server():
    import uvicorn
    from app.config import get_settings
    _setup_logging()
    settings = get_settings()
    config = uvicorn.Config("app.api.app:app", host=settings.host, port=settings.port,
                            log_level="info")
    return uvicorn.Server(config)


def start_server_background() -> threading.Thread:
    """守护线程内嵌启动（桌面端复用）。返回线程对象。"""
    from app.config import get_settings
    server = _make_server()
    settings = get_settings()

    def _run():
        logger.info("PCB_Ins v2 后台服务启动: http://%s:%d", settings.host, settings.port)
        server.run()

    t = threading.Thread(target=_run, name="pcbins-server", daemon=True)
    t.start()
    return t


def main() -> None:
    from app.config import get_settings
    settings = get_settings()
    logger.info("PCB_Ins v2 服务启动: http://%s:%d", settings.host, settings.port)
    _make_server().run()


if __name__ == "__main__":
    main()
