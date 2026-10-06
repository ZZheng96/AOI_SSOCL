"""AOI 系统后端启动入口。

用法：
    python server.py            # 前台启动 uvicorn（host/port 取自 configs/default.yaml）
    start_server_background()   # 守护线程内嵌启动（供桌面端 UI 进程内复用）
"""
from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 保证以脚本方式运行时能解析 backend 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """M11e 日志轮转（评审 §11.1）：root 追加 RotatingFileHandler →
    storage/logs/server.log（默认 10MB×5 份，logging 段可调）。
    幂等：重复调用不重复挂 handler。落盘失败不影响服务启动。"""
    root = logging.getLogger()
    if any(getattr(h, "_aoi_rotating", False) for h in root.handlers):
        return
    try:
        from backend.core.config import get_settings
        settings = get_settings()
        log_dir = settings.storage("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        max_mb = int(settings.get("logging", "max_mb", 10))
        backups = int(settings.get("logging", "backups", 5))
        fh = RotatingFileHandler(log_dir / "server.log",
                                 maxBytes=max_mb * 1024 * 1024,
                                 backupCount=backups, encoding="utf-8")
        fh._aoi_rotating = True  # type: ignore[attr-defined]
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    except Exception:  # noqa: BLE001
        logger.exception("日志轮转初始化失败（继续仅控制台输出）")


def _host_port() -> tuple[str, int]:
    from backend.core.config import get_settings
    settings = get_settings()
    return (settings.get("system", "host", "127.0.0.1"),
            int(settings.get("system", "port", 8017)))


def _make_server():
    """按配置构建 uvicorn.Server 实例。"""
    import uvicorn
    _setup_logging()
    host, port = _host_port()
    config = uvicorn.Config("backend.api.app:app", host=host, port=port,
                            log_level="info")
    return uvicorn.Server(config)


def start_server_background() -> threading.Thread:
    """在守护线程中启动服务（桌面端内嵌使用）。返回线程对象。"""
    server = _make_server()
    host, port = _host_port()

    def _run():
        logger.info("AOI 后台服务启动: http://%s:%d", host, port)
        server.run()

    t = threading.Thread(target=_run, name="aoi-server", daemon=True)
    t.start()
    return t


def main() -> None:
    host, port = _host_port()
    logger.info("AOI 服务启动: http://%s:%d", host, port)
    _make_server().run()


if __name__ == "__main__":
    main()
