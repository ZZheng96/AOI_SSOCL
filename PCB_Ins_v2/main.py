"""PCB_Ins v2 桌面入口：内嵌启动检测服务（守护线程）+ 主窗口。

用法：
    python main.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import ensure_dirs  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


def _start_backend_if_port_free() -> None:
    """端口已被占用时复用现有服务，不再重复起线程。

    2026-09-01：原实现无条件 start_server_background()，第二次启动桌面端会因
    端口冲突在守护线程里抛异常，UI 仍打开但请求全部失败，现象难排查。
    """
    from app.config import get_settings

    settings = get_settings()
    host = "127.0.0.1" if settings.host == "0.0.0.0" else settings.host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, settings.port))
        except OSError:
            print(f"[main] 端口 {settings.port} 已被占用，直接复用现有服务")
            return

    from server import start_server_background
    start_server_background()


def main() -> int:
    ensure_dirs()
    _start_backend_if_port_free()

    app = QApplication(sys.argv)
    app.setApplicationName("PCB 缺陷检测系统 v2")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
