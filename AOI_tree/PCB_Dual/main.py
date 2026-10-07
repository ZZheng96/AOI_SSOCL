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


def _start_aoi_core_async() -> None:
    """后台线程启动/复用树干 AOI_Core，不阻塞 UI；首次特征检测时 FeatureClient 会等待就绪。"""
    import threading

    def _run():
        try:
            from app.core.aoi_core_launcher import ensure_core_running
            ok, msg = ensure_core_running()
            print(f"[main] AOI_Core {'就绪' if ok else '不可用'}：{msg}")
        except Exception as exc:  # noqa: BLE001
            print(f"[main] AOI_Core 启动异常：{exc}")

    threading.Thread(target=_run, name="aoi-core-launcher", daemon=True).start()


def main() -> int:
    from app.core.single_instance import acquire_lock, lock_path_for
    if not acquire_lock("pcbins_ui"):
        app = QApplication(sys.argv)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(None, "PCB 缺陷检测系统 v2",
                            f"已有实例在运行（lock: {lock_path_for('pcbins_ui')}）")
        return 2

    ensure_dirs()
    _start_backend_if_port_free()
    _start_aoi_core_async()

    app = QApplication(sys.argv)
    app.setApplicationName("PCB 缺陷检测系统 v2")
    from app.core.aoi_core_launcher import stop_core
    app.aboutToQuit.connect(stop_core)
    window = MainWindow()
    if hasattr(window.model_page, "shutdown"):
        app.aboutToQuit.connect(window.model_page.shutdown)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
