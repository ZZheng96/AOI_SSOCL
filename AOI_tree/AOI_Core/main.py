"""AOI 实时在线 AI 质检系统 —— 桌面端入口。

启动流程：
1. 守护线程内嵌启动 FastAPI 后端（server.start_server_background，
   由后端代理实现；若 import 失败则提示并假设后端已独立启动）；
2. 轮询 /api/health 直至就绪（超时则提示并继续，UI 离线可用）；
3. 创建 QApplication + MainWindow，应用 LIGHT_QSS 浅色主题，
   字体 Microsoft YaHei UI。
"""
from __future__ import annotations

import os
import sys
import threading
import time

# 保证可以 `python main.py` 直接运行（包根加入 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://127.0.0.1:8017"
HEALTH_URL = BASE_URL + "/api/health"


def start_backend() -> bool:
    """守护线程内嵌启动后端；返回是否成功启动。

    U-opsflow(2026-08-26)：端口 8017 已被占用（后端已在运行，如先跑了
    server.py）时**直接复用现有后端**，不再起第二个 uvicorn——此前二次
    启动会因端口绑定失败直接 exit 1 崩溃且无提示。
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.bind((BASE_URL.split("//")[-1].rsplit(":", 1)[0], 8017))
        port_free = True
    except OSError:
        port_free = False
    finally:
        sock.close()
    if not port_free:
        print("[main] 端口 8017 已被占用（后端已在运行），直接复用现有后端")
        return False
    try:
        from server import start_server_background  # type: ignore
    except ImportError:
        print("[main] 未找到 server.start_server_background，"
              "假设后端已独立启动（%s）" % BASE_URL)
        return False
    thread = threading.Thread(target=start_server_background,
                              name="aoi-backend", daemon=True)
    thread.start()
    print("[main] 后端已在守护线程中启动")
    return True


def wait_backend_ready(timeout_s: float = 20.0) -> bool:
    """轮询 /api/health 直至就绪或超时。"""
    import requests

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.ok and r.json().get("status") == "ok":
                print("[main] 后端已就绪（%.1fs）" % (time.monotonic() - t0))
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.8)
    print("[main] 等待后端就绪超时，UI 将以离线容错模式启动")
    return False


def main() -> int:
    start_backend()
    wait_backend_ready()

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from ui.main_window import MainWindow
    from ui.theme import apply_light_theme

    app = QApplication(sys.argv)
    app.setApplicationName("AOI 实时在线 AI 质检系统")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    apply_light_theme(app)

    window = MainWindow(BASE_URL)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
