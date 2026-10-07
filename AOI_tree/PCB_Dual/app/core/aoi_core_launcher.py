"""树干 AOI_Core 后端的启动 / 复用（子进程）。

树枝 PCB_Dual 与树干 AOI_Core 都有顶层包 ``algo``，同进程 import 会撞名，
且 torch/DINO 只应由 Core 加载一次，因此 Core 后端必须以独立子进程运行，
树枝只通过 HTTP 调用。（Core 的 ``ui`` 页面只走 HTTP/WS，可在树枝进程内
嵌入，见 app/ui/core_pages.py。）

- 已有实例（/api/health 返回 ok）→ 直接复用，不拉起、退出时也不关闭
- 端口空闲 → ``python server.py``（cwd=AOI_Core）拉起子进程，轮询健康检查
- 退出时只 terminate 自己拉起的进程
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def core_healthy(base_url: str, timeout: float = 1.5) -> bool:
    import requests
    try:
        r = requests.get(f"{base_url}/api/health", timeout=timeout)
        return r.ok and (r.json() or {}).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def ensure_core_running(wait_s: float = 90.0) -> tuple[bool, str]:
    """确保 Core 可用。返回 (ok, 说明)。不抛异常，便于 UI 启动时兜底。"""
    from app.config import get_settings

    settings = get_settings()
    url = settings.aoi_core_url
    if core_healthy(url):
        return True, f"复用已运行的 AOI_Core：{url}"
    if not settings.aoi_core_autostart:
        return False, f"AOI_Core 未运行且 autostart=false：{url}"

    core_dir = settings.aoi_core_dir
    if not (core_dir / "server.py").is_file():
        return False, f"找不到 AOI_Core/server.py：{core_dir}"

    host = (urlparse(url).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost"):
        return False, f"AOI_Core 指向远端 {url} 且不可达，不在本机拉起"

    global _proc
    with _lock:
        if _proc is None or _proc.poll() is not None:
            log_dir = settings.logs_dir
            log_f = open(log_dir / "aoi_core_child.log", "ab")  # noqa: SIM115
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            _proc = subprocess.Popen(
                [sys.executable, "server.py"], cwd=str(core_dir),
                stdout=log_f, stderr=subprocess.STDOUT, env=env, creationflags=flags)
            atexit.register(stop_core)
            logger.info("已拉起 AOI_Core 子进程 pid=%s", _proc.pid)

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _proc.poll() is not None:
            return False, (f"AOI_Core 子进程退出 code={_proc.returncode}，"
                           f"见 {settings.logs_dir / 'aoi_core_child.log'}")
        if core_healthy(url):
            return True, f"已启动 AOI_Core 子进程 pid={_proc.pid}：{url}"
        time.sleep(0.5)
    return False, f"AOI_Core 启动超时（{wait_s:.0f}s）：{url}"


def stop_core() -> None:
    """只关闭本进程拉起的 Core。"""
    global _proc
    with _lock:
        p, _proc = _proc, None
    if p is None or p.poll() is not None:
        return
    try:
        p.terminate()
        p.wait(timeout=8)
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
