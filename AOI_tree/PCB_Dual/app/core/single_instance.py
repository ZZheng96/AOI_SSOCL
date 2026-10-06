"""单实例防护（P2：UI / 内嵌服务双入口防双开）。

Windows 原生 msvcrt.locking 对 storage/{name}.lock 非阻塞锁 1 字节；
文件句柄随进程退出自动释放，无需显式解锁。
"""
from __future__ import annotations

import logging
import msvcrt
from pathlib import Path
from typing import Optional, TextIO

log = logging.getLogger(__name__)

_lock_handle: Optional[TextIO] = None


def acquire_lock(name: str) -> bool:
    """尝试获取名为 name 的单实例锁；成功返回 True（进程退出自动释放），已占用返回 False。"""
    global _lock_handle
    from app.config import get_settings
    lock_path = get_settings().storage("locks") / f"{name}.lock"
    try:
        fh = open(lock_path, "a+b")
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log.warning("[single_instance] 已有实例在运行（lock: %s）", lock_path)
        try:
            fh.close()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
        return False
    fh.seek(0)
    _lock_handle = fh
    return True


def lock_path_for(name: str) -> Path:
    """返回锁文件路径（用于提示信息）。"""
    from app.config import get_settings
    return get_settings().storage("locks") / f"{name}.lock"


def ensure_single_instance(name: str) -> None:
    """获取失败则打印提示并退出（供无 UI 的入口使用）。"""
    if not acquire_lock(name):
        print(f"已有实例在运行（lock: {lock_path_for(name)}）")
        raise SystemExit(2)
