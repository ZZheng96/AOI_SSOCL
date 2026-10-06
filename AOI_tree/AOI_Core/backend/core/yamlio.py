"""YAML 配置原子写回（A16，2026-10-03）。

此前多处直接 `open(path,"w") + yaml.safe_dump`：进程崩溃/断电会在目标文件
已被截断、新内容未写全的窗口留下半个 yaml，下次启动 safe_load 即崩。
改为「写临时文件 + os.replace 原子替换」：任一时刻磁盘上要么是旧完整
配置、要么是新完整配置，不存在截断中间态。同目录临时文件保证 os.replace
为同卷 rename（原子）。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


def save_yaml_atomic(path: str | Path, cfg: Dict[str, Any]) -> None:
    """原子写回 yaml：先写同目录临时文件，fsync 后 os.replace 替换目标。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # 同卷原子替换
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
