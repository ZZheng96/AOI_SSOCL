"""pickle 反序列化加固（2026-10-03 评审问题 A10）。

pickle.load 可执行任意字节码——上传/篡改的 .pkl 即 RCE。本模块把全算法层
的 pickle 反序列化收口到唯一入口，强制两点：
1. 路径白名单：仅允许加载白名单根目录内的文件，其余直接抛
   PickleSecurityError 拒绝反序列化。默认白名单为 algo/assets（随包权重，
   如全局类型判别器 defect_clf.pkl）；引擎快照根由 backend 启动时经
   register_pickle_root 注册；临时场景可用环境变量 AOI_PICKLE_EXTRA_ROOTS
   （os.pathsep 分隔）追加。
2. 强制留痕：每次加载记录路径与字节数，事后可审计。
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, List

logger = logging.getLogger(__name__)


class PickleSecurityError(RuntimeError):
    """pickle 路径白名单外拒绝反序列化。"""


_REGISTERED_ROOTS: List[str] = []


def _norm(p: str) -> str:
    """realpath 解符号链接 + normcase 兼容 Windows 大小写。"""
    return os.path.normcase(os.path.realpath(p))


def register_pickle_root(root: str) -> None:
    """注册额外的合法 pickle 根目录（幂等）。引擎启动时注册 snap_root。"""
    r = _norm(root)
    if r not in _REGISTERED_ROOTS:
        _REGISTERED_ROOTS.append(r)


def asset_roots() -> List[str]:
    """随包资源根（查找优先级）：<包根>/assets → <包根>/../assets（AOI_tree 共享）→ algo/assets。"""
    algo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_root = os.path.dirname(algo_dir)
    return [os.path.join(pkg_root, "assets"),
            os.path.join(os.path.dirname(pkg_root), "assets"),
            os.path.join(algo_dir, "assets")]


def _allowed_roots() -> List[str]:
    roots = [_norm(r) for r in asset_roots()]
    roots.extend(_REGISTERED_ROOTS)
    extra = os.environ.get("AOI_PICKLE_EXTRA_ROOTS", "")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(_norm(item))
    return roots


def is_pickle_path_allowed(path: str) -> bool:
    rp = _norm(path)
    for root in _allowed_roots():
        try:
            if os.path.commonpath([rp, root]) == root:
                return True
        except ValueError:  # 跨盘符（Windows）
            continue
    return False


def safe_pickle_load(path: str) -> Any:
    """白名单校验 + 留痕后反序列化；白名单外抛 PickleSecurityError。"""
    if not is_pickle_path_allowed(path):
        logger.warning("[safe_pickle] 拒绝加载白名单外 pickle：%s", path)
        raise PickleSecurityError(
            f"拒绝反序列化白名单外 pickle：{path}（白名单根：{_allowed_roots()}）")
    rp = os.path.realpath(path)
    size = os.path.getsize(rp)
    logger.info("[safe_pickle] 加载 pickle：%s（%d 字节）", rp, size)
    with open(rp, "rb") as f:
        return pickle.load(f)
