"""外部算法仓库导入沙箱。

三个外部仓库（插件焊点 / 贴片锡焊 / 金手指）各自维护自己的一套顶层包
（比如插件焊点和贴片锡焊都各有一个自己的 ``algorithms`` 包、自己的一份
``core`` 契约拷贝），如果同时把它们的目录加进 ``sys.path`` 后各自
``import algorithms``，后一次导入会命中前一次已经缓存在
``sys.modules["algorithms"]`` 里的模块，从而互相顶掉，导入到错误的包内容。

``IsolatedImport`` 提供一个"导入即清场"的上下文管理器：进入时把需要的目录
插入 ``sys.path`` 并清空可能冲突的顶层模块名缓存；退出时把这些目录移出
``sys.path``，再次清空这些顶层模块名缓存。真正需要长期持有的类/函数/实例，
调用方应在 ``with`` 块内部取出引用后缓存到自己的模块级变量里——Python 对
已经拿到手的对象引用不受 ``sys.modules`` 清理影响，只有"以后再 import
同名模块"才会被这次清理掉，所以每个仓库只需要做一次隔离导入。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable


def _purge(prefixes: Iterable[str]) -> None:
    for name in list(sys.modules.keys()):
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
                break


class IsolatedImport:
    """临时把 ``extra_paths`` 加入 ``sys.path`` 并清空 ``top_level_names``
    对应的模块缓存；退出时还原。用法：

        with IsolatedImport([repo_dir], ["algorithms", "core"]):
            mod = importlib.import_module("algorithms.xxx.yyy")
            MyClass = mod.MyClass  # 在 with 块内取出引用
        # 出了 with 块，sys.path / sys.modules 已还原，MyClass 仍可用
    """

    def __init__(self, extra_paths: Iterable[Path | str], top_level_names: Iterable[str]):
        self.extra_paths = [str(p) for p in extra_paths]
        self.top_level_names = list(top_level_names)
        self._added: list[str] = []

    def __enter__(self) -> "IsolatedImport":
        self._added = []
        for p in self.extra_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
                self._added.append(p)
        _purge(self.top_level_names)
        importlib.invalidate_caches()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for p in self._added:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
        _purge(self.top_level_names)
        return False
