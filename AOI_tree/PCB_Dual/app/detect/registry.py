"""algorithm_id -> 适配器 注册表。

加载来源：
1. 内置 adapters 包（失败记 load_errors，不拖垮应用）
2. ``app/detect/modules/`` 下声明 ``MODULE`` 的插件（Catalog 也会扫一遍）

产线 ``allow_stub=False``：未加载的项记 ERROR，不跑占位结果。
"""
from __future__ import annotations

from app.detect.adapters.base import BaseAdapter

_adapters: dict[str, BaseAdapter] = {}
_load_errors: dict[str, str] = {}
_initialized = False


def _try_load(name: str, factory) -> None:
    try:
        adapter = factory()
    except Exception as exc:  # noqa: BLE001 - 外部仓库加载失败不能拖垮应用
        _load_errors[name] = str(exc)
        return
    ids = list(getattr(adapter, "algorithm_ids", None) or [])
    try:
        for m in adapter.manifests() or []:
            if m.id and m.id not in ids:
                ids.append(m.id)
    except Exception:
        pass
    for alg_id in ids:
        _adapters[alg_id] = adapter


def _ensure_init() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    def _smt():
        from app.detect.adapters.smt_adapter import SmtAdapter

        return SmtAdapter()

    def _gold():
        from app.detect.adapters.gold_finger_adapter import GoldFingerAdapter

        return GoldFingerAdapter()

    def _wrinkle():
        from app.detect.adapters.wrinkle_adapter import WrinkleAdapter

        return WrinkleAdapter()

    def _body():
        from app.detect.adapters.body_adapter import BodyAdapter

        return BodyAdapter()

    def _shift():
        from app.detect.adapters.shift_adapter import ShiftAdapter

        return ShiftAdapter()

    def _th():
        from app.detect.adapters.through_hole_adapter import ThroughHoleAdapter

        return ThroughHoleAdapter()

    # 插件焊点检测（tht_* 五项）：算法已从 reference5 补齐（alg_repo/through_hole，
    # 37 图全量自测 OK=17/NG=20），恢复注册进 Catalog/UI。
    _try_load("贴片锡焊检测", _smt)
    _try_load("金手指/板面检测", _gold)
    _try_load("表皮起皱检测", _wrinkle)
    _try_load("元件本体检测", _body)
    _try_load("贴片元件移位检测", _shift)
    _try_load("插件焊点检测", _th)
    _discover_plugin_modules()


def _discover_plugin_modules() -> None:
    import importlib
    import pkgutil
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent / "modules"
    if not pkg_dir.is_dir():
        return
    try:
        import app.detect.modules as pkg
    except Exception as exc:  # noqa: BLE001
        _load_errors["detect.modules"] = str(exc)
        return

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod_name = f"{pkg.__name__}.{info.name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            _load_errors[mod_name] = str(exc)
            continue
        module_obj = getattr(mod, "MODULE", None)
        if module_obj is None:
            continue
        _try_load(mod_name, lambda obj=module_obj: obj() if isinstance(obj, type) else obj)


def get_adapter_for(algorithm_id: str) -> BaseAdapter | None:
    _ensure_init()
    return _adapters.get(algorithm_id)


def all_adapters() -> dict[str, BaseAdapter]:
    _ensure_init()
    return dict(_adapters)


def load_errors() -> dict[str, str]:
    _ensure_init()
    return dict(_load_errors)


def register_adapter(adapter: BaseAdapter) -> None:
    """测试 / 热加载：把适配器挂上注册表并刷新 Catalog。"""
    _ensure_init()
    ids = list(getattr(adapter, "algorithm_ids", None) or [])
    try:
        for m in adapter.manifests() or []:
            if m.id and m.id not in ids:
                ids.append(m.id)
    except Exception:
        pass
    for alg_id in ids:
        _adapters[alg_id] = adapter
    try:
        from app.detect.catalog import refresh_catalog

        refresh_catalog()
    except Exception:
        pass


def reload() -> None:
    """强制重新加载（供设置变更/仓库路径修改后手动刷新）。"""
    global _initialized
    _adapters.clear()
    _load_errors.clear()
    _initialized = False
    _ensure_init()
    try:
        from app.detect.catalog import refresh_catalog

        refresh_catalog()
    except Exception:
        pass
