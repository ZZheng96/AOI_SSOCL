"""检测项目录：由模块 Manifest 生成，UI / 调度只问 Catalog。

发现顺序：
1. 已加载适配器的 ``manifests()``
2. 扫描 ``app/detect/modules/`` 下声明了 ``MODULE`` 的插件
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.detect.adapters.base import BaseAdapter
from app.detect.contract import ModuleManifest


@dataclass
class Catalog:
    manifests: dict[str, ModuleManifest] = field(default_factory=dict)
    adapters: dict[str, BaseAdapter] = field(default_factory=dict)
    load_errors: dict[str, str] = field(default_factory=dict)

    def get(self, item_id: str) -> ModuleManifest | None:
        return self.manifests.get(item_id)

    def require(self, item_id: str) -> ModuleManifest:
        m = self.get(item_id)
        if m is None:
            return ModuleManifest(
                id=item_id,
                display_name=item_id,
                group_id="unknown",
                group_name="其他",
            )
        return m

    def items(self, *, job_only: bool = True, trigger: str | None = None) -> list[ModuleManifest]:
        out = []
        for m in self.manifests.values():
            if job_only and not m.is_job_item:
                continue
            if trigger and m.trigger != trigger:
                continue
            out.append(m)
        return out

    def groups(self, *, job_only: bool = True) -> list[tuple[str, str, list[ModuleManifest]]]:
        buckets: dict[str, list[ModuleManifest]] = defaultdict(list)
        names: dict[str, str] = {}
        order: list[str] = []
        for m in self.items(job_only=job_only):
            if m.group_id not in names:
                names[m.group_id] = m.group_name
                order.append(m.group_id)
            buckets[m.group_id].append(m)
        return [(gid, names[gid], buckets[gid]) for gid in order]

    def labels(self) -> dict[str, str]:
        return {m.id: m.display_name for m in self.manifests.values()}

    def color(self, item_id: str) -> str:
        m = self.get(item_id)
        return m.color if m else "#dc2626"

    def region_kind(self, item_id: str) -> str:
        m = self.get(item_id)
        return (m.region_kind if m else "") or "none"

    def group_name(self, item_id: str) -> str:
        return self.require(item_id).group_name

    def group_id(self, item_id: str) -> str:
        return self.require(item_id).group_id

    def requires_standard(self, item_ids: list[str]) -> bool:
        for item_id in item_ids:
            m = self.get(item_id)
            if m is not None:
                if m.requires_standard:
                    return True
                continue
            adapter = self.adapters.get(item_id)
            if adapter is None or getattr(adapter, "requires_standard", True):
                return True
        return False

    def shared_param_id(self, item_id: str) -> str | None:
        m = self.get(item_id)
        return m.shared_param_id if m else None

    def recommended_ids(self, category: str) -> list[str]:
        if not category:
            return self.default_selected_ids()
        ids = [m.id for m in self.items(job_only=True) if category in m.recommended_categories]
        return ids or self.default_selected_ids()

    def default_selected_ids(self) -> list[str]:
        return [m.id for m in self.items(job_only=True) if m.default_selected]

    def is_job_item(self, item_id: str) -> bool:
        m = self.get(item_id)
        return bool(m is None or m.is_job_item)


_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = build_catalog()
    return _catalog


def refresh_catalog() -> Catalog:
    global _catalog
    _catalog = build_catalog()
    return _catalog


def build_catalog() -> Catalog:
    from app.detect.registry import all_adapters, load_errors

    cat = Catalog()
    cat.load_errors = dict(load_errors())
    adapters = all_adapters()
    cat.adapters = dict(adapters)

    for alg_id, adapter in adapters.items():
        manifests = []
        if hasattr(adapter, "manifests"):
            try:
                manifests = list(adapter.manifests() or [])
            except Exception as exc:  # noqa: BLE001
                cat.load_errors[f"manifest:{alg_id}"] = str(exc)
                manifests = []
        by_id = {m.id: m for m in manifests if getattr(m, "id", None)}
        if alg_id in by_id:
            cat.manifests[alg_id] = by_id[alg_id]
        elif alg_id not in cat.manifests:
            cat.manifests[alg_id] = _fallback_manifest(alg_id, adapter)
        for m in by_id.values():
            cat.manifests.setdefault(m.id, m)
            cat.adapters.setdefault(m.id, adapter)

    return cat


def _fallback_manifest(alg_id: str, adapter: BaseAdapter) -> ModuleManifest:
    label = alg_id.split("_", 1)[-1] if "_" in alg_id else alg_id
    return ModuleManifest(
        id=alg_id,
        display_name=label,
        group_id="unknown",
        group_name="其他",
        requires_standard=bool(getattr(adapter, "requires_standard", True)),
    )
