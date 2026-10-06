"""检测算法参数配方：模板+算法 > 类别+算法 > 算法默认。

跟 :class:`app.recipe.store.RecipeStore`（管预处理配方）是平行的一套存储，
只是这里管的是"每个 algorithm_id 的可调检测参数"（比如插件焊点孔洞的
`void_min_area`），不是预处理链路。"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import RECIPES_DIR

_SAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _safe(name: str) -> str:
    return _SAFE_RE.sub("_", name.strip()) if name else ""


@dataclass
class ParamRecord:
    scope: str
    template_name: str | None = None
    category: str | None = None
    algorithm: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ParamRecord":
        return cls(
            scope=data.get("scope", "algorithm_default"),
            template_name=data.get("template_name"),
            category=data.get("category"),
            algorithm=data.get("algorithm", ""),
            params=dict(data.get("params") or {}),
            version=int(data.get("version", 1)),
        )


class ParamStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or RECIPES_DIR) / "params"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, scope: str, template_name: str | None, category: str | None, algorithm: str) -> Path:
        alg = _safe(algorithm)
        if scope == "template_algorithm":
            return self.root / "template_algorithm" / f"{_safe(template_name or '')}__{alg}.json"
        if scope == "category_algorithm":
            return self.root / "category_algorithm" / f"{_safe(category or '')}__{alg}.json"
        if scope == "algorithm_default":
            return self.root / "algorithm_default" / f"{alg}.json"
        raise ValueError(f"unknown scope: {scope}")

    def _read(self, path: Path) -> ParamRecord | None:
        if not path.exists():
            return None
        try:
            return ParamRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None

    def _write(self, path: Path, record: ParamRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def save(
        self,
        scope: str,
        algorithm: str,
        params: dict[str, Any],
        template_name: str | None = None,
        category: str | None = None,
    ) -> ParamRecord:
        record = ParamRecord(
            scope=scope,
            template_name=_safe(template_name) if template_name else None,
            category=_safe(category) if category else None,
            algorithm=algorithm,
            params=dict(params),
        )
        self._write(self._path(scope, template_name, category, algorithm), record)
        return record

    def clear(self, scope: str, algorithm: str, template_name: str | None = None, category: str | None = None) -> None:
        path = self._path(scope, template_name, category, algorithm)
        if path.exists():
            path.unlink()

    def resolve(
        self,
        template_name: str | None,
        category: str | None,
        algorithm: str,
        algorithm_defaults: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """返回 (最终生效参数, 命中层级描述)。永远以 algorithm_defaults 为底，
        逐层用已保存的覆盖值叠加，缺的键继续用默认值（不要求用户存全量参数）。
        层级优先级：模板+算法 > 类别+算法 > 算法默认（已保存的全局覆盖）>
        算法默认（代码内置）。"""
        merged = dict(algorithm_defaults or {})
        hit = "算法默认（内置）"

        record_ad = self._read(self._path("algorithm_default", None, None, algorithm))
        if record_ad is not None:
            merged.update(record_ad.params)
            hit = "算法默认（已保存全局覆盖）"

        record = self._read(self._path("category_algorithm", None, category, algorithm)) if category else None
        if record is not None:
            merged.update(record.params)
            hit = f"类别配方（{category} / {algorithm}）"

        record_t = (
            self._read(self._path("template_algorithm", template_name, None, algorithm))
            if template_name
            else None
        )
        if record_t is not None:
            merged.update(record_t.params)
            hit = f"模板配方（{template_name} / {algorithm}）"

        return merged, hit

    def get_saved(
        self, scope: str, algorithm: str, template_name: str | None = None, category: str | None = None
    ) -> dict[str, Any] | None:
        record = self._read(self._path(scope, template_name, category, algorithm))
        return dict(record.params) if record is not None else None

    def export_for_template(
        self,
        template_name: str | None,
        algorithm_ids: list[str],
        *,
        category: str | None = None,
        defaults_by_alg: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """把当前 resolve 结果冻结为模板内 params 字典。"""
        defaults_by_alg = defaults_by_alg or {}
        out: dict[str, dict[str, Any]] = {}
        for alg_id in algorithm_ids:
            defaults = dict(defaults_by_alg.get(alg_id) or {})
            resolved, _hit = self.resolve(template_name, category, alg_id, defaults)
            out[alg_id] = resolved
        return out

    def import_from_template(
        self,
        template_name: str,
        params_by_alg: dict[str, dict[str, Any]],
    ) -> None:
        """把模板内冻结参数写回 template_algorithm 层（兼容调试页）。"""
        for alg_id, params in params_by_alg.items():
            if not alg_id or not isinstance(params, dict):
                continue
            self.save("template_algorithm", alg_id, params, template_name=template_name)
