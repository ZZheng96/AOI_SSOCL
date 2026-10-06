from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import RECIPES_DIR
from app.preprocess.engine import PreprocessParams


@dataclass
class RecipeRecord:
    scope: str
    category: str | None = None
    algorithm: str | None = None
    preprocess: dict = field(default_factory=lambda: PreprocessParams().to_dict())
    version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RecipeRecord":
        return cls(
            scope=data.get("scope", "global_default"),
            category=data.get("category"),
            algorithm=data.get("algorithm"),
            preprocess=data.get("preprocess") or PreprocessParams().to_dict(),
            version=int(data.get("version", 1)),
        )


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


class RecipeStore:
    """v1.1: category+algorithm > category_default > global_default"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or RECIPES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_global_default()

    def _ensure_global_default(self) -> None:
        path = self.root / "global_default.json"
        if not path.exists():
            record = RecipeRecord(
                scope="global_default",
                preprocess=PreprocessParams(color="NONE").to_dict(),
            )
            self._write(path, record)

    def _write(self, path: Path, record: RecipeRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _read(self, path: Path) -> RecipeRecord | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RecipeRecord.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None

    def _path_for(
        self,
        scope: str,
        category: str | None = None,
        algorithm: str | None = None,
    ) -> Path:
        category = _norm(category)
        algorithm = _norm(algorithm)

        if scope == "global_default":
            return self.root / "global_default.json"
        if scope == "algorithm_default":
            return self.root / "algorithm" / f"{algorithm}.json"
        if scope == "category_default":
            return self.root / "category" / f"{category}.json"
        if scope == "category_algorithm":
            return self.root / "category_algorithm" / f"{category}__{algorithm}.json"
        raise ValueError(f"unknown scope: {scope}")

    def save(
        self,
        scope: str,
        params: PreprocessParams | dict,
        category: str | None = None,
        algorithm: str | None = None,
    ) -> RecipeRecord:
        preprocess = params.to_dict() if isinstance(params, PreprocessParams) else dict(params)
        record = RecipeRecord(
            scope=scope,
            category=_norm(category),
            algorithm=_norm(algorithm),
            preprocess=preprocess,
        )
        path = self._path_for(scope, record.category, record.algorithm)
        self._write(path, record)
        return record

    def resolve(
        self,
        category: str | None = None,
        algorithm: str | None = None,
    ) -> tuple[RecipeRecord, str]:
        """v1.3 起不再区分"类别"，优先级：算法专属（已保存全局覆盖） > 全局默认。
        仍兼容旧的 category_* 记录（如果存在且传入了 category），但检测页
        已不再传 category，因此这两层实际上不会再被命中。"""
        category = _norm(category)
        algorithm = _norm(algorithm)

        candidates: list[tuple[Path, str]] = []
        if category and algorithm:
            candidates.append(
                (
                    self._path_for("category_algorithm", category=category, algorithm=algorithm),
                    f"类别+算法（{category} / {algorithm}）",
                )
            )
        if algorithm:
            candidates.append(
                (
                    self._path_for("algorithm_default", algorithm=algorithm),
                    f"算法专属（{algorithm}）",
                )
            )
        if category:
            candidates.append(
                (
                    self._path_for("category_default", category=category),
                    f"类别默认（{category}）",
                )
            )
        candidates.append((self._path_for("global_default"), "全局默认"))

        for path, desc in candidates:
            record = self._read(path)
            if record is not None:
                return record, desc

        fallback = RecipeRecord(scope="global_default", preprocess=PreprocessParams().to_dict())
        return fallback, "全局默认（内存回退）"

    def clear(self, scope: str, category: str | None = None, algorithm: str | None = None) -> None:
        path = self._path_for(scope, category, algorithm)
        if path.exists():
            path.unlink()

    def get_saved(self, scope: str, category: str | None = None, algorithm: str | None = None) -> RecipeRecord | None:
        return self._read(self._path_for(scope, category, algorithm))

    def list_all(self) -> list[dict[str, Any]]:
        items = []
        for path in sorted(self.root.rglob("*.json")):
            record = self._read(path)
            if record:
                items.append({"path": str(path.relative_to(self.root)), **record.to_dict()})
        return items
