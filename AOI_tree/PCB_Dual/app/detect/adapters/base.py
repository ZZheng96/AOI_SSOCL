"""适配器统一接口：每个外部算法仓库包一个 Adapter。

契约：``manifests / param_specs / default_params / validate_params /
is_ready / run``。主程序只问 Catalog 与本接口，不写死算法列表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DetectRequest


@dataclass
class ParamSpec:
    """一个可调参数的界面展示信息。由模块声明，UI 只认识 kind。"""

    key: str
    label: str
    kind: str  # "int" | "float" | "enum" | "bool"
    default: Any = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    hint: str = ""
    choices: list[str] = field(default_factory=list)
    tolerance: dict[str, Any] | None = None


class BaseAdapter:
    """算法适配器基类。子类实现 ``manifests`` / ``param_specs`` / ``run``。"""

    algorithm_ids: list[str] = []
    requires_standard: bool = True

    def manifests(self) -> list[ModuleManifest]:
        return []

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return []

    def default_params(self, algorithm_id: str) -> dict[str, Any]:
        return {p.key: p.default for p in self.param_specs(algorithm_id)}

    def migrate_params(self, algorithm_id: str, old: dict[str, Any] | None) -> dict[str, Any]:
        """未知键忽略，缺键填 default。"""
        specs = {p.key: p for p in self.param_specs(algorithm_id)}
        merged = dict(self.default_params(algorithm_id))
        for key, value in (old or {}).items():
            if key in specs:
                merged[key] = value
        return merged

    def validate_params(self, algorithm_id: str, params: dict[str, Any]) -> tuple[bool, str]:
        specs = {p.key: p for p in self.param_specs(algorithm_id)}
        if not specs:
            return True, ""
        for key, spec in specs.items():
            if key not in params:
                continue
            val = params[key]
            if spec.kind == "enum" and spec.choices and str(val) not in spec.choices:
                return False, f"{spec.label} 取值不在允许列表：{val}"
            if spec.kind in ("int", "float"):
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    return False, f"{spec.label} 不是有效数字：{val}"
                if num < spec.minimum or num > spec.maximum:
                    return False, f"{spec.label} 超出范围 [{spec.minimum}, {spec.maximum}]"
        return True, ""

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        """只根据 rois / 依赖回答是否可跑，不要靠全局文件名猜标定。"""
        _ = algorithm_id, template_name, rois
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        raise NotImplementedError
