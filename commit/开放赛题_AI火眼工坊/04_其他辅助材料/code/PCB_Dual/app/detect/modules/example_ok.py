"""示例检测项：仅用于验证 Catalog 发现机制。

复制本文件、改 id / Manifest / run，即可在建模与检测列表中出现，无需改 Page。
默认不作为产线项（trigger=manual）；契约测试会直接实例化。
"""
from __future__ import annotations

from typing import Any

from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DetectRequest


class ExampleOkAdapter(BaseAdapter):
    algorithm_ids = ["example_ok"]
    requires_standard = False

    def manifests(self) -> list[ModuleManifest]:
        return [
            ModuleManifest(
                id="example_ok",
                display_name="示例OK（接入样例）",
                group_id="example",
                group_name="示例模块",
                requires_standard=False,
                region_kind="none",
                color="#64748b",
                algorithm_version="0.1.0",
                trigger="manual",
                is_job_item=False,
            )
        ]

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return []

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        _ = algorithm_id, template_name, rois
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        _ = req, params
        return AlgorithmResult(
            algorithm=algorithm_id,
            ok=True,
            message="示例模块：始终 OK",
            status="OK",
            display_name="示例OK（接入样例）",
        )


MODULE = ExampleOkAdapter
