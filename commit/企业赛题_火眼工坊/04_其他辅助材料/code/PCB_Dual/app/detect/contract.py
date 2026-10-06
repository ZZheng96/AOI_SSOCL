"""检测契约：模块 Manifest、标准输入 Context、结果字段约定。

主程序只依赖本模块与 Catalog；禁止按 algorithm_id 前缀写业务分支。
契约版本 ``INSPECT_CONTRACT = 1``。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

INSPECT_CONTRACT = 1

Trigger = Literal["on_job", "live", "manual", "external"]
ItemStatus = Literal["OK", "NG", "SKIP", "ERROR"]
JobStatus = Literal["OK", "NG", "ERROR"]


@dataclass
class ModuleManifest:
    """检测项自描述。内容由模块填写，UI / 调度只读字段。"""

    id: str
    display_name: str
    group_id: str
    group_name: str
    requires_standard: bool = True
    region_kind: str = "none"  # body / th / smt / gold / none
    color: str = "#dc2626"
    algorithm_version: str = "1.0.0"
    shared_param_id: str | None = None
    trigger: Trigger = "on_job"
    inspect_contract: int = INSPECT_CONTRACT
    #: False = 参数组（如焊锡共用），不作为产线检测项执行
    is_job_item: bool = True
    default_selected: bool = False
    recommended_categories: tuple[str, ...] = ()
    param_schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Roi:
    id: str
    kind: str
    shape: dict[str, Any] = field(default_factory=dict)
    layer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "layer": self.layer, "shape": dict(self.shape)}


@dataclass
class DetectionContext:
    """统一输入：模块不得再靠全局文件名猜标定。"""

    item_id: str
    template_id: str | None = None
    template_version: int | None = None
    image_test: np.ndarray | None = None
    image_std: np.ndarray | None = None
    image_test_raw: np.ndarray | None = None
    image_std_raw: np.ndarray | None = None
    rois: list[Roi] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    preprocess_meta: dict[str, Any] = field(default_factory=dict)
    trigger: Trigger = "on_job"
    category: str | None = None
    template_name: str | None = None


def item_status(*, ok: bool, skipped: bool = False, error_code: str | None = None) -> ItemStatus:
    if error_code:
        return "ERROR"
    if skipped:
        return "SKIP"
    return "OK" if ok else "NG"


def job_status(*, gate_blocked: bool, items: list[Any]) -> JobStatus:
    if gate_blocked:
        return "ERROR"
    if any(getattr(r, "status", None) == "ERROR" or getattr(r, "error_code", None) for r in items):
        return "ERROR"
    if any((not getattr(r, "ok", True)) and (not getattr(r, "skipped", False)) for r in items):
        return "NG"
    return "OK"
