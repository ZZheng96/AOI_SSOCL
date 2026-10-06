"""检测调度用的公共数据结构（供 scheduler / adapters / registry / gate 共用）。

在现有 AlgorithmResult / DetectSummary / DefectBox 上增字段、不推翻，
避免历史 result.json 断裂。ItemOutcome / JobOutcome 为语义别名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.detect.contract import Roi


@dataclass
class DefectBox:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float | None = None
    roi_id: str | None = None
    color: str = ""
    item_id: str = ""


@dataclass
class DetectRequest:
    image_test: np.ndarray
    image_std: np.ndarray | None = None
    image_std_raw: np.ndarray | None = None
    image_test_raw: np.ndarray | None = None
    category: str | None = None
    template_name: str | None = None
    algorithm: str = ""
    params: dict = field(default_factory=dict)
    rois: list[Roi] = field(default_factory=list)
    template_id: str | None = None
    template_version: int | None = None
    trigger: str = "on_job"
    preprocess_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmResult:
    algorithm: str
    ok: bool
    message: str
    boxes: list[DefectBox] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    preprocess_std: np.ndarray | None = None
    preprocess_test: np.ndarray | None = None
    hit_layer: str = ""
    diff_image: np.ndarray | None = None
    manual_verdict: str | None = None  # None | "confirmed_ng" | "false_positive"
    # 契约增补（旧归档可缺这些键）
    item_id: str = ""
    display_name: str = ""
    status: str = ""  # OK / NG / SKIP / ERROR
    defect_type: str = ""
    defect_count: int = 0
    elapsed_ms: int = 0
    algorithm_version: str = ""
    error_code: str | None = None
    error_message: str | None = None
    param_schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.item_id:
            self.item_id = self.algorithm
        if self.defect_count == 0 and self.boxes:
            self.defect_count = len(self.boxes)
        if not self.status:
            if self.error_code:
                self.status = "ERROR"
            elif self.skipped:
                self.status = "SKIP"
            else:
                self.status = "OK" if self.ok else "NG"


ItemOutcome = AlgorithmResult


@dataclass
class DetectSummary:
    overall_ok: bool
    results: list[AlgorithmResult] = field(default_factory=list)
    vis_image: np.ndarray | None = None
    gate_blocked: bool = False
    gate_message: str = ""
    gate_note: str = ""
    overall: str = ""  # OK / NG / ERROR
    elapsed_ms: int = 0
    ng_count: int = 0
    template_id: str | None = None
    template_version: int | None = None
    source: str = "inspect"  # inspect | debug | trial
    stop_on_error: bool = False

    def __post_init__(self) -> None:
        if not self.overall:
            if self.gate_blocked:
                self.overall = "ERROR"
            elif any(r.status == "ERROR" or r.error_code for r in self.results):
                self.overall = "ERROR"
            elif self.overall_ok:
                self.overall = "OK"
            else:
                self.overall = "NG"
        if self.ng_count == 0:
            self.ng_count = len(self.ng_list)

    @property
    def ng_list(self) -> list[AlgorithmResult]:
        return [r for r in self.results if (not r.ok) and (not r.skipped) and r.status != "ERROR"]

    @property
    def error_list(self) -> list[AlgorithmResult]:
        return [r for r in self.results if r.status == "ERROR" or r.error_code]


JobOutcome = DetectSummary
