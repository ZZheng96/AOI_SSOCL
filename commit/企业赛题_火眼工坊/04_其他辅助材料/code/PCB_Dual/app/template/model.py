"""InspectionTemplate：标准图为基本单位的检测配方。

关联链：标准图 → 模板 → 检测项 → 检测区域
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.template.regions import (
    calibration_from_region_sets,
    count_regions,
    region_kind_for,
    region_sets_from_calibration,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class StandardImageRef:
    path: str = ""
    sha1: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "StandardImageRef":
        data = data or {}
        return cls(
            path=str(data.get("path") or ""),
            sha1=str(data.get("sha1") or ""),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
        )


@dataclass
class DetectionItem:
    """模板内一项检测：缺陷类型 + 绑定的区域种类 + 参数。"""

    algorithm_id: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    preprocess_recipe: dict[str, Any] | None = None
    region_kind: str = ""

    def __post_init__(self) -> None:
        if not self.region_kind:
            self.region_kind = region_kind_for(self.algorithm_id) or ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "enabled": self.enabled,
            "params": dict(self.params),
            "preprocess_recipe": deepcopy(self.preprocess_recipe),
            "region_kind": self.region_kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionItem":
        return cls(
            algorithm_id=str(data.get("algorithm_id") or ""),
            enabled=bool(data.get("enabled", True)),
            params=dict(data.get("params") or {}),
            preprocess_recipe=deepcopy(data.get("preprocess_recipe")),
            region_kind=str(data.get("region_kind") or ""),
        )


# 兼容旧代码引用
TemplateAlgorithm = DetectionItem


@dataclass
class GateConfig:
    size_gate_enabled: bool = True
    size_gate_max_diff_px: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "GateConfig":
        data = data or {}
        return cls(
            size_gate_enabled=bool(data.get("size_gate_enabled", True)),
            size_gate_max_diff_px=int(data.get("size_gate_max_diff_px", 10)),
        )


@dataclass
class InspectionTemplate:
    id: str
    display_name: str = ""
    version: int = 1
    status: str = "draft"  # draft | published
    created_at: str = ""
    updated_at: str = ""
    standard_image: StandardImageRef = field(default_factory=StandardImageRef)
    category: str = ""
    #: 检测项（缺陷类型）；序列化主字段为 detection_items，algorithms 为兼容别名
    detection_items: list[DetectionItem] = field(default_factory=list)
    #: 检测区域：按种类存放几何（body/th/smt_*/gold）
    region_sets: dict[str, list[dict]] = field(default_factory=dict)
    #: 适配器兼容用的 legacy 标定快照（由 region_sets 同步）
    calibration: dict[str, Any] = field(default_factory=dict)
    gate: GateConfig = field(default_factory=GateConfig)
    notes: str = ""
    #: 与模板一一对应，固定为 template.id，避免多模板共享标定
    calib_key: str = ""
    #: v2 双检融合模式：traditional=仅传统算法 / feature=仅特征学习 / dual=双检并行
    #: 任一 NG 即 NG。AOI 特征引擎按模板 category 关联品类模型。
    engine_mode: str = "dual"
    #: v2 绑定的 AOI 品类模型名（与 PCB 分类 category 解耦；空则回退用 category）
    model_category: str = ""
    #: v2 双坐标标定（pxPerMM/原点，mm）；独立于 ROI 标定 calibration
    px_calibration: dict = field(default_factory=dict)
    #: v2 CAD 器件表（P2：位号/物料号/封装/坐标 mm/角度）
    cad_items: list = field(default_factory=list)
    #: v2 拼板配置（P2：panel_dx/dy/cols/rows/rotation）
    panels: dict = field(default_factory=dict)
    #: v2 FOV 划分（P2：网格 row/col/x/y）
    fov_grid: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id
        # 强制 calib_key 与模板 id 一致，保证一图一模板一标定
        self.calib_key = self.id
        if not self.created_at:
            self.created_at = _now_iso()
        if not self.updated_at:
            self.updated_at = self.created_at
        if not self.region_sets and self.calibration:
            self.region_sets = region_sets_from_calibration(self.calibration)
        if self.region_sets and not self.calibration:
            self.calibration = calibration_from_region_sets(self.region_sets)

    # ---- 兼容属性：algorithms ----
    @property
    def algorithms(self) -> list[DetectionItem]:
        return self.detection_items

    @algorithms.setter
    def algorithms(self, value: list[DetectionItem]) -> None:
        self.detection_items = list(value or [])

    def enabled_algorithm_ids(self) -> list[str]:
        return [a.algorithm_id for a in self.detection_items if a.enabled and a.algorithm_id]

    def enabled_detection_items(self) -> list[DetectionItem]:
        return [a for a in self.detection_items if a.enabled and a.algorithm_id]

    def params_map(self) -> dict[str, dict[str, Any]]:
        return {a.algorithm_id: dict(a.params) for a in self.detection_items if a.algorithm_id}

    def set_algorithm_ids(self, algorithm_ids: list[str], *, keep_params: bool = True) -> None:
        existing = {a.algorithm_id: a for a in self.detection_items}
        next_items: list[DetectionItem] = []
        for alg_id in algorithm_ids:
            if keep_params and alg_id in existing:
                prev = existing[alg_id]
                next_items.append(
                    DetectionItem(
                        algorithm_id=alg_id,
                        enabled=True,
                        params=dict(prev.params),
                        preprocess_recipe=deepcopy(prev.preprocess_recipe),
                        region_kind=prev.region_kind or (region_kind_for(alg_id) or ""),
                    )
                )
            else:
                next_items.append(DetectionItem(algorithm_id=alg_id, enabled=True))
        self.detection_items = next_items

    def upsert_params(self, algorithm_id: str, params: dict[str, Any]) -> None:
        for alg in self.detection_items:
            if alg.algorithm_id == algorithm_id:
                alg.params = dict(params)
                alg.enabled = True
                return
        self.detection_items.append(
            DetectionItem(algorithm_id=algorithm_id, enabled=True, params=dict(params))
        )

    def set_region_shapes(self, target: str, shapes: list[dict]) -> None:
        self.region_sets[target] = [dict(s) for s in shapes]
        self.sync_calibration_from_regions()

    def get_region_shapes(self, target: str) -> list[dict]:
        return [dict(s) for s in (self.region_sets.get(target) or [])]

    def sync_calibration_from_regions(self) -> None:
        self.calibration = calibration_from_region_sets(self.region_sets, base=self.calibration)

    def sync_regions_from_calibration(self) -> None:
        self.region_sets = region_sets_from_calibration(self.calibration)

    def association_summary(self) -> dict[str, Any]:
        items = self.enabled_detection_items()
        kinds = sorted({i.region_kind for i in items if i.region_kind})
        return {
            "standard_image": self.standard_image.path,
            "template_id": self.id,
            "detection_item_count": len(items),
            "region_kinds": kinds,
            "region_count": count_regions(self.region_sets),
        }

    def rule_snapshot(self) -> dict[str, Any]:
        """写入结果归档的规则摘要，便于追溯。"""
        return {
            "template_id": self.id,
            "template_version": self.version,
            "display_name": self.display_name,
            "category": self.category,
            "calib_key": self.calib_key,
            "detection_items": [a.to_dict() for a in self.detection_items if a.enabled],
            "algorithms": [a.to_dict() for a in self.detection_items if a.enabled],
            "region_sets": deepcopy(self.region_sets),
            "calibration": deepcopy(self.calibration),
            "gate": self.gate.to_dict(),
        }

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        self.calib_key = self.id
        self.sync_calibration_from_regions()
        item_dicts = [a.to_dict() for a in self.detection_items]
        return {
            "id": self.id,
            "display_name": self.display_name,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "standard_image": self.standard_image.to_dict(),
            "category": self.category,
            "detection_items": item_dicts,
            # 兼容旧读取方
            "algorithms": item_dicts,
            "region_sets": deepcopy(self.region_sets),
            "calibration": deepcopy(self.calibration),
            "gate": self.gate.to_dict(),
            "notes": self.notes,
            "calib_key": self.calib_key,
            "engine_mode": self.engine_mode,
            "model_category": self.model_category,
            "px_calibration": self.px_calibration,
            "cad_items": deepcopy(self.cad_items),
            "panels": deepcopy(self.panels),
            "fov_grid": deepcopy(self.fov_grid),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InspectionTemplate":
        raw_items = data.get("detection_items") or data.get("algorithms") or []
        items = [DetectionItem.from_dict(a) for a in raw_items if isinstance(a, dict)]
        region_sets = data.get("region_sets")
        if isinstance(region_sets, dict):
            parsed_sets = {
                str(k): [dict(s) for s in (v or []) if isinstance(s, dict)]
                for k, v in region_sets.items()
            }
        else:
            parsed_sets = {}
        calibration = dict(data.get("calibration") or {})
        if not parsed_sets and calibration:
            parsed_sets = region_sets_from_calibration(calibration)
        tid = str(data.get("id") or "")
        return cls(
            id=tid,
            display_name=str(data.get("display_name") or data.get("id") or ""),
            version=int(data.get("version") or 1),
            status=str(data.get("status") or "draft"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            standard_image=StandardImageRef.from_dict(data.get("standard_image")),
            category=str(data.get("category") or ""),
            detection_items=items,
            region_sets=parsed_sets,
            calibration=calibration,
            gate=GateConfig.from_dict(data.get("gate")),
            notes=str(data.get("notes") or ""),
            calib_key=tid,
            engine_mode=str(data.get("engine_mode") or "dual"),
            model_category=str(data.get("model_category") or ""),
            px_calibration=dict(data.get("px_calibration") or {}),
            cad_items=list(data.get("cad_items") or []),
            panels=dict(data.get("panels") or {}),
            fov_grid=list(data.get("fov_grid") or []),
        )
