"""插件焊点检测适配器：包装 `PCB Through-Hole Solder Inspection` 仓库。

复用该仓库自带的 `pcb_solder_app.metadata.DEFECT_METAS` 作为参数元数据来源，
检测入口是
`algorithms.pcb_through_hole.pcb_through_hole_solder_alg.PcbThroughHoleSolderAlg.run()`。

仓库更新后：
- 算法目录在 ``detect/src``，契约参考在 ``detect/contract_reference``；
- 五类缺陷：孔洞 / 少锡 / 多锡 / **连锡** / 不出脚；
- ≥2 个焊点框时算法只跑连锡，盘内四类需单焊点或自动锡面模式。
"""
from __future__ import annotations

import importlib
import threading
from typing import Any

from app.calibration.th_roi import get_th_pad_count, th_run_config, th_run_config_from_pads
from app.config import TH_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

DEFECT_ID_BY_ALG: dict[str, int] = {
    "tht_孔洞": 12,
    "tht_少锡": 13,
    "tht_多锡包锡": 14,
    "tht_连锡": 15,
    "tht_不出脚": 16,
}

DEFECT_ID_BRIDGE = 15
_PRESET_KEYS = ("void_preset", "insuf_preset")

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load_th() -> dict[str, Any]:
    with _lock:
        if _cache:
            return _cache
        src_dir = TH_REPO_DIR / "detect" / "src"
        contract_dir = TH_REPO_DIR / "detect" / "contract_reference"
        app_dir = TH_REPO_DIR / "app"
        with IsolatedImport(
            [src_dir, contract_dir, app_dir],
            ["core", "algorithms"],
        ):
            alg_mod = importlib.import_module(
                "algorithms.pcb_through_hole.pcb_through_hole_solder_alg"
            )
            metadata_mod = importlib.import_module("pcb_solder_app.metadata")

            _cache["klass"] = alg_mod.PcbThroughHoleSolderAlg
            _cache["instance"] = alg_mod.PcbThroughHoleSolderAlg()
            _cache["defect_by_id"] = dict(metadata_mod.DEFECT_BY_ID)
            _cache["src_dir"] = src_dir
            _cache["contract_dir"] = contract_dir
            _cache["app_dir"] = app_dir
        return _cache


def _th_runtime():
    """每次真正跑算法时重新挂上 TH 的 algorithms，避免被其它仓库同名包顶掉。"""
    th = _load_th()
    return IsolatedImport(
        [th["src_dir"], th["contract_dir"], th["app_dir"]],
        ["core", "algorithms"],
    )


_TH_DISPLAY = {
    "tht_孔洞": "孔洞",
    "tht_少锡": "少锡",
    "tht_多锡包锡": "多锡(包锡)",
    "tht_连锡": "连锡",
    "tht_不出脚": "不出脚",
}


def _th_pad_count(template_name: str | None, rois: list[Roi] | None) -> int:
    if rois:
        return len([r for r in rois if r.kind in ("th", "") or r.layer == "th"])
    return get_th_pad_count(template_name)


class ThroughHoleAdapter(BaseAdapter):
    algorithm_ids = list(DEFECT_ID_BY_ALG.keys())
    requires_standard = True

    def __init__(self) -> None:
        _load_th()

    def manifests(self) -> list[ModuleManifest]:
        return [
            ModuleManifest(
                id=alg_id,
                display_name=label,
                group_id="through_hole",
                group_name="插件焊点检测",
                requires_standard=True,
                region_kind="th",
                color="#15803d",
                algorithm_version="2.1.0",
                trigger="on_job",
                recommended_categories=("插件器件",),
            )
            for alg_id, label in _TH_DISPLAY.items()
        ]

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        th = _load_th()
        defect_id = DEFECT_ID_BY_ALG.get(algorithm_id)
        meta = th["defect_by_id"].get(defect_id)
        if meta is None:
            return []
        specs: list[ParamSpec] = []
        if meta.preset_key:
            specs.append(
                ParamSpec(
                    key=meta.preset_key,
                    label="灵敏度挡位",
                    kind="enum",
                    default="MED",
                    choices=["LOW", "MED", "HIGH"],
                    hint="LOW=宽松(少报) / MED=标准(默认) / HIGH=严格(多报)",
                )
            )
        for p in meta.params:
            specs.append(
                ParamSpec(
                    key=p.key,
                    label=p.label,
                    kind=p.kind,
                    default=p.default,
                    minimum=p.minimum,
                    maximum=p.maximum,
                    step=p.step,
                    hint=p.hint,
                )
            )
        return specs

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        defect_id = DEFECT_ID_BY_ALG.get(algorithm_id)
        n_pads = _th_pad_count(template_name, rois)
        if defect_id == DEFECT_ID_BRIDGE:
            if n_pads < 2:
                return False, "连锡需要在「标定」中框选至少 2 个焊点区域"
            return True, ""
        if n_pads >= 2:
            return False, "当前标定了多个焊点，算法只会跑连锡；盘内缺陷请改用单框或自动锡面"
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        th = _load_th()
        defect_id = DEFECT_ID_BY_ALG.get(algorithm_id)
        meta = th["defect_by_id"].get(defect_id)
        label = meta.name if meta else algorithm_id
        if defect_id is None:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"未知的插件焊点缺陷类型: {algorithm_id}")

        ready, reason = self.is_ready(algorithm_id, req.template_name, rois=list(req.rois or []))
        if not ready:
            return AlgorithmResult(
                algorithm=algorithm_id, ok=True, skipped=True, skip_reason=reason,
                message=f"[跳过] {label}：{reason}",
            )

        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test
        if std_bgr is None or test_bgr is None:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"[适配器] {label}：缺少标准图或测试图")

        advanced = {k: v for k, v in params.items() if k not in _PRESET_KEYS}
        cfg: dict[str, Any] = {
            "template_id": req.template_name,
            "enabled_defects": [defect_id],
            "advanced": advanced,
        }
        pad_shapes = [r.shape for r in (req.rois or []) if r.kind in ("th", "") or r.layer == "th"]
        if pad_shapes:
            cfg.update(th_run_config_from_pads(pad_shapes))
        else:
            cfg.update(th_run_config(req.template_name))
        if "void_preset" in params:
            cfg["void_preset"] = params["void_preset"]
        if "insuf_preset" in params:
            cfg["insuf_preset"] = params["insuf_preset"]

        try:
            with _th_runtime():
                # 每次在隔离环境里新建实例，避免其它仓库顶掉 algorithms 后旧实例延迟 import 失败
                alg_mod = importlib.import_module(
                    "algorithms.pcb_through_hole.pcb_through_hole_solder_alg"
                )
                instance = alg_mod.PcbThroughHoleSolderAlg()
                result = instance.run(test_bgr, cfg, original_template_image=std_bgr)
        except Exception as exc:  # noqa: BLE001
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"[适配器异常] {label}: {exc}")

        if result.code != 0:
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=False,
                status="ERROR",
                error_code="ALG_ERROR",
                error_message=result.message,
                message=f"[错误] {label}：{result.message}",
            )

        meta_out = result.metadata or {}
        status = meta_out.get("status", "OK")
        boxes = [
            DefectBox(x=int(p.x), y=int(p.y), w=int(p.width), h=int(p.height), label=p.label or label)
            for p in result.parts
        ]
        vis = meta_out.get("output_image")

        if status == "REVIEW":
            reason = meta_out.get("review_reason") or "配准/覆盖率不可靠，需人工复检"
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=True,
                skipped=True,
                skip_reason=reason,
                message=f"[待复检] {label}：{reason}",
                diff_image=vis,
            )

        coverage = meta_out.get("coverage")
        coverage_txt = f"{coverage:.2f}" if isinstance(coverage, (int, float)) else "-"
        ok = status != "NG"
        message = f"{label} 判定={status}，缺陷数={meta_out.get('num_defects', 0)}，锡面覆盖率={coverage_txt}"
        return AlgorithmResult(algorithm=algorithm_id, ok=ok, message=message, boxes=boxes, diff_image=vis)
