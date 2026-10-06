"""元件本体检测适配器：包装 `元件本体检测` 仓库的 `PCBDefectDetector`
（错件/缺件/极反/移位/立碑/翻件/破损 共 7 类）。

参数元数据直接读取 PCB_APP 的 ``pcb_app.metadata``，与验证台 UI 保持一致；
检测入口为 ``PCBDefectDetector.detect(algorithm_code, template, test, config)``。

若模板已标定 body_rois，则按 ROI 裁剪后逐区送检（当前算法包不支持 roi=
参数，应用侧裁剪并把缺陷框坐标偏移回整图）。未框选时退回整图检测。
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Any, Mapping

import numpy as np

from app.calibration.body_roi import get_body_rois
from app.config import BODY_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

CODE_BY_ALG: dict[str, str] = {
    "body_错件": "component_wrong_part",
    "body_缺件": "component_missing",
    "body_极反": "component_reverse_polarity",
    "body_移位": "component_shift",
    "body_立碑": "component_tombstone",
    "body_翻件": "component_flipped",
    "body_破损": "component_damage",
}

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _normalize_roi(roi: Mapping[str, Any] | None, img_w: int, img_h: int):
    if not roi:
        return None
    shape = str(roi.get("shape", "rect")).lower()
    if shape == "circle":
        cx = float(roi.get("cx", 0.0))
        cy = float(roi.get("cy", 0.0))
        r = float(roi.get("r", 0.0))
        if r <= 0:
            return None
        x0, y0, w, h = cx - r, cy - r, 2.0 * r, 2.0 * r
    else:
        x0 = float(roi.get("x", 0.0))
        y0 = float(roi.get("y", 0.0))
        w = float(roi.get("w", 0.0))
        h = float(roi.get("h", 0.0))
        if w <= 0 or h <= 0:
            return None
    xi = max(0, int(round(x0)))
    yi = max(0, int(round(y0)))
    x2 = min(img_w, int(round(x0 + w)))
    y2 = min(img_h, int(round(y0 + h)))
    if x2 - xi < 2 or y2 - yi < 2:
        return None
    cw, ch = x2 - xi, y2 - yi
    mask = None
    if shape == "circle":
        yy, xx = np.ogrid[:ch, :cw]
        ccx = float(roi.get("cx", 0.0)) - xi
        ccy = float(roi.get("cy", 0.0)) - yi
        r = float(roi.get("r", 0.0))
        mask = ((xx - ccx) ** 2 + (yy - ccy) ** 2 <= r * r).astype(np.uint8) * 255
    return (xi, yi, cw, ch, mask)


def _crop_with_roi(img: np.ndarray, norm_roi) -> np.ndarray:
    xi, yi, cw, ch, mask = norm_roi
    crop = img[yi : yi + ch, xi : xi + cw].copy()
    if mask is not None:
        crop[mask == 0] = 0
    return crop


def _apply_roi(template: np.ndarray, test: np.ndarray, roi: Mapping[str, Any] | None):
    if not roi:
        return template, test, 0, 0
    norm = _normalize_roi(roi, test.shape[1], test.shape[0])
    if norm is None:
        return template, test, 0, 0
    off_x, off_y = norm[0], norm[1]
    tst = _crop_with_roi(test, norm)
    tnorm = _normalize_roi(roi, template.shape[1], template.shape[0])
    tpl = _crop_with_roi(template, tnorm if tnorm is not None else norm)
    return tpl, tst, off_x, off_y


def _load_metadata_module():
    import sys

    meta_path = BODY_REPO_DIR / "PCB_APP" / "pcb_app" / "metadata.py"
    if not meta_path.is_file():
        raise FileNotFoundError(f"找不到本体算法元数据: {meta_path}")
    mod_name = "body_defect_metadata"
    spec = importlib.util.spec_from_file_location(mod_name, meta_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载本体算法元数据: {meta_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 在装饰阶段会查 sys.modules[cls.__module__]，必须先登记
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_body() -> dict[str, Any]:
    with _lock:
        if _cache:
            return _cache
        pkg_root = BODY_REPO_DIR / "pcb_defect" / "pcb_defect_detector"
        if not (pkg_root / "pcb_defect_detector" / "__init__.py").is_file():
            raise FileNotFoundError(f"找不到 pcb_defect_detector 包: {pkg_root}")

        meta_mod = _load_metadata_module()
        with IsolatedImport([pkg_root], ["core", "algorithms", "pcb_defect_detector", "utils"]):
            det_mod = __import__("pcb_defect_detector", fromlist=["PCBDefectDetector"])
            detector = det_mod.PCBDefectDetector()
            available = set(detector.available_algorithms())
            _cache["detector"] = detector
            _cache["available"] = available
            _cache["metas"] = dict(meta_mod.DEFECT_BY_CODE)
        return _cache


def _param_to_spec(p) -> ParamSpec | None:
    # hidden / advanced：界面不展示，检测时仍走 metadata 默认值
    if getattr(p, "hidden", False) or getattr(p, "advanced", False):
        return None
    kind = p.kind
    tolerance = getattr(p, "tolerance", None)
    if isinstance(tolerance, dict) and tolerance:
        tolerance = {str(k): v for k, v in tolerance.items()}
    else:
        tolerance = None
    if kind == "bool":
        return ParamSpec(
            key=p.key,
            label=p.label,
            kind="enum",
            default="true" if p.default else "false",
            choices=["true", "false"],
            hint=p.hint or "",
            tolerance=tolerance,
        )
    if kind == "choice":
        return ParamSpec(
            key=p.key,
            label=p.label,
            kind="enum",
            default=str(p.default),
            choices=list(p.choices or []),
            hint=p.hint or "",
            tolerance=tolerance,
        )
    if kind not in ("int", "float"):
        return None
    return ParamSpec(
        key=p.key,
        label=p.label,
        kind=kind,
        default=p.default,
        minimum=float(p.minimum),
        maximum=float(p.maximum),
        step=float(p.step),
        hint=p.hint or "",
        tolerance=tolerance,
    )


def _coerce_config(meta, params: dict[str, Any]) -> dict[str, Any]:
    """合并判据默认值 + UI 参数；bool 枚举转回真正 bool。"""
    cfg = meta.default_config() if meta is not None else {}
    cfg.update(params)
    for key, val in list(cfg.items()):
        if isinstance(val, str) and val.strip().lower() in ("true", "false"):
            cfg[key] = val.strip().lower() == "true"
    return cfg


_BODY_DISPLAY = {
    "body_错件": "错件",
    "body_缺件": "缺件",
    "body_极反": "极反",
    "body_移位": "移位",
    "body_立碑": "立碑",
    "body_翻件": "翻件",
    "body_破损": "破损",
}

_BODY_RECOMMEND = {
    "body_错件": ("电阻", "电容", "二极管", "LED", "IC"),
    "body_缺件": ("电阻", "电容", "二极管", "LED", "IC"),
    "body_极反": ("二极管", "LED"),
    "body_移位": ("电阻", "电容", "二极管", "LED", "IC"),
    "body_立碑": ("电阻", "电容"),
    "body_翻件": ("电容",),
    "body_破损": ("IC",),
}


class BodyAdapter(BaseAdapter):
    algorithm_ids = list(CODE_BY_ALG.keys())
    requires_standard = True

    def __init__(self) -> None:
        _load_body()

    def manifests(self) -> list[ModuleManifest]:
        return [
            ModuleManifest(
                id=alg_id,
                display_name=label,
                group_id="body",
                group_name="元件本体检测",
                requires_standard=True,
                region_kind="body",
                color="#0f766e",
                algorithm_version="1.0.0",
                trigger="on_job",
                default_selected=alg_id in ("body_错件", "body_缺件", "body_移位", "body_破损"),
                recommended_categories=_BODY_RECOMMEND.get(alg_id, ()),
            )
            for alg_id, label in _BODY_DISPLAY.items()
        ]

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        code = CODE_BY_ALG.get(algorithm_id)
        if not code:
            return []
        body = _load_body()
        meta = body["metas"].get(code)
        if meta is None:
            return []
        specs: list[ParamSpec] = []
        for p in meta.params:
            spec = _param_to_spec(p)
            if spec is not None:
                specs.append(spec)
        return specs

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        code = CODE_BY_ALG.get(algorithm_id)
        body = _load_body()
        if code not in body["available"]:
            return False, f"算法包未成功加载该项：{code}"
        has_roi = bool(
            [r for r in (rois or []) if r.kind == "body" or r.layer == "body"]
        ) or bool(get_body_rois(template_name))
        if not has_roi:
            return False, "需在标定中框选元件本体 ROI（缺件/错件/移位等须逐元件判定）"
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        code = CODE_BY_ALG.get(algorithm_id)
        if code is None:
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=False,
                message=f"未知的本体缺陷类型: {algorithm_id}",
            )

        body = _load_body()
        meta = body["metas"].get(code)
        label = meta.name if meta else algorithm_id
        if code not in body["available"]:
            # 算法包未加载该项属内部失败（非配置缺失），不能静默 SKIP
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=False,
                status="ERROR",
                error_code="ALG_NOT_LOADED",
                error_message=f"算法包未加载该项: {code}",
                message=f"[错误] {label}：算法包未加载该项（{code}）",
            )

        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test
        if std_bgr is None or test_bgr is None:
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=False,
                message=f"[适配器] {label}：缺少标准图或测试图",
            )

        cfg = _coerce_config(meta, params)
        if req.rois:
            rois = [r.shape for r in req.rois if r.kind == "body" or r.layer == "body"]
        else:
            rois = get_body_rois(req.template_name)
        # 未框选 -> 整图；有框选 -> 逐 ROI 裁剪送检后合并
        jobs: list[tuple[str, Mapping[str, Any] | None]] = (
            [(f"ROI{i + 1}", r) for i, r in enumerate(rois)] if rois else [("整图", None)]
        )

        all_boxes: list[DefectBox] = []
        statuses: list[str] = []
        errors: list[str] = []
        last_vis = None
        for roi_name, roi in jobs:
            tpl_in, tst_in, ox, oy = _apply_roi(std_bgr, test_bgr, roi)
            try:
                result = body["detector"].detect(code, tpl_in, tst_in, config=cfg)
            except Exception as exc:  # noqa: BLE001
                return AlgorithmResult(
                    algorithm=algorithm_id,
                    ok=False,
                    message=f"[适配器异常] {label}: {exc}",
                )
            if result.status == "ERROR":
                errors.append(f"{roi_name}:{result.error or '引擎错误'}")
                if result.output_image is not None:
                    last_vis = result.output_image
                continue
            statuses.append(str(result.status).upper())
            for d in result.defects or []:
                all_boxes.append(
                    DefectBox(
                        x=int(d.x) + ox,
                        y=int(d.y) + oy,
                        w=int(d.width),
                        h=int(d.height),
                        label=f"{label}·{roi_name}" if rois else label,
                    )
                )
            if result.output_image is not None:
                last_vis = result.output_image

        if not statuses and errors:
            return AlgorithmResult(
                algorithm=algorithm_id,
                ok=False,
                status="ERROR",
                error_code="ALG_ERROR",
                error_message="; ".join(errors),
                message=f"[错误] {label}：{errors[0]}",
                diff_image=last_vis,
            )

        ok = "NG" not in statuses
        status_text = "NG" if not ok else ("OK" if statuses else "ERROR")
        roi_note = f"，区域={len(rois)}" if rois else ""
        err_note = f"，部分区域失败={len(errors)}" if errors else ""
        message = f"{label} 判定={status_text}，缺陷数={len(all_boxes)}{roi_note}{err_note}"
        return AlgorithmResult(
            algorithm=algorithm_id,
            ok=ok,
            message=message,
            boxes=all_boxes,
            diff_image=last_vis,
        )
