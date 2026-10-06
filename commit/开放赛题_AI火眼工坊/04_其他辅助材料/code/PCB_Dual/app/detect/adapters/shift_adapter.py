"""贴片元件移位检测适配器：包装 `shift_smt` 仓库的 `ShiftSmtAllAlg`
（shift2：贴片电容底座移位/缺件检测）。

一次 `run()` 同时完成底座定位、缺件判定与移位判定，输出 label 区分
``shift`` / ``misspart`` / ``base``（正常）。这里按系统的"一检测项一缺陷"
约定拆成两个检测项（移位、缺件），共用参数挂在 ``shift_共用参数`` 上。

底座框（base 框）由模板标定（``app.calibration.shift_base``），检测时按框
逐个送检；未标定时跳过。底座框既是滑窗模板尺寸，也是移位判定参考框。
"""
from __future__ import annotations

import importlib
import threading
from typing import Any

from app.calibration.shift_base import get_shift_bases
from app.config import SHIFT_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

SHIFT_SHARED_ID = "shift_共用参数"

LABEL_KEY_BY_ALG: dict[str, str] = {
    "shift_移位": "shift",
    "shift_缺件": "misspart",
}

LABEL_CN_BY_ALG: dict[str, str] = {
    "shift_移位": "移位",
    "shift_缺件": "缺件",
}

# bool 参数在 UI 上用 enum("true"/"false") 表示，传给算法前转回 bool。
_BOOL_KEYS = {"use_rotation", "use_autogamma"}


def _coerce_bools(cfg: dict[str, Any]) -> None:
    for key in _BOOL_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = cfg[key].strip().lower() == "true"


# 现场真正需要调的参数：底座 HSV 颜色范围 + 缺件/移位阈值 + 旋转/亮度开关。
# 深层滑窗参数（grid_size/mode/use_edges/edge_ratio/whole_weight/allow_out_of_bounds）
# 与 LUT 高级颜色查找表不暴露，保持算法默认。
_SHARED_SPECS: list[ParamSpec] = [
    ParamSpec("feature_h_low", "底座色相 HSV-H 下限", "int", 0, 0, 179, 1,
              "底座（元件本体）颜色范围的下限。配合上限框定底座颜色区间。"),
    ParamSpec("feature_h_high", "底座色相 HSV-H 上限", "int", 179, 0, 179, 1,
              "底座颜色范围上限。"),
    ParamSpec("feature_s_low", "底座饱和度 HSV-S 下限", "int", 0, 0, 255, 1,
              "饱和度下限，调低可识别更浅的底座。"),
    ParamSpec("feature_s_high", "底座饱和度 HSV-S 上限", "int", 147, 0, 255, 1,
              "饱和度上限。"),
    ParamSpec("feature_v_low", "底座亮度 HSV-V 下限", "int", 0, 0, 255, 1,
              "亮度下限，调低可识别更暗的底座。"),
    ParamSpec("feature_v_high", "底座亮度 HSV-V 上限", "int", 36, 0, 255, 1,
              "亮度上限，配合下限框定底座亮度区间。"),
    ParamSpec("feature_thresh", "缺件判定阈值", "float", 0.05, 0.0, 1.0, 0.01,
              "滑窗特征峰值低于此值判定为缺件。调大：更易报缺件；调小：更少报。"),
    ParamSpec("shift_thresh", "移位阈值（像素）", "float", 10.0, 1.0, 50.0, 0.5,
              "检测框中心与模板框中心距离超过此值判定为移位。调大：更少报；调小：更易报。"),
    ParamSpec("use_rotation", "启用旋转精修", "enum", "true", choices=["true", "false"],
              hint="开启后先轴对齐定位再小范围试角度，可检出带旋转的元件。"),
    ParamSpec("rotation_range", "旋转搜索范围（±度）", "float", 10.0, 1.0, 90.0, 1.0,
              "旋转精修的搜索角度范围。"),
    ParamSpec("rotation_step", "旋转搜索步长（度）", "float", 1.0, 0.1, 30.0, 0.1,
              "旋转精修的搜索步长，越小越精细但越慢。"),
    ParamSpec("use_autogamma", "启用自动 Gamma 亮度归一化", "enum", "false", choices=["true", "false"],
              hint="开启后对图像做亮度归一化，改善光照不一致时的底座提取。"),
]

_PARAM_SPECS: dict[str, list[ParamSpec]] = {
    SHIFT_SHARED_ID: _SHARED_SPECS,
    "shift_移位": [],
    "shift_缺件": [],
}

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load_shift() -> dict[str, Any]:
    with _lock:
        if _cache:
            return _cache
        src_dir = SHIFT_REPO_DIR / "src"
        contract_dir = SHIFT_REPO_DIR / "contract_reference"
        with IsolatedImport([src_dir, contract_dir], ["core", "algorithms"]):
            alg_mod = importlib.import_module(
                "algorithms.component_detect.shift_smt_all_alg"
            )
            _cache["instance"] = alg_mod.ShiftSmtAllAlg()
        return _cache


def _shift_bases(template_name: str | None, rois: list[Roi] | None) -> list[tuple[int, int, int, int]]:
    """底座框提取：优先 req.rois 的 shift 层，回退标定存储。"""
    rects: list[tuple[int, int, int, int]] = []
    if rois:
        for roi in rois:
            if roi.kind != "shift" and roi.layer != "shift" and roi.layer != "shift_base":
                continue
            shape = roi.shape or {}
            if str(shape.get("shape") or "rect") != "rect":
                continue
            try:
                x, y, w, h = (float(shape[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                rects.append((int(x), int(y), int(w), int(h)))
    if rects:
        return rects
    return get_shift_bases(template_name)


class ShiftAdapter(BaseAdapter):
    algorithm_ids = [SHIFT_SHARED_ID, *LABEL_KEY_BY_ALG.keys()]
    requires_standard = True

    def __init__(self) -> None:
        _load_shift()

    def manifests(self) -> list[ModuleManifest]:
        items = [
            ModuleManifest(
                id=SHIFT_SHARED_ID,
                display_name="移位共用参数",
                group_id="shift_smt",
                group_name="贴片元件移位检测",
                requires_standard=True,
                region_kind="shift",
                color="#0891b2",
                algorithm_version="1.0.0",
                trigger="manual",
                is_job_item=False,
            )
        ]
        for alg_id, label in LABEL_CN_BY_ALG.items():
            items.append(
                ModuleManifest(
                    id=alg_id,
                    display_name=label,
                    group_id="shift_smt",
                    group_name="贴片元件移位检测",
                    requires_standard=True,
                    region_kind="shift",
                    color="#0891b2",
                    algorithm_version="1.0.0",
                    shared_param_id=SHIFT_SHARED_ID,
                    trigger="on_job",
                    default_selected=True,
                    recommended_categories=("电阻", "电容", "二极管", "LED", "IC"),
                )
            )
        return items

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return list(_PARAM_SPECS.get(algorithm_id, []))

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        bases = _shift_bases(template_name, rois)
        if not bases:
            return False, "未标定底座框（base 框），请先在“标定”中框选元件底座区域"
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        if algorithm_id == SHIFT_SHARED_ID:
            return self._run_shared_preview(req, params)

        label_key = LABEL_KEY_BY_ALG.get(algorithm_id)
        label_cn = LABEL_CN_BY_ALG.get(algorithm_id, algorithm_id)
        if label_key is None:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"未知的移位缺陷类型: {algorithm_id}")

        bases = _shift_bases(req.template_name, list(req.rois or []))
        if not bases:
            reason = "未标定底座框（base 框）"
            return AlgorithmResult(
                algorithm=algorithm_id, ok=True, skipped=True, skip_reason=reason,
                message=f"[跳过] {label_cn}：{reason}，请先标定",
            )

        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test
        if test_bgr is None:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"[适配器] {label_cn}：缺少测试图")

        alg = _load_shift()
        cfg: dict[str, Any] = dict(alg["instance"].default_config())
        cfg.update(params)
        _coerce_bools(cfg)

        boxes: list[DefectBox] = []
        errors: list[str] = []
        last_vis = None
        for base in bases:
            try:
                result = alg["instance"].run(test_bgr, cfg, roi_bbox=list(base))
            except Exception as exc:  # noqa: BLE001
                return AlgorithmResult(algorithm=algorithm_id, ok=False, message=f"[适配器异常] {label_cn}: {exc}")

            if result.code == 2:  # 算法出错
                errors.append(str(result.metadata.get("error", "算法错误")))
                continue
            if label_key == "misspart":
                # 缺件：parts 里 label == "misspart"；缺件框用底座框位置（元件应在此却缺失）
                if any(getattr(p, "label", None) == "misspart" for p in result.parts):
                    boxes.append(DefectBox(x=base[0], y=base[1], w=base[2], h=base[3], label=label_cn))
            else:
                # 移位：metadata.is_shifted；缺陷框用检测到的底座实际位置
                if result.metadata.get("is_shifted"):
                    bb = result.metadata.get("base_box") or list(base)
                    boxes.append(DefectBox(x=int(bb[0]), y=int(bb[1]), w=int(bb[2]), h=int(bb[3]), label=label_cn))
            vis = result.metadata.get("output_image")
            if vis is not None:
                last_vis = vis

        if bases and len(errors) == len(bases):
            return AlgorithmResult(
                algorithm=algorithm_id, ok=False, status="ERROR",
                error_code="ALG_ERROR", error_message="; ".join(errors),
                message=f"[错误] {label_cn}：所有底座框算法执行失败（{errors[0]}）",
                diff_image=last_vis,
            )
        ok = len(boxes) == 0
        err_note = f"，部分底座框失败={len(errors)}" if errors else ""
        message = f"{label_cn} 判定={'OK' if ok else 'NG'}，缺陷数={len(boxes)}，底座框={len(bases)}{err_note}"
        return AlgorithmResult(algorithm=algorithm_id, ok=ok, message=message, boxes=boxes, diff_image=last_vis)

    def _run_shared_preview(self, req: DetectRequest, shared_params: dict[str, Any]) -> AlgorithmResult:
        """共用参数预览：对每个底座框同时跑一次，把移位+缺件结果一起画出来。"""
        bases = _shift_bases(req.template_name, list(req.rois or []))
        if not bases:
            reason = "未标定底座框（base 框）"
            return AlgorithmResult(
                algorithm=SHIFT_SHARED_ID, ok=True, skipped=True, skip_reason=reason,
                message=f"[跳过] 共用参数预览：{reason}，请先标定",
            )

        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test
        if test_bgr is None:
            return AlgorithmResult(algorithm=SHIFT_SHARED_ID, ok=False, message="[适配器] 共用参数预览：缺少测试图")

        alg = _load_shift()
        cfg: dict[str, Any] = dict(alg["instance"].default_config())
        cfg.update(shared_params)
        _coerce_bools(cfg)

        boxes: list[DefectBox] = []
        errors: list[str] = []
        last_vis = None
        for base in bases:
            try:
                result = alg["instance"].run(test_bgr, cfg, roi_bbox=list(base))
            except Exception as exc:  # noqa: BLE001
                return AlgorithmResult(algorithm=SHIFT_SHARED_ID, ok=False, message=f"[适配器异常] 共用参数预览: {exc}")
            if result.code == 2:
                # 算法内部失败要计数，全部失败时透出 ERROR，不得静默判 OK
                errors.append(str(result.metadata.get("error", "算法错误")))
                continue
            if any(getattr(p, "label", None) == "misspart" for p in result.parts):
                boxes.append(DefectBox(x=base[0], y=base[1], w=base[2], h=base[3], label="缺件"))
            elif result.metadata.get("is_shifted"):
                bb = result.metadata.get("base_box") or list(base)
                boxes.append(DefectBox(x=int(bb[0]), y=int(bb[1]), w=int(bb[2]), h=int(bb[3]), label="移位"))
            vis = result.metadata.get("output_image")
            if vis is not None:
                last_vis = vis

        if bases and len(errors) == len(bases):
            return AlgorithmResult(
                algorithm=SHIFT_SHARED_ID, ok=False, status="ERROR",
                error_code="ALG_ERROR", error_message="; ".join(errors),
                message=f"[错误] 共用参数预览：所有底座框算法执行失败（{errors[0]}）",
                diff_image=last_vis,
            )
        ok = len(boxes) == 0
        err_note = f"，部分底座框失败={len(errors)}" if errors else ""
        message = f"共用参数预览 判定={'OK' if ok else 'NG'}，缺陷数={len(boxes)}，底座框={len(bases)}{err_note}"
        return AlgorithmResult(algorithm=SHIFT_SHARED_ID, ok=ok, message=message, boxes=boxes, diff_image=last_vis)
