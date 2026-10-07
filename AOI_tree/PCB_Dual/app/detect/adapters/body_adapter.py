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


# ── 差分前置门控 ──
# 本体 7 类都是"测试图 vs 标准图"差分判定：若测试 ROI 经亚像素配准 + 亮度归一后
# 与标准 ROI 在噪声水平内一致，则不可能存在本体缺陷，直接判 OK，不再送算法包。
# 算法包内部的 body_mask/轮廓/估姿对 JPEG、噪声、亮度、1~2px 平移非常敏感（小元件尤甚），
# 门控只拦截"无变化"，有实际变化时仍走原算法，召回不受影响。
_GATE_MAX_SHIFT = 8.0     # 配准允许的最大平移（px），超出视为配准不可信，不门控
_GATE_DIFF = 22           # 平滑后灰度差阈值
_GATE_AREA = 15           # 差异连通域面积上限（px²），低于即视为无变化（算法包 blob_area_min=30）


def _gray32(img: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def _unchanged_vs_template(std: np.ndarray, test: np.ndarray, norm_roi) -> tuple[bool, str]:
    """测试图在 ROI 内是否与标准图一致（配准 + 亮度归一后）。返回 (一致, 说明)。"""
    import cv2

    if std.shape[:2] != test.shape[:2]:
        return False, "尺寸不一致"
    H, W = std.shape[:2]
    if norm_roi is None:
        xi, yi, cw, ch, mask = 0, 0, W, H, None
    else:
        xi, yi, cw, ch, mask = norm_roi
    # 在外扩上下文上估计平移，避免元件自身移位被配准吃掉
    pad = int(max(32, 0.5 * max(cw, ch)))
    x0, y0 = max(0, xi - pad), max(0, yi - pad)
    x1, y1 = min(W, xi + cw + pad), min(H, yi + ch + pad)
    g_ctx, t_ctx = _gray32(std[y0:y1, x0:x1]), _gray32(test[y0:y1, x0:x1])
    win = cv2.createHanningWindow((g_ctx.shape[1], g_ctx.shape[0]), cv2.CV_32F)
    (sx, sy), _resp = cv2.phaseCorrelate(g_ctx, t_ctx, win)
    if abs(sx) > _GATE_MAX_SHIFT or abs(sy) > _GATE_MAX_SHIFT:
        return False, f"配准偏移过大({sx:.1f},{sy:.1f})"
    g = cv2.GaussianBlur(g_ctx[yi - y0 : yi - y0 + ch, xi - x0 : xi - x0 + cw], (5, 5), 0)
    gm, gs = float(g.mean()), float(g.std())

    def _max_blob(dx: int, dy: int) -> int:
        # 整数偏移取块：getRectSubPix 起点为整数时不插值，越界部分复制边缘
        center = (xi - x0 + (cw - 1) / 2.0 + dx, yi - y0 + (ch - 1) / 2.0 + dy)
        t = cv2.GaussianBlur(cv2.getRectSubPix(t_ctx, (cw, ch), center), (5, 5), 0)
        ts = float(t.std())
        gain = min(1.25, max(0.8, gs / ts)) if ts > 1e-3 else 1.0  # 亮度归一：均值对齐 + 受限增益
        t = (t - float(t.mean())) * gain + gm
        binm = (cv2.absdiff(g, t) > _GATE_DIFF).astype(np.uint8)
        if mask is not None:
            binm[mask == 0] = 0
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0

    # phaseCorrelate 的亚像素质心在无位移时也可能给出 ±0.5，插值会制造假差异；
    # 只在零偏移与相邻整数偏移（floor/ceil 组合）处比较，取最小者（残余亚像素误差由 5x5 平滑吸收）
    import math

    cands = [(0, 0)] + [
        (dx, dy)
        for dx in sorted({math.floor(sx), math.ceil(sx)})
        for dy in sorted({math.floor(sy), math.ceil(sy)})
        if (dx, dy) != (0, 0)
    ]
    max_area = 1 << 30
    for dx, dy in cands:
        max_area = min(max_area, _max_blob(dx, dy))
        if max_area < _GATE_AREA:
            break
    return max_area < _GATE_AREA, f"偏移=({sx:.1f},{sy:.1f}) 最大差异块={max_area}px"


_gate_lock = threading.Lock()
_gate_memo: dict[tuple, tuple[np.ndarray, np.ndarray, tuple[bool, str]]] = {}


def _gate_cached(std: np.ndarray, test: np.ndarray, norm_roi) -> tuple[bool, str]:
    """同一次检测里 7 类本体算法共用同一 ROI 门控结果（按数组身份 + ROI 记忆）。"""
    key = (id(std), id(test)) + (tuple(norm_roi[:4]) if norm_roi is not None else ())
    with _gate_lock:
        hit = _gate_memo.get(key)
        if hit is not None and hit[0] is std and hit[1] is test:
            return hit[2]
    res = _unchanged_vs_template(std, test, norm_roi)
    with _gate_lock:
        if len(_gate_memo) >= 8:  # 持有图像引用，保持很小
            _gate_memo.clear()
        _gate_memo[key] = (std, test, res)
    return res


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
        gated = 0
        for roi_name, roi in jobs:
            norm = _normalize_roi(roi, test_bgr.shape[1], test_bgr.shape[0]) if roi else None
            if norm is not None:
                same, _why = _gate_cached(std_bgr, test_bgr, norm)
                if same:
                    statuses.append("OK")
                    gated += 1
                    continue
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
        gate_note = f"，与标准图一致={gated}" if gated else ""
        message = f"{label} 判定={status_text}，缺陷数={len(all_boxes)}{roi_note}{gate_note}{err_note}"
        return AlgorithmResult(
            algorithm=algorithm_id,
            ok=ok,
            message=message,
            boxes=all_boxes,
            diff_image=last_vis,
        )
