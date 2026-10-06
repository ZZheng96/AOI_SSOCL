from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from core.entities.detect import (
    AlgorithmResult as PartyAlgorithmResult,
    DetectPartBox,
    ResultType,
)
from core.entities.algorithms import IDetectionAlgorithm
from core.adapter_runtime import (
    RunSaver,
    draw_result,
    legacy_defects_to_summary,
)


class ComponentPartyAlg(IDetectionAlgorithm):
    """7 项元件算法的公共基类。

    - 提供空实现 _record_step（detect_roi 会调用，本链路不需要中间图）
    - 子类仍需实现 default_config / run
    """

    _process_dir: Optional[str] = None
    _process_images: List[Dict[str, str]] = []

    def _record_step(self, name: str, image: "np.ndarray"):
        """本链路不保存中间步骤图，空实现。"""
        return


class _LegacyConfig:
    """把 cfg dict 包装成属性访问对象。

    保留的 detect_roi 内部使用 getattr(config, key, default) 读取阈值；
    本类让 dict 也能走 getattr 语义，缺失键返回 default。
    """

    def __init__(self, cfg: Dict[str, Any]):
        object.__setattr__(self, "_cfg", cfg or {})

    def __getattr__(self, name: str) -> Any:
        cfg = object.__getattribute__(self, "_cfg")
        if name in cfg:
            return cfg[name]
        raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return (object.__getattribute__(self, "_cfg")).get(name)


def make_legacy_config(cfg: Dict[str, Any]) -> _LegacyConfig:
    return _LegacyConfig(cfg)


def _preprocess(image: np.ndarray, template: np.ndarray):
    """统一预处理：尺寸对齐 + 转 BGR 三通道。返回 (image_bgr, template_bgr)。"""
    h, w = image.shape[:2]
    if image.shape != template.shape:
        template = cv2.resize(template, (w, h))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if len(template.shape) == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    return image, template


def run_party_shell(
    algorithm_code: str,
    self_defaults: Dict[str, Any],
    image: np.ndarray,
    config: Dict[str, Any],
    original_template_image: Optional[np.ndarray],
    detect_core: Callable[[np.ndarray, np.ndarray, _LegacyConfig], Any],
    *,
    merge_config_fn,
    save_to_disk: bool = True,
) -> PartyAlgorithmResult:
    """run() 外壳通用实现。

    参数:
        algorithm_code: 算法唯一标识
        self_defaults: self._defaults（实例默认参数）
        image: 测试图
        config: 外部覆盖参数 dict
        original_template_image: 模板图
        detect_core: (image_bgr, template_bgr, legacy_cfg) -> 旧版 AlgorithmResult
        merge_config_fn: core.utils.alg_config.merge_config
        save_to_disk: 是否落盘留档
    """
    t0 = time.perf_counter()
    try:
        # ---- 1. 输入校验 ----
        if image is None or image.size == 0:
            return _fail_result(algorithm_code, "empty image", t0)
        if original_template_image is None or original_template_image.size == 0:
            # 无模板：原逻辑返回 OK，转为 code=0 空结果
            return PartyAlgorithmResult(
                code=0,
                message="ok",
                algorithm_code=algorithm_code,
                cost_time=time.perf_counter() - t0,
                result_type=ResultType.PARTS,
                parts=[],
                metadata={"mode": "no_template"},
            )

        # ---- 2. 合并参数 + 预处理 ----
        cfg = merge_config_fn(self_defaults, config)
        full_image_bgr, full_template_bgr = _preprocess(image, original_template_image)

        # 可选全局降采样：proc_max_side>0 时，按最长边缩到该上限再检测，缺陷框最后还原回原图。
        # 默认 0=关闭（检测结果与原分辨率完全一致）。极限提速时按需开启（精度换速度）。
        proc_max_side = int(cfg.get("proc_max_side", 0) or 0)
        proc_scale = 1.0
        det_image, det_template = full_image_bgr, full_template_bgr
        fh, fw = full_image_bgr.shape[:2]
        if proc_max_side > 0 and max(fh, fw) > proc_max_side:
            proc_scale = proc_max_side / float(max(fh, fw))
            new_w = max(1, int(round(fw * proc_scale)))
            new_h = max(1, int(round(fh * proc_scale)))
            det_image = cv2.resize(full_image_bgr, (new_w, new_h))
            det_template = cv2.resize(full_template_bgr, (new_w, new_h))
        # 把缩放因子透传给检测核心：含像素绝对阈值的算法（移位 dx/dy、破损 blob 面积）据此换算
        cfg["_proc_scale"] = proc_scale
        legacy_cfg = make_legacy_config(cfg)

        # ---- 3. 调用保留的检测核心 ----
        legacy_result = detect_core(det_image, det_template, legacy_cfg)

        # ---- 3b. 缺陷框从检测分辨率还原回原图分辨率 ----
        legacy_defects = getattr(legacy_result, "defects", None) or []
        if proc_scale != 1.0:
            inv = 1.0 / proc_scale
            for d in legacy_defects:
                bb = getattr(d, "bounding_box", None)
                if bb is not None:
                    bb.x *= inv; bb.y *= inv; bb.width *= inv; bb.height *= inv

        # ---- 4. 结果转换: DefectInfo -> DetectPartBox ----
        parts: List[DetectPartBox] = []
        for d in legacy_defects:
            bb = getattr(d, "bounding_box", None)
            parts.append(DetectPartBox(
                x=int(bb.x) if bb is not None else 0,
                y=int(bb.y) if bb is not None else 0,
                width=int(bb.width) if bb is not None else 0,
                height=int(bb.height) if bb is not None else 0,
                confidence=float(getattr(d, "confidence", 0.0) or 0.0),
                label=getattr(d, "defect_type", "defect"),
                box_type="defect",
                metadata={"severity": float(getattr(d, "severity", 0.0) or 0.0),
                          "description": getattr(d, "description", "")},
            ))

        # ---- 5. 可视化（在原分辨率图上画还原后的框）----
        image_bgr, template_bgr = full_image_bgr, full_template_bgr
        vis = draw_result(full_image_bgr, legacy_defects)
        status = getattr(legacy_result, "status", "OK")
        cost = time.perf_counter() - t0

        # 落盘留档已禁用
        # if save_to_disk:
        #     result_dict = {
        #         "algorithm_code": algorithm_code,
        #         "code": 0,
        #         "status": status,
        #         "defect_count": len(parts),
        #         "defects": legacy_defects_to_summary(
        #             getattr(legacy_result, "defects", None)),
        #         "cost_time": cost,
        #     }
        #     try:
        #         RunSaver.save(algorithm_code, image_bgr, template_bgr, vis, result_dict)
        #     except Exception:
        #         pass  # 落盘失败不影响返回

        # ---- 6. 返回 AlgorithmResult ----
        meta = {
            "output_image": vis,
            "status": status,
            "defect_count": len(parts),
        }
        # 保留 legacy metadata 中的非图字段（如 polarity_checks / skip / algo_version）
        legacy_meta = getattr(legacy_result, "metadata", None) or {}
        for k, v in legacy_meta.items():
            if k not in meta and not isinstance(v, np.ndarray):
                meta[k] = v

        return PartyAlgorithmResult(
            code=0,
            message="ok",
            algorithm_code=algorithm_code,
            cost_time=cost,
            result_type=ResultType.PARTS,
            parts=parts,
            metadata=meta,
        )
    except Exception as exc:
        return _fail_result(algorithm_code, str(exc), t0, exc=exc)


def _fail_result(algorithm_code: str, message: str, t0: float,
                 exc: Optional[BaseException] = None) -> PartyAlgorithmResult:
    meta: Dict[str, Any] = {"error": message}
    if exc is not None:
        meta["traceback"] = traceback.format_exc()
    return PartyAlgorithmResult(
        code=1,
        message=message,
        algorithm_code=algorithm_code,
        cost_time=time.perf_counter() - t0,
        result_type=ResultType.ERROR,
        metadata=meta,
    )
