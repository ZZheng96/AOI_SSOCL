"""插件焊点缺陷检测 - 算法控件封装"""
from __future__ import annotations

import hashlib
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.entities.detect import AlgorithmResult, BoundingBox, DetectPartBox, ResultType
from core.utils.alg_config import merge_config

from . import config as INS_C
from .align import map_roi_to_test, scale_preview_roi, solder_mask_to_preview_roi
from .bridge_joints import pad_joints, parse_bridge_joints
from .defect_mode import effective_enabled_defects
from .pipeline import align_preview, inspect as run_inspect
from .solder_extract import extract_solder_mask
from .template import TemplateModel
from .visualize import draw_annotation
from .yaml_config import apply_yaml_data

_VALID_SOLDER_MODES = ("ellipse", "contour", "roi")
_VALID_ROI_SHAPES = ("rect", "circle")


class PcbThroughHoleSolderAlg(IDetectionAlgorithm):
    """插件焊点缺陷检测（孔洞/少锡/多锡/连锡/不出脚）。
    """

    algorithm_code = "pcb_through_hole_solder_defect"
    version = 1

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        # 串行化 config 全局状态，并保护模板 LRU
        self._lock = threading.Lock()
        self._template_cache: "OrderedDict[str, TemplateModel]" = OrderedDict()

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """算法默认参数。配置 JSON 缺省时即用这里的值。"""
        return {
            "enable_review": True,
            "solder_mode": "ellipse",
            "solder_roi": None,
            "template_id": None,
            "enabled_defects": [12, 13, 14, 15, 16],
            "bridge_joints": None,
            "void_preset": "MED",
            "insuf_preset": "MED",
            "template_cache_size": 32,
            "advanced": {},
        }

    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox: Optional[BoundingBox] = None,
        original_template_image: Optional[np.ndarray] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> AlgorithmResult:
        """检测入口。算法池只调用此方法。"""
        t0 = time.perf_counter()
        try:
            if not self._is_valid_image(image):
                return self._fail_result("image (测试图) 为空或无效", t0)
            if not self._is_valid_image(original_template_image):
                return self._fail_result(
                    "original_template_image (标准图) 为空或无效："
                    "本算法为模板对比型算法，必须提供标准图", t0)

            try:
                test_full = self._ensure_bgr(image)
                template_bgr = self._ensure_bgr(original_template_image)
            except ValueError as exc:
                return self._fail_result(str(exc), t0)

            # 外部 config 覆盖默认参数（None 值自动忽略）
            cfg = merge_config(self._defaults, config)

            solder_mode = str(cfg.get("solder_mode") or "ellipse")
            if solder_mode not in _VALID_SOLDER_MODES:
                return self._fail_result(
                    f"solder_mode must be one of {_VALID_SOLDER_MODES}, got {solder_mode!r}", t0)

            if solder_mode == "roi":
                roi_error = self._validate_manual_pads(cfg)
                if roi_error is not None:
                    return self._fail_result(roi_error, t0)

            # roi_bbox 可选裁剪；坐标系仍相对调用方传入的完整 image
            work_image, offset_x, offset_y = self._crop_by_bbox(test_full, roi_bbox)
            if work_image.size == 0:
                return self._fail_result("roi_bbox 裁剪结果为空", t0)

            # 本算法为单模板<->单测试图逐对比对，不支持 multiple_roi_images，
            # 忽略但不报错（契约允许算法只使用部分参数）
            ignored_multi_roi = bool(multiple_roi_images)

            # 核心判定逻辑在同目录各缺陷模块内，调整应改那边的对应模块而非本文件。
            # 临界区：模板缓存读写 + config 全局调参 + inspect() 执行放
            # 在同一把锁内，避免并发调用互相覆盖全局阈值/竞争缓存写入。
            with self._lock:
                try:
                    tm, template_key = self._get_or_build_template(template_bgr, cfg, solder_mode)
                except Exception as exc:
                    return self._fail_result(f"模板预处理失败: {exc}", t0, exc=exc)

                n_pads = len(pad_joints(self._bridge_joints_from_cfg(cfg)))
                enabled = cfg.get("enabled_defects")
                if enabled is None:
                    enabled = tm.enabled_defects
                tm = replace(
                    tm,
                    enabled_defects=effective_enabled_defects(enabled, n_pads),
                )

                try:
                    self._apply_global_params(cfg)
                except Exception as exc:
                    return self._fail_result(f"参数配置无效: {exc}", t0, exc=exc)

                res = run_inspect(
                    work_image, tm, name=str(template_key),
                    enable_review=bool(cfg.get("enable_review", True)),
                )
                vis = draw_annotation(res)

            # 组织结果：坐标映射回完整 image
            parts = self._build_parts(res, offset_x, offset_y)
            metadata = self._build_metadata(
                res, cfg, template_key, vis, test_full, offset_x, offset_y,
                roi_bbox, ignored_multi_roi)

            return AlgorithmResult(
                code=0,
                message="ok",
                algorithm_code=self.algorithm_code,
                cost_time=time.perf_counter() - t0,
                result_type=ResultType.PARTS,
                parts=parts,
                metadata=metadata,
            )
        except Exception as exc:
            # 兜底：任何意外都转成 code=1 的结果，绝不向算法池抛异常
            return self._fail_result(str(exc), t0, exc=exc)

    def prebuild_template(
        self, template_image: np.ndarray, config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """仅生成/复用标准图锡面 Mask 缓存，不做检测（不需要测试图）。
        """
        try:
            if not self._is_valid_image(template_image):
                return {"ok": False, "error": "标准图为空或无效"}
            try:
                template_bgr = self._ensure_bgr(template_image)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

            cfg = merge_config(self._defaults, config)
            solder_mode = str(cfg.get("solder_mode") or "ellipse")
            if solder_mode not in _VALID_SOLDER_MODES:
                return {"ok": False,
                        "error": f"solder_mode must be one of {_VALID_SOLDER_MODES}, got {solder_mode!r}"}
            if solder_mode == "roi":
                roi_error = self._validate_manual_pads(cfg)
                if roi_error is not None:
                    return {"ok": False, "error": roi_error}

            with self._lock:
                tm, template_key = self._get_or_build_template(template_bgr, cfg, solder_mode)
            # 锡面 Mask 可能已按 WORK_MAX_DIM 降采样；回传到调用方原始标准图分辨率，
            # 供界面在标准图上叠加显示生成的锡面区域。
            solder_mask = None
            if tm.masks is not None and tm.masks.solder is not None:
                std_h, std_w = template_bgr.shape[:2]
                mask = tm.masks.solder
                if mask.shape[0] != std_h or mask.shape[1] != std_w:
                    mask = cv2.resize(mask, (std_w, std_h), interpolation=cv2.INTER_NEAREST)
                solder_mask = mask
            return {"ok": True, "template_id": template_key,
                    "roi_rect": list(tm.roi.rect) if tm.roi is not None else None,
                    "solder_mask": solder_mask}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def preview_roi(
        self, image: np.ndarray, config: Dict[str, Any],
        original_template_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """配准后把标准图几何映射到测试图。成功含 ``roi``/``rois``（多框时逐项）。"""
        try:
            if not self._is_valid_image(image):
                return {"ok": False, "error": "image (测试图) 为空或无效"}
            if not self._is_valid_image(original_template_image):
                return {"ok": False, "error": "original_template_image (标准图) 为空或无效"}
            try:
                test_full = self._ensure_bgr(image)
                template_bgr = self._ensure_bgr(original_template_image)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

            cfg = merge_config(self._defaults, config)
            solder_mode = str(cfg.get("solder_mode") or "ellipse")
            if solder_mode not in _VALID_SOLDER_MODES:
                return {"ok": False,
                        "error": f"solder_mode must be one of {_VALID_SOLDER_MODES}, got {solder_mode!r}"}
            if solder_mode == "roi":
                roi_error = self._validate_manual_pads(cfg)
                if roi_error is not None:
                    return {"ok": False, "error": roi_error}

            with self._lock:
                try:
                    tm, template_key = self._get_or_build_template(template_bgr, cfg, solder_mode)
                except Exception as exc:
                    return {"ok": False, "error": f"模板预处理失败: {exc}"}
                meta, method, work_wh = align_preview(test_full, tm)

            std_h, std_w = template_bgr.shape[:2]
            pads = pad_joints(self._bridge_joints_from_cfg(cfg))
            src_rois: list[dict] = []
            if pads:
                src_rois = [j.to_preview_roi() for j in pads]
            elif solder_mode == "roi" and cfg.get("solder_roi"):
                src_rois.append(dict(cfg["solder_roi"]))
            else:
                mask = tm.masks.solder if tm.masks is not None else None
                roi_work = solder_mask_to_preview_roi(mask)
                tm_h, tm_w = tm.bgr.shape[:2]
                rs_x, rs_y = std_w / max(1, tm_w), std_h / max(1, tm_h)
                src_rois.append(scale_preview_roi(roi_work, rs_x, rs_y))

            mapped_rois = [
                map_roi_to_test(r, (std_w, std_h), work_wh, meta) for r in src_rois
            ]
            return {
                "ok": True,
                "roi": mapped_rois[0] if mapped_rois else None,
                "rois": mapped_rois,
                "template_id": template_key,
                "align_method": method,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ---- 模板缓存：同一模板仅在首次调用时生成一次锡面 Mask/派生特征，随后复用 ----

    @staticmethod
    def _bridge_joints_from_cfg(cfg: Dict[str, Any]):
        """解析 bridge_joints；兼容 Tin 标注字段名 joints。"""
        label = {
            "bridge_joints": cfg.get("bridge_joints"),
            "joints": cfg.get("joints"),
        }
        return parse_bridge_joints(label)

    def _template_key_for(self, template_bgr: np.ndarray, cfg: Dict[str, Any]) -> str:
        tid = cfg.get("template_id")
        joints = self._bridge_joints_from_cfg(cfg)
        joints_sig = hashlib.sha1(
            repr([j.to_dict() for j in joints]).encode("utf-8")
        ).hexdigest()[:8] if joints else "noj"
        if tid is not None:
            # solder_mode 纳入 key：同一 template_id 切换 ellipse/contour/roi 时
            # 锡面 Mask 生成逻辑不同，缺了这段会误命中旧方式的缓存。
            solder_mode = str(cfg.get("solder_mode") or "ellipse")
            key = f"{tid}:{solder_mode}:{joints_sig}"
            if solder_mode == "roi":
                # "roi" 框可能被用户重新调整，需纳入 solder_roi 内容避免误用旧框
                roi_sig = hashlib.sha1(
                    repr(sorted((cfg.get("solder_roi") or {}).items())).encode("utf-8")
                ).hexdigest()[:8]
                key = f"{key}:{roi_sig}"
            return key
        # 未提供 template_id 时以标准图内容摘要兜底 (roi 模式下纳入 solder_roi)
        solder_mode = str(cfg.get("solder_mode") or "ellipse")
        extra = repr(sorted((cfg.get("solder_roi") or {}).items())) if solder_mode == "roi" else ""
        extra += joints_sig
        digest = hashlib.sha1((template_bgr.tobytes() + solder_mode.encode() + extra.encode())).hexdigest()[:16]
        h, w = template_bgr.shape[:2]
        return f"auto:{h}x{w}:{digest}"

    def _get_or_build_template(
        self, template_bgr: np.ndarray, cfg: Dict[str, Any], solder_mode: str,
    ) -> tuple[TemplateModel, str]:
        key = self._template_key_for(template_bgr, cfg)
        cached = self._template_cache.get(key)
        if cached is not None:
            self._template_cache.move_to_end(key)
            return cached, key

        joints = self._bridge_joints_from_cfg(cfg)
        pads = pad_joints(joints)
        if pads:
            h, w = template_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for j in pads:
                mask = cv2.bitwise_or(mask, j.mask((h, w)))
        else:
            label = {"solder_mode": solder_mode, "roi": cfg.get("solder_roi")}
            if solder_mode == "roi" and cfg.get("solder_roi"):
                mask, _fit_meta = extract_solder_mask(template_bgr, label)
            elif solder_mode == "roi":
                raise ValueError("人工框选模式下需要提供 bridge_joints 或 solder_roi")
            else:
                mask, _fit_meta = extract_solder_mask(template_bgr, label)

        tm = TemplateModel.build(
            template_bgr, roi_mask=mask, templ_id=str(key), bridge_joints=joints)

        self._template_cache[key] = tm
        max_size = int(cfg.get("template_cache_size") or 32)
        while len(self._template_cache) > max(1, max_size):
            self._template_cache.popitem(last=False)
        return tm, key

    def clear_template_cache(self) -> None:
        """供运维/测试主动清空模板缓存（如标准图已更新，需要强制重新生成 Mask）。"""
        with self._lock:
            self._template_cache.clear()

    # ---- 人工框参数校验 (solder_mode="roi" 时) ----

    def _validate_manual_pads(self, cfg: Dict[str, Any]) -> Optional[str]:
        """人工模式：bridge_joints 有 pad 即可；否则回退校验 solder_roi。"""
        if pad_joints(self._bridge_joints_from_cfg(cfg)):
            return None
        return self._validate_solder_roi(cfg.get("solder_roi"))

    @staticmethod
    def _validate_solder_roi(solder_roi: Any) -> Optional[str]:
        """校验 config["solder_roi"]；合法返回 None，非法返回可读的错误信息。"""
        if not isinstance(solder_roi, dict) or not solder_roi:
            return ("solder_mode='roi' 时必须提供 bridge_joints 或 solder_roi，"
                    f"实际 solder_roi: {solder_roi!r}")
        shape = str(solder_roi.get("shape", "rect") or "rect").lower()
        if shape not in _VALID_ROI_SHAPES:
            return f"solder_roi.shape 必须是 {_VALID_ROI_SHAPES} 之一, 实际: {shape!r}"
        required = ("cx", "cy", "r") if shape == "circle" else ("x", "y", "w", "h")
        for key in required:
            if key not in solder_roi:
                return f"solder_roi(shape={shape!r}) 缺少字段 {key!r}: {solder_roi!r}"
            try:
                float(solder_roi[key])
            except (TypeError, ValueError):
                return f"solder_roi(shape={shape!r}) 字段 {key!r} 不是合法数值: {solder_roi!r}"
        size_ok = (float(solder_roi["r"]) > 0) if shape == "circle" else (
            float(solder_roi["w"]) > 0 and float(solder_roi["h"]) > 0)
        if not size_ok:
            return f"solder_roi(shape={shape!r}) 的尺寸必须 > 0: {solder_roi!r}"
        return None

    # ---- 全局调参（config 为进程级状态） ----

    @staticmethod
    def _apply_global_params(cfg: Dict[str, Any]) -> None:
        overrides: Dict[str, Any] = {}
        if cfg.get("void_preset"):
            overrides["void_preset"] = str(cfg["void_preset"])
        if cfg.get("insuf_preset"):
            overrides["insuf_preset"] = str(cfg["insuf_preset"])
        advanced = cfg.get("advanced")
        # advanced 是完整覆盖快照 (非增量)，每次先清空再按本次快照重建，
        # 避免恢复默认后仍残留上一次的值。
        INS_C.INSUF_OVERRIDES.clear()
        if advanced:
            # 须包成 "thresholds" 嵌套键才会被 apply_yaml_data 写入 C.TH/INSUF_OVERRIDES
            overrides["thresholds"] = dict(advanced)
        if overrides:
            apply_yaml_data(overrides)

    # ---- 结果组织：Detection -> DetectPartBox（坐标映射回原始 image） ----

    @staticmethod
    def _build_parts(res, offset_x: int, offset_y: int) -> List[DetectPartBox]:
        parts: List[DetectPartBox] = []
        meta = res.align_meta
        for det in res.detections:
            x, y, w, h = meta.map_bbox_to_original(det.bbox)
            parts.append(DetectPartBox(
                x=int(x + offset_x), y=int(y + offset_y),
                width=int(w), height=int(h),
                confidence=float(det.score), angle=0.0,
                box_type="defect",
                label=INS_C.DEFECT_NAMES.get(det.defect_id, str(det.defect_id)),
                class_id=int(det.defect_id),
                metadata={"reason": det.reason, "pad_id": int(det.pad_id)},
            ))
        return parts

    @staticmethod
    def _build_metadata(
        res, cfg: Dict[str, Any], template_key: str, vis: np.ndarray,
        test_full: np.ndarray, offset_x: int, offset_y: int,
        roi_bbox: Optional[BoundingBox], ignored_multi_roi: bool,
    ) -> Dict[str, Any]:
        status = res.status or ("NG" if res.is_defect else "OK")
        if status == "待复检":
            status = "REVIEW"
        # `vis` 是在裁剪后的 work_image 上画的标注，这里贴回完整 image 尺寸的
        # 画布，保证 output_image 与 parts/image_shape 坐标系一致 (契约要求)。
        if vis.shape[:2] == test_full.shape[:2]:
            output_image = vis
        else:
            output_image = test_full.copy()
            vh, vw = vis.shape[:2]
            output_image[offset_y:offset_y + vh, offset_x:offset_x + vw] = vis
        metadata: Dict[str, Any] = {
            "status": status,
            "defect_ids": list(res.defect_ids),
            "num_defects": len(res.detections),
            "coverage": res.coverage,
            "review_reason": res.review_reason,
            "align_method": res.align_method,
            "total_ms": res.total_ms,
            "template_id": template_key,
            "solder_mode": cfg.get("solder_mode"),
            "image_shape": [int(test_full.shape[0]), int(test_full.shape[1])],
            "output_image": output_image,
        }
        if roi_bbox is not None:
            metadata["roi_bbox"] = {
                "x": roi_bbox.x, "y": roi_bbox.y,
                "width": roi_bbox.width, "height": roi_bbox.height,
            }
        if ignored_multi_roi:
            metadata["multiple_roi_images_ignored"] = True
        return metadata

    def _fail_result(
        self, message: str, t0: float, exc: Optional[BaseException] = None,
    ) -> AlgorithmResult:
        """统一的失败出口：code=1 + ResultType.ERROR。"""
        meta: Dict[str, Any] = {"error": message}
        if exc is not None:
            meta["traceback"] = traceback.format_exc()
        return AlgorithmResult(
            code=1,
            message=message,
            algorithm_code=self.algorithm_code,
            cost_time=time.perf_counter() - t0,
            result_type=ResultType.ERROR,
            metadata=meta,
        )

    # ---- 内部工具 ----

    @staticmethod
    def _is_valid_image(img: Optional[np.ndarray]) -> bool:
        return isinstance(img, np.ndarray) and img.size > 0 and img.ndim in (2, 3)

    @staticmethod
    def _ensure_bgr(img: np.ndarray) -> np.ndarray:
        """统一输入格式为 3 通道 BGR uint8，兼容灰度图/带 alpha 通道输入。"""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise ValueError(f"unsupported image shape: {img.shape}")

    @staticmethod
    def _crop_by_bbox(
        image: np.ndarray, bbox: Optional[BoundingBox],
    ) -> tuple[np.ndarray, int, int]:
        """按 roi_bbox 裁剪测试图；返回 (裁剪后图像, x偏移, y偏移)。
        bbox 为 None 时原样返回 (偏移 0, 0)，坐标系仍相对调用方传入的完整 image。"""
        if bbox is None:
            return image, 0, 0
        h, w = image.shape[:2]
        x0 = max(0, min(int(bbox.x), w))
        y0 = max(0, min(int(bbox.y), h))
        x1 = max(x0, min(x0 + int(bbox.width), w))
        y1 = max(y0, min(y0 + int(bbox.height), h))
        return image[y0:y1, x0:x1], x0, y0
