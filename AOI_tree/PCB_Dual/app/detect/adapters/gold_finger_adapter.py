"""金手指/板面检测适配器：包装 `金手指检测` 仓库。

该仓库不是一个 `IDetectionAlgorithm` 类，而是一组脚本级函数（配准 + 板面
掩膜提取 + 光照归一 + AbsDiff 缺陷检测 + 可选缺陷分类），本适配器按
`main_pipeline.main()` 的流程顺序直接调用这些函数，等价于把该脚本的主流程
改写成一次 `run()` 调用（不弹交互取色窗口、不写磁盘缓存）。

板面颜色识别支持两种模式：
  - 已通过 `app.calibration.gold_seed` 标定种子点：用种子点生成的 HSV 范围
    （"custom" 模式，等价于仓库里的 "seed" 模式，只是不弹窗）
  - 未标定：回退到仓库默认的 "gold" 模式（固定沉金 HSV 范围）

该仓库的顶层模块名（peizhun/seed_extract/main_pipeline/defect_features/
forest_infer）不会跟插件焊点、贴片锡焊两个仓库的 "algorithms"/"core" 冲突，
所以这里直接把仓库目录永久加入 `sys.path`，不需要 `IsolatedImport` 隔离。
"""
from __future__ import annotations

import importlib
import sys
import threading
from typing import Any

import cv2
import numpy as np

from app.calibration.gold_seed import get_gold_seed
from app.config import GOLD_REPO_DIR
from app.detect.adapters.base import BaseAdapter, ParamSpec
from app.detect.contract import ModuleManifest, Roi
from app.detect.types import AlgorithmResult, DefectBox, DetectRequest

ALGORITHM_ID = "board_板面金手指"

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load_gold() -> dict[str, Any]:
    with _lock:
        if _cache:
            return _cache
        repo_dir = str(GOLD_REPO_DIR)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        peizhun_mod = importlib.import_module("peizhun")
        pipeline_mod = importlib.import_module("main_pipeline")
        _cache["align_image"] = peizhun_mod.align_image
        _cache["get_pcb_alpha_mask"] = pipeline_mod.get_pcb_alpha_mask
        _cache["refine_alignment_ecc"] = pipeline_mod.refine_alignment_ecc
        _cache["compose_refined_affine"] = pipeline_mod.compose_refined_affine
        _cache["match_color_linear"] = pipeline_mod.match_color_linear
        _cache["match_illumination_local"] = pipeline_mod.match_illumination_local
        _cache["detect_defects"] = pipeline_mod.detect_defects
        _cache["detect_mask_shape_defects"] = pipeline_mod.detect_mask_shape_defects
        _cache["map_boxes_to_source"] = pipeline_mod.map_boxes_to_source
        _cache["draw_defect_boxes"] = pipeline_mod.draw_defect_boxes
        _cache["classify_defects"] = pipeline_mod.classify_defects
        _cache["annotate_defect_labels"] = pipeline_mod.annotate_defect_labels
        _cache["model_path"] = GOLD_REPO_DIR / "defect_classifier.npz"
        return _cache


def _extract_mask_and_valid(gf: dict[str, Any], img: np.ndarray, board_color: str, ranges) -> tuple[np.ndarray, np.ndarray]:
    v = (img.max(axis=2) > 0).astype(np.uint8) * 255
    v = cv2.erode(v, np.ones((5, 5), np.uint8))
    m = gf["get_pcb_alpha_mask"](img, board_color=board_color, custom_hsv_ranges=ranges)
    return m * (v[:, :, None] > 0), v


class GoldFingerAdapter(BaseAdapter):
    algorithm_ids = [ALGORITHM_ID]
    requires_standard = True

    def __init__(self) -> None:
        _load_gold()

    def manifests(self) -> list[ModuleManifest]:
        return [
            ModuleManifest(
                id=ALGORITHM_ID,
                display_name="板面及金手指区域",
                group_id="board",
                group_name="板面及金手指",
                requires_standard=True,
                region_kind="gold",
                color="#2563eb",
                algorithm_version="1.0.0",
                trigger="on_job",
            )
        ]

    def param_specs(self, algorithm_id: str) -> list[ParamSpec]:
        return [
            ParamSpec("diff_thresh", "颜色差异阈值", "int", 25, 5, 100, 1,
                      "配准归一化后逐像素颜色差异阈值。调大：更少报；调小：更多报。"),
            ParamSpec("min_defect_area", "最小缺陷面积", "int", 5, 1, 500, 1,
                      "小于该像素面积的差异区域忽略。调大：更少报；调小：更多报。"),
            ParamSpec("shift_tolerance", "配准容差(像素)", "int", 1, 0, 5, 1,
                      "容忍的配准残余像素偏移，用于消除边缘处的配准误差误报。调大：更少报；调小：更多报。"),
        ]

    def is_ready(
        self,
        algorithm_id: str,
        template_name: str | None,
        rois: list[Roi] | None = None,
    ) -> tuple[bool, str]:
        _ = algorithm_id, template_name, rois
        return True, ""

    def run(self, algorithm_id: str, req: DetectRequest, params: dict[str, Any]) -> AlgorithmResult:
        gf = _load_gold()
        std_bgr = req.image_std_raw if req.image_std_raw is not None else req.image_std
        test_bgr = req.image_test_raw if req.image_test_raw is not None else req.image_test

        if std_bgr is None or std_bgr.size == 0 or test_bgr is None or test_bgr.size == 0:
            return AlgorithmResult(algorithm=algorithm_id, ok=False, message="板面及金手指：标准图/测试图为空",
                                   status="ERROR", error_code="INPUT_EMPTY", error_message="标准图/测试图为空")

        seed = get_gold_seed(req.template_name)
        if req.rois:
            pts = []
            for r in req.rois:
                if r.kind == "gold" or r.layer == "gold":
                    shp = r.shape or {}
                    if "x" in shp and "y" in shp:
                        pts.append([float(shp["x"]), float(shp["y"])])
            if pts:
                seed = dict(seed)
                seed["seeds"] = pts
        if seed.get("ranges"):
            board_color = "custom"
            custom_ranges = [(tuple(lo), tuple(hi)) for lo, hi in seed["ranges"]]
        else:
            board_color = "gold"
            custom_ranges = None

        diff_thresh = float(params.get("diff_thresh", 25))
        min_defect_area = float(params.get("min_defect_area", 5))
        shift_tolerance = int(params.get("shift_tolerance", 1))

        try:
            _ok_align, aligned_test, align_meta = gf["align_image"](test_bgr, std_bgr)
            align_affine = np.array(align_meta.get("affine", [[1, 0, 0], [0, 1, 0]]), dtype=np.float32)

            alpha_mask = gf["get_pcb_alpha_mask"](std_bgr, board_color=board_color, custom_hsv_ranges=custom_ranges)
            alpha_mask_test, valid = _extract_mask_and_valid(gf, aligned_test, board_color, custom_ranges)

            area_good = max(1, int(np.count_nonzero(alpha_mask > 0.5)))
            area_ratio = np.count_nonzero(alpha_mask_test > 0.5) / area_good
            if not (0.5 <= area_ratio <= 1.5):
                alpha_mask_test = None
            else:
                ok_ecc, ecc_warp = gf["refine_alignment_ecc"](alpha_mask, alpha_mask_test)
                if ok_ecc:
                    align_affine = gf["compose_refined_affine"](align_affine, ecc_warp)
                    th, tw = std_bgr.shape[:2]
                    moving = test_bgr
                    if moving.shape[:2] != (th, tw):
                        interp = cv2.INTER_AREA if (moving.shape[0] > th or moving.shape[1] > tw) else cv2.INTER_CUBIC
                        moving = cv2.resize(moving, (tw, th), interpolation=interp)
                    aligned_test = cv2.warpAffine(
                        moving, align_affine, (tw, th), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )
                    alpha_mask_test, valid = _extract_mask_and_valid(gf, aligned_test, board_color, custom_ranges)

            stats_mask = (valid > 0) & (alpha_mask.squeeze() > 0.99)
            norm_test = gf["match_color_linear"](aligned_test, std_bgr, valid_mask=stats_mask)
            norm_test = gf["match_illumination_local"](norm_test, std_bgr, valid_mask=stats_mask)

            blur_good = cv2.GaussianBlur(std_bgr, (3, 3), 0)
            blur_test = cv2.GaussianBlur(norm_test, (3, 3), 0)
            im_good_extracted = (blur_good.astype(np.float32) * alpha_mask).astype(np.uint8)
            im_test_extracted = (blur_test.astype(np.float32) * alpha_mask).astype(np.uint8)
            im_test_display = (norm_test.astype(np.float32) * alpha_mask).astype(np.uint8)

            long_side = max(std_bgr.shape[:2])
            edge_band = max(6, int(round(long_side / 300)))

            aligned_vis, diff_mask, defect_boxes, _defect_count = gf["detect_defects"](
                im_good_extracted, im_test_extracted, alpha_mask,
                diff_thresh=diff_thresh, min_defect_area=min_defect_area,
                shift_tolerance=shift_tolerance, edge_margin=1, edge_band=edge_band,
                edge_penalty=1.0, edge_thresh_cap=None,
                alpha_mask_test=alpha_mask_test, display_img=im_test_display,
            )

            if alpha_mask_test is not None:
                shape_tol = max(2, int(round(long_side / 800)))
                shape_min_area = int(max(20, 20 * (long_side / 1000.0) ** 2))
                shape_mask, shape_boxes = gf["detect_mask_shape_defects"](
                    alpha_mask, alpha_mask_test, valid_mask=valid, tol_px=shape_tol, min_area=shape_min_area,
                )
                if shape_boxes:
                    diff_mask = cv2.bitwise_or(diff_mask, shape_mask)
                    defect_boxes = defect_boxes + shape_boxes
                    gf["draw_defect_boxes"](aligned_vis, shape_boxes)

            src_boxes = gf["map_boxes_to_source"](defect_boxes, align_affine, test_bgr.shape, std_bgr.shape)
            final_vis = gf["draw_defect_boxes"](test_bgr.copy(), src_boxes)

            labels = ["异常"] * len(src_boxes)
            model_path = gf["model_path"]
            if defect_boxes and model_path.exists():
                try:
                    classified = gf["classify_defects"](diff_mask, norm_test, str(model_path))
                    mapped_cls_boxes = gf["map_boxes_to_source"](
                        [b for b, _lbl, _p in classified], align_affine, test_bgr.shape, std_bgr.shape
                    )
                    labels = [lbl for _b, lbl, _p in classified]
                    gf["annotate_defect_labels"](
                        final_vis,
                        [(mapped_cls_boxes[i], lbl, prob) for i, (_b, lbl, prob) in enumerate(classified)],
                    )
                except Exception:  # noqa: BLE001 - 分类失败不影响主检测结果
                    pass

            boxes = [
                DefectBox(x=int(x), y=int(y), w=int(w), h=int(h), label=lbl)
                for (x, y, w, h), lbl in zip(src_boxes, labels)
            ]
            ok = len(boxes) == 0
            message = f"板面及金手指 判定={'OK' if ok else 'NG'}，缺陷数={len(boxes)}，配准方式={align_meta.get('method')}"
            return AlgorithmResult(
                algorithm=algorithm_id, ok=ok, message=message, boxes=boxes,
                diff_image=final_vis,
                metadata={
                    "alignment": {
                        "ok": bool(_ok_align),
                        "method": str(align_meta.get("method") or ""),
                        "score": float(align_meta.get("score") or 0.0),
                        "affine": align_affine.tolist(),
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            return AlgorithmResult(algorithm=algorithm_id, ok=False,
                                   message=f"[适配器异常] 板面及金手指: {exc}",
                                   status="ERROR", error_code="ADAPTER_EXCEPTION", error_message=str(exc))
