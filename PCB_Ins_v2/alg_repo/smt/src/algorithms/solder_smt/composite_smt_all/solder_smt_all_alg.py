from __future__ import annotations
import os, json, time, traceback
from typing import Any, Dict, List, Optional, Tuple
import cv2, numpy as np
from core.entities.algorithms import IDetectionAlgorithm
from core.entities.detect import AlgorithmResult, BoundingBox, DetectPartBox, ResultType
from core.utils.alg_config import merge_config

from ._solder_core import (
    _remove_interference,
    classify_pads,
    _expand_rect,
    _merge_overlapping_rects,
    extract_solder,
    judge_insufficient,
    judge_excess,
    judge_bridge,
    judge_cold_solder,
    align_template,
    map_rect_to_original,
    map_rect_from_original,
    map_point_to_original,
    map_mask_to_original,
    compute_diff,
    auto_resize_image,
    scale_pad_rects,
    scale_group_rects,
)


# 算法主体

class SolderSmtAllAlg(IDetectionAlgorithm):
    """SMT焊锡全缺陷检测(改进版) - 集成 _solder_core 纯算法函数。

    一次 run() 检测 excess/insufficient/bridge/cold_solder 四类缺陷。
    支持7种模板模式, 有模板时可做差分对比(精度高), 无模板时用绝对阈值。
    """

    algorithm_code = "SolderSmtAllAlg"
    version = 2
    SUPPORT_DEFECTS = ["excess", "insufficient", "bridge", "cold_solder"]
    description = "SMT焊锡全缺陷检测(改进版): 多锡/少锡/连锡/虚焊, 集成_solder_core"
    principle = "干扰排除 | 三轮聚类焊盘定位 | H校正W分层膨胀焊锡提取 | 模板差分RB_score"
    frontend_params = []

    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            # ── Step1: 干扰排除(阻焊+丝印+元件) ──
            "mask_h_low": 38, "mask_h_high": 78,
            "mask_s_min": 30, "mask_v_min": 30,
            "silk_h_low": 82, "silk_h_high": 97,
            "silk_s_low": 75, "silk_s_high": 155,
            "silk_v_low": 75, "silk_v_high": 175,
            "comp_v_max": 60,
            "comp_h_low": 0, "comp_h_high": 179,
            "comp_s_min": 0, "comp_s_max": 255,
            "min_interference_area": 30,
            "interference_close_ks": 5,

            # ── Step1+: 焊盘框定位(形态学+连通域+三轮聚类) ──
            "morph_close_ks": 9, "morph_open_ks": 5,
            "min_pad_area": 20,
            "n_classes": "auto",
            "cluster_ar_thresh": 0.5,
            "cluster_area_ratio": 0.3,
            "cluster_ar_thresh_r2": 0.15,
            "cluster_area_ratio_r2": 0.15,
            "pad_expand_ratio": 0.30,
            "pad_expand_short_ratio": 0.30,
            "pad_expand_aspect_thresh": 2.0,
            "pad_expand_match_aspect": True,

            # ── Step2: 焊锡提取(H校正W + 分层膨胀 + 局部W距离 + SV约束) ──
            "solder_blue_h_low": 85, "solder_blue_h_high": 115,
            "solder_blue_s_min": 60, "solder_blue_v_min": 150,
            "solder_noninter_dilate_ks": 15,
            "solder_growzone_open_ks": 5,
            "solder_grow_dilate_ks": 3,
            "solder_grow_max_iters": 0,
            "solder_enable_noninter_process": False,
            "solder_enable_sv_ok": False,
            "solder_enable_texture_filter": False,
            "skip_rb_score": True,
            "merge_compute_diff": True,
            "skip_diff_heatmap": True,
            "enable_auto_resize": True,
            "auto_resize_max_size": 400,
            "auto_resize_threshold": 450,
            "solder_max_growth_ratio": 0,
            "solder_growth_check_interval": 0,
            "solder_min_blue_ratio": 0.1,
            "solder_v_min_blue": 150,
            "solder_v_min_interf": 150,
            "solder_v_min_red": 80,
            "solder_s_min_blue": 60,
            "solder_s_min_interf": 30,
            "solder_s_min_red": 0,
            "solder_s_max_red": 100,
            "solder_v_min_global": 110,
            "solder_wv_w_scale": 45.0,
            "solder_wv_v_scale": 80.0,
            "solder_wv_thresh": 1.0,
            "solder_wv_w_weight": 0.6,
            "solder_wv_v_weight": 0.4,
            "solder_close_ks": 7,
            "solder_texture_ksize": 7,
            "solder_texture_thresh": 3.0,
            # W校正LUT段边界
            "w_lut_a_hi": 25, "w_lut_b_hi": 35,
            "w_lut_c_hi": 77, "w_lut_d_hi": 100,
            "w_lut_w_a": 60, "w_lut_w_b": 3,
            "w_lut_w_c": 4, "w_lut_w_d": 3,

            # ── 模板对齐 ──
            "ECC_EUCLIDEAN": False,
            "crack_cpp_min_pads": 4,  # C++批量crack检测的最小pad数(0=始终用Python)
            "scale_min": 0.5,
            "scale_max": 2.0,

            "grow_outside_region": True,
            "seed_outside_region": True,
            # ── Step3: 缺陷判定 ──
            "insufficient_thresh": 0.09,
            "excess_thresh": 0.15,
            "bridge_min_area": 20,
            "cold_solder_dark_ratio": 0.30,
            "cold_solder_v_dark": 80,
            "pin_type": "gull-wing",
            # ── 虚焊3规则参数 (gull-wing专用) ──
            "cold_solder_rim_ratio": 0.5,
            "toe_metal_ratio_thresh": 0.3,
            "cold_diff_ratio_thresh": 0.38,
            "toe_metal_h_low": 80, "toe_metal_h_high": 124,
            "toe_metal_s_min": 39, "toe_metal_s_max": 78,
            "toe_metal_v_min": 125, "toe_metal_v_max": 222,

            # ── Step4: 模板差分(RB_score) ──
            "RB_thresh_ins": 100,
            "RB_thresh_exc": 100,
            "RB_red_ratio_min": 0.03,
            "diff_insufficient_thresh": 0.05,
            "insufficient_use_diff": True,
            "bluer_h_thresh": 30,
            "diff_thresh": 80,

            # ── 模式控制 ──
            "template_mode": "grow_local",
            "provide_pad_frames": False,

            # ── 缺陷开关 ──
            "detect_excess": True,
            "detect_insufficient": True,
            "detect_bridge": True,
            "detect_cold_solder": True,

            # ── 可视化开关 ──
            "enable_visualization": True,

            # ── 性能优化开关 ──
            "skip_diff_morphology": False,  # True=跳过diff形态学去噪(提速~2ms)
            "diff_min_area": 10,            # 0=跳过连通域过滤(提速~1.5ms)
        }

    # 主入口

    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox: Optional[BoundingBox] = None,
        original_template_image: Optional[np.ndarray] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> AlgorithmResult:
        """检测入口。算法池只调用此方法。

        固定模式: diff_grow_ROI
        焊盘来源: 仅从 pad.json (cfg["pad_frames"]) 获取
        """
        t0 = time.perf_counter()
        try:
            # ── 1. 输入校验 ──
            if image is None or image.size == 0:
                return self._fail_result("empty image", t0)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]

            # ── 2. ROI裁剪 ──
            roi_offset = (0, 0)
            if roi_bbox is not None:
                rx, ry = int(roi_bbox.x), int(roi_bbox.y)
                rw, rh = int(roi_bbox.width), int(roi_bbox.height)
                image = image[ry:ry + rh, rx:rx + rw]
                roi_offset = (rx, ry)
                if image.size == 0:
                    return self._fail_result("empty roi image", t0)

            # ── 3. 合并参数 + 解析缺陷开关 ──
            cfg = merge_config(self._defaults, config)
            cfg.pop("pad_source", None)
            det_excess = bool(cfg.get("detect_excess", True))
            det_insuf = bool(cfg.get("detect_insufficient", True))
            det_bridge = bool(cfg.get("detect_bridge", True))
            det_cold = bool(cfg.get("detect_cold_solder", True))
            any_defect = det_excess or det_insuf or det_bridge or det_cold
            if not any_defect:
                return AlgorithmResult(
                    code=0, message="ok",
                    algorithm_code=self.algorithm_code,
                    cost_time=time.perf_counter() - t0,
                    result_type=ResultType.PARTS,
                    parts=[],
                    metadata={
                        "num_defects": 0,
                        "mode": "diff_grow_ROI",
                        "has_template": False,
                        "pad_count": 0,
                        "solder_area": 0,
                        "defects_enabled": {
                            "excess": det_excess, "insufficient": det_insuf,
                            "bridge": det_bridge, "cold_solder": det_cold,
                        },
                    },
                )

            # ── 3b. 自动缩放(可选): 当图片宽或高超过阈值时等比缩小 ──
            _resize_scale = 1.0  # 待检图缩放比例
            _tpl_resize_scale = 1.0  # 模板图缩放比例(可能与待检图不同)
            if cfg.get("enable_auto_resize", False):
                _rs_max = cfg.get("auto_resize_max_size", 400)
                _rs_thresh = cfg.get("auto_resize_threshold", 450)
                image, _resize_scale = auto_resize_image(image, _rs_max, _rs_thresh)
                # pad_frames先不缩放, 等模板处理后再用正确的scale缩放
                if _resize_scale < 1.0 and cfg.get("group_frames"):
                    cfg["group_frames"] = scale_group_rects(cfg["group_frames"], _resize_scale)
                cfg["_resize_scale"] = _resize_scale

            h, w = image.shape[:2]
            template_mode = "diff_grow_ROI"  # 固定模式
            provide_pad_frames = True  # 始终从pad.json获取

            # ── 4. 模板预处理 ──
            has_tpl = (original_template_image is not None
                       and original_template_image.size > 0)
            tpl = None
            if has_tpl:
                tpl = original_template_image.copy()
                if tpl.ndim == 2:
                    tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
                elif tpl.ndim == 3 and tpl.shape[2] == 4:
                    tpl = tpl[:, :, :3]
                # 同步缩放模板图(可能得到与待检图不同的scale)
                if _resize_scale < 1.0:
                    tpl, _tpl_resize_scale = auto_resize_image(tpl, cfg.get("auto_resize_max_size", 400),
                                               cfg.get("auto_resize_threshold", 450))
                # pad_frames来自pad.json(模板坐标系), 用模板的scale缩放
                _pad_scale = _tpl_resize_scale if has_tpl else _resize_scale
                if _pad_scale < 1.0 and cfg.get("pad_frames"):
                    cfg["pad_frames"] = scale_pad_rects(cfg["pad_frames"], _pad_scale)
                cfg["_tpl_resize_scale"] = _tpl_resize_scale
            else:
                # 无模板: pad_frames在待检图坐标系, 用待检图scale
                if _resize_scale < 1.0 and cfg.get("pad_frames"):
                    cfg["pad_frames"] = scale_pad_rects(cfg["pad_frames"], _resize_scale)

            # ── 5. Step1: 干扰排除 (预计算HSV供后续复用) ──
            loose_inter = cfg.get("loose", has_tpl or provide_pad_frames)
            _hsv_shared = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
            interference = _remove_interference(image, cfg, loose=loose_inter, hsv=_hsv_shared)

            # ── 6. 焊盘框定位 (仅从pad.json) ──
            tpl_info: Optional[Dict[str, Any]] = None
            pad_rects: List[Tuple[int, int, int, int]] = []
            outer_rects: List[Tuple[int, int, int, int]] = []
            group_rects: List[Tuple[int, int, int, int]] = []

            pad_source = "ROI"
            raw_pads = cfg.get("pad_frames", [])
            if not raw_pads:
                return self._fail_result("no pad_frames in config", t0)

            # 有模板时做对齐映射
            if has_tpl:
                tpl_info = align_template(tpl, image, cfg)
                if tpl_info is not None:
                    pad_rects = [tuple(int(v) for v in map_rect_to_original(r, tpl_info))
                                 for r in raw_pads]
                    pad_rects = [(x, y, rw, rh) for (x, y, rw, rh) in pad_rects
                                 if rw > 5 and rh > 5]
                else:
                    pad_rects = [tuple(int(v) for v in r) for r in raw_pads]
            else:
                pad_rects = [tuple(int(v) for v in r) for r in raw_pads]
            outer_rects = self._compute_outers(pad_rects, w, h, cfg)
            group_rects = _merge_overlapping_rects(outer_rects)

            # ── 7. 模板对齐(如果还未对齐) ──
            if has_tpl and tpl_info is None:
                tpl_info = align_template(tpl, image, cfg)

            # ── 8. Step2: 焊锡提取 ──
            # HSV已在Step1预计算, 只需Gray
            _gray_shared = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            (mask_blue, mask_merged, mask_filled,
             mask_texture, mask_final, std_map,
             non_inter_heavy, w_channel) = extract_solder(
                image, interference, pad_rects, group_rects, cfg,
                hsv=_hsv_shared, gray=_gray_shared)
            solder_mask = mask_final

            # DEBUG: 保存焊锡mask+pad中间图(可选)
            if cfg.get("debug_save_solder_mask", False):
                import os as _os
                _dbg_dir = cfg.get("debug_output_dir", "_debug_solder_mask")
                _os.makedirs(_dbg_dir, exist_ok=True)
                _dbg_vis = image.copy()
                _dbg_vis[solder_mask > 0] = (0, 255, 255)  # 焊锡涂黄
                for (px, py, pw, ph) in pad_rects:
                    cv2.rectangle(_dbg_vis, (px, py), (px+pw, py+ph), (0, 0, 255), 1)
                _dbg_tag = cfg.get("_debug_pair_tag", "unknown")
                cv2.imwrite(_os.path.join(_dbg_dir, f"{_dbg_tag}.png"), _dbg_vis)

            # ── 9. Step3: 缺陷判定(grow) ──
            grow_ins: List[Tuple] = []
            grow_exc: List[Tuple] = []
            grow_brg: List[Tuple] = []
            grow_cold: List[Tuple] = []
            exc_mask: np.ndarray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if det_insuf:
                grow_ins = judge_insufficient(solder_mask, pad_rects, image.shape, cfg)
            # 多锡/连锡: use_diff=True时跳过第一次judge，由Step4b的合并判定覆盖
            _use_diff_later = ("diff" in template_mode) and has_tpl
            if det_excess and not _use_diff_later:
                exc_mask, exc_is_ng, exc_total_ratio, exc_blobs = judge_excess(
                    solder_mask, pad_rects, group_rects, image.shape, cfg)
            if det_bridge and not _use_diff_later:
                grow_brg = judge_bridge(solder_mask, pad_rects, group_rects, image.shape, cfg)

            # ── 10. Step4: 模板差分(可选) ──
            diff_ins_flags: List[bool] = []
            diff_exc_flags: List[bool] = []
            diff_brg_flags: List[bool] = []
            rb_blue_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            diff_mask_full: Optional[np.ndarray] = None
            use_diff = ("diff" in template_mode) and has_tpl and (tpl_info is not None)
            skip_rb_score = bool(cfg.get("skip_rb_score", False))
            if use_diff and not skip_rb_score:
                try:
                    diff_ins_flags, diff_exc_flags, diff_brg_flags, rb_blue_full = self._run_diff_detection(
                        tpl, image, tpl_info, pad_rects, group_rects, cfg)
                except Exception:
                    use_diff = False

            # ── 10b. 合并判定: grow焊锡mask + RB_blue -> 统一judge ──
            if use_diff or skip_rb_score:
                combined_solder = cv2.bitwise_or(solder_mask, rb_blue_full) if not skip_rb_score else solder_mask
                # 计算低阈值diff_mask约束excess (仅在有模板时计算)
                diff_mask_exc = None
                if has_tpl and tpl_info is not None:
                    try:
                        _diff_p = dict(cfg)
                        _diff_p["diff_thresh"] = cfg.get("diff_thresh_exc", 30)
                        _dm_low, _, _ = compute_diff(tpl, image, tpl_info, _diff_p)
                        diff_mask_exc = map_mask_to_original(_dm_low, tpl_info) if _dm_low is not None else None
                    except Exception:
                        diff_mask_exc = None
                if det_excess:
                    exc_mask, exc_is_ng, exc_total_ratio, exc_blobs = judge_excess(
                        combined_solder, pad_rects, group_rects, image.shape, cfg,
                        diff_mask_full=diff_mask_exc)
                if det_bridge:
                    grow_brg = judge_bridge(combined_solder, pad_rects, group_rects, image.shape, cfg)

            # ── 10c. 虚焊检测 (复用低阈值diff_mask或独立计算) ──
            if det_cold:
                diff_mask_cold = None
                merge_compute_diff = bool(cfg.get("merge_compute_diff", False))
                if merge_compute_diff and diff_mask_exc is not None:
                    diff_mask_cold = diff_mask_exc
                elif use_diff and has_tpl and tpl_info is not None:
                    try:
                        _dm_cold, _, _ = compute_diff(tpl, image, tpl_info, cfg)
                        diff_mask_cold = map_mask_to_original(_dm_cold, tpl_info) if _dm_cold is not None else None
                    except Exception:
                        diff_mask_cold = None
                toe_rects, rim_rects = self._load_toe_rim_annotations(
                    pad_rects, cfg, tpl_info, has_tpl)
                grow_cold = judge_cold_solder(
                    image, solder_mask, pad_rects, image.shape, cfg,
                    toe_rects=toe_rects, rim_rects=rim_rects,
                    diff_mask_full=diff_mask_cold,
                    ins_results=grow_ins if det_insuf else None,
                    pad_source=pad_source,
                    hsv=_hsv_shared, gray=_gray_shared)

            # ── 11. 融合grow+diff结果(OR逻辑) + 转换为DetectPartBox ──
            parts: List[DetectPartBox] = []

            if det_insuf:
                for i, (x, y, rw, rh, ratio, g_ng) in enumerate(grow_ins):
                    if g_ng:
                        parts.append(DetectPartBox(
                            x=x, y=y, width=rw, height=rh,
                            confidence=min(1.0, max(0.0, 1.0 - ratio)),
                            box_type="defect", label="insufficient",
                            metadata={"coverage": float(ratio),
                                      "method": "grow"}))

            if det_excess and exc_is_ng:
                for (x, y, bw, bh) in exc_blobs:
                    parts.append(DetectPartBox(
                        x=int(x), y=int(y), width=int(bw), height=int(bh),
                        confidence=min(1.0, exc_total_ratio),
                        box_type="defect", label="excess",
                        metadata={"excess_ratio": float(exc_total_ratio),
                                  "method": "grow+diff" if use_diff else "grow"}))

            if det_bridge:
                for i, (gr, bridge_pairs, g_ng, bridge_blobs) in enumerate(grow_brg):
                    if g_ng:
                        method = "grow+diff" if use_diff else "grow"
                        if bridge_blobs:
                            for (bx, by, bw, bh) in bridge_blobs:
                                parts.append(DetectPartBox(
                                    x=int(bx), y=int(by), width=int(bw), height=int(bh),
                                    confidence=0.7,
                                    box_type="defect", label="bridge",
                                    metadata={"method": method,
                                              "bridge_blobs": bridge_blobs}))
                        else:
                            gx, gy, gw, gh = gr
                            parts.append(DetectPartBox(
                                x=int(gx), y=int(gy), width=int(gw), height=int(gh),
                                confidence=0.7,
                                box_type="defect", label="bridge",
                                metadata={"method": method,
                                          "bridge_blobs": bridge_blobs}))

            if det_cold:
                for (x, y, rw, rh, score, is_ng, reasons, ratio) in grow_cold:
                    if is_ng:
                        parts.append(DetectPartBox(
                            x=x, y=y, width=rw, height=rh,
                            confidence=min(1.0, score),
                            box_type="defect", label="cold_solder",
                            metadata={"cold_score": float(score),
                                      "cold_ratio": float(ratio),
                                      "method": "grow",
                                      "reasons": reasons}))

            # ── 12. ROI坐标还原 ──
            if roi_offset != (0, 0):
                for p in parts:
                    p.x += roi_offset[0]
                    p.y += roi_offset[1]

            # ── 13. 构建输出可视化图 ──
            vis = None
            if cfg.get("enable_visualization", True):
                vis = self._build_output_image(image, parts)

            # ── 13b. auto_resize坐标还原: 把parts坐标缩放回原图尺寸 ──
            if _resize_scale < 1.0:
                _inv = 1.0 / _resize_scale
                for p in parts:
                    p.x = int(p.x * _inv)
                    p.y = int(p.y * _inv)
                    p.width = int(p.width * _inv)
                    p.height = int(p.height * _inv)

            # ── 14. 返回结果 ──
            return AlgorithmResult(
                code=0, message="ok",
                algorithm_code=self.algorithm_code,
                cost_time=time.perf_counter() - t0,
                result_type=ResultType.PARTS,
                parts=parts,
                metadata={
                    "num_defects": len(parts),
                    "mode": "diff_grow_ROI",
                    "has_template": has_tpl,
                    "pad_count": len(pad_rects),
                    "solder_area": int(np.count_nonzero(solder_mask)),
                    "output_image": vis,
                    "defects_enabled": {
                        "excess": det_excess, "insufficient": det_insuf,
                        "bridge": det_bridge, "cold_solder": det_cold,
                    },
                    "pad_rects_aligned": scale_pad_rects(pad_rects, 1.0 / _resize_scale) if _resize_scale < 1.0 else pad_rects,
                },
            )
        except Exception as exc:
            return self._fail_result(str(exc), t0, exc=exc)

    # 辅助方法
   
    _DEFECT_COLORS = {
        "insufficient": (0, 255, 255),   # 黄色: 少锡
        "excess": (255, 0, 0),           # 蓝色: 多锡
        "bridge": (0, 0, 255),           # 红色: 连锡
        "cold_solder": (255, 0, 255),    # 品红: 虚焊
    }
    _DEFECT_LABELS = {
        "insufficient": "少锡 insufficient",
        "excess": "多锡 excess",
        "bridge": "连锡 bridge",
        "cold_solder": "虚焊 cold_solder",
    }

    # 中文字体缓存
    _font_cache = None

    @classmethod
    def _get_font(cls):
        if cls._font_cache is not None:
            return cls._font_cache
        from PIL import ImageFont
        import os, sys
        # 跨平台字体搜索路径
        _font_paths = []
        if sys.platform == "win32":
            _font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simsun.ttc",
                "C:/Windows/Fonts/arial.ttf",
            ]
        elif sys.platform == "darwin":
            _font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        else:  # Linux
            _font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        for fp in _font_paths:
            if os.path.isfile(fp):
                try:
                    cls._font_cache = ImageFont.truetype(fp, 16)
                    return cls._font_cache
                except Exception:
                    pass
        cls._font_cache = ImageFont.load_default()
        return cls._font_cache

    def _build_output_image(self, image: np.ndarray, parts: List) -> np.ndarray:
        """绘制缺陷框+中文标签（PIL渲染，标签超界时自动放下方）"""
        from PIL import Image, ImageDraw
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_h, img_w = vis.shape[:2]
        font = self._get_font()

        # 第一遍: OpenCV画框 (虚焊框内缩, 避免与少锡框完全重合)
        for p in parts:
            x, y, w, h = int(p.x), int(p.y), int(p.width), int(p.height)
            label = getattr(p, "label", "")
            color_bgr = self._DEFECT_COLORS.get(label, (0, 255, 0))
            if label == "cold_solder":
                inset = max(4, min(w, h) // 8)
                cv2.rectangle(vis, (x + inset, y + inset),
                              (x + w - inset, y + h - inset), color_bgr, 2)
            else:
                cv2.rectangle(vis, (x, y), (x + w, y + h), color_bgr, 2)

        if not parts:
            return vis

        # 第二遍: PIL画文字（单次BGR->RGB->BGR往返）
        pil_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for p in parts:
            x, y, w, h = int(p.x), int(p.y), int(p.width), int(p.height)
            label = getattr(p, "label", "")
            color_bgr = self._DEFECT_COLORS.get(label, (0, 255, 0))
            text = self._DEFECT_LABELS.get(label, label)

            # 虚焊框内缩, 标签也跟着内缩后的框
            if label == "cold_solder":
                inset = max(4, min(w, h) // 8)
                x += inset; y += inset; w -= 2 * inset; h -= 2 * inset

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            space_above = y
            space_below = img_h - (y + h)

            if space_above >= th + 6:
                # 放框上方
                ty = y - th - 4
                label_top = max(0, y - th - 4)
            elif space_below >= th + 6:
                # 放框下方
                ty = y + h + 2
                label_top = y + h
            else:
                # 上下都不够 -> 放框内顶部
                ty = y + 2
                label_top = y

            rx = min(x + tw + 8, img_w)
            rb = min(label_top + th + 6, img_h)
            draw.rectangle([x, label_top, rx, rb],
                           fill=color_bgr[::-1])  # BGR->RGB
            draw.text((x + 3, ty), text, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # def _detect_pads_local(
    #     self,
    #     img: np.ndarray,
    #     interference: np.ndarray,
    #     cfg: Dict[str, Any],
    # ) -> Tuple[List[Tuple[int, int, int, int]],
    #            List[Tuple[int, int, int, int]],
    #            List[Tuple[int, int, int, int]]]:
    #     """Step1+: 在非干扰区上检测焊盘框(形态学+连通域+三轮聚类)。
    #
    #     返回: (pad_rects, outer_rects, group_rects)
    #     """
    #     h, w = img.shape[:2]
    #     non_inter = cv2.bitwise_not(interference)
    #     k_close = cv2.getStructuringElement(
    #         cv2.MORPH_RECT, (int(cfg["morph_close_ks"]), int(cfg["morph_close_ks"])))
    #     k_open = cv2.getStructuringElement(
    #         cv2.MORPH_RECT, (int(cfg["morph_open_ks"]), int(cfg["morph_open_ks"])))
    #     nir = cv2.morphologyEx(non_inter, cv2.MORPH_CLOSE, k_close)
    #     nir = cv2.morphologyEx(nir, cv2.MORPH_OPEN, k_open)
    #
    #     n_cc, _, stats, _ = cv2.connectedComponentsWithStats(nir, 8)
    #     raw: List[Tuple[int, int, int, int]] = []
    #     min_pad_area = int(cfg.get("min_pad_area", 20))
    #     for i in range(1, n_cc):
    #         if int(stats[i, cv2.CC_STAT_AREA]) < min_pad_area:
    #             continue
    #         raw.append((int(stats[i, cv2.CC_STAT_LEFT]),
    #                     int(stats[i, cv2.CC_STAT_TOP]),
    #                     int(stats[i, cv2.CC_STAT_WIDTH]),
    #                     int(stats[i, cv2.CC_STAT_HEIGHT])))
    #
    #     result = classify_pads(raw, w, h, cfg)
    #     pad_rects = [(x, y, rw, rh) for (x, y, rw, rh, ci) in result["final"]]
    #     outer_rects = self._compute_outers(pad_rects, w, h, cfg)
    #     group_rects = _merge_overlapping_rects(outer_rects)
    #     return pad_rects, outer_rects, group_rects

    @staticmethod
    def _compute_outers(
        pad_rects: List[Tuple[int, int, int, int]],
        img_w: int, img_h: int,
        cfg: Dict[str, Any],
    ) -> List[Tuple[int, int, int, int]]:
        """计算焊盘外扩框(outer_rects)。"""
        expand_ratio = float(cfg.get("pad_expand_ratio", 0.2))
        short_ratio = float(cfg.get("pad_expand_short_ratio", 0.30))
        aspect_thresh = float(cfg.get("pad_expand_aspect_thresh", 2.0))
        match_aspect = bool(cfg.get("pad_expand_match_aspect", False))
        return [_expand_rect(x, y, rw, rh, expand_ratio, img_w, img_h,
                             short_ratio, aspect_thresh, match_aspect)
                for (x, y, rw, rh) in pad_rects]

    @staticmethod
    def _match_annotations_to_pads(
        pad_rects: List[Tuple[int, int, int, int]],
        ann_rects: List[Tuple[int, int, int, int]],
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """Match annotation rectangles to their corresponding pads by center point.

        Returns: dict {pad_index: ann_rect}
        """
        if not ann_rects:
            return {}
        result = {}
        for ann_rect in ann_rects:
            ax, ay, aw, ah = ann_rect
            acx = ax + aw / 2.0
            acy = ay + ah / 2.0
            best_pad = -1
            best_overlap = 0
            for pi, (px, py, pw, ph) in enumerate(pad_rects):
                if px <= acx <= px + pw and py <= acy <= py + ph:
                    ix1 = max(ax, int(px))
                    iy1 = max(ay, int(py))
                    ix2 = min(ax + aw, int(px + pw))
                    iy2 = min(ay + ah, int(py + ph))
                    overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_pad = pi
            if best_pad >= 0:
                result[best_pad] = ann_rect
        return result

    def _load_toe_rim_annotations(
        self,
        pad_rects: List[Tuple[int, int, int, int]],
        cfg: Dict[str, Any],
        tpl_info: Optional[Dict[str, Any]],
        has_tpl: bool,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """加载toe/rim标注 (用于虚焊3规则检测).

        优先级:
          1. cfg中的 toe_frames / rim_frames (直接矩形列表)
          2. cfg中的 pad_json_path (从pad.json加载所有shape)

        如果标注来自模板(source含"_OK"), 映射到待检图坐标。

        Returns: (toe_rects_matched, rim_rects_matched)
            每个是 {pad_index: (x,y,w,h)} 或 None
        """
        component_type = cfg.get("pin_type", "gull-wing")
        is_terminal = component_type == "terminal"
        if is_terminal:
            return None, None

        toe_rects_matched: Optional[Dict] = None
        rim_rects_matched: Optional[Dict] = None

        # ── 方式1: 直接从config读取toe_frames/rim_frames ──
        _rs = cfg.get("_tpl_resize_scale", cfg.get("_resize_scale", 1.0))  # pad.json来自模板, 用模板scale
        toe_frames = cfg.get("toe_frames")
        rim_frames = cfg.get("rim_frames")
        if toe_frames and _rs < 1.0:
            toe_frames = scale_pad_rects(toe_frames, _rs)
        if rim_frames and _rs < 1.0:
            rim_frames = scale_pad_rects(rim_frames, _rs)

        # ── 方式2: 从pad_json_path加载 ──
        pad_json_path = cfg.get("pad_json_path")
        if pad_json_path is None and cfg.get("pad_frames_source"):
            pad_json_path = cfg.get("pad_frames_source")

        if pad_json_path and os.path.isfile(pad_json_path):
            with open(pad_json_path, 'r', encoding='utf-8') as pf:
                _pad_meta = json.load(pf)
                _src = _pad_meta.get('source', '')
                _ann_from_template = '_OK' in _src or '_template' in _src

            # 加载所有rectangle shapes (应用缩放)
            _all_shapes = []
            for s in _pad_meta.get('shapes', []):
                if s.get('shape_type') != 'rectangle':
                    continue
                pts = s['points']
                x = int(min(pts[0][0], pts[1][0]) * _rs)
                y = int(min(pts[0][1], pts[1][1]) * _rs)
                w = int(abs(pts[1][0] - pts[0][0]) * _rs)
                h = int(abs(pts[1][1] - pts[0][1]) * _rs)
                _all_shapes.append((s.get('label', ''), x, y, w, h))

            if toe_frames is None:
                toe_frames = [(x, y, w, h) for (lbl, x, y, w, h) in _all_shapes if lbl == 'toe']
            if rim_frames is None:
                rim_frames = [(x, y, w, h) for (lbl, x, y, w, h) in _all_shapes if lbl == 'rim']

            # 模板标注映射到待检图坐标
            if _ann_from_template and has_tpl and tpl_info is not None:
                if toe_frames:
                    toe_frames = [tuple(int(v) for v in map_rect_to_original(r, tpl_info))
                                  for r in toe_frames]
                if rim_frames:
                    rim_frames = [tuple(int(v) for v in map_rect_to_original(r, tpl_info))
                                  for r in rim_frames]

        # 匹配到pad
        if toe_frames:
            _matched = self._match_annotations_to_pads(pad_rects, toe_frames)
            toe_rects_matched = _matched if _matched else None
        if rim_frames:
            _matched = self._match_annotations_to_pads(pad_rects, rim_frames)
            rim_rects_matched = _matched if _matched else None

        # 警告由 judge_cold_solder 内部打印 (知道diff_mask_full是否可用)
        return toe_rects_matched, rim_rects_matched

    def _run_diff_detection(
        self,
        tpl: np.ndarray,
        image: np.ndarray,
        tpl_info: Dict[str, Any],
        pad_rects: List[Tuple[int, int, int, int]],
        group_rects: List[Tuple[int, int, int, int]],
        cfg: Dict[str, Any],
    ) -> Tuple[List[bool], List[bool], List[bool], np.ndarray]:
        """模板差分检测(RB_score方向性差异)。

        在重叠空间计算, 返回每个pad/group的NG标志。
        返回: (diff_ins_flags, diff_exc_flags, diff_brg_flags, rb_blue_full)
        """
        template_crop = tpl_info['template_crop']
        aligned_inspect = tpl_info['aligned_inspect']
        ox, oy, ow, oh = tpl_info['overlap_rect']

        if ow < 20 or oh < 20:
            return [], [], [], np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # ── HSV方向性差异 ──
        t_hsv = cv2.cvtColor(template_crop, cv2.COLOR_BGR2HSV)
        i_hsv = cv2.cvtColor(aligned_inspect, cv2.COLOR_BGR2HSV)
        t_h = t_hsv[:, :, 0].astype(np.float32)
        i_h = i_hsv[:, :, 0].astype(np.float32)
        t_v = t_hsv[:, :, 2].astype(np.float32)
        i_v = i_hsv[:, :, 2].astype(np.float32)

        # H环形差值: 正=向蓝偏移, 负=向红偏移
        h_diff = i_h - t_h
        h_diff = np.where(h_diff > 90, h_diff - 180, h_diff)
        h_diff = np.where(h_diff < -90, h_diff + 180, h_diff)
        v_diff = i_v - t_v

        # ── RB_score: redder + darker - bluer - brighter ──
        RB_thresh_ins = float(cfg.get("RB_thresh_ins", 100))
        RB_thresh_exc = float(cfg.get("RB_thresh_exc", 100))
        RB_red_ratio_min = float(cfg.get("RB_red_ratio_min", 0.03))
        diff_insufficient_thresh = float(cfg.get("diff_insufficient_thresh", 0.05))
        excess_thresh = float(cfg.get("excess_thresh", 0.3))
        bridge_min_area = int(cfg.get("bridge_min_area", 20))

        redder = np.clip(-h_diff, 0, None)
        darker = np.clip(-v_diff, 0, None)
        bluer = np.clip(h_diff, 0, None)
        brighter = np.clip(v_diff, 0, None)
        RB_score = redder + darker - bluer - brighter

        k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        RB_red = cv2.morphologyEx(
            (RB_score > RB_thresh_ins).astype(np.uint8) * 255,
            cv2.MORPH_OPEN, k_noise)
        RB_blue = cv2.morphologyEx(
            (RB_score < -RB_thresh_exc).astype(np.uint8) * 255,
            cv2.MORPH_OPEN, k_noise)

        # diff_ins_mask: (redder|darker) & ~(bluer|brighter)
        h_neg = (h_diff < -30).astype(np.uint8) * 255
        v_neg = (v_diff < -80).astype(np.uint8) * 255
        h_pos = (h_diff > 30).astype(np.uint8) * 255
        v_pos = (v_diff > 80).astype(np.uint8) * 255
        R_evidence = cv2.bitwise_or(h_neg, v_neg)
        not_R = cv2.bitwise_or(h_pos, v_pos)
        diff_ins_mask = cv2.bitwise_and(R_evidence, cv2.bitwise_not(not_R))
        diff_ins_mask = cv2.morphologyEx(diff_ins_mask, cv2.MORPH_OPEN, k_noise)

        # ── pad框映射到重叠空间 ──
        pad_overlap = []
        pad_indices = []
        for idx, r in enumerate(pad_rects):
            mx, my, mw, mh = map_rect_from_original(r, tpl_info)
            if mx + mw > 0 and mx < ow and my + mh > 0 and my < oh:
                pad_overlap.append((int(mx), int(my), int(mw), int(mh)))
                pad_indices.append(idx)

        if not pad_overlap:
            return [], [], [], np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # ── 少锡: diff_ins_mask + RB_red ──
        diff_ins_flags = [False] * len(pad_rects)
        for i, (px, py, pw, ph) in enumerate(pad_overlap):
            pad_mask = np.zeros((oh, ow), dtype=np.uint8)
            cv2.rectangle(pad_mask, (px, py), (px + pw, py + ph), 255, -1)
            ins_ratio = np.count_nonzero(
                cv2.bitwise_and(diff_ins_mask, pad_mask)) / max(pw * ph, 1)
            RB_red_ratio = np.count_nonzero(
                cv2.bitwise_and(RB_red, pad_mask)) / max(pw * ph, 1)
            diff_ins_flags[pad_indices[i]] = (
                ins_ratio > diff_insufficient_thresh and
                RB_red_ratio > RB_red_ratio_min)

        # ── 多锡: RB_blue in outer-pad ──
        diff_exc_flags = [False] * len(pad_rects)
        expand_ratio = float(cfg.get("pad_expand_ratio", 0.2))
        short_ratio = float(cfg.get("pad_expand_short_ratio", 0.30))
        aspect_thresh = float(cfg.get("pad_expand_aspect_thresh", 2.0))
        match_aspect = bool(cfg.get("pad_expand_match_aspect", False))
        for i, (px, py, pw, ph) in enumerate(pad_overlap):
            ox2, oy2, ow2, oh2 = _expand_rect(
                px, py, pw, ph, expand_ratio, ow, oh, short_ratio, aspect_thresh, match_aspect)
            outer_mask = np.zeros((oh, ow), dtype=np.uint8)
            cv2.rectangle(outer_mask, (int(ox2), int(oy2)),
                          (int(ox2 + ow2), int(oy2 + oh2)), 255, -1)
            cv2.rectangle(outer_mask, (px, py), (px + pw, py + ph), 0, -1)
            RB_blue_outer = cv2.bitwise_and(RB_blue, outer_mask)
            outer_area = max(np.count_nonzero(outer_mask), 1)
            blue_ratio = np.count_nonzero(RB_blue_outer) / outer_area
            diff_exc_flags[pad_indices[i]] = blue_ratio > excess_thresh

        # ── 连锡: RB_blue between pads + 连通两pad ──
        diff_brg_flags = [False] * len(group_rects)
        pad_union = np.zeros((oh, ow), dtype=np.uint8)
        for (px, py, pw, ph) in pad_overlap:
            cv2.rectangle(pad_union, (px, py), (px + pw, py + ph), 255, -1)
        pad_dilated = cv2.dilate(pad_union, k_noise)

        for gi in range(len(group_rects)):
            gx, gy, gw, gh = map_rect_from_original(group_rects[gi], tpl_info)
            gx, gy, gw, gh = int(gx), int(gy), int(gw), int(gh)
            group_mask = np.zeros((oh, ow), dtype=np.uint8)
            cv2.rectangle(group_mask, (gx, gy), (gx + gw, gy + gh), 255, -1)
            between_pads = cv2.subtract(group_mask, pad_union)
            blue_between = cv2.bitwise_and(RB_blue, between_pads)
            blue_between = cv2.morphologyEx(blue_between, cv2.MORPH_OPEN, k_noise)
            if np.count_nonzero(blue_between) <= bridge_min_area:
                continue
            n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
                blue_between, 8)
            for j in range(1, n_cc):
                if int(cc_stats[j, cv2.CC_STAT_AREA]) < bridge_min_area:
                    continue
                comp_mask = np.zeros_like(blue_between)
                comp_mask[cc_labels == j] = 255
                comp_dilated = cv2.dilate(comp_mask, k_noise)
                touched = cv2.bitwise_and(comp_dilated, pad_dilated)
                pad_count = 0
                for (px, py, pw, ph) in pad_overlap:
                    pad_rect_mask = np.zeros((oh, ow), dtype=np.uint8)
                    cv2.rectangle(pad_rect_mask, (px, py),
                                  (px + pw, py + ph), 255, -1)
                    if np.any(cv2.bitwise_and(touched, pad_rect_mask)):
                        pad_count += 1
                if pad_count >= 2:
                    diff_brg_flags[gi] = True
                    break

        # RB_blue映射到全图(供excess合并使用)
        rb_blue_full = map_mask_to_original(RB_blue, tpl_info)

        return diff_ins_flags, diff_exc_flags, diff_brg_flags, rb_blue_full

    # 失败出口
  
    def _fail_result(self, message: str, t0: float,
                     exc: Optional[BaseException] = None) -> AlgorithmResult:
        meta: Dict[str, Any] = {"error": message}
        if exc is not None:
            meta["traceback"] = traceback.format_exc()
        return AlgorithmResult(
            code=1, message=message,
            algorithm_code=self.algorithm_code,
            cost_time=time.perf_counter() - t0,
            result_type=ResultType.ERROR,
            metadata=meta,
        )


# CLI demo

if __name__ == "__main__":
    import sys, os, json, argparse

    # 修复相对导入 + core 模块路径
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
    _PROJECT_ROOT = os.path.dirname(_SRC_ROOT)
    for _d in (_SRC_ROOT, _PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "test_algo")):
        if _d not in sys.path:
            sys.path.insert(0, _d)

    ap = argparse.ArgumentParser(description="SolderSmtAllAlg demo")
    ap.add_argument("image", type=str, help="输入图片路径")
    ap.add_argument("-t", "--template", type=str, default=None, help="模板图片路径")
    ap.add_argument("--pad", type=str, default=None, help="pad.json 路径")
    ap.add_argument("--params", type=str, default=None, help="params.json 路径(加载额外参数)")
    ap.add_argument("--no-excess", action="store_true", help="关闭多锡检测")
    ap.add_argument("--no-insufficient", action="store_true", help="关闭少锡检测")
    ap.add_argument("--no-bridge", action="store_true", help="关闭连锡检测")
    ap.add_argument("--no-cold", action="store_true", help="关闭虚焊检测")
    ap.add_argument("--no-vis", action="store_true", help="关闭可视化(提速~3ms)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read: {args.image}")
    tpl = cv2.imread(args.template) if args.template else None

    # 从 pad.json 加载焊盘框
    pad_json_path = args.pad
    if pad_json_path is None:
        # 尝试在同目录查找 pad.json
        _guess = os.path.join(os.path.dirname(args.image), "pad.json")
        if os.path.isfile(_guess):
            pad_json_path = _guess

    pad_frames = []
    if pad_json_path and os.path.isfile(pad_json_path):
        with open(pad_json_path, 'r', encoding='utf-8') as pf:
            _meta = json.load(pf)
        for s in _meta.get("shapes", []):
            if s.get("label") == "pad" and s.get("shape_type") == "rectangle":
                (x1, y1), (x2, y2) = s["points"]
                pad_frames.append((
                    int(min(x1, x2)), int(min(y1, y2)),
                    int(abs(x2 - x1)), int(abs(y2 - y1))
                ))

    if not pad_frames:
        raise SystemExit("no pad_frames found (use --pad or put pad.json next to image)")

    alg = SolderSmtAllAlg()
    cfg = {
        "detect_excess": not args.no_excess,
        "detect_insufficient": not args.no_insufficient,
        "detect_bridge": not args.no_bridge,
        "detect_cold_solder": not args.no_cold,
        "pad_frames": pad_frames,
        "pad_json_path": pad_json_path,
        "enable_visualization": not args.no_vis,
    }
    # 加载 params.json (如果指定)
    if args.params and os.path.isfile(args.params):
        with open(args.params, 'r', encoding='utf-8') as pf:
            cfg.update(json.load(pf))
    r = alg.run(img, cfg, original_template_image=tpl)

    if r.code != 0:
        raise SystemExit(r.message)

    enabled = r.metadata["defects_enabled"]
    active = [k for k, v in enabled.items() if v]
    print(f"缺陷数={r.metadata['num_defects']} "
          f"模式={r.metadata['mode']} "
          f"焊盘={r.metadata['pad_count']} "
          f"耗时={r.cost_time * 1000:.1f}ms "
          f"启用={active}")
    for p in r.parts:
        print(f"  {p.label}:({p.x},{p.y}){p.width}x{p.height} conf={p.confidence:.2f}")
