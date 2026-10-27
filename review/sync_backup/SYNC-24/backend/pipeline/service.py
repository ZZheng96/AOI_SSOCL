"""检测服务：检测引擎推理 + 可视化产物 + 入库 + 统计。

对应赛题问题一（实时视频/图片异常检测）：
  - 图片：引擎多槽位融合打分（CDF 在线校准 + gray 三态），输出分数/热力图/缺陷框/耗时
  - 视频：按间隔抽帧逐帧检测，供 WebSocket 实时推流或后台批处理

M2 起统一走 backend.engine.DetectionEngine（demo1/demo4 适配层已下线）。
Detection 表字段映射（不加新列，避免迁移）：
  final_score=score、is_anomaly、latency_ms
  level1/2/3_score ← slot_scores["sem"]/["disc"]/["shead"]（无则 None）
  triggered_level  ← triggered_slot（字符串复用旧列）
  n_tiles(JSON)    ← {"slots","raw","weights","router_w","triggered_slot","decision"}
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..api.schemas import detection_interpretation
from ..core.alarm import get_alarm_monitor
from ..core.audit import log_action
from ..core.config import get_settings
from ..core.imaging import imread, imwrite
from ..core.notify import notify_event
from ..db.database import session_scope
from ..db.models import (Dataset, Detection, Image as ImageRow, Model,
                         StatsDaily, Video, VideoFrame)
from ..engine import get_engine

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ── 可视化 ───────────────────────────────────────────────────
def _best_heatmap(result) -> Optional[np.ndarray]:
    """引擎结果的主热力图：AC-063 优先连续融合热图（工作网格），回退二值 mask。"""
    for attr in ("fused_heatmap", "mask"):
        m = getattr(result, attr, None)
        if isinstance(m, np.ndarray):
            return m.astype(np.float32)
    return None


def _heatmap_u8(heatmap: np.ndarray, size_wh) -> np.ndarray:
    """热图 → 显示分辨率 uint8（INTER_LINEAR + 图内 min-max 归一化）。"""
    hm = cv2.resize(np.asarray(heatmap, np.float32), tuple(int(v) for v in size_wh),
                    interpolation=cv2.INTER_LINEAR)
    lo, hi = float(hm.min()), float(hm.max())
    # 原地运算（与 (hm-lo)/d*255 逐位一致，省去临时数组分配）
    np.subtract(hm, lo, out=hm)
    np.divide(hm, max(hi - lo, 1e-8), out=hm)
    np.multiply(hm, 255, out=hm)
    return hm.astype(np.uint8)


def _mask_and_boxes(hm_u8: np.ndarray, rel_thresh: float, min_area: int,
                    morph_kernel: int):
    """uint8 热图 → (形态学后二值 mask 0/255, 连通域框 [x,y,w,h] 显示坐标)。"""
    # cv2.threshold(THRESH_BINARY) 即 src > t → 255，与 (hm_u8 > t)*255 等价
    _, mask = cv2.threshold(hm_u8, int(255 * rel_thresh), 255, cv2.THRESH_BINARY)
    if morph_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((morph_kernel, morph_kernel), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area:
            boxes.append([int(x), int(y), int(bw), int(bh)])
    return mask, boxes


def quantize_raw_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """AC-063：留存用量化（float16，非有限/溢出时保留 float32）。

    在线渲染与留存使用同一份量化值，保证读回复算与在线输出逐位一致。"""
    with np.errstate(over="ignore", invalid="ignore"):
        hm16 = np.asarray(heatmap).astype(np.float16)
    if not np.isfinite(hm16).all():
        return np.asarray(heatmap, np.float32)
    return hm16


def recompute_heatmap_mask(raw_heatmap: np.ndarray, display_wh,
                           rel_thresh: float = 0.6, min_area: int = 16,
                           morph_kernel: int = 3,
                           display_scale: float = 1.0) -> dict:
    """AC-063 纯函数：由留存原始热图按给定阈值复算 mask/框。

    与在线 render_heatmap_overlay 同一实现（_heatmap_u8 + _mask_and_boxes），
    默认参数下与在线输出一致。返回 mask（显示分辨率 0/255）、boxes_display
    （显示坐标 x,y,w,h）与 boxes_image（原图像素 xyxy，换算口径同在线
    int(v/display_scale)）。"""
    hm_u8 = _heatmap_u8(raw_heatmap, display_wh)
    mask, boxes = _mask_and_boxes(hm_u8, rel_thresh, min_area, morph_kernel)
    s = float(display_scale)
    img_boxes = boxes if s >= 1.0 else [
        [int(x / s), int(y / s), int(w / s), int(h / s)] for x, y, w, h in boxes]
    return {"mask": mask, "boxes_display": boxes,
            "boxes_image": [[x, y, x + w, y + h] for x, y, w, h in img_boxes],
            "params": {"rel_thresh": float(rel_thresh), "min_area": int(min_area),
                       "morph_kernel": int(morph_kernel),
                       "display_size": [int(display_wh[0]), int(display_wh[1])],
                       "display_scale": s}}


def load_raw_heatmap(path, base_dir) -> np.ndarray:
    """读回留存原始热图；路径必须位于 base_dir（热图目录）内，拒绝穿越。"""
    import os
    base = Path(base_dir).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else base / p).resolve()
    try:
        inside = os.path.commonpath([str(base), str(p)]) == str(base)
    except ValueError:  # 不同盘符
        inside = False
    if not inside or p.suffix.lower() != ".npy":
        raise PermissionError(f"raw heatmap 路径不在热图目录内: {path}")
    return np.load(str(p), allow_pickle=False)


def recompute_from_postproc(postproc: dict, base_dir, rel_thresh=None,
                            min_area=None, morph_kernel=None) -> dict:
    """按 heatmap_postproc 元数据读回 raw 并复算；未给的参数取在线留存值。"""
    raw = load_raw_heatmap(postproc["raw_heatmap_path"], base_dir)
    return recompute_heatmap_mask(
        raw, postproc["display_size"],
        rel_thresh=postproc["rel_thresh"] if rel_thresh is None else rel_thresh,
        min_area=postproc["min_area"] if min_area is None else min_area,
        morph_kernel=postproc["morph_kernel"] if morph_kernel is None else morph_kernel,
        display_scale=postproc.get("display_scale", 1.0))


def _save_npy(path, arr: np.ndarray) -> bool:
    try:
        np.save(str(path), np.ascontiguousarray(arr), allow_pickle=False)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("原始热图写盘失败 %s", path)
        return False


def render_heatmap_overlay(image_bgr: np.ndarray, heatmap: np.ndarray,
                           alpha: float = 0.5, engine_boxes=None,
                           rel_thresh: float = 0.6,
                           min_area: int = 16,
                           morph_kernel: int = 3,
                           return_mask: bool = False):
    """返回 (纯热力图, 叠加图, 热图连通域框 [x,y,w,h] 显示坐标)。

    AC-063：叠加图上画引擎框（engine_boxes 显示坐标 xyxy，红）；热图连通域框
    （heatmap_cc，仅可视化来源，黄细线）单独返回，不再覆盖引擎 defect_boxes。"""
    h, w = image_bgr.shape[:2]
    hm_u8 = _heatmap_u8(heatmap, (w, h))
    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    mask, boxes = _mask_and_boxes(hm_u8, rel_thresh, min_area, morph_kernel)
    overlay = _compose_overlay(image_bgr, heat_color, alpha, engine_boxes, boxes)
    if return_mask:  # AC-063：形态学后的二值 mask（显示分辨率，0/255）供留存
        return heat_color, overlay, boxes, mask
    return heat_color, overlay, boxes


def _compose_overlay(image_bgr: np.ndarray, heat_color: np.ndarray, alpha: float,
                     engine_boxes, cc_boxes) -> np.ndarray:
    """叠加图：原图×(1-α)+着色热图×α；无引擎框时画热图连通域框（黄），否则画引擎框（红）。"""
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heat_color, alpha, 0)
    if engine_boxes is None:
        for x, y, bw, bh in cc_boxes:
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 255), 1)
    for x0, y0, x1, y1 in engine_boxes or []:
        cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return overlay


def _display_bgr(arr_rgb: np.ndarray, disp_wh) -> np.ndarray:
    """原图 RGB → 显示分辨率 BGR（先缩放后换通道，与先换后缩逐位一致）。"""
    if (int(disp_wh[0]), int(disp_wh[1])) != (arr_rgb.shape[1], arr_rgb.shape[0]):
        arr_rgb = cv2.resize(arr_rgb, (int(disp_wh[0]), int(disp_wh[1])),
                             interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)


# ── 产物后台渲染/写盘（TEST-01 dev_async）────────────────────
# 着色/叠加/画框/编码与落库并行（不影响响应里的框：框由主线程同步算）。
# 未完成产物登记在 _pending，供 /api/file 在文件写完前等待而非 404。
_pending: Dict[str, object] = {}
_pending_lock = threading.Lock()


def _track(paths, fut) -> None:
    keys = [str(Path(p)) for p in paths]
    with _pending_lock:
        for k in keys:
            _pending[k] = fut

    def _done(_f, keys=keys):
        with _pending_lock:
            for k in keys:
                if _pending.get(k) is _f:
                    del _pending[k]
    fut.add_done_callback(_done)


def wait_artifact(path, timeout: float = 5.0) -> Optional[bool]:
    """等待产物写盘：None=非在途产物；True=已写成；False=写盘失败。
    超时抛 concurrent.futures.TimeoutError。"""
    with _pending_lock:
        fut = _pending.get(str(Path(path)))
    if fut is None:
        return None
    return bool(fut.result(timeout=timeout))


def _safe_job(fn, *args) -> bool:
    try:
        return bool(fn(*args))
    except Exception:  # noqa: BLE001
        logger.exception("检测产物渲染/写盘异常 %s", getattr(fn, "__name__", fn))
        return False


def _job_heat(path, hm_u8: np.ndarray) -> bool:
    return imwrite(path, cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET))


def _job_overlay(path, arr_rgb: np.ndarray, disp_wh, hm_u8: np.ndarray,
                 alpha: float, engine_boxes) -> bool:
    heat_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = _compose_overlay(_display_bgr(arr_rgb, disp_wh), heat_color,
                               alpha, engine_boxes, ())
    return imwrite(path, overlay)


def _submit(fn, path, *args):
    fut = _io_exec.submit(_safe_job, fn, path, *args)
    _track([path], fut)
    return fut


HEATMAP_POSTPROC_ALGO = "minmax_rel_thresh+morph_open+cc_bbox/v1"


def heatmap_postproc_meta(rel_thresh: float, min_area: int, morph_kernel: int,
                          display_scale: float) -> dict:
    """AC-065：热图后处理参数及确定性 version_id（同参数同 id，可追溯/对比）。"""
    import hashlib
    params = {"algo": HEATMAP_POSTPROC_ALGO, "rel_thresh": round(float(rel_thresh), 6),
              "threshold_basis": "per_image_minmax", "min_area": int(min_area),
              "min_area_unit": "display_px", "morph_kernel": int(morph_kernel)}
    key = "|".join(f"{k}={params[k]}" for k in sorted(params))
    params["version_id"] = "hpp-" + hashlib.sha1(key.encode()).hexdigest()[:12]
    params["display_scale"] = display_scale
    return params


_CV2_SAFE_FORMATS = {"JPEG", "BMP", "PNG"}
_CV2_SAFE_MODES = {"RGB", "L"}
# PNG 仅放行 8 位真彩/灰度（rawmode 精确匹配）；16 位、1/2/4 位灰度、带 tRNS、
# APNG 多帧的降位/透明语义两库可能不同，仍走 PIL。
_PNG_SAFE_RAWMODES = {"RGB", "L"}
# OPT-DECODE：cv2>=4.11 可直接解码为 RGB，省一次 BGR2RGB（2500² 约 6ms）
_CV2_RGB_FLAG = getattr(cv2, "IMREAD_COLOR_RGB", None)
_io_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="det-io")


def _cv2_decodable(pil) -> bool:
    if pil.format not in _CV2_SAFE_FORMATS or pil.mode not in _CV2_SAFE_MODES:
        return False
    if pil.format == "PNG":
        if "transparency" in pil.info or getattr(pil, "n_frames", 1) != 1:
            return False
        tile = pil.tile
        return len(tile) == 1 and tile[0][3] in _PNG_SAFE_RAWMODES
    return True


def _decode_rgb(image_path) -> np.ndarray:
    """解码为 RGB uint8，像素与 PIL ``convert("RGB")`` 一致。

    G1：2500² JPEG 上 PIL 解码约 74ms，cv2 约 42ms，实测 maxdiff=0。
    OPT-DECODE（2026-10-13）：8 位 PNG（RGB/L）也走 cv2（libpng，2500²
    PNG PIL 约 110ms → cv2 直出 RGB 约 89ms，maxdiff=0）。PIL 仅读头
    （约 0.1ms）做格式门控；CMYK、调色板、16 位、透明等语义可能不同的
    格式仍用 PIL。cv2 默认按 EXIF 旋转而 PIL 不旋转，因此加
    IGNORE_ORIENTATION 保持一致；尺寸与头信息不符时回退 PIL。
    """
    with Image.open(str(image_path)) as pil:
        if _cv2_decodable(pil):
            flags = cv2.IMREAD_IGNORE_ORIENTATION | (
                _CV2_RGB_FLAG if _CV2_RGB_FLAG is not None else cv2.IMREAD_COLOR)
            img = imread(image_path, flags)
            if (img is not None and img.ndim == 3 and img.shape[2] == 3
                    and img.shape[1::-1] == pil.size):
                if _CV2_RGB_FLAG is None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        return np.array(pil.convert("RGB"))


# ── 检测服务 ─────────────────────────────────────────────────
class DetectionService:
    def __init__(self):
        self.settings = get_settings()

    @property
    def engine(self):
        return get_engine()

    def get_pipeline(self, category: str):
        """按品类返回 引擎 pipeline（未准备时抛错）。"""
        return self.engine.get_pipeline(category)

    # ---- 图片检测 ----
    def detect_image(self, image_path: str | Path, category: str,
                     image_id: Optional[int] = None,
                     target_type: str = "image",
                     persist: bool = True,
                     with_heatmap: bool = True,
                     workorder_id: Optional[int] = None,
                     claim_token: Optional[str] = None) -> Dict:
        engine = self.engine
        t0 = time.perf_counter()
        # with_heatmap 时复用同一次解码与数组（predict + overlay 共用）
        arr_rgb = None
        try:
            if with_heatmap:
                arr_rgb = _decode_rgb(image_path)
                result = engine.predict_ndarray(category, arr_rgb,
                                                path_hint=str(image_path))
            else:
                result = engine.predict_image_path(category, str(image_path))
        except RuntimeError as e:
            raise ValueError(
                f"品类 '{category}' 未准备，请先调用 /models/prepare（{e}）")
        return self._finish_detect(result, arr_rgb, str(image_path), category,
                                   t0, image_id=image_id,
                                   target_type=target_type, persist=persist,
                                   with_heatmap=with_heatmap,
                                   workorder_id=workorder_id,
                                   claim_token=claim_token)

    def detect_frame(self, data: bytes, category: str, persist: bool = True,
                     with_heatmap: bool = True) -> Dict:
        """内存帧零拷贝检测（产线"相机取图直入内存"形态，不足清单 #5）。

        与 detect_pil（先 PNG 落盘再读回）相对：bytes -> imdecode ->
        ndarray 直入引擎，全程不落盘；对应接口 POST /api/detect/frame。
        供上位机/相机 SDK 拿到帧缓冲后直接送检，消除磁盘 IO 往返。
        """
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("图像解码失败（仅支持 jpg/png/bmp/webp 帧数据）")
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        try:
            result = self.engine.predict_ndarray(category, arr_rgb,
                                                 path_hint=None)
        except RuntimeError as e:
            raise ValueError(
                f"品类 '{category}' 未准备，请先调用 /models/prepare（{e}）")
        out = self._finish_detect(result, arr_rgb, "<memory_frame>", category,
                                  t0, image_id=None, target_type="upload",
                                  persist=persist, with_heatmap=with_heatmap)
        out["zero_copy"] = True   # 口径标记：内存直入（无磁盘 IO）
        return out

    def _finish_detect(self, result, arr_rgb, image_path: str, category: str,
                       t0: float, image_id: Optional[int] = None,
                       target_type: str = "image", persist: bool = True,
                       with_heatmap: bool = True,
                       workorder_id: Optional[int] = None,
                       claim_token: Optional[str] = None) -> Dict:
        """检测结果统一收尾：字段组装 + 可视化 + e2e 口径 + 落库 + 告警。"""
        total_ms = (time.perf_counter() - t0) * 1000

        slot_scores = dict(result.slot_scores or {})
        n_tiles = {
            "slots": slot_scores,
            "raw": dict(result.raw_scores or {}),
            "weights": dict(result.weights or {}),
            "router_w": dict(result.router_w or {}),
            "actual_router": dict(result.router_w or {}),
            "candidate_router": (result.extra or {}).get("candidate_router"),
            "feature_metadata": (result.extra or {}).get("feature_metadata"),
            "triggered_slot": result.triggered_slot,
            "cal": dict(slot_scores),
            "decision": result.decision,
            "decision_trace": list((result.extra or {}).get("decision_trace", [])),
            # AC-014：引擎框（原图 x0,y0,x1,y1）+ 逐框区域证据；与渲染框分开保存
            "box_evidence": [{"bbox": list(b["bbox"]), "dominant_slot": b.get("dominant_slot", ""),
                              "source": b.get("source", "engine_box"), "evidence": b.get("evidence")}
                             for b in (result.defect_boxes or [])],
        }
        # AC-063：定位容器——区域记录（原图 xyxy、来源、分数、后处理版本、候选状态）
        loc = dict((result.extra or {}).get("localization") or {"status": "legacy_unknown"})
        loc["regions"] = [{k: b.get(k) for k in (
            "region_id", "bbox", "area", "source", "score_peak", "score_mean",
            "dominant_slot", "postproc", "status", "confirmed")}
            for b in (result.defect_boxes or [])]
        loc["heatmap_boxes"] = []
        # AC-063：坐标系=原图像素 xyxy；记录原图尺寸与产生定位的模型版本，缺失为 None
        loc["coord"] = "image_xyxy"
        loc["image_size"] = ([int(arr_rgb.shape[1]), int(arr_rgb.shape[0])]
                             if arr_rgb is not None else None)
        loc["model_version"] = result.model_version
        n_tiles["localization"] = loc
        out = {
            "image_path": str(image_path),
            "category": category,
            "final_score": result.score,
            "is_anomaly": bool(result.is_anomaly),
            "decision": result.decision,
            **detection_interpretation(result.decision, bool(result.is_anomaly)),
            "threshold": result.threshold,
            "gray_threshold": result.gray_threshold,
            # 前端兼容：旧三档分数位复用为 sem/disc/shead 槽位分
            "level1_score": slot_scores.get("sem"),
            "level2_score": slot_scores.get("disc"),
            "level3_score": slot_scores.get("shead"),
            "triggered_level": result.triggered_slot or None,
            "slots": slot_scores,
            "weights": dict(result.weights or {}),
            "raw_scores": dict(result.raw_scores or {}),
            "cal_scores": dict(slot_scores),
            "router_w": dict(result.router_w or {}),
            "actual_router": dict(result.router_w or {}),
            "candidate_router": (result.extra or {}).get("candidate_router"),
            "feature_metadata": (result.extra or {}).get("feature_metadata"),
            "triggered_slot": result.triggered_slot,
            "box_evidence": n_tiles["box_evidence"],
            "localization": loc,
            "open_alert": bool(result.open_alert),
            "e_open_score": (result.extra or {}).get("e_open_score"),
            "e_open_threshold": (result.extra or {}).get("e_open_threshold"),
            # M11b：缺陷类型归因（U96 有监督判别器 top-1，未激活时回退规则映射）
            "defect_types": list(result.types or []),
            # M11d：对位偏移预警（治具/传送带漂移，offset 为参考分辨率像素）
            "align_offset": result.align_offset,
            "align_warn": bool(result.align_warn),
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
            "total_latency_ms": total_ms,
            # AC-069：等品类锁时间（被同品类学习/巩固阻塞的部分，含于 total_latency_ms）
            "lock_wait_ms": (result.extra or {}).get("lock_wait_ms"),
            "n_tiles": n_tiles,
        }

        # 引擎缺陷框（bbox x0,y0,x1,y1 → 前端契约 x,y,w,h）
        out["defect_boxes"] = [
            [int(b["bbox"][0]), int(b["bbox"][1]),
             int(b["bbox"][2] - b["bbox"][0]),
             int(b["bbox"][3] - b["bbox"][1])]
            for b in (result.defect_boxes or [])]

        # 可视化产物（高速模式跳过：省大图二次处理与存盘 ~100ms+）
        # 展示级分辨率（≤1920）+ JPEG：12MP PNG 存盘是热力图路径最大开销
        io_futs: list = []
        if with_heatmap and arr_rgb is not None:
            heatmap = _best_heatmap(result)
            if heatmap is not None:
                h0, w0 = arr_rgb.shape[:2]
                scale = 1.0
                disp_wh = (int(w0), int(h0))
                max_disp = int(self.settings.get("pipeline",
                                                 "overlay_max_dim", 1920))
                if max(h0, w0) > max_disp:
                    scale = max_disp / max(h0, w0)
                    disp_wh = (int(w0 * scale), int(h0 * scale))
                eng_disp = [[v * scale for v in b["bbox"]]
                            for b in (result.defect_boxes or [])]
                rel_thresh = float(self.settings.get(
                    "pipeline", "overlay_box_rel_thresh", 0.6))
                min_area = int(self.settings.get(
                    "pipeline", "overlay_box_min_area", 16))
                morph_k = int(self.settings.get(
                    "pipeline", "overlay_box_morph_kernel", 3))
                save_raw = bool(self.settings.get(
                    "pipeline", "overlay_save_raw_heatmap", True))
                if save_raw:  # 在线渲染与留存同一份量化热图，复算逐位一致
                    heatmap = quantize_raw_heatmap(heatmap)
                hm_dir = self.settings.storage("heatmaps")
                stem = Path(image_path).stem + f"_{int(time.time()*1000)}"
                hm_path = hm_dir / f"{stem}_heat.jpg"
                ov_path = hm_dir / f"{stem}_overlay.jpg"
                # TEST-01 dev_async：与 render_heatmap_overlay 同一实现拆分——
                # 显示缩放/换通道/着色/叠加/画框/JPEG 编码进线程池（cv2 释放
                # GIL），与下方连通域框计算及落库并行；响应框仍由主线程同步算。
                hm_u8 = _heatmap_u8(heatmap, disp_wh)
                io_futs = [("overlay", _submit(_job_overlay, ov_path, arr_rgb,
                                               disp_wh, hm_u8, 0.5, eng_disp)),
                           ("heat", _submit(_job_heat, hm_path, hm_u8))]
                bin_mask, boxes = _mask_and_boxes(hm_u8, rel_thresh, min_area,
                                                  morph_k)
                # AC-063/065：热图连通域后处理参数随结果留存（min_area 为展示分辨率像素）
                loc["heatmap_postproc"] = heatmap_postproc_meta(
                    rel_thresh, min_area, morph_k, scale)
                # 原始热图值域（min-max 归一化基准），便于由热图 JPEG 外复现阈值
                loc["heatmap_postproc"]["raw_range"] = [float(heatmap.min()),
                                                        float(heatmap.max())]
                if scale < 1.0:  # 热图连通域框换算回原图像素坐标
                    boxes = [[int(x / scale), int(y / scale),
                              int(w / scale), int(h / scale)]
                             for x, y, w, h in boxes]
                # AC-063：热图连通域框仅作可视化来源（heatmap_cc），不覆盖引擎框
                loc["heatmap_boxes"] = [
                    {"bbox": [x, y, x + bw, y + bh], "source": "heatmap_cc",
                     "status": "visual_only", "confirmed": False}
                    for x, y, bw, bh in boxes]
                # G1：返回前 join，调用方拿到结果时文件必已就绪。
                out.update(heatmap_path=str(hm_path), overlay_path=str(ov_path))
                # AC-063：二值 mask 无损 PNG 留存（可关）；与 JPEG 并行写盘
                if bool(self.settings.get("pipeline", "overlay_save_mask", True)):
                    mk_path = hm_dir / f"{stem}_mask.png"
                    io_futs.append(("mask", _submit(imwrite, mk_path, bin_mask)))
                    loc["heatmap_postproc"]["mask_path"] = str(mk_path)
                    out["mask_path"] = str(mk_path)
                # AC-063：原始热图（模型输出/工作网格分辨率，float16 .npy）留存，
                # 供离线按不同阈值复算 mask/框（recompute_from_postproc）
                pp = loc["heatmap_postproc"]
                dw, dh = disp_wh
                rh, rw = heatmap.shape[:2]
                pp["display_size"] = [int(dw), int(dh)]
                pp["image_size"] = [int(w0), int(h0)]
                if save_raw:
                    raw_path = hm_dir / f"{stem}_raw.npy"
                    io_futs.append(("raw", _submit(_save_npy, raw_path, heatmap)))
                    pp.update(
                        raw_heatmap_path=str(raw_path),
                        raw_heatmap_file=raw_path.name,
                        raw_shape=[int(rh), int(rw)],
                        raw_dtype=str(heatmap.dtype),
                        raw_resolution="model_output_work_grid",
                        # 坐标关系：raw(列,行)×scale_raw_to_display → 显示像素；
                        # 显示像素 /display_scale → 原图像素（框取 int 截断）
                        resize="INTER_LINEAR",
                        scale_raw_to_display=[dw / rw, dh / rh],
                        scale_raw_to_image=[w0 / rw, h0 / rh],
                        box_to_image="int(v/display_scale)")
                    out["raw_heatmap_path"] = str(raw_path)

        # 端到端口径（不足清单 #5 补强，2026-08-30）：解码+推理+热力图渲染。
        # 此前 latency_e2e_ms DB 列/StatsDaily 列无人写入，统计全部回退纯
        # 推理口径（routes_stats coalesce），名不副实；persist/alarm 属检测
        # 后记账，不计入。TEST-01 dev_async 起渲染只含同步段（热图上采样+
        # 连通域框），着色/叠加/编码在线程池与落库并行（返回前仍 join）。
        out["latency_e2e_ms"] = (time.perf_counter() - t0) * 1000
        out["persisted"] = False
        out["feedback_supported"] = False
        out["detection_id"] = None

        try:
            # 严格验证而非清洗：非法数值必须在落库、统计和告警之前失败。
            json.dumps(out, allow_nan=False)
            if persist:
                out["detection_id"] = self._persist(
                    out, image_id, target_type, workorder_id=workorder_id,
                    claim_token=claim_token)
                out["persisted"] = True
                out["feedback_supported"] = True
        finally:
            failed = [kind for kind, f in io_futs if not f.result()]
            if failed:
                logger.warning("热力图写盘失败 %s %s", image_path, failed)
                self._mark_artifact_error(out.get("detection_id"), failed)
        # 持久化已在上方事务中提交；告警属于旁路副作用，失败不得让
        # pipeline worker 把已成功的 claim 改写成 failed。
        try:
            out["alarm"] = self._record_alarm(out["category"], out["is_anomaly"])
        except Exception:  # noqa: BLE001
            logger.exception("连续异常告警记录失败 category=%s", out["category"])
            out["alarm"] = {"active": False, "window_hits": 0}
        return out

    def detect_pil(self, pil_image: Image.Image, category: str,
                   save_dir: Optional[Path] = None) -> Dict:
        """对内存中的 PIL 图像检测（上传场景）。先落盘再检测，保证热力图一致。"""
        save_dir = save_dir or self.settings.storage("uploads")
        path = save_dir / f"upload_{int(time.time()*1000)}.png"
        pil_image.convert("RGB").save(path)
        return self.detect_image(path, category, target_type="upload")

    # ---- 视频抽帧检测（生成器，供WS/后台任务消费）----
    def iter_video_detections(self, video_path: str | Path, category: str,
                              interval: Optional[int] = None,
                              video_id: Optional[int] = None,
                              persist: bool = True,
                              with_heatmap: bool = True) -> Iterator[Dict]:
        interval = interval or self.settings.get("pipeline", "video_frame_interval", 5)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        idx, sampled = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % interval == 0:
                    tmp = self.settings.storage("uploads") / f"frame_{video_id or 0}_{idx}.jpg"
                    imwrite(tmp, frame)
                    det = self.detect_image(tmp, category, target_type="video_frame",
                                            persist=persist,
                                            with_heatmap=with_heatmap)
                    det.update(frame_idx=idx, sampled_idx=sampled, video_id=video_id)
                    if persist:
                        # M8a：帧图登记 Image（挂视频批次）+ 唤醒 VideoFrame 表
                        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                        frame_image_id = self._register_frame(
                            video_id, idx, tmp, category, ts_ms)
                        if frame_image_id is not None:
                            det["image_id"] = frame_image_id
                            self._attach_frame_image(det, frame_image_id)
                        self._attach_video(det, video_id, idx)
                    sampled += 1
                    yield det
                idx += 1
        finally:
            cap.release()

    # ---- 入库与统计 ----
    def _persist(self, det: Dict, image_id: Optional[int], target_type: str,
                 workorder_id: Optional[int] = None,
                 claim_token: Optional[str] = None) -> int:
        with session_scope() as s:
            claim_requeued_at = None
            if claim_token is not None:
                claim = s.query(ImageRow.claim_requeued_at).filter(
                    ImageRow.id == image_id,
                    ImageRow.claim_token == claim_token).first()
                if claim is not None:
                    claim_requeued_at = claim[0]
                # 同一事务内核对认领、写检测及完成状态；拒绝已撤销的 worker。
                claimed = (s.query(ImageRow)
                           .filter(ImageRow.id == image_id,
                                   ImageRow.processing_status == "claiming",
                                   ImageRow.claim_token == claim_token,
                                   ImageRow.claim_workorder_id == workorder_id)
                           .update({ImageRow.processing_status: "succeeded",
                                    ImageRow.claim_token: None,
                                    ImageRow.claim_workorder_id: None,
                                    ImageRow.claim_requeued_at: None,
                                    ImageRow.processed_at: datetime.now(),
                                    ImageRow.last_error: None},
                                   synchronize_session=False))
                if claimed != 1:
                    raise ValueError("产线 claim 已失效，拒绝持久化检测")
            # M7 修补：记录检测当时实际使用的模型版本（追溯链"哪个版本判的"）。
            # 版本由引擎在品类锁内捕获并随 DetectionResult 透传，不能在此重读
            # current.json，否则并发激活可能把检测归档到另一版本。
            model_id = None
            try:
                ver = det.get("model_version")
                if ver is not None:
                    m = (s.query(Model)
                         .filter(Model.category == det["category"],
                                 Model.version == f"v{ver}")
                         .order_by(Model.id.desc()).first())
                    model_id = m.id if m is not None else None
            except Exception:  # noqa: BLE001 版本缺失不阻塞落库
                pass
            row = Detection(
                target_type=target_type, image_id=image_id,
                workorder_id=workorder_id, claim_token=claim_token,
                claim_requeued_at=claim_requeued_at,
                image_path=det["image_path"], category=det["category"],
                model_id=model_id,
                final_score=det["final_score"], is_anomaly=det["is_anomaly"],
                level1_score=det.get("level1_score"), level2_score=det.get("level2_score"),
                level3_score=det.get("level3_score"),
                triggered_level=det.get("triggered_level"),
                latency_ms=det["latency_ms"],
                # 端到端口径独立列（旧数据为 NULL，统计 coalesce 回退推理口径）
                latency_e2e_ms=det.get("latency_e2e_ms"),
                n_tiles={**(det.get("n_tiles") or {}),
                         "model_version": det.get("model_version"),
                         "threshold": det.get("threshold"),
                         "gray_threshold": det.get("gray_threshold"),
                         "boxes": det.get("defect_boxes") or [],
                         "types": det.get("defect_types") or [],  # M11b 归因落库
                         "decision_trace": det.get("decision_trace") or [],
                         "open_alert": bool(det.get("open_alert")),  # M14c 不完备报告原料
                         "align_warn": bool(det.get("align_warn")),
                         "align_offset": det.get("align_offset"),  # M11d 对位落库
                         "infer_latency_ms": det.get("latency_ms"),  # 模型推理耗时（赛题口径）
                         "total_latency_ms": det.get("total_latency_ms")},  # 端到端全链路（用户感知口径）
                heatmap_path=det.get("heatmap_path"), overlay_path=det.get("overlay_path"),
            )
            if det.get("open_alert"):
                row.n_tiles = {**row.n_tiles,
                               "e_open_score": det.get("e_open_score"),
                               "e_open_threshold": det.get("e_open_threshold"),
                               "open_alert_item": {
                                   "status": "queued", "revision": 0,
                                   "e_open_score": det.get("e_open_score"),
                                   "e_open_threshold": det.get("e_open_threshold"),
                                   "operator": None, "disposed_at": None,
                                   "reason": None, "feedback_id": None, "events": []}}
            s.add(row)
            s.flush()
            det_id = row.id
            stats = s.query(StatsDaily).filter(StatsDaily.date == date.today()).first()
            if stats is None:
                stats = StatsDaily(date=date.today())
                s.add(stats)
                s.flush()
            stats.n_inspected += 1
            stats.n_anomaly += int(det["is_anomaly"])
            stats.total_latency_ms += det.get("total_latency_ms", det["latency_ms"])
            stats.total_latency_e2e_ms += float(
                det.get("latency_e2e_ms") or det["latency_ms"])

        # M5a：webhook 推送（异步，不阻塞检测路径；未配置即禁用）
        decision = (det.get("n_tiles") or {}).get("decision", "")
        event = "anomaly" if det["is_anomaly"] else ("gray" if decision == "gray" else None)
        def _notify_safely(event_name: str, payload: dict) -> None:
            try:
                notify_event(event_name, payload)
            except Exception:  # noqa: BLE001
                # 通知是提交后的旁路副作用，不能反向改变已提交结果。
                logger.exception("检测事件通知失败 event=%s detection_id=%s",
                                 event_name, det_id)

        if event:
            _notify_safely(event, {
                "category": det["category"], "detection_id": det_id,
                "score": det["final_score"], "decision": decision,
                "image_path": det["image_path"],
                "defect_boxes": det.get("defect_boxes") or [],
                "defect_types": det.get("defect_types") or [],   # M11b 归因（MES 判废依据）
                "latency_ms": det["latency_ms"],
            })
        # M6a：开放集未知缺陷事件（open 槽哨兵命中：不像任何已知正常）
        if det.get("open_alert"):
            _notify_safely("open_set", {
                "category": det["category"], "detection_id": det_id,
                "score": det["final_score"], "image_path": det["image_path"],
            })
        # M11d：对位偏移预警事件（治具/传送带漂移，提醒现场检查定位）
        if det.get("align_warn"):
            logger.warning("对位偏移预警 detection_id=%s category=%s offset=%s",
                           det_id, det["category"], det.get("align_offset"))
            _notify_safely("align_warn", {
                "category": det["category"], "detection_id": det_id,
                "align_offset": det.get("align_offset"),
                "image_path": det["image_path"],
            })
        # M10b：缺陷自动归档（证据留存/复盘回流，不经反馈；失败不阻塞检测）
        try:
            self._auto_archive(det, det_id)
        except Exception:  # noqa: BLE001
            logger.exception("缺陷自动归档失败 detection_id=%s", det_id)
        return det_id

    @staticmethod
    def _mark_artifact_error(detection_id: Optional[int], failed: List[str]) -> None:
        """产物写盘失败可查询：Detection.n_tiles["artifact_error"]（仅失败时出现）。"""
        if not detection_id:
            return
        try:
            with session_scope() as s:
                row = s.get(Detection, detection_id)
                if row is not None:
                    row.n_tiles = {**(row.n_tiles or {}), "artifact_error": {
                        "failed": list(failed), "at": datetime.now().isoformat()}}
        except Exception:  # noqa: BLE001
            logger.exception("产物失败标记写入失败 detection_id=%s", detection_id)

    # ---- 缺陷自动归档（M10b，评审自检 §9 P1）----
    def _auto_archive(self, det: Dict, detection_id: int) -> None:
        """命中归档决策（默认 anomaly）的帧自动拷贝到
        storage/archive/{category}/{yyyymmdd}/ 并登记 Image
        （split=archive / label=anomaly / source=auto_archive，挂当日
        "自动归档"批次），供复盘、训练回流与证据留存。"""
        from ..core.datasets import get_or_create_dataset
        cfg = self.settings.section("archive") or {}
        if not cfg.get("enabled", True):
            return
        decision = det.get("decision") or ""
        if decision not in (cfg.get("decisions") or ["anomaly"]):
            return
        src = Path(det["image_path"])
        if not src.is_file():
            return
        day = date.today().strftime("%Y%m%d")
        dst_dir = self.settings.storage("archive") / det["category"] / day
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"det{detection_id}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        with session_scope() as s:
            ds_id = get_or_create_dataset(
                name=f"自动归档-{day}", category=det["category"],
                source_type="auto_archive")
            img = s.query(ImageRow).filter(ImageRow.path == str(dst)).first()
            if img is None:
                img = ImageRow(path=str(dst), category=det["category"],
                               split="archive", label="anomaly",
                               source="auto_archive", dataset_id=ds_id)
                s.add(img)
                s.flush()
                ds = s.get(Dataset, ds_id)
                if ds is not None:
                    ds.n_images = (ds.n_images or 0) + 1
        det["archive_path"] = str(dst)

    # ---- 连续异常告警（M6a：滑窗异常率，产线批量缺陷信号）----
    @staticmethod
    def _record_alarm(category: str, is_anomaly: bool) -> Dict:
        """每帧滑窗判定（O(窗口) 内存操作，无推理开销）。

        fired 时记 OperationLog + webhook "alarm"（未订阅/未配置即静默）。
        返回 {active, window_hits}，随 detect 响应与 WS 帧下发。
        """
        mon = get_alarm_monitor()
        fired, hits = mon.record(category, is_anomaly)
        if fired:
            try:
                log_action("alarm",
                           f"category={category} window_hits={hits} "
                           f"window={mon.window} k={mon.k}")
                notify_event("alarm", {
                    "category": category, "window_hits": hits,
                    "window": mon.window, "k": mon.k,
                })
            except Exception:  # noqa: BLE001
                # 告警记录/通知是提交后的旁路副作用。
                logger.exception("连续异常告警副作用失败 category=%s", category)
        return {"active": hits >= mon.k, "window_hits": hits}

    @staticmethod
    def _attach_video(det: Dict, video_id: Optional[int], frame_idx: int) -> None:
        if video_id is None or not det.get("detection_id"):
            return
        with session_scope() as s:
            row = s.get(Detection, det["detection_id"])
            if row:
                row.video_id = video_id
                row.frame_idx = frame_idx

    # ---- 视频帧登记（M8a：唤醒 VideoFrame 死表，帧图挂视频批次）----
    @staticmethod
    def _register_frame(video_id: Optional[int], frame_idx: int,
                        frame_path, category: str,
                        timestamp_ms: float = 0.0) -> Optional[int]:
        """帧图登记 Image（split=unlabeled, source=video, 挂视频导入时建的批次）
        + VideoFrame 行（幂等：video_id+frame_idx 已存在则跳过）。返回 image_id。"""
        if video_id is None:
            return None
        from ..core import datasets as ds_helper
        from ..core.ingest import register_image
        with session_scope() as s:
            video = s.get(Video, video_id)
            if video is None:
                return None
            ds_id = video.dataset_id
            vf = (s.query(VideoFrame)
                  .filter(VideoFrame.video_id == video_id,
                          VideoFrame.frame_idx == frame_idx).first())
        image_id = register_image(
            frame_path, category=category, split="unlabeled", label="unknown",
            source="video", copy_to_storage=False, dataset_id=ds_id)
        if vf is None:
            with session_scope() as s:
                s.add(VideoFrame(video_id=video_id, frame_idx=frame_idx,
                                 timestamp_ms=timestamp_ms, image_id=image_id))
        if ds_id is not None:
            ds_helper.refresh_dataset_count(ds_id)
        return image_id

    @staticmethod
    def _attach_frame_image(det: Dict, image_id: int) -> None:
        """把帧检测记录关联到登记后的 Image 行。"""
        if not det.get("detection_id"):
            return
        with session_scope() as s:
            row = s.get(Detection, det["detection_id"])
            if row:
                row.image_id = image_id


_service: DetectionService | None = None


def get_detection_service() -> DetectionService:
    global _service
    if _service is None:
        _service = DetectionService()
    return _service
