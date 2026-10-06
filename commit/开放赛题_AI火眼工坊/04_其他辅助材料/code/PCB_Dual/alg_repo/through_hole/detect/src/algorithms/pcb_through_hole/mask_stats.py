"""相对标准/锡面统计的自适应阈值工具 + 标准/测试锡面尺寸公差圈处理"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from . import config as C


@lru_cache(maxsize=64)
def _ellipse_kernel(ksize: int) -> np.ndarray:
    """按核尺寸缓存的椭圆结构元。
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))


@dataclass
class SolderRelStats:
    """标准或测试锡面 HSV 相对统计。"""
    v_median: float = 128.0
    v_std: float = 20.0
    v_p10: float = 100.0
    v_p25: float = 115.0
    s_median: float = 80.0
    s_p75: float = 120.0
    area: int = 1


def compute_solder_rel_stats(hsv: np.ndarray, solder: np.ndarray) -> SolderRelStats:
    """从锡面 mask 统计 V/S 分布，供相对阈值使用。"""
    sel = solder > 0
    if sel.sum() < 30:
        return SolderRelStats(area=int(sel.sum()))
    v = hsv[:, :, 2][sel].astype(np.float32)
    s = hsv[:, :, 1][sel].astype(np.float32)
    return SolderRelStats(
        v_median=float(np.median(v)),
        v_std=float(max(v.std(), 8.0)),
        v_p10=float(np.percentile(v, 10)),
        v_p25=float(np.percentile(v, 25)),
        s_median=float(np.median(s)),
        s_p75=float(np.percentile(s, 75)),
        area=int(sel.sum()),
    )


def void_v_threshold(stats: SolderRelStats, *, abs_cap: int | None = None) -> int:
    """孔洞暗区 V 上限：相对锡面 median/std，并以绝对上限兜底。"""
    cap = int(abs_cap if abs_cap is not None else C.VOID_V_MAX)
    k = float(C.TH.get("void_v_rel_median_k", 0.70))
    rel = stats.v_median - k * stats.v_std
    rel = min(rel, stats.v_p25 - float(C.TH.get("void_v_rel_margin", 3.0)))
    floor = float(C.TH.get("void_v_abs_floor", 60))
    return int(np.clip(rel, floor, cap))


def void_s_threshold(stats: SolderRelStats, *, abs_cap: int | None = None) -> int:
    """孔洞暗区 S 上限：相对锡面饱和度，低饱和暗区。"""
    cap = int(abs_cap if abs_cap is not None else C.TH.get("void_dark_s_max", 130))
    rel = stats.s_median + float(C.TH.get("void_s_rel_margin", 50.0))
    return int(min(cap, max(40, rel)))


def void_desat_threshold(stats: SolderRelStats) -> int:
    """孔洞「真实低饱和裸露暗面」判定 S 上限：相对标准锡面自身饱和度基线下调
    (而非绝对阈值)，区别于少锡露铜/曲面阴影等仍带原色调的「有色暗区」伪影。"""
    margin = float(C.TH.get("void_desat_s_rel_margin", 40.0))
    floor = float(C.TH.get("void_desat_s_floor", 40.0))
    return int(max(floor, stats.s_median - margin))


def highlight_v_threshold(stats: SolderRelStats) -> int:
    """高亮 V 下限：相对锡面中位数偏移。"""
    abs_min = int(C.HIGHLIGHT_V_MIN)
    rel = stats.v_median + float(C.TH.get("highlight_v_rel_delta", 85.0))
    rel = max(rel, stats.v_p25 + float(C.TH.get("highlight_v_rel_p25_delta", 60.0)))
    return int(min(255, max(abs_min, rel)))


def highlight_s_threshold(stats: SolderRelStats) -> int:
    """高亮 S 上限：相对锡面低饱和。"""
    abs_max = int(C.HIGHLIGHT_S_MAX)
    rel = stats.s_median - float(C.TH.get("highlight_s_rel_margin", 20.0))
    return int(max(20, min(abs_max, rel)))


def exposed_pixel_ok(v: np.ndarray, s: np.ndarray, stats_g: SolderRelStats) -> np.ndarray:
    """露底像素：相对标准锡面 V/S，辅以绝对兜底。"""
    v_min = max(
        float(C.TH.get("insuf_exposed_v_min", 90)),
        stats_g.v_median - float(C.TH.get("insuf_exposed_v_rel_drop", 40.0)),
    )
    s_max = min(
        float(C.TH.get("insuf_exposed_s_max", 110)),
        stats_g.s_median + float(C.TH.get("insuf_exposed_s_rel_margin", 30.0)),
    )
    return (v > v_min) & (s < s_max)


def rel_min_area(abs_min: float, frac: float, ref_area: float) -> float:
    """面积下限：绝对像素与相对参考面积取较大值。"""
    return float(max(abs_min, frac * max(1.0, ref_area)))


def pin_core_min_pixels(roi_w: int, roi_h: int, pin_std: int) -> int:
    """引脚顶最小面积：绝对下限与 ROI/标准相对下限取较大值。"""
    abs_min = int(C.TH.get("pin_top_core_std_min", 200))
    roi_frac = float(C.TH.get("pin_top_core_roi_frac", 0.008))
    std_frac = float(C.TH.get("pin_top_core_std_frac", 0.35))
    rel = max(roi_w * roi_h * roi_frac, pin_std * std_frac)
    return int(max(abs_min, rel))


def fuse_min_votes(evidence: dict[str, bool], min_votes: int) -> bool:
    """多特征投票融合。"""
    return sum(1 for v in evidence.values() if v) >= min_votes


# 标准/测试锡面尺寸公差圈：剔除边缘浅环伪影，保留明显内凹。

def align_erode_ksize(w: int, h: int) -> int:
    """尺寸公差圈腐蚀核 (奇数)，标准/测试整体差一圈时不应判 NG。"""
    px = int(C.TH.get("edge_align_tol_px", 4))
    frac = float(C.TH.get("edge_align_erode_frac", 0.035))
    k = max(px, int(min(w, h) * frac))
    return k | 1


def align_erode_kernel(w: int, h: int) -> np.ndarray:
    k = align_erode_ksize(w, h)
    return _ellipse_kernel(k)


def solder_core_mask(solder: np.ndarray, w: int, h: int) -> np.ndarray:
    """扣除边缘公差圈后的锡面核心 (仍为标准轮廓内)。 """
    return cv2.erode(solder, align_erode_kernel(w, h))


def solder_outer_tolerance_band(solder_g: np.ndarray, w: int, h: int) -> np.ndarray:
    """标准锡面外圈公差带 = 标准锡面 − 核心。"""
    core = solder_core_mask(solder_g, w, h)
    return cv2.subtract(solder_g, core)


def core_inner_tolerance_ring(solder_g: np.ndarray, w: int, h: int) -> np.ndarray:
    """标准核心内缘公差环 = 核心 − 再腐蚀一圈；尺寸差一圈的浅环伪影多落在此环带。"""
    k = align_erode_kernel(w, h)
    core = solder_core_mask(solder_g, w, h)
    inner = cv2.erode(core, k)
    return cv2.subtract(core, inner)


@dataclass
class EdgeArtifactContext:
    """`is_shallow_edge_artifact` 所需的中间结果，仅依赖 (solder_g, w, h)。"""
    deep: np.ndarray
    ring: np.ndarray
    outer: np.ndarray


def build_edge_artifact_context(solder_g: np.ndarray, w: int, h: int) -> EdgeArtifactContext:
    k = align_erode_kernel(w, h)
    core = cv2.erode(solder_g, k)
    deep = cv2.erode(core, k)
    ring = cv2.subtract(core, deep)      # 等价原 core_inner_tolerance_ring()
    outer = cv2.subtract(solder_g, core)  # 等价原 solder_outer_tolerance_band()
    return EdgeArtifactContext(deep=deep, ring=ring, outer=outer)


def is_shallow_edge_artifact(comp: np.ndarray, solder_g: np.ndarray,
                             w: int, h: int, *,
                             ctx: "EdgeArtifactContext | None" = None) -> bool:
    """候选是否仅为标准/测试锡面差一圈导致的浅环伪影，而非明显向内凹进。
    """
    area = cv2.countNonZero(comp)
    if area <= 0:
        return True

    if ctx is None:
        ctx = build_edge_artifact_context(solder_g, w, h)
    deep, ring, outer = ctx.deep, ctx.ring, ctx.outer

    in_deep = cv2.countNonZero(cv2.bitwise_and(comp, deep))
    if in_deep > 0:
        return False

    min_frac = float(C.TH.get("edge_artifact_ring_min_frac", 0.75))
    in_ring = cv2.countNonZero(cv2.bitwise_and(comp, ring))
    if in_ring >= area * min_frac:
        return True

    in_outer = cv2.countNonZero(cv2.bitwise_and(comp, outer))
    if in_outer >= area * min_frac and in_deep == 0:
        return True
    return False
