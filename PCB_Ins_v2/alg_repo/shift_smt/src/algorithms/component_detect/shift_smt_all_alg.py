"""贴片电容移位检测算法（单文件实现）。

本文件由原算法的两个文件整合而成：
- shift_smt_all_alg.py（算法主类 ShiftSmtAllAlg）
- _shift_core.py（核心 CV 函数，已并入本文件）

shift2 版本输入图协议：
- 输入图 = 待检图（原图，不裁切小图）；
- 底座框（base 框）由调用方通过 run() 的 roi_bbox 参数显式传入（原图坐标），
  算法不再从图像尺寸按 roi_margin 反推底座框；
- roi_bbox 传入的 base 框同时充当滑窗模板尺寸与移位判定参考框（原 template_box 语义）；
- 底座可能部分/全部超出图像范围，可开启 allow_out_of_bounds（默认关闭）配合越界滑窗定位。

说明：
- auto_gamma（gamma 归一化）已移出，作为上位机调参工具放在
  src/algorithms/gamma/gamma.py；算法内为惰性引用：仅当 use_autogamma=True 时
  才在运行期导入，use_autogamma=False（默认）不产生任何依赖，部署环境缺
  src/algorithms/gamma 也不会导致算法导入失败。
- 算法与上位机的契约接口（IDetectionAlgorithm / AlgorithmResult /
  DetectPartBox / ResultType / merge_config）来自 core.*（contract_reference），
  运行平台需保证 core 包可导入。
"""

from __future__ import annotations

import base64
import json
import time
import traceback
import zlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.entities.algorithms import IDetectionAlgorithm
from core.entities.detect import (
    AlgorithmResult,
    DetectPartBox,
    ResultType,
)
from core.utils.alg_config import merge_config


# ============================================================
# 移位检测核心函数（原 _shift_core.py，auto_gamma 除外）
# ============================================================

def _decode_hs_lut(b64str: str) -> np.ndarray:
    """
    解码 base64 编码的 HS 查找表为 180x256 布尔矩阵。

    编码格式：36x64 降采样 -> packbits -> zlib -> base64。
    解码后通过最近邻插值恢复到 180x256。

    Args:
        b64str: base64 编码的 LUT 字符串

    Returns:
        (180, 256) bool 数组，True 表示该 (H,S) 组合在选区内
    """
    raw = zlib.decompress(base64.b64decode(b64str))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    small = bits[:36 * 64].reshape(36, 64).astype(bool)
    # 用 np.repeat 上采样（与 pcba_tuner 编解码格式一致）
    lut = np.repeat(np.repeat(small, 5, axis=0), 4, axis=1)[:180, :256]
    return lut


def _apply_hs_lut_mask(
    hsv: np.ndarray, lut: np.ndarray, v_min: int, v_max: int,
) -> np.ndarray:
    """
    用 LUT 查表 + V 范围生成掩码。

    H/S 通过 LUT 查表判定（支持任意形状选区），V 仍用范围判定。

    Args:
        hsv: HSV 图像
        lut: (180, 256) bool 查找表
        v_min: V 通道下限
        v_max: V 通道上限

    Returns:
        uint8 掩码 (0/255)
    """
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    hs_mask = lut[h_ch, s_ch]                       # H/S 查表
    v_mask = (v_ch >= v_min) & (v_ch <= v_max)      # V 范围
    return (hs_mask & v_mask).astype(np.uint8) * 255


def extract_base_mask(
    image: np.ndarray,
    cfg: Dict[str, Any],
    hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    提取底座掩码。

    支持两种模式：
    1. inRange 模式（默认）：用 HSV 6 值范围 cv2.inRange 提取掩码
    2. LUT 模式：用色板生成的 180x256 查找表提取掩码，支持任意形状选区

    两种模式都使用 V 通道范围 (v_low ~ v_high) 做亮度过滤。
    轻量形态学开运算去除孤立噪点后返回。

    Args:
        image: BGR 图像
        cfg: 配置字典
            - use_hs_lut: 是否启用 LUT 模式
            - hs_lut: base64 编码的 LUT 字符串（LUT 模式下必需）
            - feature_h_low/high, feature_s_low/high, feature_v_low/high: inRange 模式参数
        hsv: 预计算的 HSV 图像（可选，避免重复转换）

    Returns:
        uint8 掩码 (0/255)，255 表示底座颜色区域
    """
    if hsv is None:
        # AutoGamma 预处理（可选，由上位机调参工具 src/algorithms/gamma 提供）。
        # 惰性导入：use_autogamma=False（默认）不依赖 gamma 模块，部署环境缺依赖也不报错；
        # use_autogamma=True 且模块缺失时给出明确错误，由 run() 兜底成 code=2。
        if cfg.get("use_autogamma", False):
            try:
                from algorithms.gamma.gamma import auto_gamma
            except ImportError as exc:
                raise RuntimeError(
                    "use_autogamma=True 但 gamma 模块不可用（未部署 "
                    "src/algorithms/gamma），请关闭 use_autogamma 或补齐依赖"
                ) from exc
            target = float(cfg.get("autogamma_target", 0.35))
            image = auto_gamma(image[:, :, :3], target)
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)

    # LUT 模式：H/S 用查找表，V 用范围
    if cfg.get("use_hs_lut", False) and cfg.get("hs_lut", ""):
        lut = _decode_hs_lut(cfg["hs_lut"])
        v_min = int(cfg.get("feature_v_low", 0))
        v_max = int(cfg.get("feature_v_high", 36))
        mask = _apply_hs_lut_mask(hsv, lut, v_min, v_max)
    else:
        # inRange 模式：HSV 6 值范围
        lower = np.array([
            cfg.get("feature_h_low", 0),
            cfg.get("feature_s_low", 0),
            cfg.get("feature_v_low", 0),
        ], dtype=np.uint8)
        upper = np.array([
            cfg.get("feature_h_high", 179),
            cfg.get("feature_s_high", 147),
            cfg.get("feature_v_high", 36),
        ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

    # 3x3 开运算去除孤立噪点
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    return mask


def _integral_block_cov(
    integral: np.ndarray, ya: int, xa: int, yb: int, xb: int, ny: int, nx: int,
) -> np.ndarray:
    """
    用积分图计算某子块在所有滑窗位置的覆盖率。

    积分图(Summed Area Table)可以在 O(1) 时间内计算任意矩形区域的像素和。
    对于每个滑窗位置 (px, py)，子块 [ya:yb, xa:xb] 的像素和为：
      sum = I[yb,xb] - I[ya,xb] - I[yb,xa] + I[ya,xa]
    其中 I 为积分图。覆盖率 = sum / 子块面积。

    通过 numpy 切片，一次性计算所有滑窗位置，实现全向量化。

    Args:
        integral: 积分图 (h+1) x (w+1)
        ya, xa: 子块左上角相对于窗口原点的偏移
        yb, xb: 子块右下角相对于窗口原点的偏移
        ny, nx: 滑窗在 y/x 方向的有效位置数

    Returns:
        (ny, nx) float64 数组，每个元素为该位置子块的覆盖率 [0, 1]
    """
    bh = yb - ya  # 子块高度
    bw = xb - xa  # 子块宽度
    if bh <= 0 or bw <= 0:
        return np.ones((ny, nx), dtype=np.float64)
    # 积分图四角求和：所有滑窗位置一次性向量化计算
    return (integral[yb:yb + ny, xb:xb + nx]
            - integral[ya:ya + ny, xb:xb + nx]
            - integral[yb:yb + ny, xa:xa + nx]
            + integral[ya:ya + ny, xa:xa + nx]) / (bw * bh)


def _subpixel_refine(fm: np.ndarray, px: int, py: int) -> Tuple[float, float]:
    """
    抛物线插值亚像素细化。

    在整数峰值 (px, py) 附近，用三点抛物线拟合特征图，
    求抛物线顶点得到亚像素精度的峰值位置。

    公式：dx = 0.5 * (f(x-1) - f(x+1)) / (f(x-1) - 2*f(x) + f(x+1))

    Args:
        fm: 特征图 (ny, nx)
        px, py: 整数峰值坐标

    Returns:
        (refined_x, refined_y) 亚像素精度的坐标
    """
    h, w = fm.shape
    dx, dy = 0.0, 0.0
    # x 方向：用 (px-1, px, px+1) 三点拟合抛物线
    if 0 < px < w - 1:
        denom = fm[py, px - 1] - 2 * fm[py, px] + fm[py, px + 1]
        if abs(denom) > 1e-10:
            dx = 0.5 * (fm[py, px - 1] - fm[py, px + 1]) / denom
    # y 方向：同理
    if 0 < py < h - 1:
        denom = fm[py - 1, px] - 2 * fm[py, px] + fm[py + 1, px]
        if abs(denom) > 1e-10:
            dy = 0.5 * (fm[py - 1, px] - fm[py + 1, px]) / denom
    return px + dx, py + dy


def _compute_feature_map(
    integral: np.ndarray, win_w: int, win_h: int, ny: int, nx: int,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    用积分图计算所有滑窗位置的特征得分图。

    特征构成（与 find_base_box 一致）：
      - 特征1: 整窗覆盖率（grid_size=1 时 1 个块）
      - 特征2-5: 四条边缘覆盖率（上/下/左/右条带）
    聚合模式: avg（算术平均）或 min（取最小值）。

    Args:
        integral: 积分图 (h+1) x (w+1)
        win_w, win_h: 窗口尺寸
        ny, nx: 滑窗在 y/x 方向的有效位置数
        cfg: 配置字典

    Returns:
        (ny, nx) float64 特征得分图
    """
    n_blocks = int(cfg.get("feature_grid_size", 1))
    mode = cfg.get("feature_mode", "avg")
    use_edges = bool(cfg.get("feature_use_edges", True))
    edge_ratio = float(cfg.get("feature_edge_ratio", 0.08))
    whole_weight = float(cfg.get("feature_whole_weight", 0.5))

    features = []

    # 特征1: 分块覆盖率（grid_size=1 时为整窗覆盖率）
    for by in range(n_blocks):
        for bx in range(n_blocks):
            ya = by * win_h // n_blocks
            yb = (by + 1) * win_h // n_blocks
            xa = bx * win_w // n_blocks
            xb = (bx + 1) * win_w // n_blocks
            features.append(_integral_block_cov(integral, ya, xa, yb, xb, ny, nx))

    # 特征2-5: 四条边缘覆盖率（锐化峰值，提升定位精度）
    if use_edges:
        eh = max(1, int(win_h * edge_ratio))
        ew = max(1, int(win_w * edge_ratio))
        features.append(_integral_block_cov(integral, 0, 0, eh, win_w, ny, nx))              # 上边缘
        features.append(_integral_block_cov(integral, win_h - eh, 0, win_h, win_w, ny, nx))  # 下边缘
        features.append(_integral_block_cov(integral, 0, 0, win_h, ew, ny, nx))              # 左边缘
        features.append(_integral_block_cov(integral, 0, win_w - ew, win_h, win_w, ny, nx))  # 右边缘

    # 聚合
    n_whole = n_blocks * n_blocks  # 整窗/分块特征数
    n_edge = len(features) - n_whole

    if mode == "avg":
        if n_edge > 0 and whole_weight != 0.2:
            # 加权平均：整窗权重 whole_weight，边缘权重 (1-whole_weight)/n_edge
            w_edge = (1.0 - whole_weight) / n_edge
            feature_map = features[0] * whole_weight / n_whole
            for i in range(1, n_whole):
                feature_map += features[i] * whole_weight / n_whole
            for i in range(n_whole, len(features)):
                feature_map += features[i] * w_edge
        else:
            # 默认等权平均（向后兼容）
            feature_map = features[0].copy()
            for f in features[1:]:
                feature_map += f
            feature_map /= len(features)
    else:
        feature_map = features[0].copy()
        for f in features[1:]:
            np.minimum(feature_map, f, out=feature_map)

    return feature_map


def _rotation_refine(
    mask: np.ndarray,
    win_w: int, win_h: int,
    cfg: Dict[str, Any],
    peak_px: int = 0, peak_py: int = 0,
) -> Tuple[int, int, float, float]:
    """
    固定位置旋转搜索：在轴对齐峰值位置，对每个角度旋转 mask 区域副本，
    评估该固定位置的覆盖率特征得分，取最佳角度。不做平移滑窗。

    优化原理：不计算完整特征图，每个角度只评估一个位置（~0.05ms/角度），
    比滑窗搜索（~0.6ms/角度）快 12 倍。对于小角度旋转，轴对齐峰值位置
    足够接近最优位置，精度损失可忽略。

    进一步优化：由于评估的是覆盖率特征（整窗 + 4 条边缘，均为比例统计量），
    对分辨率不敏感，因此先在降采样图上旋转评估（窗口越大降采样越多，目标
    窗口 ~100px），速度提升约 scale^2 倍，得分数值与全分辨率近似一致。

    对每个角度：
      1. 裁剪峰值附近的小区域（窗口 + margin），降采样
      2. 旋转区域副本（绕窗口中心，-angle 抵消底座旋转）
      3. 在固定位置提取窗口，计算 5 个特征（整窗 + 4 条边缘）
      4. 取最高得分的角度

    Args:
        mask: 底座掩码 (uint8, 0/255)
        win_w, win_h: 窗口尺寸
        cfg: 配置字典
        peak_px, peak_py: 轴对齐搜索的峰值位置（窗口左上角，原图坐标）

    Returns:
        (px, py, best_angle, best_score) 位置不变，最佳角度、得分
    """
    h, w = mask.shape[:2]

    rot_range = float(cfg.get("rotation_range", 10.0))
    rot_step = float(cfg.get("rotation_step", 1.0))
    angles = np.arange(-rot_range, rot_range + rot_step / 2, rot_step)

    # 窗口中心（原图坐标）
    cx = peak_px + win_w / 2.0
    cy = peak_py + win_h / 2.0

    # 裁剪区域：窗口 + margin（确保旋转后窗口内无黑边）
    max_rad = np.radians(rot_range)
    margin = int(max(win_w, win_h) * np.sin(max_rad) / 2 + 20)
    x0 = max(0, int(cx - win_w / 2 - margin))
    y0 = max(0, int(cy - win_h / 2 - margin))
    x1 = min(w, int(cx + win_w / 2 + margin))
    y1 = min(h, int(cy + win_h / 2 + margin))

    rw = x1 - x0
    rh = y1 - y0
    if rw < win_w + 10 or rh < win_h + 10:
        return peak_px, peak_py, 0.0, 0.0

    # 窗口中心在区域坐标中的位置
    cx_reg = cx - x0
    cy_reg = cy - y0

    # 裁剪区域（不修改原 mask）
    region = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)

    # 降采样：覆盖率特征对分辨率不敏感，缩小图评估显著降低 warpAffine 开销
    scale = max(1, min(win_w, win_h) // 100)
    if scale > 1:
        rw_s = rw // scale
        rh_s = rh // scale
        if rw_s >= win_w // scale + 3 and rh_s >= win_h // scale + 3:
            region = cv2.resize(region, (rw_s, rh_s), interpolation=cv2.INTER_NEAREST)
            rw, rh = rw_s, rh_s
            win_w, win_h = win_w // scale, win_h // scale
            cx_reg, cy_reg = cx_reg / scale, cy_reg / scale

    # 边缘参数
    use_edges = bool(cfg.get("feature_use_edges", True))
    edge_ratio = float(cfg.get("feature_edge_ratio", 0.08))
    whole_weight = float(cfg.get("feature_whole_weight", 0.5))
    eh = max(1, int(win_h * edge_ratio))
    ew = max(1, int(win_w * edge_ratio))

    # 窗口在区域坐标中的左上角
    wx = int(round(cx_reg - win_w / 2))
    wy = int(round(cy_reg - win_h / 2))
    if wx < 0 or wy < 0 or wx + win_w > rw or wy + win_h > rh:
        return peak_px, peak_py, 0.0, 0.0

    best_score = -1.0
    best_angle = 0.0

    for angle in angles:
        # 旋转区域副本（绕窗口中心，-angle 抵消底座旋转）
        M = cv2.getRotationMatrix2D((cx_reg, cy_reg), -angle, 1.0)
        rotated = cv2.warpAffine(region, M, (rw, rh), flags=cv2.INTER_NEAREST)

        # 在固定位置提取窗口（不滑窗）
        window = rotated[wy:wy + win_h, wx:wx + win_w]
        if window.shape[0] < win_h or window.shape[1] < win_w:
            continue

        # 计算特征（与滑窗特征一致：整窗 + 4 条边缘）
        feats = [np.count_nonzero(window) / (win_w * win_h)]  # 整窗
        if use_edges:
            feats.append(np.count_nonzero(window[:eh, :]) / max(1, eh * win_w))       # 上
            feats.append(np.count_nonzero(window[win_h - eh:, :]) / max(1, eh * win_w))  # 下
            feats.append(np.count_nonzero(window[:, :ew]) / max(1, win_h * ew))       # 左
            feats.append(np.count_nonzero(window[:, win_w - ew:]) / max(1, win_h * ew))  # 右

        # 聚合（与 _compute_feature_map 一致）
        n_f = len(feats)
        if n_f > 1 and whole_weight != 0.2:
            w_edge = (1.0 - whole_weight) / (n_f - 1)
            score = feats[0] * whole_weight + sum(f * w_edge for f in feats[1:])
        else:
            score = sum(feats) / n_f
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    # 位置不变，只返回最佳角度
    return peak_px, peak_py, best_angle, best_score


def find_base_box(
    mask: np.ndarray,
    cfg: Dict[str, Any],
    template_box: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """
    滑窗特征匹配，找底座峰值位置。

    算法流程：
    1. 用模板底座框的宽高作为滑窗尺寸
    2. 构建掩码积分图，加速覆盖率计算
    3. 计算每个滑窗位置的特征得分：
       - 整窗覆盖率（grid_size=1 时 1 个块）
       - 4 条边缘覆盖率（上/下/左/右条带）
    4. 取所有特征的算术平均(avg)作为最终得分
    5. 得分最高的位置即为底座位置
    6. 亚像素抛物线插值细化位置
    7. 峰值得分低于阈值 -> 缺件(None)

    启用 use_rotation 时，先轴对齐定位峰值，再在峰值附近限定区域
    做旋转搜索，取全局峰值作为结论。限定区域大幅减少旋转耗时。

    allow_out_of_bounds=True（默认 False）时，允许滑窗位置部分/整体超出图像：
    掩码四周零填充后扩展滑窗域，窗口与图的重叠部分参与覆盖率计算，
    返回的检测框坐标可为负值或超出图像范围——用于底座大幅移位、部分移出输入图的场景。
    注意越界模式计算量显著增大，仅在确有需要时开启。

    Args:
        mask: 底座掩码 (uint8, 0/255)
        cfg: 配置字典
        template_box: 模板底座框 (x, y, w, h)，其宽高定义滑窗尺寸

    Returns:
        (x, y, w, h, angle) 底座框+旋转角度，或 None（峰值低于阈值=缺件）
    """
    if template_box is None:
        return None

    # 滑窗尺寸：默认取模板框宽高与图像尺寸的较小值；允许越界时用完整模板框尺寸
    _, _, tw, th = template_box
    h, w = mask.shape[:2]
    allow_oob = bool(cfg.get("allow_out_of_bounds", False))
    if allow_oob:
        win_w, win_h = tw, th
    else:
        win_w = min(tw, w)
        win_h = min(th, h)
    if win_w <= 0 or win_h <= 0:
        return None

    thresh = float(cfg.get("feature_thresh", 0.05))

    # ===== 步骤1：轴对齐搜索（始终执行，为旋转搜索提供峰值参考）=====
    mask_01 = mask // 255

    if allow_oob:
        # 允许越界：掩码四周零填充 (win-1)，滑窗域扩展到 [-win+1, size-1]。
        # 窗口与图重叠的部分参与覆盖率计算，窗口外的填充区域贡献 0 覆盖率，
        # 因此返回框坐标可为负值或超出图像范围。
        pad_w, pad_h = win_w - 1, win_h - 1
        integral = cv2.integral(cv2.copyMakeBorder(
            mask_01, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=0))
        ny = h + win_h - 1
        nx = w + win_w - 1
    else:
        integral = cv2.integral(mask_01)
        ny = h - win_h + 1
        nx = w - win_w + 1
    if ny <= 0 or nx <= 0:
        return None

    feature_map = _compute_feature_map(integral, win_w, win_h, ny, nx, cfg)

    # 找特征图峰值
    best_idx = np.argmax(feature_map)
    best_y, best_x = np.unravel_index(best_idx, feature_map.shape)
    best_score = float(feature_map[best_y, best_x])

    # 亚像素细化：抛物线插值提升定位精度
    # （越界模式不 clamp，细化后统一换算回原图坐标，允许负值/超界）
    px, py = int(best_x), int(best_y)
    if 0 < px < nx - 1 and 0 < py < ny - 1:
        refined_x, refined_y = _subpixel_refine(feature_map, px, py)
        px, py = int(round(refined_x)), int(round(refined_y))
        if not allow_oob:
            px = max(0, min(nx - 1, px))
            py = max(0, min(ny - 1, py))
    if allow_oob:
        # 填充坐标 -> 原图坐标（可为负/超界）
        px, py = px - pad_w, py - pad_h

    # ===== 步骤2：旋转搜索（可选，在轴对齐峰值附近限定区域搜索）=====
    if bool(cfg.get("use_rotation", True)):
        # 越界定位出的峰值窗口不完整落图内时，旋转精修裁剪会越界且无意义，跳过
        if (not allow_oob
                or (0 <= px and px + win_w <= w and 0 <= py and py + win_h <= h)):
            rot_px, rot_py, rot_angle, rot_score = _rotation_refine(
                mask, win_w, win_h, cfg, px, py)
            # 取轴对齐和旋转搜索中的全局最优
            if rot_score > best_score:
                best_score = rot_score
                px, py, rot_final = rot_px, rot_py, rot_angle
            else:
                rot_final = 0.0
        else:
            rot_final = 0.0
    else:
        rot_final = 0.0

    # 峰值得分低于阈值 -> 缺件
    if best_score < thresh:
        return None

    return (px, py, win_w, win_h, rot_final)


def judge_shift(
    base_box: Tuple[int, int, int, int],
    template_box: Optional[Tuple[int, int, int, int]],
    cfg: Dict[str, Any],
) -> Tuple[bool, float, float, float]:
    """
    判断元件是否移位。

    比较检测底座框中心与模板底座框中心的偏移量。
    偏移距离超过阈值则判定为移位。

    Args:
        base_box: 检测到的底座框 (x, y, w, h)
        template_box: 模板底座框 (x, y, w, h)，作为参考位置
        cfg: 配置字典，含 shift_thresh

    Returns:
        (is_shifted, dx, dy, distance)
        - is_shifted: 是否移位
        - dx: x 方向中心偏移量（像素）
        - dy: y 方向中心偏移量（像素）
        - distance: 中心偏移距离（像素）
    """
    if template_box is None:
        return False, 0.0, 0.0, 0.0

    bx, by, bw, bh = base_box
    tx, ty, tw, th = template_box

    # 计算两个框的中心点偏移
    dx = (bx + bw / 2.0) - (tx + tw / 2.0)
    dy = (by + bh / 2.0) - (ty + th / 2.0)
    dist = float((dx ** 2 + dy ** 2) ** 0.5)

    # 偏移距离超过阈值则判定为移位
    shift_thresh = float(cfg.get("shift_thresh", 10.0))
    is_shifted = dist > shift_thresh

    return is_shifted, float(dx), float(dy), dist


def build_output_image(
    image: np.ndarray,
    base_box: Optional[Tuple[int, int, int, int]],
    is_shifted: bool = False,
    is_misspart: bool = False,
    dx: float = 0.0,
    dy: float = 0.0,
    rot_angle: float = 0.0,
) -> np.ndarray:
    """
    构建可视化输出图。

    检测框颜色编码：
      - 蓝色 (255,0,0): 底座正常，未移位
      - 红色 (0,0,255): 底座移位
      - "MissPart" 文字: 缺件（底座不存在）

    当 rot_angle != 0 时绘制旋转矩形（4 角点连线）。

    Args:
        image: 原始 BGR 图像
        base_box: 底座框 (x, y, w, h) 或 None
        is_shifted: 是否移位
        is_misspart: 是否缺件
        dx, dy: 移位量（像素）
        rot_angle: 旋转角度（度）

    Returns:
        BGR 可视化图像
    """
    vis = image.copy()
    # 格式统一：灰度转BGR，RGBA转BGR
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 4:
        vis = vis[:, :, :3]

    if base_box is not None:
        x, y, w, h = base_box
        # 移位=红色 (BGR: 0,0,255)，正常=蓝色 (BGR: 255,0,0)
        color = (0, 0, 255) if is_shifted else (255, 0, 0)

        if abs(rot_angle) > 0.1:
            # 旋转矩形：计算 4 角点并连线
            cx, cy = x + w / 2.0, y + h / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                               dtype=np.float64)
            ones = np.ones((4, 1))
            pts = np.hstack([corners, ones])
            rotated = M @ pts.T  # (2, 4)
            pts_int = np.int32(rotated.T.reshape(-1, 1, 2))
            cv2.polylines(vis, [pts_int], True, color, 2)
        else:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # 在框上方绘制标签和移位量
        label = "shift" if is_shifted else "base"
        angle_str = f" a={rot_angle:.1f}" if abs(rot_angle) > 0.1 else ""
        label_text = f"{label} dx={dx:.1f} dy={dy:.1f}{angle_str}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 黑底白字，确保可读性
        cv2.rectangle(vis, (x - 1, y - th - 6), (x + tw + 4, y), (20, 20, 20), -1)
        cv2.putText(vis, label_text, (x + 1, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    elif is_misspart:
        # 缺件：图中央显示红色 "MissPart" 文字
        h, w = vis.shape[:2]
        cv2.putText(vis, "MissPart", (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return vis


# ============================================================
# 算法主类
# ============================================================

class ShiftSmtAllAlg(IDetectionAlgorithm):
    """贴片电容移位检测算法。"""

    algorithm_code: str = "ShiftSmtAllAlg"
    version: int = 1

    def __init__(self, config_path: str | None = None):
        # 必须调用基类构造：基类负责读 JSON → self._defaults / self.algorithm_code / self.version
        super().__init__(config_path)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """默认算法参数（shift2 新协议：输入图为待检原图，底座框由 run() 的 roi_bbox 传入）。"""
        return {
            # 输入图协议：待检原图（不裁切小图）；底座框（base 框）由 run(roi_bbox=...) 显式传入
            # （原图坐标），同时充当滑窗模板尺寸与移位判定参考框。不再需要 roi_margin。

            # HSV 抽色范围：inRange 模式下的 6 值范围
            "feature_h_low": 0, "feature_h_high": 179,      # H: 全范围
            "feature_s_low": 0, "feature_s_high": 147,      # S: 低-中饱和度
            "feature_v_low": 0, "feature_v_high": 36,       # V: 低亮度

            # LUT 模式（可选，由 tuner 色板生成，比 inRange 更精确）
            "use_hs_lut": False,                              # 是否启用 LUT 模式
            "hs_lut": "",                                     # base64 编码的 180x256 查找表

            # 滑窗特征参数：控制底座定位的精度与稳定性
            "feature_grid_size": 1,                           # 分块数（1=整窗）
            "feature_mode": "avg",                            # 聚合模式（avg/min）
            "feature_use_edges": True,                        # 启用四条边缘特征
            "feature_edge_ratio": 0.08,                       # 边缘条带占比（8%）
            "feature_whole_weight": 0.5,                      # 整窗覆盖率权重（开发者参数，0.5抑制多峰竞争）
            "feature_thresh": 0.05,                           # 缺件判定阈值
            "allow_out_of_bounds": False,                     # 允许检测框超出图像范围（大移位场景：底座部分移出图外）
                                                              # False=滑窗必须完整落在图内（默认）；
                                                              # True=掩码零填充后越界滑窗，返回框坐标可为负/超界。
                                                              # 注意：越界模式特征图面积约 (h+win-1)*(w+win-1)，计算量明显增大。

            # 移位判定参数
            "shift_thresh": 10.0,                             # 移位阈值（像素）

            # 旋转精修参数（可选，两阶段：先轴对齐定位再小范围试角度）
            "use_rotation": True,                             # 是否启用旋转精修
            "rotation_range": 10.0,                           # 旋转搜索范围（±度）
            "rotation_step": 1.0,                             # 旋转搜索步长（度）

            "use_autogamma": False,                           # 是否启用 AutoGamma 亮度归一化
            "autogamma_target": 0.35,                         # AutoGamma 目标亮度（0~1）
        }

    def run(
        self,
        image: np.ndarray,
        config: Dict[str, Any],
        roi_bbox: Optional[Any] = None,
        original_template_image: Optional[np.ndarray] = None,
        multiple_roi_images: Optional[List[np.ndarray]] = None,
    ) -> AlgorithmResult:
        """
        执行移位检测。

        Args:
            image: 待检图 BGR 图像（shift2 协议：原图，不裁切小图）
            config: 配置字典（外部参数覆盖默认参数）
            roi_bbox: 底座框 base 框 (x, y, w, h)，原图坐标系，**必传**。
                      同时充当滑窗模板尺寸与移位判定参考框（原 template_box 语义）。
                      支持 4 元组/列表或带 x/y/width/height 属性的 BoundingBox 对象。
                      上位机通道（detectSinglePart）不传此关键字，底座框通过
                      config["template_roi"] 下发，故未传 roi_bbox 时自动回退读取。
            original_template_image: 未使用（接口兼容）
            multiple_roi_images: 未使用（接口兼容）

        Returns:
            AlgorithmResult（code/message 三态约定）:
              - code=0 / message="ok":    算法正常运行且未检出异常
              - code=1 / message="fail":  算法检出异常（移位/缺件）
              - code=2 / message="error": 算法出错/未正常运行（具体原因在 metadata["error"]）
        """
        t0 = time.perf_counter()
        try:
            # 1. 输入校验
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                return self._fail_result("image is None or empty", t0)

            # 格式统一：RGBA->BGR，灰度->BGR
            img = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 2. 合并配置（默认参数 + 外部传入参数）
            cfg = merge_config(self._defaults, config)

            # 3. 底座框（base 框）由调用方显式传入（原图坐标），既作滑窗模板尺寸，
            #    也作移位判定参考框。不再从图像尺寸反推。
            #    来源兼容两种通道：roi_bbox 关键字优先；上位机 detectSinglePart
            #    通过 threshold JSON 下发 template_roi，服务端解析后进 config，故兜底读取。
            if roi_bbox is None:
                roi_bbox = cfg.get("template_roi")
            if roi_bbox is None:
                return self._fail_result(
                    "roi_bbox/template_roi is required (base box: x,y,w,h in image coords)", t0)
            if hasattr(roi_bbox, "x") and hasattr(roi_bbox, "width"):
                rb = (int(roi_bbox.x), int(roi_bbox.y),
                      int(roi_bbox.width), int(roi_bbox.height))
            else:
                rb = tuple(int(v) for v in roi_bbox)
            if len(rb) != 4 or rb[2] <= 0 or rb[3] <= 0:
                return self._fail_result(f"invalid roi_bbox: {roi_bbox}", t0)
            template_box = rb

            # 4. HSV 抽色提取底座掩码
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            base_mask = extract_base_mask(img, cfg, hsv=hsv)

            # 5. 滑窗特征匹配，定位底座位置（返回 5-tuple: x, y, w, h, rot_angle）
            result_box = find_base_box(base_mask, cfg, template_box)

            # 6. 缺件检测：结果为 None 表示峰值低于阈值
            if result_box is None:
                vis = build_output_image(img, None, is_misspart=True)
                part = DetectPartBox(
                    x=0, y=0, width=0, height=0,
                    confidence=1.0,
                    label="misspart",
                )
                return AlgorithmResult(
                    code=1, message="fail",
                    algorithm_code=self.algorithm_code,
                    cost_time=time.perf_counter() - t0,
                    result_type=ResultType.PARTS,
                    parts=[part],
                    metadata={"output_image": vis, "base_mask": base_mask},
                )

            # 解包检测结果
            base_box = (int(result_box[0]), int(result_box[1]),
                        int(result_box[2]), int(result_box[3]))
            rot_angle = float(result_box[4]) if len(result_box) > 4 else 0.0

            # 7. 移位检测：比较检测框中心与模板框中心
            is_shifted, dx, dy, dist = judge_shift(base_box, template_box, cfg)

            # 8. 构建输出（协议约定：检出异常 code=1/message=fail，
            #    未检出 code=0/message=ok，算法出错 code=2/message=error）
            label = "shift" if is_shifted else "base"
            code = 1 if is_shifted else 0
            message = "fail" if is_shifted else "ok"
            vis = build_output_image(img, base_box, is_shifted=is_shifted,
                                     dx=dx, dy=dy, rot_angle=rot_angle)

            part = DetectPartBox(
                x=int(base_box[0]), y=int(base_box[1]),
                width=int(base_box[2]), height=int(base_box[3]),
                confidence=1.0,
                angle=rot_angle,
                label=label,
                metadata={
                    "dx": dx, "dy": dy, "shift_distance": dist,
                    "is_shifted": is_shifted, "rotation_angle": rot_angle,
                },
            )

            return AlgorithmResult(
                code=code, message=message,
                algorithm_code=self.algorithm_code,
                cost_time=time.perf_counter() - t0,
                result_type=ResultType.PARTS,
                parts=[part],
                metadata={
                    "output_image": vis,
                    "base_mask": base_mask,
                    "base_box": base_box,
                    "template_box": template_box,
                    "is_shifted": is_shifted,
                    "dx": dx, "dy": dy, "shift_distance": dist,
                    "rotation_angle": rot_angle,
                },
            )
        except Exception as exc:
            # 兜底：任何意外都转成 code=2/message="error" 的结果（见 _fail_result），绝不向算法池抛异常
            return self._fail_result(str(exc), t0, exc=exc)

    def _fail_result(
        self,
        message: str,
        t0: float,
        exc: Optional[BaseException] = None,
    ) -> AlgorithmResult:
        """统一的失败出口：code=2/message="error" + ResultType.ERROR（具体原因进 metadata）。"""
        meta: Dict[str, Any] = {"error": message}
        if exc is not None:
            meta["traceback"] = traceback.format_exc()
        return AlgorithmResult(
            code=2,
            message="error",
            algorithm_code=self.algorithm_code,
            cost_time=time.perf_counter() - t0,
            result_type=ResultType.ERROR,
            metadata=meta,
        )
