# -*- coding: utf-8 -*-
"""
缺陷区域手工特征提取模块（训练与在线推理共用，保证两边口径一致）。

对单个缺陷区域（二值掩膜 + 对应 BGR 图像 ROI）提取三组传统特征：
  1. 几何形状: 相对尺寸 / 伸长率 / 实心度 / 圆度 / Hu 矩
     —— 负责区分划痕(HuaHen)这类细长缺陷与块状缺陷
  2. 颜色: 区域内 HSV/Lab 统计 + 区域内外环带的颜色差
     —— 负责区分沾锡(ZhanXi,亮金属色) / 异色(YiSe,色相偏移) / 脏污(ZangWu,偏暗)
     内外差可以抵消不同板子、不同光照的整体差异
  3. 纹理: 梯度统计 + 旋转不变均匀 LBP 直方图
     —— 负责区分异物(YiWu,有自身纹理和清晰边界)与压痕(YaHen,颜色变化小但有局部明暗)

几何特征全部用整图长边做归一化，保证 300px 小图与 5120px 大图口径一致。
"""
import cv2
import numpy as np


# ---------------------------------------------------------------
# 旋转不变均匀 LBP (riu2)：256 种 8bit 模式映射到 10 个 bin
# ---------------------------------------------------------------
def _build_lbp_lut() -> np.ndarray:
    lut = np.zeros(256, np.uint8)
    for v in range(256):
        bits = [(v >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        lut[v] = sum(bits) if transitions <= 2 else 9
    return lut


_LBP_LUT = _build_lbp_lut()

_LBP_SHIFTS = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]


def _lbp_hist(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """区域内 riu2 LBP 归一化直方图 (10 维)。"""
    g = gray.astype(np.int16)
    code = np.zeros(g.shape, np.uint8)
    for k, (dy, dx) in enumerate(_LBP_SHIFTS):
        shifted = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        code |= ((shifted >= g).astype(np.uint8) << k)
    codes = _LBP_LUT[code][mask > 0]
    hist = np.bincount(codes, minlength=10).astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


# ---------------------------------------------------------------
# 特征名列表（与 extract_region_features 返回向量一一对应）
# ---------------------------------------------------------------
FEATURE_NAMES = (
    # 几何 (13)
    ["log_area_rel", "rel_diameter", "elongation", "extent",
     "solidity", "circularity"]
    + [f"hu{i + 1}" for i in range(7)]
    # 颜色 (15)
    + ["h_cos_in", "h_sin_in", "s_mean_in", "v_mean_in",
       "s_std_in", "v_std_in",
       "l_mean_in", "a_mean_in", "b_mean_in",
       "d_l", "d_a", "d_b", "d_s", "d_v", "d_hue"]
    # 纹理 (13)
    + ["grad_mean", "grad_std", "lap_std"]
    + [f"lbp{i}" for i in range(10)]
)


def _circular_hue_mean(hue_deg_ocv: np.ndarray):
    """OpenCV 的 H 通道取值 0~180，映射到单位圆求均值向量。"""
    ang = hue_deg_ocv.astype(np.float32) * (2.0 * np.pi / 180.0)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    return float(c), float(s)


def extract_region_features(image_bgr: np.ndarray, mask: np.ndarray,
                            image_long_side: float):
    """
    提取单个缺陷区域的特征向量。

    image_bgr / mask: 已裁剪到缺陷附近的 ROI（同尺寸），mask 为 0/255 二值图，
                      ROI 四周需留出一圈背景（外扩环带要用）。
    image_long_side:  原始整图长边像素数，几何特征据此归一到相对尺度。

    返回 np.float32 向量（与 FEATURE_NAMES 对应）；区域无效时返回 None。
    """
    area = int(cv2.countNonZero(mask))
    if area < 4:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)

    # ---------- 几何 ----------
    long_side = max(float(image_long_side), 1.0)
    log_area_rel = float(np.log10(area / (long_side * long_side) + 1e-12))
    rel_diameter = float(np.sqrt(area) / long_side)

    (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
    lo, hi = (min(rw, rh), max(rw, rh))
    elongation = float(hi / max(lo, 1e-3))
    extent = float(area / max(rw * rh, 1e-3))

    hull = cv2.convexHull(cnt)
    solidity = float(area / max(cv2.contourArea(hull), 1e-3))
    perimeter = cv2.arcLength(cnt, True)
    circularity = float(4.0 * np.pi * area / max(perimeter * perimeter, 1e-3))

    m = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # 对数压缩，跨尺度可比

    # ---------- 内外环带 ----------
    ring_w = int(np.clip(round(np.sqrt(area) * 0.4), 3, 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_w + 1, 2 * ring_w + 1))
    ring = cv2.subtract(cv2.dilate(mask, kernel), mask)
    if cv2.countNonZero(ring) < 4:
        ring = cv2.bitwise_not(mask)  # ROI 里掩膜占满时退化为全部外围像素

    inside = mask > 0
    outside = ring > 0

    # ---------- 颜色 ----------
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    h_in, s_in, v_in = (hsv[..., 0][inside], hsv[..., 1][inside].astype(np.float32),
                        hsv[..., 2][inside].astype(np.float32))
    h_out = hsv[..., 0][outside]
    s_out = hsv[..., 1][outside].astype(np.float32)
    v_out = hsv[..., 2][outside].astype(np.float32)

    h_cos_in, h_sin_in = _circular_hue_mean(h_in)
    h_cos_out, h_sin_out = _circular_hue_mean(h_out)
    # 内外色相均值向量的夹角 (0~pi)，对跨 0/180 边界的红色也正确
    d_hue = float(np.arccos(np.clip(
        (h_cos_in * h_cos_out + h_sin_in * h_sin_out)
        / max(np.hypot(h_cos_in, h_sin_in) * np.hypot(h_cos_out, h_sin_out), 1e-6),
        -1.0, 1.0)))

    lab_in = lab.reshape(-1, 3)[inside.ravel()].astype(np.float32)
    lab_out = lab.reshape(-1, 3)[outside.ravel()].astype(np.float32)
    l_mean_in, a_mean_in, b_mean_in = lab_in.mean(axis=0) / 255.0
    d_l, d_a, d_b = (lab_in.mean(axis=0) - lab_out.mean(axis=0)) / 255.0

    color_feats = [
        h_cos_in, h_sin_in,
        float(s_in.mean() / 255.0), float(v_in.mean() / 255.0),
        float(s_in.std() / 255.0), float(v_in.std() / 255.0),
        float(l_mean_in), float(a_mean_in), float(b_mean_in),
        float(d_l), float(d_a), float(d_b),
        float((s_in.mean() - s_out.mean()) / 255.0),
        float((v_in.mean() - v_out.mean()) / 255.0),
        d_hue,
    ]

    # ---------- 纹理 ----------
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)[inside] / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)[inside] / 255.0

    texture_feats = [float(grad.mean()), float(grad.std()), float(lap.std())]
    lbp = _lbp_hist(gray, mask)

    feats = np.array(
        [log_area_rel, rel_diameter, elongation, extent, solidity, circularity]
        + hu.tolist() + color_feats + texture_feats + lbp.tolist(),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(feats)):
        return None
    return feats


def crop_region(image_bgr: np.ndarray, points: np.ndarray, pad: int = 48):
    """
    按多边形顶点裁剪 ROI 并栅格化掩膜。
    points: (N,2) float/int 多边形顶点（整图坐标系）。
    返回 (roi_bgr, roi_mask)；多边形退化时返回 (None, None)。
    """
    h, w = image_bgr.shape[:2]
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    if len(pts) < 3:
        return None, None

    x0 = max(0, int(pts[:, 0].min()) - pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    x1 = min(w, int(pts[:, 0].max()) + pad + 1)
    y1 = min(h, int(pts[:, 1].max()) + pad + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None

    roi = image_bgr[y0:y1, x0:x1]
    roi_mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(roi_mask, [pts - np.array([x0, y0])], 255)
    return roi, roi_mask
