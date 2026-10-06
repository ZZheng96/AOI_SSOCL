from __future__ import annotations

import json
import os

import cv2
import numpy as np


# 标注 (.json) 加载

def label_path_for(std_dir: str, num: str) -> str:
    return os.path.join(std_dir, f"templ_{num}.json")


def load_label_required(std_dir: str, num: str) -> dict:
    """严格读取 templ_{num}.json；缺失或无法解析时直接报错（run.py 模板预处理用）。"""
    json_path = label_path_for(std_dir, num)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"缺少锡面轮廓标注文件: {json_path}\n"
            f"请先为该标准图补充标注 (solder_mode: ellipse/contour) 后重试")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            label = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise FileNotFoundError(f"无法解析标注文件: {json_path} ({e})") from e
    return label


# 公共工具函数

_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def largest_external_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fill_holes(mask):
    """外轮廓填充，去掉内部镂空。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8) * 255


def smooth_contour(mask, blur_ksize=5):
    """高斯模糊后再二值化，平滑锯齿。"""
    k = blur_ksize | 1
    blur = cv2.GaussianBlur(mask, (k, k), 0)
    _, out = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    return out


def solidity_of(mask):
    """轮廓面积 / 凸包面积（越接近 1 越规则）。"""
    contour = largest_external_contour(mask)
    if contour is None:
        return 0.0
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


# 空间位置先验（四角必为背景 / 中心区域必为锡面）

_CORNER_EXCLUDE_FRAC = 0.16   # 四角硬背景排除区大小（相对 min(h,w) 的三角区）
_CENTER_PROTECT_FRAC = 0.40   # 中心保护区半轴（相对 w、h），区内像素禁止被判为背景


def corner_exclusion_mask(h, w, frac=_CORNER_EXCLUDE_FRAC):
    """四角硬背景排除区：由"锡面不会触及四个角"的空间先验得到。"""
    margin = max(2, int(round(frac * min(h, w))))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (
        (xx + yy < margin)                               # 左上角三角区
        | ((w - 1 - xx) + yy < margin)                   # 右上角三角区
        | (xx + (h - 1 - yy) < margin)                   # 左下角三角区
        | ((w - 1 - xx) + (h - 1 - yy) < margin)          # 右下角三角区
    )
    return mask


def center_protection_mask(h, w, frac=_CENTER_PROTECT_FRAC):
    """中心保护区：由"锡面通常位于中心区域"的空间先验得到。"""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ay, ax = max(1.0, frac * h), max(1.0, frac * w)
    yy, xx = np.mgrid[0:h, 0:w]
    dist2 = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    return dist2 <= 1.0


# 四角色差超过此值视为角部被锡面/高光污染（合法底板渐变通常远低于此）
_CORNER_OUTLIER_THRESH = 60.0
_OPPOSITE_CORNER = {0: 3, 1: 2, 2: 1, 3: 0}  # TL/TR/BL/BR 对角


def _corner_patch_slices(h, w, patch=10):
    p = min(patch, h // 4, w // 4) or 1
    return [
        (slice(0, p), slice(0, p)),         # TL
        (slice(0, p), slice(w - p, w)),     # TR
        (slice(h - p, h), slice(0, p)),     # BL
        (slice(h - p, h), slice(w - p, w)),  # BR
    ]


def _bilinear_background_model(img_bgr, patch=10):
    """由四角背景采样构建随空间平滑变化的背景颜色场，而非假设整篇背景颜色唯一：
    """
    h, w = img_bgr.shape[:2]
    slices = _corner_patch_slices(h, w, patch)
    patches = [img_bgr[sy, sx] for sy, sx in slices]
    means = np.array([c.reshape(-1, 3).astype(np.float32).mean(axis=0) for c in patches])
    stds = np.array([c.reshape(-1, 3).astype(np.float32).std(axis=0) for c in patches])

    median = np.median(means, axis=0)
    dist = np.linalg.norm(means - median, axis=1)
    valid = dist < _CORNER_OUTLIER_THRESH
    if valid.sum() < 2:  # 极端退化保护：至少保留距离中位色最近的两角
        valid = np.zeros(4, dtype=bool)
        valid[np.argsort(dist)[:2]] = True

    fixed_means = means.copy()
    invalid_idx = np.where(~valid)[0]
    if len(invalid_idx) == 1:
        bad = int(invalid_idx[0])
        opp = _OPPOSITE_CORNER[bad]
        adj = [i for i in range(4) if i not in (bad, opp)]
        # 平行四边形关系：TL+BR = TR+BL（双线性场对角颜色和相等），由其余三角估算异常角
        fixed_means[bad] = means[adj[0]] + means[adj[1]] - means[opp]
    elif len(invalid_idx) >= 2:
        fallback = means[valid].mean(axis=0) if valid.any() else median
        fixed_means[~valid] = fallback

    TL, TR, BL, BR = fixed_means
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(w - 1, 1))[..., None]
    v = (yy / max(h - 1, 1))[..., None]
    top = TL * (1 - u) + TR * u
    bottom = BL * (1 - u) + BR * u
    bg_mean_map = top * (1 - v) + bottom * v

    global_std = np.maximum(stds[valid].mean(axis=0) if valid.any() else stds.mean(axis=0), 3.0)
    bg_std_map = np.broadcast_to(global_std, (h, w, 3)).astype(np.float32)

    return bg_mean_map, bg_std_map, slices, valid


def background_distance2(img_bgr, bg_mean_map=None, bg_std_map=None):
    """逐像素计算与其所在位置的局部背景估计之间的归一化颜色距离平方。"""
    if bg_mean_map is None or bg_std_map is None:
        bg_mean_map, bg_std_map, _, _ = _bilinear_background_model(img_bgr)
    diff = (img_bgr.astype(np.float32) - bg_mean_map) / bg_std_map
    return np.sum(diff * diff, axis=2)


# 分支二：不规则异形轮廓类（区域生长）

def extract_solder_mask_contour(img_bgr, k=4.2, close_frac=0.05):
    """分支二：确定四角种子点 -> 区域生长（背景/锡面分离） -> 形态学处理 -> 轮廓优化 -> 生成 Mask。"""
    h, w = img_bgr.shape[:2]
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    bg_mean_map, bg_std_map, seed_regions, valid = _bilinear_background_model(img_bgr)

    dist2 = background_distance2(blurred, bg_mean_map, bg_std_map)
    bg_candidate = (dist2 <= (k * k)).astype(np.uint8) * 255

    # 空间位置约束：中心保护区内像素禁止被判为背景候选，避免区域生长渗漏进锡面中心
    protect = center_protection_mask(h, w)
    bg_candidate[protect] = 0

    # 连通域分析：只保留与种子区域相连的背景连通域，等价于从四角种子点做区域生长；
    # 被判定为异常（可能被锡面污染）的角不作为种子
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(bg_candidate, connectivity=8)
    seed_labels = set()
    for i, (sy, sx) in enumerate(seed_regions):
        if not valid[i]:
            continue
        region_labels = labels[sy, sx]
        vals, counts = np.unique(region_labels, return_counts=True)
        for v, c in zip(vals, counts):
            if v != 0 and c >= region_labels.size * 0.5:  # 该角多数像素都属于此背景连通域才采信
                seed_labels.add(int(v))

    background = np.isin(labels, list(seed_labels)).astype(np.uint8) * 255 if seed_labels else np.zeros((h, w), dtype=np.uint8)
    foreground = cv2.bitwise_not(background)

    # 形态学处理：开运算去噪 -> 闭运算连接断裂边缘 -> 孔洞填充
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, _OPEN_KERNEL)
    diag = (h ** 2 + w ** 2) ** 0.5
    close_ksize = max(3, int(diag * close_frac)) | 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel)
    foreground = fill_holes(foreground)

    # 轮廓优化：最大连通区域筛选 + 轮廓平滑 + 边界优化（再次填充确保连续）
    foreground = keep_largest_component(foreground)
    foreground = smooth_contour(foreground)
    foreground = fill_holes(foreground)

    # 空间位置约束兜底：四角硬背景排除区强制置为背景
    foreground[corner_exclusion_mask(h, w)] = 0

    return foreground


# 分支一：规则圆/椭圆类（Canny + 几何拟合 + 评分）

def _enhance_gray(img_bgr):
    """灰度转换 + 图像增强：双边滤波抑制噪声同时保边，CLAHE 提升局部对比度。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _auto_canny(gray, sigma=0.33):
    """依据灰度中值自适应设定 Canny 双阈值，减少人工调参依赖。"""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _fit_circle_algebraic(pts):
    """代数最小二乘圆拟合（Kasa method）。"""
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1.0)))
    return (cx, cy), r


def _circle_residuals(pts, params):
    (cx, cy), r = params
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return d - r


def _ellipse_residuals(pts, ellipse):
    (cx, cy), (major, minor), angle = ellipse
    theta = np.deg2rad(angle)
    ct, st = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    a, b = max(major / 2.0, 1e-3), max(minor / 2.0, 1e-3)
    norm = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return (norm - 1.0) * ((a + b) / 2.0)  # 换算为近似像素距离，便于与圆的残差同尺度比较


def _iterative_refit(all_pts, init_pts, fit_func, residual_func, scale_func,
                      n_iters=4, rel_thresh=0.10, min_pts=8):
    """类 RANSAC 迭代拟合校正：以凸包点做初拟合 -> 用初拟合结果在**全部边缘点**中
    按"残差 <= 相对阈值(有效半径的一定比例)"筛选内点（真实边界点应普遍落在此邻域内，
    背景噪声/引脚高光/内部纹理边缘则会被排除） -> 用内点重新拟合，迭代收紧。
    """
    params = fit_func(init_pts)
    cur_pts = init_pts
    for _ in range(n_iters):
        scale = max(scale_func(params), 1e-3)
        residual = np.abs(residual_func(all_pts, params))
        thresh = max(2.0, rel_thresh * scale)
        keep = residual <= thresh
        if keep.sum() < min_pts:
            break
        new_pts = all_pts[keep]
        if len(new_pts) == len(cur_pts):
            cur_pts = new_pts
            break
        cur_pts = new_pts
        params = fit_func(cur_pts)
    return params, cur_pts


def _visible_boundary_bins(center, w, h, boundary_dist_func, n_bins=24, margin=2):
    """判定拟合边界在各角度扇区的理论落点是否落在图像范围内。"""
    cx, cy = center
    bin_centers = -np.pi + (np.arange(n_bins) + 0.5) * (2 * np.pi / n_bins)
    visible = np.zeros(n_bins, dtype=bool)
    for i, theta in enumerate(bin_centers):
        t = boundary_dist_func(theta)
        bx, by = cx + t * np.cos(theta), cy + t * np.sin(theta)
        visible[i] = (-margin <= bx <= w - 1 + margin) and (-margin <= by <= h - 1 + margin)
    return visible


def _ring_coverage(pts, center, w, h, boundary_dist_func, n_bins=24):
    """环形覆盖度：以拟合中心为原点，统计"图像可见范围内"的角度扇区中，
    有实际边缘证据支撑的比例。值越接近1，说明拟合边界在可见范围内被边缘证据
    充分环绕支撑；偏低则拟合可能只由局部弧段支撑，不可信（如背景干扰导致的
    偏移拟合）。"""
    cx, cy = center
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    covered = np.zeros(n_bins, dtype=bool)
    covered[np.unique(bins)] = True
    visible = _visible_boundary_bins(center, w, h, boundary_dist_func, n_bins=n_bins)
    denom = max(1, int(visible.sum()))
    return float(np.count_nonzero(covered & visible)) / denom


def _ellipse_boundary_dist_func(ellipse_params):
    (cx, cy), (major, minor), angle = ellipse_params
    phi = np.deg2rad(angle)
    a, b = max(major / 2.0, 1e-3), max(minor / 2.0, 1e-3)

    def _dist(theta):
        d = theta - phi
        denom = (np.cos(d) / a) ** 2 + (np.sin(d) / b) ** 2
        return 1.0 / np.sqrt(max(denom, 1e-9))

    return _dist


_BG_RELATION_K = 4.2  # 与分支二一致的背景颜色距离阈值，用于识别"背景内部变化"产生的假边缘


def _fit_ellipse_candidates(img_bgr):
    """Canny边缘检测 -> 轮廓提取（外边界凸包点，仅排除四角硬背景区） -> 圆拟合 / 椭圆拟合
    -> 类RANSAC迭代校正。"""
    h, w = img_bgr.shape[:2]
    gray = _enhance_gray(img_bgr)
    edges = _auto_canny(gray)
    edges = cv2.dilate(edges, _OPEN_KERNEL, iterations=1)
    edges[corner_exclusion_mask(h, w)] = 0

    bg_mean_map, bg_std_map, _, _ = _bilinear_background_model(img_bgr)
    bg_like = background_distance2(img_bgr, bg_mean_map, bg_std_map) <= (_BG_RELATION_K ** 2)
    edges[bg_like] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0).astype(np.float64)
    if len(pts) < 8:
        return None

    # 稳健初始定位：用中位数估计粗略中心/半径，剔除远离典型环带的离群点
    # （如残留的背景纹理噪声点），避免其进入凸包后把初拟合结果拉偏、过度放大。
    rough_center = np.median(pts, axis=0)
    rough_dist = np.linalg.norm(pts - rough_center, axis=1)
    rough_radius = np.median(rough_dist)
    band = (rough_dist >= 0.4 * rough_radius) & (rough_dist <= 1.6 * rough_radius)
    seed_pts = pts[band] if band.sum() >= 8 else pts

    hull_pts = cv2.convexHull(seed_pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull_pts) < 5:
        return None

    def _fit_circle(p):
        return _fit_circle_algebraic(p)

    def _fit_ellipse(p):
        return cv2.fitEllipse(p.astype(np.float32))

    circle_params, circle_inliers = _iterative_refit(
        pts, hull_pts, _fit_circle, _circle_residuals, scale_func=lambda p: p[1])
    ellipse_params, ellipse_inliers = _iterative_refit(
        pts, hull_pts, _fit_ellipse, _ellipse_residuals, scale_func=lambda p: (p[1][0] + p[1][1]) / 4.0)

    center_c, radius_c = circle_params
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(round(center_c[0])), int(round(center_c[1]))), int(round(radius_c)), 255, thickness=-1)
    circle_res = _circle_residuals(circle_inliers, circle_params)
    circle_fit_error = float(np.sqrt(np.mean(circle_res ** 2))) / max(radius_c, 1e-3)
    circle_ring_cov = _ring_coverage(circle_inliers, center_c, w, h, lambda theta: radius_c)

    (cx_e, cy_e), (major, minor), angle = ellipse_params
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse_params, 255, thickness=-1)
    ellipse_res = _ellipse_residuals(ellipse_inliers, ellipse_params)
    eff_radius = (major + minor) / 4.0
    ellipse_fit_error = float(np.sqrt(np.mean(ellipse_res ** 2))) / max(eff_radius, 1e-3)
    ellipse_ring_cov = _ring_coverage(
        ellipse_inliers, (cx_e, cy_e), w, h, _ellipse_boundary_dist_func(ellipse_params))
    axis_ratio = float(min(major, minor) / max(major, minor)) if max(major, minor) > 0 else 0.0

    diag_half = 0.5 * min(h, w)
    img_center = ((w - 1) / 2.0, (h - 1) / 2.0)
    circle_pos_offset = float(np.hypot(center_c[0] - img_center[0], center_c[1] - img_center[1])) / max(diag_half, 1e-3)
    ellipse_pos_offset = float(np.hypot(cx_e - img_center[0], cy_e - img_center[1])) / max(diag_half, 1e-3)

    return {
        "circle": {
            "mask": circle_mask, "params": circle_params, "fit_error": circle_fit_error,
            "ring_coverage": circle_ring_cov, "axis_ratio": 1.0, "pos_offset": circle_pos_offset,
        },
        "ellipse": {
            "mask": ellipse_mask, "params": ellipse_params, "fit_error": ellipse_fit_error,
            "ring_coverage": ellipse_ring_cov, "axis_ratio": axis_ratio, "pos_offset": ellipse_pos_offset,
        },
    }


# 拟合结果评分通过阈值（均基于拟合本身的几何证据，独立于分支二的参照 Mask，
# 避免参照 Mask 在背景干扰下的缺陷被传递进本分支的判定）：
_FIT_ERROR_THRESH = 0.16     # 轮廓拟合误差：残差RMS / 有效半径
_RING_COVERAGE_THRESH = 0.70  # 轮廓覆盖率：边缘证据环绕拟合边界的角度覆盖比例
_POS_OFFSET_THRESH = 0.35     # 区域位置约束：拟合中心与图像中心的偏移量（相对半边长）
_AXIS_RATIO_THRESH = 0.55     # 圆度指标：椭圆长短轴比下限，过低视为拟合异常而非真实锡面形态


def extract_solder_mask_ellipse(img_bgr, reference_mask=None):
    """分支一：Canny边缘检测 -> 轮廓提取 -> 圆/椭圆拟合 -> 拟合结果评分。
    """
    candidates = _fit_ellipse_candidates(img_bgr)
    if candidates is None:
        meta = {"reason": "edge_points_insufficient"}
        if reference_mask is not None:
            meta["solidity"] = solidity_of(reference_mask)
        return None, meta

    # 每个候选（圆/椭圆）独立评判是否达标，而非只看综合评分最高者：
    # 综合评分较高的候选可能是被离群点拉偏产生的异常拟合（如长短轴比失真），
    # 需要用各项硬性阈值单独过滤，避免其掩盖另一个真正合格的候选。
    passing = {}
    for shape, cand in candidates.items():
        passed = (
            cand["fit_error"] <= _FIT_ERROR_THRESH
            and cand["ring_coverage"] >= _RING_COVERAGE_THRESH
            and cand["pos_offset"] <= _POS_OFFSET_THRESH
            and cand["axis_ratio"] >= _AXIS_RATIO_THRESH
        )
        if passed:
            score = cand["ring_coverage"] - cand["fit_error"] - 0.5 * cand["pos_offset"]
            passing[shape] = (score, cand)

    meta = {
        "candidates": {
            shape: {k: v for k, v in cand.items() if k != "mask"}
            for shape, cand in candidates.items()
        },
    }
    if reference_mask is not None:
        meta["solidity"] = solidity_of(reference_mask)

    if not passing:
        meta["reason"] = "score_below_threshold"
        return None, meta

    best_shape, (best_score, best) = max(passing.items(), key=lambda kv: kv[1][0])
    meta.update({
        "shape": best_shape,
        "fit_error": best["fit_error"],
        "ring_coverage": best["ring_coverage"],
        "pos_offset": best["pos_offset"],
        "axis_ratio": best["axis_ratio"],
    })

    final_mask = best["mask"].copy()
    final_mask[corner_exclusion_mask(*img_bgr.shape[:2])] = 0  # 空间位置约束兜底
    return fill_holes(final_mask), meta


# 分支三：人工框选 → 直接栅格化为 Mask（不套用自动分支启发式）

_VALID_ROI_SHAPES = ("rect", "circle")
_MIN_ANNOTATED_MASK_PX = 30


def extract_solder_mask_roi(img_bgr: np.ndarray, roi: dict) -> np.ndarray:
    """人工框栅格化为锡面 Mask。

    坐标为 ``img_bgr`` 像素：矩形 ``{"shape":"rect","x","y","w","h"}``，
    圆形 ``{"shape":"circle","cx","cy","r"}``。越界裁剪；非法框抛 ``ValueError``。
    """
    if not isinstance(roi, dict):
        raise ValueError(f"solder_mode='roi' 需要 dict 格式的 roi 描述, 实际: {type(roi)!r}")

    shape = str(roi.get("shape", "rect") or "rect").lower()
    if shape not in _VALID_ROI_SHAPES:
        raise ValueError(f"roi.shape 必须是 {_VALID_ROI_SHAPES} 之一, 实际: {shape!r}")

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if shape == "circle":
        try:
            cx, cy, r = float(roi["cx"]), float(roi["cy"]), float(roi["r"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"roi(shape=circle) 缺少合法的 cx/cy/r 字段: {roi!r}") from e
        if r <= 0:
            raise ValueError(f"roi(shape=circle) 的 r 必须 > 0, 实际: {r!r}")
        cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, thickness=-1)
    else:
        try:
            x, y = float(roi["x"]), float(roi["y"])
            rw, rh = float(roi["w"]), float(roi["h"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"roi(shape=rect) 缺少合法的 x/y/w/h 字段: {roi!r}") from e
        if rw <= 0 or rh <= 0:
            raise ValueError(f"roi(shape=rect) 的 w/h 必须 > 0, 实际: w={rw!r} h={rh!r}")
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + rw)), int(round(y + rh))
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        if x1c <= x0c or y1c <= y0c:
            raise ValueError(f"roi(shape=rect) 裁剪到图像范围后为空: {roi!r} (图像尺寸 {w}x{h})")
        mask[y0c:y1c, x0c:x1c] = 255

    if cv2.countNonZero(mask) < _MIN_ANNOTATED_MASK_PX:
        raise ValueError(f"roi 生成的锡面 Mask 像素数过少 (<{_MIN_ANNOTATED_MASK_PX}): {roi!r}")

    return mask


# 标签驱动的分发逻辑

def extract_solder_mask(img_bgr: np.ndarray, label: dict | None) -> tuple[np.ndarray, dict]:
    """依据标签 solder_mode 分发到对应分支：

        - "ellipse" (默认): 圆/椭圆几何拟合，评分不足时自动回退到 contour 分支
        - "contour": 区域生长 (背景/锡面分离)
        - "roi": 人工框选，直接使用 `label["roi"]` 描述的矩形/圆形区域，不做任何
          拟合/生长 (见 `extract_solder_mask_roi`)
    """
    mode = (label or {}).get("solder_mode", "ellipse")

    if mode == "roi":
        roi = (label or {}).get("roi")
        if not roi:
            raise ValueError("solder_mode='roi' 需在 label 中提供 'roi' 字段 (人工框选的焊点区域)")
        mask = extract_solder_mask_roi(img_bgr, roi)
        return mask, {"mode": mode, "method": "manual_roi", "roi": dict(roi)}

    # 分支二结果始终计算：既是 contour 模式的直接输出，也是 ellipse 模式的评分参照 / 回退兜底
    contour_mask = extract_solder_mask_contour(img_bgr)

    if mode == "contour":
        touches_border = _touches_border(contour_mask)
        return contour_mask, {"mode": mode, "method": "contour(flood_fill)", "touches_border": touches_border}

    # mode == "ellipse"
    ellipse_mask, fit_meta = extract_solder_mask_ellipse(img_bgr, contour_mask)
    if ellipse_mask is not None:
        final_mask = ellipse_mask
        method = f"ellipse({fit_meta['shape']})"
    else:
        final_mask = contour_mask
        method = "contour(flood_fill)[fallback]"

    touches_border = _touches_border(final_mask)
    meta = {"mode": mode, "method": method, "touches_border": touches_border, "fit": fit_meta}
    return final_mask, meta


def _touches_border(mask):
    return bool(
        np.any(mask[0, :]) or np.any(mask[-1, :])
        or np.any(mask[:, 0]) or np.any(mask[:, -1])
    )


# 模板预处理入口：标注加载 + Mask 生成一步完成

def build_template_mask(std_dir: str, num: str, std_bgr: np.ndarray) -> np.ndarray:
    """模板预处理阶段调用入口：读取 `templ_{num}.json` 标注，按其 `solder_mode`
    分发到对应算法分支，在 `std_bgr` 原始分辨率下生成一次锡面 Mask 并返回。
    """
    label = load_label_required(std_dir, num)
    mask, _meta = extract_solder_mask(std_bgr, label)
    return mask


def load_template_label(std_dir: str, num: str) -> dict:
    """读取 templ_{num}.json（含 solder_mode / bridge_joints）。"""
    return load_label_required(std_dir, num)


def build_template_assets(std_dir: str, num: str, std_bgr: np.ndarray):
    """模板预处理：锡面 Mask + 连锡焊点组。返回 (mask, bridge_joints)。"""
    from .bridge_joints import parse_bridge_joints

    label = load_label_required(std_dir, num)
    mask, _meta = extract_solder_mask(std_bgr, label)
    return mask, parse_bridge_joints(label)
