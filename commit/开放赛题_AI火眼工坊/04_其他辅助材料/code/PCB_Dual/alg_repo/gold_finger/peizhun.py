# -*- coding: utf-8 -*-
"""
图像配准模块 - 独立精简版
结合相位相关（Phase Correlation）与 ORB 特征匹配的混合配准算法。
"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List

def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def _clip_bbox(
    bbox: Tuple[int, int, int, int],
    shape_hw: Tuple[int, int],
    min_size: int = 64,
) -> Tuple[int, int, int, int]:
    h, w = shape_hw
    x, y, bw, bh = bbox
    x = max(0, min(int(round(x)), max(0, w - 1)))
    y = max(0, min(int(round(y)), max(0, h - 1)))
    bw = max(int(min_size), int(round(bw)))
    bh = max(int(min_size), int(round(bh)))
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)
    x = max(0, x1 - bw)
    y = max(0, y1 - bh)
    return int(x), int(y), int(x1 - x), int(y1 - y)

def _center_crop_bbox(
    shape_hw: Tuple[int, int],
    crop_ratio: float = 0.8,
    min_size: int = 128,
) -> Tuple[int, int, int, int]:
    """返回图像中心 crop_ratio 区域的 bbox"""
    h, w = shape_hw
    crop_ratio = float(max(0.1, min(1.0, crop_ratio)))
    bw = max(int(round(w * crop_ratio)), min_size)
    bh = max(int(round(h * crop_ratio)), min_size)
    x = max(0, (w - bw) // 2)
    y = max(0, (h - bh) // 2)
    return _clip_bbox((x, y, bw, bh), shape_hw, min_size=min_size)

def _warp_affine(image: np.ndarray, mat_aff: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    # INTER_CUBIC 比 INTER_LINEAR 保留更多高频细节，
    # 亚像素平移重采样后缺陷不至于被明显软化
    return cv2.warpAffine(
        image,
        mat_aff,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def _apply_bbox_mask(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return image
    out = np.zeros_like(image)
    x, y, bw, bh = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image.shape[1], x + bw), min(image.shape[0], y + bh)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    return out

def _decompose_affine_2x3(matrix: np.ndarray) -> Tuple[Tuple[float, float], float, float]:
    a = float(matrix[0, 0])
    c = float(matrix[1, 0])
    tx = float(matrix[0, 2])
    ty = float(matrix[1, 2])
    scale = float(np.sqrt(max(1e-12, a * a + c * c)))
    rotation = float(np.degrees(np.arctan2(c, a)))
    return (tx, ty), rotation, scale

def _shift_within_limit(shift_xy: Tuple[float, float], shape_hw: Tuple[int, int], max_shift_ratio: float) -> bool:
    h, w = shape_hw
    dx, dy = float(shift_xy[0]), float(shift_xy[1])
    return abs(dx) <= float(w) * float(max_shift_ratio) and abs(dy) <= float(h) * float(max_shift_ratio)

def _phase_translate_center(
    moving_gray_f32: np.ndarray,
    template_gray_f32: np.ndarray,
    center_crop_ratio: float = 0.8,
    min_crop_size: int = 128,
) -> Tuple[Tuple[float, float], float, Tuple[int, int, int, int]]:
    """在中心区域进行相位相关计算"""
    h, w = template_gray_f32.shape[:2]
    crop_bbox = _center_crop_bbox((h, w), crop_ratio=center_crop_ratio, min_size=min_crop_size)
    x, y, bw, bh = crop_bbox
    moving_crop = moving_gray_f32[y:y + bh, x:x + bw]
    template_crop = template_gray_f32[y:y + bh, x:x + bw]

    # 汉宁窗抑制 FFT 周期化带来的边界伪峰：不加窗时裁剪边界的强边缘
    # 会污染相关峰，平移量和 response 在临界情况下不稳定
    hann = cv2.createHanningWindow((bw, bh), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(moving_crop, template_crop, hann)
    return (float(shift[0]), float(shift[1])), float(response), crop_bbox

def _warp_residual_error(
    moving_gray: np.ndarray,
    template_gray: np.ndarray,
    mat_aff: np.ndarray,
    weight: Optional[np.ndarray] = None,
) -> float:
    """用给定仿射 warp 后的（可加权）平均绝对灰度残差，作为候选变换的
    实测配准质量。warp 引入的边界外区域不计入统计。

    weight: 独特性权重图。周期性图案（金手指条纹）错开整数个周期后
    平均残差几乎不变，必须用独特结构（基准标记、板边异形处）加权，
    否则无法区分真对齐与"锁错周期的假对齐"。"""
    h, w = template_gray.shape[:2]
    warped = cv2.warpAffine(moving_gray, mat_aff, (w, h),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    cover = cv2.warpAffine(np.full((moving_gray.shape[0], moving_gray.shape[1]), 255, np.uint8),
                           mat_aff, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    m = cover > 0
    if int(m.sum()) < 100:
        return float("inf")
    diff = np.abs(warped.astype(np.float32) - template_gray.astype(np.float32))
    if weight is not None:
        wsum = float(weight[m].sum())
        if wsum > 1e-3:
            return float((diff[m] * weight[m]).sum() / wsum)
    return float(diff[m].mean())


def _horizontal_period(gray: np.ndarray) -> Optional[float]:
    """估计图像的主水平周期（金手指条纹间距）。无显著周期返回 None。"""
    g = gray.astype(np.float32)
    prof = g.mean(axis=0)
    prof -= prof.mean()
    spec = np.abs(np.fft.rfft(prof))
    if spec.size <= 4:
        return None
    spec[:3] = 0.0  # 排除直流与超低频（整体明暗渐变）
    k = int(np.argmax(spec))
    if k < 3:
        return None
    period = g.shape[1] / float(k)
    if not (4.0 <= period <= g.shape[1] / 4.0):
        return None
    if spec[k] < 5.0 * (spec.mean() + 1e-6):  # 峰不显著 = 无强周期
        return None
    return period


def _uniqueness_weight(gray: np.ndarray, period: float) -> np.ndarray:
    """独特性权重图：与"自身平移一个周期"的差异。周期性条纹区域差异
    趋零（权重小），基准标记、板边异形处差异大（权重大）。"""
    g = gray.astype(np.float32)
    p = max(1, int(round(period)))
    right = np.abs(g - np.roll(g, p, axis=1))
    left = np.abs(g - np.roll(g, -p, axis=1))
    u = np.minimum(right, left)  # 任一方向能周期自重合就视为周期区
    u[:, :p] = 0.0   # roll 环绕的边界列不可信
    u[:, -p:] = 0.0
    return cv2.GaussianBlur(u, (0, 0), 3)


def _distinctive_shift_candidate(
    moving_gray: np.ndarray,
    template_gray: np.ndarray,
    weight: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """锚点平移候选：裁出模板中独特性最强的窗口（基准标记所在处），
    在待测图全图内模板匹配。周期性主体不参与，锚点唯一 => 平移无歧义。

    独特性峰值必须显著高于全图均值（实测真锚点约 3x，纯周期图案的
    噪声峰仅 1.3~1.4x）才启用；否则返回 None，调用方也不应使用
    独特性加权——纯周期图案里权重图只是噪声，会把择优带偏。"""
    h, w = template_gray.shape[:2]
    win_w = min(w, max(64, w // 6))
    win_h = min(h, max(48, int(h * 0.8)))
    density = cv2.boxFilter(weight, ddepth=-1, ksize=(win_w, win_h),
                            normalize=True, borderType=cv2.BORDER_CONSTANT)
    _, dmax, _, dloc = cv2.minMaxLoc(density)
    if dmax < 2.0 * float(weight.mean()) or dmax <= 1e-3:
        return None, 0.0
    cx, cy = dloc
    x0 = int(np.clip(cx - win_w // 2, 0, w - win_w))
    y0 = int(np.clip(cy - win_h // 2, 0, h - win_h))
    patch = template_gray[y0:y0 + win_h, x0:x0 + win_w]
    res = cv2.matchTemplate(moving_gray, patch, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    # 模板中 (x0,y0) 的内容位于待测图 loc 处 => 待测图需平移 (x0-loc.x, y0-loc.y)
    aff = np.array([[1.0, 0.0, x0 - loc[0]], [0.0, 1.0, y0 - loc[1]]], dtype=np.float32)
    return aff, float(peak)


def _ncc_translation_candidate(
    moving_gray: np.ndarray,
    template_gray: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    """中央大块归一化互相关平移搜索。

    周期性图案（金手指条纹）上相位相关常锁到伪峰、ORB 常错配特征；
    空间域 NCC 用接近整幅的大模板搜索，包络（板边、基准标记）能压制
    周期歧义，对"基本已对齐、只差小平移"的图给出可靠的候选。
    搜索半径为图像尺寸的 1/8。"""
    h, w = template_gray.shape[:2]
    mx, my = max(16, w // 8), max(16, h // 8)
    if w - 2 * mx < 32 or h - 2 * my < 16:
        return None, 0.0
    tpl = moving_gray[my:h - my, mx:w - mx]
    res = cv2.matchTemplate(template_gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(res)
    aff = np.array([[1.0, 0.0, loc[0] - mx], [0.0, 1.0, loc[1] - my]], dtype=np.float32)
    return aff, float(peak)


def _orb_affine_fallback(
    moving_gray: np.ndarray,
    template_gray: np.ndarray,
    max_features: int,
    ratio_test: float,
    min_matches: int,
    ransac_thresh: float,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[bool, np.ndarray, Dict[str, object]]:
    """ORB 特征匹配配准（备用方案）"""
    moving_orb_gray = _apply_bbox_mask(moving_gray, roi_bbox)
    template_orb_gray = _apply_bbox_mask(template_gray, roi_bbox)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_t, des_t = orb.detectAndCompute(template_orb_gray, None)
    kp_m, des_m = orb.detectAndCompute(moving_orb_gray, None)
    
    if des_t is None or des_m is None or len(kp_t) == 0 or len(kp_m) == 0:
        return False, np.eye(2, 3, dtype=np.float32), {"message": "ORB features missing"}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_m, des_t, k=2)
    good = [m0 for m0, m1 in knn if m0.distance < ratio_test * m1.distance]

    if len(good) < min_matches:
        return False, np.eye(2, 3, dtype=np.float32), {
            "message": "Not enough ORB matches",
            "good_matches": len(good),
        }

    src = np.float32([kp_m[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)

    aff, inlier = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=200, confidence=0.90
    )
    if aff is None:
        return False, np.eye(2, 3, dtype=np.float32), {
            "message": "Affine RANSAC failed",
            "good_matches": len(good),
        }

    inliers = int(inlier.sum()) if inlier is not None else 0
    return True, aff.astype(np.float32), {
        "message": "ORB affine success",
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(good), 1)),
    }

def align_image(
    moving: np.ndarray,
    template: np.ndarray,
    enable_fast_mode: bool = True,
    phase_only: bool = False,
    max_features: int = 800,
    ratio_test: float = 0.75,
    min_matches: int = 8,
    ransac_thresh: float = 3.0,
    orb_max_rotation_deg: float = 15.0,
    orb_min_scale: float = 0.85,
    orb_max_scale: float = 1.15,
    orb_max_shift_ratio: float = 0.25,
    center_crop_ratio: float = 0.8,
    fast_scale: float = 0.5,
) -> Tuple[bool, np.ndarray, Dict[str, object]]:
    """
    通用快配准主流程
    """
    template_gray = to_gray(template)
    h, w = template_gray.shape[:2]

    # 尺寸对齐：估计与最终 warp 必须在同一坐标系。
    # 旧实现只把灰度图 resize 到模板尺寸去估计平移，最后却对"原始尺寸"的
    # 彩色图做 warp，两图分辨率不同时平移量整体错位——模板图与待测图
    # 互换后，被 resize 的一方跟着互换，就会出现"一个方向能对齐、
    # 反过来对不齐"的现象。这里直接把彩色图先统一到模板尺寸。
    if moving.shape[:2] != template.shape[:2]:
        interp = cv2.INTER_AREA if (moving.shape[0] > h or moving.shape[1] > w) else cv2.INTER_CUBIC
        moving = cv2.resize(moving, (w, h), interpolation=interp)
    moving_gray = to_gray(moving)

    # 快速模式降采样处理
    if enable_fast_mode:
        scale = float(max(0.1, min(1.0, fast_scale)))
        new_w, new_h = max(64, int(round(w * scale))), max(64, int(round(h * scale)))
        moving_gray_small = cv2.resize(moving_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        template_gray_small = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        orb_roi_bbox_small = _center_crop_bbox((new_h, new_w), crop_ratio=center_crop_ratio, min_size=64)
    else:
        scale = 1.0
        moving_gray_small = moving_gray
        template_gray_small = template_gray
        orb_roi_bbox_small = _center_crop_bbox((h, w), crop_ratio=center_crop_ratio, min_size=128)

    # 阶段一：整板中心区域 phase correlation (相位相关)
    shift_small, response, phase_bbox_small = _phase_translate_center(
        moving_gray_small.astype(np.float32),
        template_gray_small.astype(np.float32),
        center_crop_ratio=center_crop_ratio,
        min_crop_size=64 if enable_fast_mode else 128,
    )

    shift = (shift_small[0] / scale, shift_small[1] / scale)
    phase_aff = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)

    phase_crop_w, phase_crop_h = float(phase_bbox_small[2]) / scale, float(phase_bbox_small[3]) / scale
    phase_response_ok = float(response) >= 0.2
    phase_shift_ok = (abs(float(shift[0])) <= phase_crop_w * 0.25 and abs(float(shift[1])) <= phase_crop_h * 0.25)
    phase_ok = phase_response_ok and phase_shift_ok

    phase_meta = {
        "method": "phase",
        "score": float(response),
        "affine": phase_aff.tolist(),
        "phase_fallback_reason": "ok" if phase_ok else ("low_response" if not phase_response_ok else "shift_too_large")
    }

    if phase_ok:
        return True, _warp_affine(moving, phase_aff, (h, w)), phase_meta

    if phase_only:
        return False, _warp_affine(moving, phase_aff, (h, w)), phase_meta

    # 阶段二：ORB 特征匹配兜底
    ok_orb, aff_orb, meta_orb = _orb_affine_fallback(
        moving_gray_small, template_gray_small,
        max_features=max_features, ratio_test=ratio_test, min_matches=min_matches, 
        ransac_thresh=ransac_thresh, roi_bbox=orb_roi_bbox_small
    )

    if enable_fast_mode and aff_orb is not None:
        aff_orb = aff_orb.copy()
        aff_orb[0, 2] /= scale
        aff_orb[1, 2] /= scale

    orb_offset, orb_rotation, orb_scale = _decompose_affine_2x3(aff_orb)
    
    orb_ok = (
        bool(ok_orb)
        and int(meta_orb.get("inliers", 0)) >= max(3, min_matches // 2)
        and float(meta_orb.get("inlier_ratio", 0.0)) >= 0.10
        and abs(float(orb_rotation)) <= float(orb_max_rotation_deg)
        and (float(orb_min_scale) <= float(orb_scale) <= float(orb_max_scale))
        and _shift_within_limit(orb_offset, (h, w), orb_max_shift_ratio)
    )

    meta_orb["method"] = "orb_affine"
    meta_orb["score"] = meta_orb.get("inlier_ratio", 0.0)
    meta_orb["phase_fallback_reason"] = phase_meta["phase_fallback_reason"]
    meta_orb["affine"] = aff_orb.tolist()

    # 阶段三：候选择优。周期性图案上相位相关会锁到伪峰，ORB 即使内点
    # 达标也可能整体错配（把特征匹到错开一个周期/错误基准上），得到
    # "看似自洽实则全错"的仿射。因此不盲目信任任何单一方法：
    #   候选 = 恒等 / 相位平移 / ORB仿射 / NCC平移 / 锚点平移，
    #   逐一在降采样灰度图上实测配准残差，残差最小者生效。
    # 残差用"独特性权重"加权：条纹错开整数个周期后平均残差几乎不变，
    # 只有基准标记等独特结构能区分真对齐与假对齐。
    ncc_aff, ncc_peak = _ncc_translation_candidate(moving_gray_small, template_gray_small)

    period = _horizontal_period(template_gray_small)
    weight = None
    distinct_aff, distinct_peak = None, 0.0
    if period is not None:
        u = _uniqueness_weight(template_gray_small, period)
        distinct_aff, distinct_peak = _distinctive_shift_candidate(
            moving_gray_small, template_gray_small, u)
        if distinct_aff is not None:
            weight = u  # 锚点显著时才用独特性加权，否则权重图只是噪声

    identity_aff = np.eye(2, 3, dtype=np.float32)
    candidates = [
        ("identity", identity_aff, False, 1.0),
        ("phase", phase_aff, False, 1.0),   # 相位达标时已提前返回，走到这必然低置信
        ("orb", aff_orb, orb_ok, 1.0),
    ]
    if ncc_aff is not None:
        candidates.append(("ncc_shift", ncc_aff, ncc_peak >= 0.5, 1.0 / scale))
    if distinct_aff is not None:
        candidates.append(("anchor_shift", distinct_aff, distinct_peak >= 0.5, 1.0 / scale))

    errors = {}
    best_name, best_aff, best_err, best_conf = None, None, float("inf"), False
    for name, aff, conf, to_full in candidates:
        aff_small = aff.copy()
        if to_full != 1.0:
            aff = aff.copy()
            aff[:, 2] *= to_full   # 小图上估计的平移换算到全尺寸
        else:
            aff_small[:, 2] *= scale  # 全尺寸候选换算到小图评估
        err = _warp_residual_error(moving_gray_small.astype(np.float32),
                                   template_gray_small.astype(np.float32),
                                   aff_small, weight=weight)
        errors[name] = round(err, 2)
        if err < best_err:
            best_name, best_aff, best_err, best_conf = name, aff, err, conf

    if best_name == "orb":
        meta = meta_orb
    else:
        meta = phase_meta
        meta["method"] = {"identity": "lowconf_identity", "phase": "lowconf_phase",
                          "ncc_shift": "ncc_shift", "anchor_shift": "anchor_shift"}[best_name]
        if best_name == "ncc_shift":
            meta["score"] = float(ncc_peak)
        elif best_name == "anchor_shift":
            meta["score"] = float(distinct_peak)
        meta["orb_meta"] = {k: meta_orb[k] for k in ("message", "inliers", "inlier_ratio", "good_matches") if k in meta_orb}
    meta["affine"] = best_aff.tolist()
    meta["fallback_errors"] = errors
    meta["weighted_residual"] = weight is not None
    return bool(best_conf), _warp_affine(moving, best_aff, (h, w)), meta


if __name__ == "__main__":
    # ==========================================
    # 1. 路径设置 (请修改为你的实际路径)
    # ==========================================
    TEMPLATE_IMG_PATH = r"D:\pycode\mlabel\6\templ.jpg"   # 模板图
    DEFECT_IMG_PATH = r"D:\pycode\mlabel\6\test.jpg"      # 待配准图 (瑕疵图)
    OUTPUT_DIR = "registration_results" 

    # 简单生成两张测试图（如果没有提供真实图片，避免代码崩溃）
    if not os.path.exists(TEMPLATE_IMG_PATH):
        print("未找到图片，自动生成测试样本进行演示...")
        test_template = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(test_template, (100, 100), (400, 400), (255, 255, 255), -1)
        cv2.imwrite(TEMPLATE_IMG_PATH, test_template)
        
        # 瑕疵图平移 + 轻微瑕疵
        test_defect = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(test_defect, (120, 110), (420, 410), (255, 255, 255), -1)
        cv2.circle(test_defect, (250, 250), 20, (0, 0, 255), -1) # 模拟瑕疵
        cv2.imwrite(DEFECT_IMG_PATH, test_defect)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 读取图片
    template_img = cv2.imread(TEMPLATE_IMG_PATH)
    defect_img = cv2.imread(DEFECT_IMG_PATH)

    print("正在进行图像配准计算，请稍候...")

    # 3. 调用配准函数
    ok, aligned_img, meta = align_image(
        moving=defect_img,
        template=template_img,
        enable_fast_mode=True,       
        phase_only=False             
    )

    # 4. 打印配准结果信息
    print("-" * 30)
    print(f"配准是否成功: {'是' if ok else '否'}")
    print(f"最终生效方法: {meta.get('method')}")
    print(f"置信度/得分 : {meta.get('score'):.4f}")
    if meta.get('method') == 'orb_affine':
         print(f"相位配准被拒原因: {meta.get('phase_fallback_reason')}")
    print("-" * 30)

    # 5. 效果可视化与保存
    if ok:
        if template_img.shape == aligned_img.shape:
            # 图像融合 (50% 模板图 + 50% 变换后的瑕疵图)
            blended = cv2.addWeighted(template_img, 0.5, aligned_img, 0.5, 0)

            # 导出路径
            path_template = os.path.join(OUTPUT_DIR, "1_template.jpg")
            path_original = os.path.join(OUTPUT_DIR, "2_original_defect.jpg")
            path_aligned = os.path.join(OUTPUT_DIR, "3_aligned_defect.jpg")
            path_overlay = os.path.join(OUTPUT_DIR, "4_overlay_blended.jpg")

            cv2.imwrite(path_template, template_img)
            cv2.imwrite(path_original, defect_img)
            cv2.imwrite(path_aligned, aligned_img)
            cv2.imwrite(path_overlay, blended)

            print(f"✅ 成功！四张效果图已保存至: {os.path.abspath(OUTPUT_DIR)}")

            # 图像弹窗展示
            h, w = blended.shape[:2]
            show_w = 600
            show_h = int(show_w * h / w)

            cv2.imshow("1 - Template", cv2.resize(template_img, (show_w, show_h)))
            cv2.imshow("2 - Original Defect", cv2.resize(defect_img, (show_w, show_h)))
            cv2.imshow("3 - Aligned Defect", cv2.resize(aligned_img, (show_w, show_h)))
            cv2.imshow("4 - Overlay", cv2.resize(blended, (show_w, show_h)))

            print("按键盘任意键关闭图像窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("警告: 配准后图像尺寸不匹配，无法进行叠加显示。")
    else:
        print("配准失败，请检查图像差异是否过大。")