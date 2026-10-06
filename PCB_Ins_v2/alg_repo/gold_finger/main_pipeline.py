# -*- coding: utf-8 -*-
"""
PCB 缺陷检测总控流水线 - 整合版
执行顺序: 图像配准 -> 模板统一提取 -> 全局光照匹配 -> 缺陷检测 (AbsDiff)
"""
import os
import time
import cv2
import numpy as np

# 导入你写好的配准模块
from peizhun import align_image

# 种子点自适应取色
from seed_extract import load_or_pick_seed_ranges


# ==========================================
# 模块改写 1：从 tiqu_liangse.py 改写为内存处理版
# ==========================================
def remove_small_components(mask, min_area=50):
    # 向量化过滤：用 keep 查找表一次映射整个 labels 图，
    # 避免逐连通域做 labels == i 的全图布尔运算（大图上极慢）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # 背景
    return keep[labels].astype(np.uint8) * 255

def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff_mask, (0, 0), 255)
    outside = padded[1:-1, 1:-1]
    return cv2.bitwise_or(mask, cv2.bitwise_not(outside))

def _mask_gold(hsv: np.ndarray) -> np.ndarray:
    return cv2.inRange(hsv, np.array([10, 25, 60]), np.array([50, 255, 255]))

def _mask_custom(hsv: np.ndarray, hsv_ranges) -> np.ndarray:
    """自定义目标颜色提取：hsv_ranges 为 [(lower, upper), ...] 列表，
    多段范围取并集（红色这类跨越 Hue 0/180 边界的颜色需要两段）。"""
    mask = None
    for lower, upper in hsv_ranges:
        m = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        raise ValueError("custom 模式需要至少一段 HSV 范围")
    return mask

def _extract_color_mask(hsv: np.ndarray, board_color: str,
                        custom_hsv_ranges=None) -> np.ndarray:
    if board_color == "custom":
        return _mask_custom(hsv, custom_hsv_ranges)
    if board_color == "gold":
        return _mask_gold(hsv)
    raise ValueError(f"未知的板面类型: {board_color}，可选 'gold' / 'custom' / 'seed'")

def get_pcb_alpha_mask(image: np.ndarray, board_color: str = "gold",
                       custom_hsv_ranges=None, min_region_area=None) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)

    if long_side <= 800:
        scale = 8
    elif long_side <= 2000:
        scale = 2
    else:
        scale = 1

    # 分辨率倍率：形态学核、去噪面积等参数按图像实际分辨率等比放大，
    # 保证 500px 小图与 4096px 大图在"物理板面"尺度上行为一致
    ref = max(1.0, long_side / 500.0)
    eff = scale * ref  # 有效尺度 = 处理放大倍数 x 分辨率倍率

    if scale > 1:
        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        enlarged = image
    blurred = cv2.GaussianBlur(enlarged, (5, 5), 0)
    
    # HSV 提取
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = _extract_color_mask(hsv, board_color, custom_hsv_ranges)
    
    # 形态学去噪。闭运算尺寸在"原图像素"尺度上限 8px：再大就可能桥接
    # 金手指指缝（指缝常见 15~25px），把逐条指条糊成一整块，破坏掩膜
    # 拓扑；封闭小孔由后面的 fill_mask_holes 兜底，不依赖大核闭运算
    k_close = int(round(min(4.0 * ref, 8.0) * scale)) | 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # 面积过滤噪点：min_region_area 以原图像素为单位，掩膜此时在 scale 倍
    # 处理分辨率上，因此换算为 scale^2 倍后再过滤
    if min_region_area is None:
        min_area_internal = int(20 * eff * eff)  # 自适应默认值
    else:
        min_area_internal = int(min_region_area * scale * scale)
    mask = remove_small_components(mask, min_area=min_area_internal)
    mask = cv2.medianBlur(mask, 5)

    mask = fill_mask_holes(mask)
    
    feather = int(round(2 * scale)) * 2 + 1
    mask_soft_hr = cv2.GaussianBlur(mask, (feather, feather), 0)
    if scale > 1:
        alpha = cv2.resize(mask_soft_hr, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    else:
        alpha = mask_soft_hr.astype(np.float32) / 255.0

    lo, hi = 0.35, 0.65
    t = np.clip((alpha - lo) / (hi - lo), 0.0, 1.0)
    alpha = t * t * (3.0 - 2.0 * t)

    return alpha[:, :, None] # 返回 (H, W, 1) 形状

# ==========================================
# 全局光照匹配：线性增益匹配（替代 match_histograms）
# ==========================================
def match_color_linear(source: np.ndarray, reference: np.ndarray,
                       valid_mask: np.ndarray = None) -> np.ndarray:

    src = source.astype(np.float32)
    ref = reference.astype(np.float32)
    if valid_mask is not None:
        src_px = src[valid_mask]
        ref_px = ref[valid_mask]
    else:
        src_px = src.reshape(-1, src.shape[-1])
        ref_px = ref.reshape(-1, ref.shape[-1])

    mu_s, sigma_s = src_px.mean(axis=0), src_px.std(axis=0)
    mu_r, sigma_r = ref_px.mean(axis=0), ref_px.std(axis=0)
    gain = sigma_r / np.maximum(sigma_s, 1e-6)
    # 限制增益范围，防止某通道方差异常时把噪声整体放大
    gain = np.clip(gain, 0.5, 2.0)

    out = (src - mu_s) * gain + mu_r
    return np.clip(out, 0, 255).astype(np.uint8)


def _masked_lowpass(gray: np.ndarray, mask_f: np.ndarray,
                    scale: int = 16, k: int = 31) -> np.ndarray:
    """掩膜加权大尺度低通：blur(img*m)/blur(m)，无效区(背景/黑边)不参与平滑，
    避免边缘亮度被背景黑区拉低。先缩小再模糊再放大，等效大核低通且速度快。"""
    h, w = gray.shape[:2]
    sw, sh = max(1, w // scale), max(1, h // scale)
    g = cv2.resize(gray * mask_f, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask_f, (sw, sh), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (k, k), 0)
    m = cv2.GaussianBlur(m, (k, k), 0)
    low = g / np.maximum(m, 1e-4)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def refine_alignment_ecc(alpha_good: np.ndarray, alpha_test: np.ndarray,
                         max_side: int = 1500):
    """基于板面掩膜的 ECC 仿射精配准。

    粗配准（相位相关/ORB）在周期性金手指图案上容易失败或残留数像素
    ~十几像素的系统性偏移，该偏移在块外轮廓处产生成排误检。掩膜是
    平滑的 alpha 场，梯度稳定，ECC 能可靠收敛到亚像素精度并同时
    校正平移/微旋转/微缩放。

    返回 (ok, warp)。warp 为 2x3 矩阵，语义与 findTransformECC 一致：
    refined(x) = aligned_test(warp @ x)。
    """
    g = alpha_good.squeeze().astype(np.float32)
    t = alpha_test.squeeze().astype(np.float32)
    h, w = g.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 1.0:
        size = (int(round(w * s)), int(round(h * s)))
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        t = cv2.resize(t, size, interpolation=cv2.INTER_AREA)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(g, t, warp, cv2.MOTION_AFFINE,
                                        criteria, None, 5)
    except cv2.error:
        return False, None

    # 收敛质量复核：精修必须实际降低掩膜残差才采纳。周期性金手指图案上
    # ECC 可能滑进"错一个周期"的局部最优——相关系数看似收敛，残差反而变大
    t_ref = cv2.warpAffine(t, warp, (t.shape[1], t.shape[0]),
                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    err_before = float(np.abs(g - t).mean())
    err_after = float(np.abs(g - t_ref).mean())
    improved = err_after < err_before * 1.02  # 允许恒等附近的插值微扰

    warp = warp.copy()
    warp[:, 2] /= s  # 平移量换算回原分辨率；线性部分与缩放无关

    # 合理性校验：精配准只应做小幅修正，大幅变换说明掩膜本身不可靠
    a, b = float(warp[0, 0]), float(warp[0, 1])
    c, d = float(warp[1, 0]), float(warp[1, 1])
    sx = float(np.hypot(a, c))
    sy = float(np.hypot(b, d))
    rot = float(np.degrees(np.arctan2(c, a)))
    tx, ty = float(warp[0, 2]), float(warp[1, 2])
    ok = (cc > 0.5 and improved and abs(rot) <= 5.0
          and 0.9 <= sx <= 1.1 and 0.9 <= sy <= 1.1
          and abs(tx) <= 0.1 * w / s and abs(ty) <= 0.1 * h / s)
    return ok, (warp if ok else None)


def compose_refined_affine(coarse_affine: np.ndarray, ecc_warp: np.ndarray) -> np.ndarray:
    """合成粗配准仿射 A 与 ECC 精修 W 为单个前向仿射 M。

    aligned(x) = moving(A^-1 x)，refined(x) = aligned(W x) = moving(A^-1 W x)，
    warpAffine 前向语义 refined(x) = moving(M^-1 x)  =>  M = W^-1 ∘ A。
    用合成矩阵对原图一次 warp，避免二次重采样损失高频细节。
    """
    A3 = np.vstack([np.asarray(coarse_affine, np.float64), [0, 0, 1]])
    W3 = np.vstack([np.asarray(ecc_warp, np.float64), [0, 0, 1]])
    return (np.linalg.inv(W3) @ A3)[:2].astype(np.float32)


def match_illumination_local(source: np.ndarray, reference: np.ndarray,
                             valid_mask: np.ndarray,
                             max_gain: float = 1.6) -> np.ndarray:
    """
    低频光照场校正。全局线性匹配只能修正整体曝光/白平衡；当两图存在
    非均匀亮度差（灯光角度、暗角、局部阴影）时，残余亮度差会整片进入
    absdiff 造成大面积误检。这里估计两图各自的低频亮度场，按
    ref_low / src_low 的比值逐像素校正 source 的亮度。
    """
    mask_f = valid_mask.astype(np.float32)
    src_l = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_l = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)

    src_low = _masked_lowpass(src_l, mask_f)
    ref_low = _masked_lowpass(ref_l, mask_f)

    gain = ref_low / np.maximum(src_low, 1.0)
    # 限制局部增益，防止极暗区域的比值爆炸放大噪声
    gain = np.clip(gain, 1.0 / max_gain, max_gain)

    out = source.astype(np.float32) * gain[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================
# 模块 2：精简 detection.py 的检测逻辑
# ==========================================
def _build_block_mask(core_mask: np.ndarray) -> np.ndarray:
    """把逐条金手指桥接成"整块"掩膜，供边缘带定位使用。

    闭运算核必须大于指缝宽度才能桥接，而指缝宽度与图像尺寸没有固定
    比例（固定核在某些图上会让整块碎成多块，每个碎块边界都被当成
    "外轮廓"，边缘带就会切进金手指内部、误伤内部检测）。这里从小核
    开始翻倍尝试，直到连通域数量不再减少——指缝被桥接后数量骤降并
    稳定；板上确实相距很远的多个独立金区则保持独立，不会被强行合并。
    闭运算在缩小图上做，大核也能保持毫秒级。
    """
    h, w = core_mask.shape[:2]
    s = min(1.0, 500.0 / max(h, w))
    if s < 1.0:
        small = cv2.resize(core_mask, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_NEAREST)
    else:
        small = core_mask

    def _close(mask, k):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    k = 15
    block = _close(small, k)
    n = cv2.connectedComponents(block)[0] - 1
    while n > 1 and 2 * k + 1 <= max(small.shape) // 2:
        k2 = 2 * k + 1
        block2 = _close(small, k2)
        n2 = cv2.connectedComponents(block2)[0] - 1
        if n2 < n:
            block, n, k = block2, n2, k2
        else:
            break

    block = fill_mask_holes(block)
    if s < 1.0:
        block = cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST)
        block = (block > 127).astype(np.uint8) * 255
    return block


def detect_defects(im_good_masked, im_test_masked, alpha_mask, diff_thresh=25, min_defect_area=5,
                   shift_tolerance=1, edge_margin=2, edge_band=12, edge_penalty=2.0,
                   edge_thresh_cap=None, alpha_mask_test=None, display_img=None):
    """
    edge_band / edge_penalty: 板面"整块外轮廓"抗误检。外轮廓是"金面 vs
        背景"的高对比跳变区，配准亚像素残差与光照校正误差在这里最大。
        距整块外轮廓 edge_band 像素内，判定阈值从 diff_thresh*(1+edge_penalty)
        线性衰减回 diff_thresh；edge_thresh_cap 可给抬高后的阈值设绝对上限，
        保证边缘的严重缺陷（大面积缺金/崩边）仍能报出。
        注意距离按闭运算后的"整块"边界算，内部金手指之间的边缘不受影响。
    alpha_mask_test: 待测图自身的板面掩膜。提供时，仅在整块外轮廓带内
        取两掩膜交集，消除"良品是金面、待测因错位抠到背景"类假差异。
        内部不取交集——异物/污渍会让待测掩膜在缺陷处缺失，全图交集会把
        真实缺陷像素一起排除。外轮廓带内被交集排除的几何差异由
        detect_mask_shape_defects 单独兜底，不留盲区。
    """

    # 1. 邻域容差差分：只有超出良品图邻域 min/max 包络的像素才计入差异
    if shift_tolerance > 0:
        k = 2 * shift_tolerance + 1
        kernel_tol = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        good_max = cv2.dilate(im_good_masked, kernel_tol)  # 逐通道邻域最大值
        good_min = cv2.erode(im_good_masked, kernel_tol)   # 逐通道邻域最小值
        diff_hi = cv2.subtract(im_test_masked, good_max)   # 比邻域最亮还亮的部分
        diff_lo = cv2.subtract(good_min, im_test_masked)   # 比邻域最暗还暗的部分
        diff_bgr = cv2.max(diff_hi, diff_lo)
    else:
        diff_bgr = cv2.absdiff(im_good_masked, im_test_masked)

    # 2. 取三通道的最大差异，只要红、绿、蓝任一通道发生变化就捕捉
    diff_max = np.max(diff_bgr, axis=2)

    print(f"   -> [Debug] 当前图像的最大像素差值为: {diff_max.max()}")
    print(f"   -> [Debug] 设定的拦截阈值为: {diff_thresh} (只有差值大于此数值才会被判定为瑕疵)")

    # 3. 检测核心区掩膜：
    #    - alpha < 0.99 的软边缘带（羽化区）本身就会因混合系数不同产生假差异，直接排除
    #    - 用大核闭运算把逐条金手指桥接成"整块"，得到块外轮廓；
    #      到块外轮廓的距离用于区分"外边缘带"与"内部"
    #    - 仅在外边缘带内与待测掩膜取交集，排除掩膜与待测图错位的假差异
    #    - 再向内腐蚀 edge_margin 像素，消除掩膜轮廓自身的抖动
    good_core = alpha_mask.squeeze() > 0.99
    core_mask = good_core.astype(np.uint8) * 255

    # "整块"掩膜：自适应闭运算把指条桥接成整块，块外轮廓 = 边缘带基准。
    # 垫一圈零边界再算距离场：金面延伸到图像边界时，图像边缘同样存在
    # 配准残余/warp 黑边风险，必须一并算作"块外轮廓"
    block_mask = _build_block_mask(core_mask)
    dist_block = cv2.distanceTransform(
        cv2.copyMakeBorder(block_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
        cv2.DIST_L2, 5)[1:-1, 1:-1]

    if alpha_mask_test is not None:
        border_zone = dist_block < max(12, 4 * edge_band)  # 块外轮廓附近
        test_core = alpha_mask_test.squeeze() > 0.99
        core_mask[border_zone & ~test_core] = 0

    if edge_margin > 0:
        kernel_edge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * edge_margin + 1, 2 * edge_margin + 1)
        )
        core_mask = cv2.erode(core_mask, kernel_edge)

    # 4. 二值化：块外轮廓带内用距离衰减的抬高阈值，内部用固定阈值
    if edge_band > 0 and edge_penalty > 0:
        band = np.clip(1.0 - dist_block / float(edge_band), 0.0, 1.0)  # 块边界处 1，内部 0
        thresh_map = diff_thresh * (1.0 + edge_penalty * band)
        if edge_thresh_cap is not None:
            thresh_map = np.minimum(thresh_map, float(edge_thresh_cap))
        diff_binary = (diff_max.astype(np.float32) > thresh_map).astype(np.uint8) * 255
    else:
        _, diff_binary = cv2.threshold(diff_max, diff_thresh, 255, cv2.THRESH_BINARY)
    diff_binary = cv2.bitwise_and(diff_binary, core_mask)

    # 5. 形态学清理与面积过滤
    kernel_clean = np.ones((3, 3), np.uint8)
    diff_clean = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel_clean)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_clean, connectivity=8)
    # 向量化面积过滤，避免逐连通域全图布尔运算
    keep = stats[:, cv2.CC_STAT_AREA] >= min_defect_area
    keep[0] = False  # 背景
    final_diff = keep[labels].astype(np.uint8) * 255
    defect_boxes = [
        (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        for i in np.flatnonzero(keep)
    ]

    # 6. 画框标注（坐标系与 display_img / im_test_masked 一致，一般为配准后空间）
    result_img = (display_img if display_img is not None else im_test_masked).copy()
    draw_defect_boxes(result_img, defect_boxes)

    defect_pixels = np.count_nonzero(final_diff)

    return result_img, final_diff, defect_boxes, defect_pixels


def detect_mask_shape_defects(alpha_good, alpha_test, valid_mask=None,
                              tol_px=3, min_area=40):
    """比较良品/待测两张板面掩膜的几何差异，检出边缘缺料、多料类形状缺陷。

    颜色差分的检测区取了两掩膜交集（消除错位假差异），代价是"交集之外"
    成为盲区——待测板边缘真实缺金恰好落在这里。此函数补上该盲区：
    一方掩膜超出另一方 tol_px 容差带（吸收配准残差）的部分即为形状缺陷。

    valid_mask: 配准 warp 后的有效像素区（0/255）。待测图视野外的区域
        掩膜必然为空，不代表真实缺料，用它排除。
    返回 (shape_mask, boxes)。
    """
    good_bin = (alpha_good.squeeze() > 0.5).astype(np.uint8) * 255
    test_bin = (alpha_test.squeeze() > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    # 缺料：良品有金而待测在容差带外仍无金；多料反之
    missing = cv2.bitwise_and(good_bin, cv2.bitwise_not(cv2.dilate(test_bin, k)))
    extra = cv2.bitwise_and(test_bin, cv2.bitwise_not(cv2.dilate(good_bin, k)))
    if valid_mask is not None:
        missing = cv2.bitwise_and(missing, valid_mask)
        extra = cv2.bitwise_and(extra, valid_mask)
    shape_diff = cv2.bitwise_or(missing, extra)
    shape_diff = cv2.morphologyEx(shape_diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(shape_diff, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # 背景
    shape_mask = keep[labels].astype(np.uint8) * 255
    boxes = [
        (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
         int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        for i in np.flatnonzero(keep)
    ]
    return shape_mask, boxes


def map_boxes_to_source(boxes, affine, src_shape, tpl_shape):
    """把配准后（模板）坐标系下的缺陷框映回原始待测图坐标。

    peizhun.align_image 流程：先把待测图 resize 到模板尺寸，再做仿射 warp。
    因此回映 = 仿射逆变换 + 按原图/模板尺寸比例缩放。
    """
    if not boxes:
        return []
    src_h, src_w = src_shape[:2]
    tpl_h, tpl_w = tpl_shape[:2]
    aff = np.array(affine if affine is not None else [[1, 0, 0], [0, 1, 0]], np.float32)
    if aff.shape != (2, 3):
        aff = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    inv = cv2.invertAffineTransform(aff)
    sx, sy = src_w / float(tpl_w), src_h / float(tpl_h)

    mapped = []
    for (x, y, w, h) in boxes:
        corners = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float32)
        pts = corners @ inv[:, :2].T + inv[:, 2]
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        x0 = int(np.clip(np.floor(pts[:, 0].min()), 0, src_w - 1))
        y0 = int(np.clip(np.floor(pts[:, 1].min()), 0, src_h - 1))
        x1 = int(np.clip(np.ceil(pts[:, 0].max()), 1, src_w))
        y1 = int(np.clip(np.ceil(pts[:, 1].max()), 1, src_h))
        mapped.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return mapped


def draw_defect_boxes(img, boxes):
    """在图上画红色缺陷外接框（原地修改并返回）。框内像素保持不动。"""
    img_h, img_w = img.shape[:2]
    line_thick = max(1, int(round(max(img_h, img_w) / 1000)))
    pad = line_thick * 3
    for (bx, by, bw, bh) in boxes:
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(img_w - 1, bx + bw + pad)
        y1 = min(img_h - 1, by + bh + pad)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), line_thick)
    return img


# ==========================================
# 缺陷分类：加载 train_classifier.py 训练的随机森林，
# 对 diff 掩膜的每个连通域抽取与训练时同一套手工特征后预测类别
# ==========================================
_CLASSIFIER_CACHE = {}


def _load_classifier(model_path):
    """加载 train_classifier.py 导出的纯 NumPy 森林 (.npz)。
    joblib 反序列化 sklearn 模型每个进程要约 1s；NumPy 格式毫秒级加载，
    且检测端不需要安装 scikit-learn。模块级缓存避免同进程重复加载。"""
    key = os.path.abspath(model_path)
    if key not in _CLASSIFIER_CACHE:
        from forest_infer import NumpyForest
        _CLASSIFIER_CACHE[key] = NumpyForest(model_path)
    return _CLASSIFIER_CACHE[key]


def classify_defects(diff_mask, image_bgr, model_path):
    """
    diff_mask:  detect_defects 输出的二值缺陷掩膜。
    image_bgr:  抽取颜色/纹理特征的底图。用配准归一后的完整待测图
                (norm_test)，与训练时的原始标注图口径一致；
                不要用 alpha 抠过的图，否则缺陷外围环带会是黑色。
    返回 [(box, label, prob), ...]，box 为 (x, y, w, h)。
    """
    from defect_features import extract_region_features

    clf = _load_classifier(model_path)

    img_h, img_w = image_bgr.shape[:2]
    long_side = max(img_h, img_w)
    pad = 48

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)

    boxes, feat_rows, valid_idx = [], [], []
    for i in range(1, num_labels):
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)

        roi = image_bgr[y0:y1, x0:x1]
        roi_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        feats = extract_region_features(roi, roi_mask, long_side)

        boxes.append((x, y, w, h))
        if feats is not None:
            valid_idx.append(len(boxes) - 1)
            feat_rows.append(feats)

    results = [(box, "Unknown", 0.0) for box in boxes]
    if feat_rows:
        proba = clf.predict_proba(np.stack(feat_rows))
        best = np.argmax(proba, axis=1)
        for row, bi in enumerate(valid_idx):
            j = int(best[row])
            results[bi] = (boxes[bi], str(clf.classes_[j]), float(proba[row, j]))
    return results


def annotate_defect_labels(result_img, classified):
    """把类别与置信度写到缺陷红框上方，字号随分辨率自适应。"""
    img_h, img_w = result_img.shape[:2]
    scale = max(0.4, max(img_h, img_w) / 2500.0)
    thick = max(1, int(round(scale * 2)))
    for (x, y, w, h), label, prob in classified:
        text = f"{label} {prob:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx = min(max(0, x), img_w - tw - 1)
        ty = y - 6 if y - th - 6 >= 0 else min(img_h - 2, y + h + th + 6)
        cv2.putText(result_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 255, 255), thick, cv2.LINE_AA)
    return result_img


# ==========================================
# 主流程控制
# ==========================================
def main():
    # 1. 路径设置
    path_good = r"D:\pycode\mlabel\image\3\t_fov_3.jpg"
    path_test = r"D:\pycode\mlabel\image\3\test\3.jpg" 
    output_dir = "pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # 板面类型:
    #   "gold"       亮色沉金/金手指, 低分辨率
    #   "custom"     自定义目标颜色 (使用下方 CUSTOM_HSV_RANGES)
    #   "seed"       种子点取色（推荐）：首次运行弹窗，在良品图上点几个板面
    #                上的点，自动生成 HSV 范围并保存；之后直接复用配置。
    BOARD_COLOR = "seed"

    # 仅 "seed" 模式生效：True = 忽略已保存的取色配置，强制重新弹窗取色。
    # （或者直接删除 pipeline_results/_seed_ranges.json 也可以）
    SEED_REPICK = False

    # 自定义目标颜色（仅 BOARD_COLOR = "custom" 时生效）。
    # 每段为 ((H低, S低, V低), (H高, S高, V高))，OpenCV 中 H 取值 0~180。
    # 多段取并集；红色这类跨越 H=0/180 边界的颜色需要写两段，示例:
    # CUSTOM_HSV_RANGES = [((0, 40, 20), (10, 255, 255)), ((156, 40, 20), (180, 255, 255))]
    CUSTOM_HSV_RANGES = [((10, 25, 60), (50, 255, 255))]

    # 保留区域的最小面积（原图像素数），小于该面积的色块当作噪点剔除。
    # None = 按分辨率自适应；噪点干扰多时调大，细小焊盘被误删时调小。
    MIN_REGION_AREA = None

    im_good = cv2.imread(path_good)
    im_test = cv2.imread(path_test)
    
    if im_good is None or im_test is None:
        print("❌ 错误：读取图片失败。")
        return

    # "seed" 模式：把交互取色的结果转成 custom 模式的 HSV 范围，
    # 后续提取 / 缓存逻辑与 custom 完全一致
    if BOARD_COLOR == "seed":
        seed_cfg_path = os.path.join(output_dir, "_seed_ranges.json")
        seed_ranges = load_or_pick_seed_ranges(
            im_good, seed_cfg_path, image_path=path_good, force_repick=SEED_REPICK
        )
        if seed_ranges is None:
            print("⚠️ 取色被取消，回退到 gold 模式。")
            BOARD_COLOR = "gold"
        else:
            BOARD_COLOR = "custom"
            CUSTOM_HSV_RANGES = seed_ranges
            print(f" -> [seed] 生成 {len(seed_ranges)} 段 HSV 范围: {seed_ranges}")

    print("🚀 开始执行自动化缺陷检测流水线...")
    t_total = time.perf_counter()

    # [步骤 1-3 缓存]：配准 / 掩膜提取 / 光照匹配只依赖输入图片和板面类型，
    # 与 diff_thresh 等检测参数无关。调参时直接命中缓存，只重跑第 4 步检测。
    # 输入图片被修改（mtime 变化）或换图 / 换板面类型时缓存自动失效。
    # norm_v9: 新增锚点配准候选 + 掩膜闭运算上限，旧缓存的预处理结果不再可信
    cache_key = f"{path_good}|{path_test}|{BOARD_COLOR}|{CUSTOM_HSV_RANGES if BOARD_COLOR == 'custom' else ''}|{MIN_REGION_AREA}|norm_v9"
    cache_path = os.path.join(output_dir, "_preprocess_cache.npz")
    norm_test = None
    alpha_mask = None
    alpha_mask_test = None
    valid = None
    align_affine = None
    try:
        if os.path.exists(cache_path):
            src_mtime = max(os.path.getmtime(path_good), os.path.getmtime(path_test))
            if os.path.getmtime(cache_path) > src_mtime:
                cache = np.load(cache_path, allow_pickle=False)
                if str(cache["key"]) == cache_key and "alpha_mask_test" in cache.files:
                    norm_test = cache["norm_test"]
                    alpha_mask = cache["alpha_mask"]
                    alpha_mask_test = cache["alpha_mask_test"]
                    if alpha_mask_test.size == 0:  # 空数组 = 待测掩膜不可靠
                        alpha_mask_test = None
                    valid = cache["valid"]
                    align_affine = cache["affine"]
    except Exception as e:
        print(f"⚠️ 缓存读取失败，将重新计算: {e}")
        norm_test = None
        alpha_mask = None
        alpha_mask_test = None
        valid = None
        align_affine = None

    if norm_test is not None:
        print("\n[1-3/4] 命中预处理缓存，跳过配准 / 掩膜提取 / 光照匹配。")
    else:
        # [步骤 1]：图像配准 (全图视野)
        print("\n[1/4] 正在进行高精度图像配准...")
        t0 = time.perf_counter()
        ok, aligned_test, meta = align_image(im_test, im_good)
        if not ok:
            print("⚠️ 警告：配准置信度低，流水线继续执行，但结果可能存在偏差。")
        align_affine = np.array(
            meta.get("affine", [[1, 0, 0], [0, 1, 0]]), dtype=np.float32)
        print(f" -> 配准方法: {meta.get('method')}, 得分: {meta.get('score', 0):.4f}")
        print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

        # [步骤 2]：计算 Mask (解决边缘误差的核心)
        # 掩膜只依赖良品图，先算出来供光照匹配限定统计范围
        print("\n[2/4] 正在提取板面掩膜...")
        t0 = time.perf_counter()
        alpha_mask = get_pcb_alpha_mask(
            im_good,
            board_color=BOARD_COLOR,
            custom_hsv_ranges=CUSTOM_HSV_RANGES,
            min_region_area=MIN_REGION_AREA,
        )
        print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

        # [步骤 2.5]：待测图板面掩膜 + ECC 精配准。
        # 掩膜必须在色彩归一化"之前"的 aligned_test 上提取：线性色彩匹配
        # 会把背景色整体平移，可能恰好移进板面 HSV 范围，导致掩膜铺满全图。
        print("\n[2.5/4] 正在提取待测掩膜并做 ECC 精配准...")
        t0 = time.perf_counter()

        def _extract_test_mask(img):
            v = (img.max(axis=2) > 0).astype(np.uint8) * 255
            v = cv2.erode(v, np.ones((5, 5), np.uint8))
            m = get_pcb_alpha_mask(
                img,
                board_color=BOARD_COLOR,
                custom_hsv_ranges=CUSTOM_HSV_RANGES,
                min_region_area=MIN_REGION_AREA,
            )
            return m * (v[:, :, None] > 0), v  # warp 黑边视野外区域清零

        alpha_mask_test, valid = _extract_test_mask(aligned_test)

        # 可靠性检查：待测掩膜面积与良品掩膜差异过大说明取色范围不适配
        # 待测图（曝光差异等），此时禁用掩膜交集与形状检测，避免灾难性误检
        area_good = max(1, np.count_nonzero(alpha_mask > 0.5))
        area_ratio = np.count_nonzero(alpha_mask_test > 0.5) / area_good
        if not (0.5 <= area_ratio <= 1.5):
            print(f"⚠️ 待测掩膜面积异常（{area_ratio:.0%} of 良品），跳过精配准与掩膜交集。")
            alpha_mask_test = None
        else:
            # 粗配准在周期性金手指图案上常残留数像素~十几像素系统偏移，
            # 在两张掩膜上做 ECC 精配准，合成单一仿射后对原图一次性重 warp
            ok_ecc, ecc_warp = refine_alignment_ecc(alpha_mask, alpha_mask_test)
            if ok_ecc:
                align_affine = compose_refined_affine(align_affine, ecc_warp)
                moving = im_test
                th, tw = im_good.shape[:2]
                if moving.shape[:2] != (th, tw):
                    interp = (cv2.INTER_AREA
                              if (moving.shape[0] > th or moving.shape[1] > tw)
                              else cv2.INTER_CUBIC)
                    moving = cv2.resize(moving, (tw, th), interpolation=interp)
                aligned_test = cv2.warpAffine(
                    moving, align_affine, (tw, th), flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                alpha_mask_test, valid = _extract_test_mask(aligned_test)
                print(f" -> ECC 精配准生效: {np.round(ecc_warp, 4).tolist()}")
            else:
                print(" -> ECC 精配准未收敛或修正量异常，保留粗配准结果。")
        print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

        # [步骤 3]：全局光照与色彩归一化 (线性增益匹配)
        print("\n[3/4] 正在进行全局光照匹配...")
        t0 = time.perf_counter()
        stats_mask = (valid > 0) & (alpha_mask.squeeze() > 0.99)
        norm_test = match_color_linear(aligned_test, im_good, valid_mask=stats_mask)
        # 低频光照场校正：修正灯光角度/暗角/局部阴影等非均匀亮度差
        norm_test = match_illumination_local(norm_test, im_good, valid_mask=stats_mask)
        print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

        t0 = time.perf_counter()
        np.savez(
            cache_path,
            key=cache_key,
            norm_test=norm_test,
            alpha_mask=alpha_mask,
            # None（待测掩膜不可靠）用空数组占位，加载时还原
            alpha_mask_test=(alpha_mask_test if alpha_mask_test is not None
                             else np.zeros((0,), np.float32)),
            valid=valid,
            affine=align_affine,
        )
        print(f" -> 预处理结果已缓存至 {cache_path}，下次调参将直接复用。")
        print(f" -> [耗时] 缓存写入: {time.perf_counter() - t0:.3f}s")

    # 差分用图：轻度模糊降噪
    print(" -> 正在应用掩膜与模糊...")
    t0 = time.perf_counter()
    blur_good = cv2.GaussianBlur(im_good, (3, 3), 0)
    blur_test = cv2.GaussianBlur(norm_test, (3, 3), 0)
    im_good_extracted = (blur_good.astype(np.float32) * alpha_mask).astype(np.uint8)
    im_test_extracted = (blur_test.astype(np.float32) * alpha_mask).astype(np.uint8)
    im_good_display = (im_good.astype(np.float32) * alpha_mask).astype(np.uint8)
    im_test_display = (norm_test.astype(np.float32) * alpha_mask).astype(np.uint8)
    print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

    # [步骤 4]：缺陷检测（框坐标在配准后的模板坐标系）
    print("\n[4/4] 正在执行 AbsDiff 缺陷分析...")
    t0 = time.perf_counter()
    long_side = max(im_good.shape[:2])
    # 边缘带宽度按分辨率自适应：500px 图约 6px，4096px 图约 14px
    edge_band = max(6, int(round(long_side / 300)))
    aligned_vis, diff_mask, defect_boxes, defect_count = detect_defects(
        im_good_extracted,
        im_test_extracted,
        alpha_mask,
        diff_thresh=25,       # <--- 从 45 降到 20，让它变敏感
        min_defect_area=5,    # <--- 从 15 降到 5，捕捉更小的瑕疵
        shift_tolerance=1,    # <--- 容忍 1px 配准残余偏移，消除轮廓边缘误检
        edge_margin=1,        # <--- 掩膜边界向内该像素数不参与判定
        edge_band=edge_band,  # <--- 块外轮廓向内该宽度内阈值渐进抬高
        edge_penalty=1.0,     # <--- 块最外沿阈值 = diff_thresh*(1+edge_penalty)，仅作用于块外轮廓带
        edge_thresh_cap=None, # <--- 抬高阈值的绝对上限（None = 不设限）
        alpha_mask_test=alpha_mask_test,  # <--- 块外轮廓带内取良品/待测掩膜交集
        display_img=im_test_display  # 配准空间预览用；最终交付画在原测试图上
    )

    # [步骤 4.5]：掩膜几何差异检测。块外轮廓带内颜色差分取了两掩膜交集，
    # 待测板边缘真实缺金/多料会落在交集之外成为盲区，这里用掩膜对比兜底。
    if alpha_mask_test is not None:
        shape_tol = max(2, int(round(long_side / 800)))       # 吸收配准残差的容差带
        shape_min_area = int(max(20, 20 * (long_side / 1000.0) ** 2))
        shape_mask, shape_boxes = detect_mask_shape_defects(
            alpha_mask, alpha_mask_test,
            valid_mask=valid, tol_px=shape_tol, min_area=shape_min_area,
        )
        if shape_boxes:
            print(f" -> [形状] 掩膜几何差异检出 {len(shape_boxes)} 处边缘缺料/多料")
            diff_mask = cv2.bitwise_or(diff_mask, shape_mask)
            defect_boxes = defect_boxes + shape_boxes
            defect_count = int(np.count_nonzero(diff_mask))
            draw_defect_boxes(aligned_vis, shape_boxes)

    # 框映回原测试图坐标，在真实拍摄图上画框（光照/掩膜处理后的图会改像素外观）
    src_boxes = map_boxes_to_source(
        defect_boxes, align_affine, im_test.shape, im_good.shape)
    final_result = draw_defect_boxes(im_test.copy(), src_boxes)

    print(f" -> 发现缺陷区域: {len(defect_boxes)} 处, 异常像素共 {defect_count} 个")
    print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

    # [步骤 5]：缺陷分类（可选）。模型由 train_classifier.py 训练生成，
    # 不存在时自动跳过，不影响检测主流程。
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defect_classifier.npz")
    if defect_boxes and os.path.exists(model_path):
        print("\n[5/5] 正在对缺陷区域分类...")
        t0 = time.perf_counter()
        # 特征仍在配准归一空间抽取；标签文字画到原图对应框上方
        classified = classify_defects(diff_mask, norm_test, model_path)
        mapped_cls = map_boxes_to_source(
            [b for b, _, _ in classified], align_affine, im_test.shape, im_good.shape)
        classified_src = [
            (mapped_cls[i], label, prob)
            for i, (_, label, prob) in enumerate(classified)
        ]
        annotate_defect_labels(final_result, classified_src)
        for (bx, by, bw, bh), label, prob in classified_src:
            print(f" -> ({bx},{by},{bw}x{bh})  {label}  置信度 {prob:.1%}")
        print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")
    elif defect_boxes:
        print("\n[5/5] 未找到 defect_classifier.npz，跳过缺陷分类。"
              "（先运行 extract_features.py 和 train_classifier.py 生成模型）")

    # 保存关键节点的图像供复查
    print(" -> 正在保存输出图像...")
    t0 = time.perf_counter()
    cv2.imwrite(os.path.join(output_dir, "1_aligned_and_normed.jpg"), norm_test)
    cv2.imwrite(os.path.join(output_dir, "2_extracted_good.jpg"), im_good_display)
    cv2.imwrite(os.path.join(output_dir, "3_extracted_test.jpg"), im_test_display)
    cv2.imwrite(os.path.join(output_dir, "4_aligned_boxes.jpg"), aligned_vis)  # 配准空间预览
    cv2.imwrite(os.path.join(output_dir, "4_final_result.jpg"), final_result)  # 原测试图画框
    cv2.imwrite(os.path.join(output_dir, "4_diff_mask.jpg"), diff_mask)
    print(f" -> [耗时] {time.perf_counter() - t0:.3f}s")

    print(f"\n✅ 流水线执行完毕！所有产物已保存至: {output_dir}")
    print(f"⏱️ 总耗时: {time.perf_counter() - t_total:.3f}s")

    # 结果展示
    h, w = final_result.shape[:2]
    scale = min(1.0, 1000 / max(h, w))
    preview = cv2.resize(final_result, (int(w * scale), int(h * scale)))
    cv2.imshow("Final Result", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()