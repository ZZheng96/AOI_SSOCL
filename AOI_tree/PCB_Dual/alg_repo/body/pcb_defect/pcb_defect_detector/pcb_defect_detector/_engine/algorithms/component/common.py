import cv2, numpy as np
import threading
from collections import OrderedDict
from functools import wraps

# 开启 OpenCV 内置 SIMD/IPP 优化（对 Canny/matchTemplate/形态学等均有效）
cv2.setUseOptimized(True)

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# ── ndarray 特征缓存（LRU）──
# 键 = (函数名, 数组指纹)。模板每帧字节相同 → 跨帧命中；同帧重复调用 → 帧内命中。
# 采用 LRU：命中刷新到末尾、超限淘汰最旧，保证高频复用的模板特征常驻（流式场景关键）。
_FEAT_CACHE = OrderedDict()
_FEAT_CACHE_MAX = 256
_FEAT_LOCK = threading.Lock()
_MISS = object()


def _arr_fp(a):
    """数组指纹：(形状, dtype, 内容哈希)。None 返回 None。"""
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return (a.shape, a.dtype.str, hash(a.tobytes()))


def _cache_lookup(key):
    with _FEAT_LOCK:
        v = _FEAT_CACHE.get(key, _MISS)
        if v is not _MISS:
            _FEAT_CACHE.move_to_end(key)  # LRU 刷新
        return v


def _cache_store(key, val):
    with _FEAT_LOCK:
        _FEAT_CACHE[key] = val
        _FEAT_CACHE.move_to_end(key)
        while len(_FEAT_CACHE) > _FEAT_CACHE_MAX:
            _FEAT_CACHE.popitem(last=False)  # 淘汰最旧
    return val


def clear_feature_cache():
    """清空特征缓存（更换模板批次/释放内存时调用）。"""
    with _FEAT_LOCK:
        _FEAT_CACHE.clear()


def _memo1(fn):
    """缓存「单 ndarray 入参 + 仅默认其余参数」的纯函数返回值。

    注意：被缓存函数的返回值视为只读，调用方不得原地修改。
    """
    name = fn.__name__

    @wraps(fn)
    def wrapper(img, *args, **kw):
        if args or kw:
            return fn(img, *args, **kw)
        fp = _arr_fp(img)
        if fp is None:
            return fn(img)
        key = (name, fp)
        v = _cache_lookup(key)
        if v is _MISS:
            v = _cache_store(key, fn(img))
        return v

    return wrapper


def _memo2(fn):
    """缓存「两 ndarray 入参 + 仅默认其余参数」的纯函数返回值（帧内去冗余用）。"""
    name = fn.__name__

    @wraps(fn)
    def wrapper(a, b, *args, **kw):
        if args or kw:
            return fn(a, b, *args, **kw)
        fpa, fpb = _arr_fp(a), _arr_fp(b)
        if fpa is None or fpb is None:
            return fn(a, b)
        key = (name, fpa, fpb)
        v = _cache_lookup(key)
        if v is _MISS:
            v = _cache_store(key, fn(a, b))
        return v

    return wrapper

# ── 基础工具 ──
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
def clamp_box(box, shape):
    h_img, w_img = shape[:2]; x, y, w, h = (int(round(v)) for v in box)
    x = max(0, min(x, w_img-1)); y = max(0, min(y, h_img-1))
    return x, y, max(1, min(w, w_img - x)), max(1, min(h, h_img - y))
def crop(img, box):
    x, y, w, h = clamp_box(box, img.shape); return img[y:y+h, x:x+w]
def expand_box(box, margin, shape):
    x, y, w, h = box; return clamp_box((x-margin, y-margin, w+2*margin, h+2*margin), shape)
def angle_diff(a, b):
    d = abs(a-b) % 180.0
    if d > 90.0: d = 180.0 - d
    return min(d, 90.0)


# ── ROI 结构匹配（整图优先，回退局部/原位；平移 + 小角度 + 180°）──
_ROI_MATCH_MAXSIDE = 140       # 定位阶段（全角度粗扫+精修）降采样上限，只用来找候选，不参与最终打分
_ROI_MATCH_FULL_MAXSIDE = 960  # 整图粗搜降采样上限，控制粗搜耗时
_ROI_MATCH_MIN_TPL_SIDE = 24   # 粗搜降采样后模板最短边下限，避免特征丢失导致误匹配
_ROI_MATCH_COARSE_MIN = 0.30   # 整图粗搜候选可信下限（低于此分数直接放弃，退回局部搜索）
_ROI_VERIFY_MAXSIDE = 480      # 候选核验阶段降采样上限，远比定位阶段宽松；只核验0°/180°各1个
                                # 候选（至多2次），不重新扫描全部角度/位置，因此可以承受更高分辨率


def _affine_h(m):
    """2×3 仿射矩阵转 3×3 齐次矩阵。"""
    return np.vstack([np.asarray(m, np.float64), [0.0, 0.0, 1.0]])


def _affine_2x3(m):
    return np.asarray(m, np.float64)[:2, :]


def _clahe_normalize(gray):
    """CLAHE 局部对比度归一化，与 :func:`_match_feature` 共用同一步骤。"""
    if gray.ndim == 3:
        gray = to_gray(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)


def _match_feature(gray):
    """构造对光照变化更稳的灰度/边缘混合结构图。"""
    eq = _clahe_normalize(gray)
    edges = cv2.Canny(eq, 40, 120)
    return cv2.addWeighted(eq, 0.35, edges, 0.65, 0)


def _has_texture(gray, std_min=6.0, edge_density_min=0.01):
    """判断是否有可匹配的结构，而非直接用原始灰度的全局标准差判断。

    很多黑色胶体元件的激光镭雕丝印本身对比度很低（如全局 std<6、Canny 边缘几乎为
    0），但那只是"整幅原始灰度的全局对比度"低，并不代表没有可匹配的结构——真正
    参与匹配的 :func:`_match_feature` 会先做 CLAHE 局部对比度增强，丝印/引脚等
    细节在增强后通常清晰可辨。这里复用同一步 CLAHE 再判断，避免"能匹配却被判定
    低纹理而直接放弃"的误拒；对真正空白/纯色区域，CLAHE 后仍然是空白，判断结果
    不变。
    """
    eq = _clahe_normalize(gray)
    edge_density = cv2.countNonZero(cv2.Canny(eq, 40, 120)) / max(1, eq.size)
    return not (float(eq.std()) < std_min and edge_density < edge_density_min)


def _rotate_bound(gray, angle_deg):
    """不裁角地旋转图像，并返回 template→rotated 的仿射矩阵。"""
    h, w = gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos_v, sin_v = abs(m[0, 0]), abs(m[0, 1])
    out_w = max(2, int(np.ceil(h * sin_v + w * cos_v)))
    out_h = max(2, int(np.ceil(h * cos_v + w * sin_v)))
    m[0, 2] += out_w / 2.0 - cx
    m[1, 2] += out_h / 2.0 - cy
    fill = int(np.median(gray))
    rotated = cv2.warpAffine(
        gray, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return rotated, m


def _angle_branch(angle_deg):
    """返回角度所属方向分支及相对 0/180° 的普通旋转余角。"""
    a = float(angle_deg) % 360.0
    residual = ((a + 90.0) % 180.0) - 90.0
    is_180_branch = 90.0 <= a < 270.0
    return is_180_branch, residual


def _scan_angle_match(template_gray, search_gray, angles):
    """在 search 内扫描旋转模板，返回每个角度的最佳 NCC 候选。"""
    search_feat = _match_feature(search_gray)
    candidates = []
    for angle in angles:
        rotated, rot_m = _rotate_bound(template_gray, angle)
        rot_feat = _match_feature(rotated)
        rh, rw = rot_feat.shape[:2]
        sh, sw = search_feat.shape[:2]
        if rh > sh or rw > sw or min(rh, rw) < 4:
            continue
        response = cv2.matchTemplate(search_feat, rot_feat, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(response)

        # 同角度的第二空间峰值用于排除“附近有多个几乎相同元件”的歧义匹配。
        second = -1.0
        if response.size > 1:
            response2 = response.copy()
            radius_x = max(2, rw // 3)
            radius_y = max(2, rh // 3)
            x0 = max(0, loc[0] - radius_x)
            y0 = max(0, loc[1] - radius_y)
            x1 = min(response2.shape[1], loc[0] + radius_x + 1)
            y1 = min(response2.shape[0], loc[1] + radius_y + 1)
            response2[y0:y1, x0:x1] = -1.0
            second = float(response2.max())
        candidates.append({
            "angle": float(angle) % 360.0,
            "score": float(score),
            "second_score": second,
            "loc": (float(loc[0]), float(loc[1])),
            "rot_m": rot_m,
        })
    return candidates


def _empty_roi_match(reason="no_match"):
    return {
        "matched": False, "confidence": 0.0, "score": 0.0,
        "normal_score": 0.0, "reverse_score": 0.0,
        "orientation_confident": False, "reversed_180": False,
        "branch_180": False, "angle_deg": 0.0, "residual_angle_deg": 0.0,
        "dx": 0.0, "dy": 0.0, "polygon": [], "template_to_test": None,
        "crop_to_test": None, "reason": reason,
    }


def _locate_full_image(tpl_gray_full, test_image, coarse_score_min=_ROI_MATCH_COARSE_MIN):
    """整图降采样粗搜，返回原图分辨率下的近似匹配中心 ``(cx, cy)``。

    只用于给后续局部精搜提供候选窗口中心；找不到可信候选（分数太低/无候选）时
    返回 ``None``，调用方需退回原 ROI 位置附近搜索，不在整图范围内继续猜测。
    """
    ih, iw = test_image.shape[:2]
    if ih < 4 or iw < 4:
        return None
    th, tw = tpl_gray_full.shape[:2]
    long_side = max(ih, iw)
    scale = min(1.0, _ROI_MATCH_FULL_MAXSIDE / float(long_side))
    min_tpl_side = min(th, tw) * scale
    if min(th, tw) > 0 and min_tpl_side < _ROI_MATCH_MIN_TPL_SIDE:
        scale = min(1.0, _ROI_MATCH_MIN_TPL_SIDE / float(min(th, tw)))

    search_gray = to_gray(test_image)
    tpl_gray = tpl_gray_full
    if scale < 1.0:
        search_gray = cv2.resize(search_gray, (max(4, int(round(iw * scale))),
                                                    max(4, int(round(ih * scale)))))
        tpl_gray = cv2.resize(tpl_gray_full, (max(4, int(round(tw * scale))),
                                                   max(4, int(round(th * scale)))))

    # 粗搜只覆盖普通装歪（±30°）与极反（180°±30°），避免无意义的整圈扫描。
    coarse_angles = list(range(-30, 31, 10)) + list(range(150, 211, 10))
    candidates = _scan_angle_match(tpl_gray, search_gray, coarse_angles)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["score"])
    if best["score"] < coarse_score_min:
        return None

    rotated, _ = _rotate_bound(tpl_gray, best["angle"])
    rh, rw = rotated.shape[:2]
    loc_x, loc_y = best["loc"]
    center_x = (loc_x + rw / 2.0) / scale
    center_y = (loc_y + rh / 2.0) / scale
    return float(center_x), float(center_y)


def _transform_from_lowres(candidate, scale, sx, sy):
    """把低分辨率候选的 (rot_m, loc) 换算成待检整图坐标下的 2×3 变换矩阵。"""
    rot_m = np.asarray(candidate["rot_m"], np.float64).copy()
    loc_x, loc_y = candidate["loc"]
    rot_m[0, 2] = (rot_m[0, 2] + loc_x) / scale + sx
    rot_m[1, 2] = (rot_m[1, 2] + loc_y) / scale + sy
    return rot_m


def _verify_candidate(tpl_gray_full, search_full_gray, angle_deg, coarse_loc, coarse_scale,
                       maxside=_ROI_VERIFY_MAXSIDE, angle_refine_deg=3.0, angle_refine_step=0.5):
    """对锁定的单个候选做一次更高保真度的核验，替代低分辨率打分。

    粗/细搜为了速度把模板压到很小的缩略图（``_ROI_MATCH_MAXSIDE``），对大 ROI、
    低对比度丝印会丢失细节，导致分数被系统性低估。仅提高核验分辨率并不够：
    粗/细搜的 2° 角度步长对一张小缩略图足够精细，但对占比很大的模板，1° 的角度
    误差在原生分辨率下会造成边缘方向的十几像素错位，比分辨率本身对分数的影响更大。
    因此这里不只是换分辨率重算一次分数，而是在锁定的候选角度附近以更细的步长
    （默认 ±3°、0.5° 步）小范围重新寻优，兼顾保真度与速度：不重新扫描全部角度/
    全部位置，只对这一个候选局部精修，且核验窗口本身也裁得很小（受 ``maxside``
    限制），因此额外开销与 ROI 尺寸基本无关。

    返回 ``(score, template_to_test_2x3, refined_angle_deg)``；核验裁窗越界等
    极端情况下返回 ``None``，调用方需退回低分辨率结果。
    """
    th, tw = tpl_gray_full.shape[:2]
    rad = np.radians(float(angle_deg))
    cos_v, sin_v = abs(np.cos(rad)), abs(np.sin(rad))
    lrw, lrh = tw * coarse_scale, th * coarse_scale
    approx_crw = lrw * cos_v + lrh * sin_v
    approx_crh = lrh * cos_v + lrw * sin_v

    loc_x, loc_y = coarse_loc
    approx_x0 = loc_x / coarse_scale
    approx_y0 = loc_y / coarse_scale
    approx_w = approx_crw / coarse_scale
    approx_h = approx_crh / coarse_scale
    margin_x = approx_w * 0.25 + 12
    margin_y = approx_h * 0.25 + 12
    cx, cy, cw, ch = clamp_box(
        (approx_x0 - margin_x, approx_y0 - margin_y, approx_w + 2 * margin_x, approx_h + 2 * margin_y),
        search_full_gray.shape)
    verify_crop = search_full_gray[cy:cy + ch, cx:cx + cw]
    if verify_crop.size == 0:
        return None

    v_scale = min(1.0, float(maxside) / float(max(th, tw)))
    tpl_v_base = tpl_gray_full
    verify_v = verify_crop
    if v_scale < 1.0:
        tpl_v_base = cv2.resize(tpl_gray_full, (max(4, int(round(tw * v_scale))),
                                                     max(4, int(round(th * v_scale)))))
        verify_v = cv2.resize(verify_crop, (max(4, int(round(cw * v_scale))),
                                                 max(4, int(round(ch * v_scale)))))
    verify_feat = _match_feature(verify_v)

    n_steps = max(1, int(round(angle_refine_deg / angle_refine_step)))
    best = None
    for i in range(-n_steps, n_steps + 1):
        cand_angle = float(angle_deg) + i * angle_refine_step
        rotated_v, rot_m_v = _rotate_bound(tpl_v_base, cand_angle)
        rh, rw = rotated_v.shape[:2]
        if rh > verify_feat.shape[0] or rw > verify_feat.shape[1] or min(rh, rw) < 4:
            continue
        rot_feat = _match_feature(rotated_v)
        response = cv2.matchTemplate(verify_feat, rot_feat, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc_v = cv2.minMaxLoc(response)
        if best is None or score > best[0]:
            best = (score, loc_v, rot_m_v, cand_angle)
    if best is None:
        return None
    score, loc_v, rot_m_v, refined_angle = best

    template_to_test = np.asarray(rot_m_v, np.float64).copy()
    template_to_test[0, 2] = (template_to_test[0, 2] + loc_v[0]) / v_scale + cx
    template_to_test[1, 2] = (template_to_test[1, 2] + loc_v[1]) / v_scale + cy
    return float(score), template_to_test, float(refined_angle)


def _best_branch_match(candidates, tpl_gray_full, search_full_gray, scale, sx, sy):
    """取某方向分支里低分辨率打分最高的候选，做原生核验并换算到整图坐标。

    始终返回 ``(score, template_to_test, angle)``；核验失败时退回低分辨率坐标换算
    结果，保证下游一定能拿到一个可用的变换矩阵。该分支在搜索窗内完全放不下旋转后
    模板（``candidates`` 为空，常见于大 ROI 局部搜索窗裕量不足）时返回 ``-1.0``
    分数与 ``None`` 变换，调用方需按“该分支不可用”处理。
    """
    if not candidates:
        return -1.0, None, 0.0
    best_c = max(candidates, key=lambda c: c["score"])
    verified = _verify_candidate(tpl_gray_full, search_full_gray, best_c["angle"],
                                  best_c["loc"], scale)
    if verified is not None:
        score, transform, refined_angle = verified
        transform = transform.copy()
        transform[0, 2] += sx
        transform[1, 2] += sy
        return score, transform, refined_angle
    return float(best_c["score"]), _transform_from_lowres(best_c, scale, sx, sy), best_c["angle"]


def _match_in_search_box(tpl_gray_full, test_image, roi_box, search_box,
                          maxside=_ROI_MATCH_MAXSIDE):
    """在给定 ``search_box``（待检整图坐标，已 clamp）内做粗到细双分支匹配并校验。

    定位阶段仍用低分辨率缩略图做全角度粗扫+精修以保证速度；但最终是否“匹配成功”
    的打分改为对 0°/180° 两个分支各取一个最佳候选，在更高保真度下单独核验一次
    （见 :func:`_verify_candidate`），避免大 ROI/低对比度丝印被固定的粗搜分辨率
    拖累打出偏低的分数。返回结构与 ``match_roi_structure`` 相同的可序列化 dict。
    """
    x, y, w, h = (float(v) for v in roi_box)
    sx, sy, sw, sh = search_box
    search_full = test_image[sy:sy + sh, sx:sx + sw]
    if search_full.size == 0:
        return _empty_roi_match("empty_search")
    search_full_gray = to_gray(search_full)

    # 模板和搜索窗使用同一缩放比例，仿射线性项无需换算，平移项除以 scale 即可。
    th, tw = tpl_gray_full.shape[:2]
    scale = min(1.0, maxside / float(max(th, tw)))
    tpl_gray = tpl_gray_full
    search_gray = search_full_gray
    if scale < 1.0:
        tpl_gray = cv2.resize(tpl_gray, (max(4, int(round(tw * scale))),
                                              max(4, int(round(th * scale)))))
        search_gray = cv2.resize(search_gray, (max(4, int(round(sw * scale))),
                                                    max(4, int(round(sh * scale)))))

    coarse_angles = list(range(-30, 31, 10)) + list(range(150, 211, 10))
    candidates = _scan_angle_match(tpl_gray, search_gray, coarse_angles)
    if not candidates:
        return _empty_roi_match("no_candidate")
    coarse_best = max(candidates, key=lambda c: c["score"])

    # 最佳粗角附近 2°步进精修；同时保留粗搜全部候选用于0/180分数比较。
    center_angle = coarse_best["angle"]
    fine_angles = [(center_angle + d) % 360.0 for d in range(-10, 11, 2)]
    fine = _scan_angle_match(tpl_gray, search_gray, fine_angles)
    all_candidates = candidates + fine
    low_res_best = max(all_candidates, key=lambda c: c["score"])
    uniqueness = float(low_res_best["score"]) - float(low_res_best.get("second_score", -1.0))

    normal_candidates = [c for c in all_candidates if not _angle_branch(c["angle"])[0]]
    reverse_candidates = [c for c in all_candidates if _angle_branch(c["angle"])[0]]

    # 只对每个分支里低分辨率打分最高的那一个候选做原生核验（至多2次），
    # 不重新扫描全部角度/位置，核验开销与 ROI 尺寸/角度数量无关，恒定较小。
    normal_score, normal_transform, normal_angle = _best_branch_match(
        normal_candidates, tpl_gray_full, search_full_gray, scale, sx, sy)
    reverse_score, reverse_transform, reverse_angle = _best_branch_match(
        reverse_candidates, tpl_gray_full, search_full_gray, scale, sx, sy)

    if reverse_score > normal_score and reverse_transform is not None:
        branch_180, score = True, reverse_score
        template_to_test, winning_angle = reverse_transform, reverse_angle
    elif normal_transform is not None:
        branch_180, score = False, normal_score
        template_to_test, winning_angle = normal_transform, normal_angle
    else:
        # 两个分支在当前搜索窗裕量下都放不下旋转后的模板（极少见，通常是局部回退
        # 搜索窗过小），直接判为不可信匹配，交由上层回退到原始 ROI 位置逻辑。
        return _empty_roi_match("no_candidate")
    _, residual = _angle_branch(winning_angle)
    orientation_confident = (
        branch_180 and reverse_score > 0.45 and reverse_score > normal_score + 0.06)

    matched = score >= 0.42 and (uniqueness >= 0.015 or score >= 0.72)
    if not matched:
        empty = _empty_roi_match("low_score" if score < 0.42 else "ambiguous_location")
        empty.update({"score": score, "normal_score": max(normal_score, -1.0), "reverse_score": max(reverse_score, -1.0)})
        return empty

    corners = np.array(
        [[[0.0, 0.0]], [[float(tw), 0.0]], [[float(tw), float(th)]],
         [[0.0, float(th)]]], np.float32)
    polygon = cv2.transform(corners, template_to_test.astype(np.float32)).reshape(-1, 2)
    center = polygon.mean(axis=0)
    nominal_center = np.array([x + w / 2.0, y + h / 2.0], np.float32)
    dx, dy = (center - nominal_center).tolist()

    # 算法裁图只消除普通旋转余角；若最佳分支在180°，保留180°内容方向。
    crop_to_test = template_to_test.copy()
    if branch_180:
        r180 = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), 180.0, 1.0)
        crop_to_test = _affine_2x3(
            _affine_h(template_to_test) @ _affine_h(r180))

    confidence = max(0.0, min(1.0, 0.75 * score + 0.25 * min(1.0, uniqueness / 0.08)))
    return {
        "matched": True,
        "confidence": float(confidence),
        "score": score,
        "normal_score": float(normal_score),
        "reverse_score": float(reverse_score),
        "orientation_confident": bool(orientation_confident),
        "reversed_180": bool(orientation_confident),
        "branch_180": bool(branch_180),
        "angle_deg": float(winning_angle),
        "residual_angle_deg": float(residual),
        "dx": float(dx), "dy": float(dy),
        "polygon": [[float(px), float(py)] for px, py in polygon],
        "template_to_test": template_to_test.tolist(),
        "crop_to_test": crop_to_test.tolist(),
        "search_box": [int(sx), int(sy), int(sw), int(sh)],
        "reason": "matched",
    }


def match_roi_structure(template_roi, test_image, roi_box, search_expand=1.0, full_search=True):
    """定位模板 ROI 在待检整图中的位置与方向（平移 + 小角度 + 180°）。

    优先在**整张待检图**范围内粗搜定位（不局限于原 ROI 坐标附近），命中候选后在
    原分辨率局部窗口内精搜校验；整图搜索找不到可信候选、或候选精搜校验不通过时，
    退回到原 ROI 坐标附近的局部搜索（等价于旧版"原位匹配"）；局部搜索也失败则返回
    ``matched=False``，调用方必须回退到原同坐标 ROI，不做任何猜测。

    返回值是可序列化 dict。矩阵 ``template_to_test`` 把模板 ROI 的局部坐标
    映射到待检整图；``crop_to_test`` 把给算法使用的摆正裁图坐标映射回待检整图。
    """
    if template_roi is None or test_image is None or template_roi.size == 0 or test_image.size == 0:
        return _empty_roi_match("empty_input")

    x, y, w, h = (float(v) for v in roi_box)
    if w < 4 or h < 4:
        return _empty_roi_match("roi_too_small")

    tpl_gray_full = to_gray(template_roi)
    if not _has_texture(tpl_gray_full):
        return _empty_roi_match("low_texture")

    ex, ey = w * float(search_expand), h * float(search_expand)
    best_fail = None

    if full_search:
        loc = _locate_full_image(tpl_gray_full, test_image)
        if loc is not None:
            cx, cy = loc
            box = clamp_box(
                (cx - w / 2.0 - ex, cy - h / 2.0 - ey, w + 2 * ex, h + 2 * ey),
                test_image.shape)
            result = _match_in_search_box(tpl_gray_full, test_image, roi_box, box)
            if result.get("matched"):
                result["reason"] = "matched_full_image"
                return result
            best_fail = result

    # 退回原 ROI 坐标附近的局部搜索（旧版"原位匹配"行为）。
    local_box = clamp_box((x - ex, y - ey, w + 2 * ex, h + 2 * ey), test_image.shape)
    result = _match_in_search_box(tpl_gray_full, test_image, roi_box, local_box)
    if result.get("matched"):
        result["reason"] = "matched_local"
        return result
    if best_fail is None or result.get("score", 0.0) >= best_fail.get("score", 0.0):
        best_fail = result

    empty = _empty_roi_match(best_fail.get("reason", "no_match") if best_fail else "no_match")
    if best_fail is not None:
        empty.update({
            "score": best_fail.get("score", 0.0),
            "normal_score": best_fail.get("normal_score", 0.0),
            "reverse_score": best_fail.get("reverse_score", 0.0),
        })
    return empty


def extract_matched_roi(test_image, match, output_size):
    """按匹配变换提取待检 ROI；180°分支保持元件内容为180°，不被转正。"""
    if not match or not match.get("matched") or match.get("crop_to_test") is None:
        return None
    out_w, out_h = (max(2, int(round(v))) for v in output_size)
    crop_to_test = np.asarray(match["crop_to_test"], np.float64)
    test_to_crop = cv2.invertAffineTransform(crop_to_test)
    return cv2.warpAffine(
        test_image, test_to_crop, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)

# ── 本体掩膜/轮廓/位姿 ──
@_memo1
def body_mask(img, sat_thresh=40, val_thresh=60):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img.ndim == 3 else None
    gray = to_gray(img)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    if hsv is not None:
        sat = hsv[:,:,1] > sat_thresh; val = hsv[:,:,2] > val_thresh
        color_fg = (sat & val).astype(np.uint8) * 255
    else:
        color_fg = (gray > val_thresh).astype(np.uint8) * 255
    mask = cv2.bitwise_or(color_fg, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def min_area_pose(mask):
    cnt = largest_contour(mask)
    return cv2.minAreaRect(cnt) if cnt is not None else None

def centroid_of_mask(mask):
    cnt = largest_contour(mask)
    if cnt is None: return None
    m = cv2.moments(cnt)
    return (float(m["m10"]/m["m00"]), float(m["m01"]/m["m00"])) if abs(m["m00"]) >= 1e-6 else None

def body_centroid_shift(golden, test):
    gc = centroid_of_mask(body_mask(golden)); tc = centroid_of_mask(body_mask(test))
    return (tc[0]-gc[0], tc[1]-gc[1]) if gc and tc else (0.0, 0.0)

def translate_image(img, dx, dy):
    h, w = img.shape[:2]; m = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img, m, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _pca_points(xs, ys, max_pts=2000):
    """点集做 PCA 主方向角；点数过多时等间隔子采样以提速（不影响主方向）。"""
    n = len(xs)
    if n > max_pts:
        step = n // max_pts + 1
        xs = xs[::step]; ys = ys[::step]
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    _, ev = cv2.PCACompute(pts, mean=None); vx, vy = ev[0]
    return float(np.degrees(np.arctan2(vy, vx)))

@_memo1
def _pca_body_angle(img):
    mask = body_mask(img); ys, xs = np.where(mask > 0)
    if len(xs) < 20: return None
    return _pca_points(xs, ys)

@_memo1
def _pca_edge_angle(img):
    gray = to_gray(img); edges = cv2.Canny(gray, 40, 120)
    ys, xs = np.where(edges > 0)
    if len(xs) < 50: return None
    return _pca_points(xs, ys)

def _signed_angle_diff(ang_t, ang_g):
    raw = ang_t - ang_g % 180.0
    while raw > 90: raw -= 180
    while raw < -90: raw += 180
    return float(raw)

_SHIFT_CORR_MAXSIDE = 120  # matchTemplate 对齐内部工作分辨率上限

def roi_shift_offset(golden, test, margin=40):
    """估计 test 相对 golden 的平移；matchTemplate 在降采样图上做以提速，偏移还原到原分辨率。"""
    m = max(margin, 8)
    g = to_gray(golden); t = to_gray(test)
    gh, gw = g.shape[:2]
    long_side = max(gh, gw)
    f = 1.0
    if long_side > _SHIFT_CORR_MAXSIDE:
        f = _SHIFT_CORR_MAXSIDE / float(long_side)
        g = cv2.resize(g, (max(1, int(gw * f)), max(1, int(gh * f))))
        t = cv2.resize(t, (max(1, int(t.shape[1] * f)), max(1, int(t.shape[0] * f))))
    ms = max(2, int(round(m * f)))
    padded = cv2.copyMakeBorder(t, ms, ms, ms, ms, cv2.BORDER_REPLICATE)
    if padded.shape[0] < g.shape[0] or padded.shape[1] < g.shape[1]:
        return 0.0, 0.0
    res = cv2.matchTemplate(padded, g, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    return float((loc[0] - ms) / f), float((loc[1] - ms) / f)

def estimate_roi_shift(golden, test):
    dx_c, dy_c = body_centroid_shift(golden, test)
    dx_n, dy_n = roi_shift_offset(golden, test)
    return (dx_c, dy_c) if abs(dx_c)+abs(dy_c)>=abs(dx_n)+abs(dy_n) else (dx_n, dy_n)

def _pose_from_ecc_warp(warp):
    theta = float(np.degrees(np.arctan2(warp[1,0], warp[0,0])))
    return float(warp[0,2]), float(warp[1,2]), theta

def estimate_roi_pose(golden, test, iterations=30, eps=1e-4):
    dx, dy = estimate_roi_shift(golden, test)
    ang_g = _pca_edge_angle(golden); ang_t = _pca_edge_angle(test)
    if ang_g is not None and ang_t is not None: return dx, dy, _signed_angle_diff(ang_t, ang_g)
    ang_g = _pca_body_angle(golden); ang_t = _pca_body_angle(test)
    if ang_g is not None and ang_t is not None: return dx, dy, _signed_angle_diff(ang_t, ang_g)
    g = to_gray(golden).astype(np.float32)
    test_aligned = translate_image(test, -dx, -dy)
    t = to_gray(test_aligned).astype(np.float32)
    if g.shape != t.shape: t = cv2.resize(t, (g.shape[1], g.shape[0]))
    warp = np.eye(2,3,dtype=np.float32)
    try:
        cv2.findTransformECC(g, t, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,int(iterations),float(eps)), None, 1)
        _, _, theta = _pose_from_ecc_warp(warp); return dx, dy, theta
    except cv2.error: return dx, dy, 0.0

def align_by_pose(test, golden):
    # 共享 ROI 匹配可能已经把待检块摆正；高相关时直接复用，避免再次估姿把
    # 已对齐图像反向拉偏。这里只优化对齐预处理，不改变任何缺陷判定阈值。
    if test.shape[:2] == golden.shape[:2] and ncc_score(test, golden) >= 0.92:
        return test
    dx, dy, theta = estimate_roi_pose(golden, test)
    h, w = test.shape[:2]; cx, cy = w/2.0, h/2.0
    m_rot = cv2.getRotationMatrix2D((cx,cy), -theta, 1.0)
    aligned = cv2.warpAffine(test, m_rot, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return translate_image(aligned, -dx, -dy)

def align_by_centroid(test, golden):
    dx, dy = body_centroid_shift(golden, test)
    return translate_image(test, -dx, -dy)

# ── ROI 对比指标 ──
def roi_compare_metrics(golden_roi, test_roi, diff_thresh=40):
    g_gray = to_gray(golden_roi); t_gray = to_gray(test_roi)
    if g_gray.shape != t_gray.shape: t_gray = cv2.resize(t_gray, (g_gray.shape[1], g_gray.shape[0]))
    diff = cv2.absdiff(g_gray, t_gray)
    _, diff_bin = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    diff_score = cv2.countNonZero(diff_bin) / max(1, diff_bin.size)
    gm = body_mask(golden_roi); tm = body_mask(test_roi)
    tmpl_body = max(1, cv2.countNonZero(gm))
    body_ratio = cv2.countNonZero(tm) / tmpl_body
    h, w = g_gray.shape
    if w >= h:
        q = w // 4; slices = [(0,q),(q,w//2),(w//2,w-q),(w-q,w)]; axis = 1
    else:
        q = h // 4; slices = [(0,q),(q,h//2),(h//2,h-q),(h-q,h)]; axis = 0
    diffs, brs = [], []
    for s0, s1 in slices:
        if axis == 1:
            dg, gm_s, tm_s = diff[:,s0:s1], gm[:,s0:s1], tm[:,s0:s1]
        else:
            dg, gm_s, tm_s = diff[s0:s1,:], gm[s0:s1,:], tm[s0:s1,:]
        _, b = cv2.threshold(dg, diff_thresh, 255, cv2.THRESH_BINARY)
        diffs.append(cv2.countNonZero(b)/max(1,b.size))
        brs.append(cv2.countNonZero(tm_s)/max(1,cv2.countNonZero(gm_s)))
    end_hi = max(diffs[0], diffs[-1])
    end_diff_asym = min(diffs[0],diffs[-1])/end_hi if end_hi>1e-6 else 1.0
    cg = largest_contour(gm); ct = largest_contour(tm)
    shape_dist = float(cv2.matchShapes(cg,ct,cv2.CONTOURS_MATCH_I1,0.0)) if cg is not None and ct is not None else 0.0
    t_pose = min_area_pose(tm); g_pose = min_area_pose(gm)
    def _aspect(pose):
        if pose is None: return 0.0
        pw, ph = pose[1]; lo, hi = min(pw,ph), max(pw,ph)
        return hi/lo if lo>1e-6 else 0.0
    g_ar, t_ar = _aspect(g_pose), _aspect(t_pose)
    aspect_dev = abs(t_ar-g_ar)/g_ar if g_ar>1e-6 else 0.0
    return {
        "body_ratio": body_ratio, "diff_score": diff_score,
        "min_quarter_body_ratio": min(brs) if brs else 1.0,
        "br_std": float(np.std(brs)) if brs else 0.0,
        "end_diff_asym": end_diff_asym, "shape_dist": shape_dist,
        "ncc": ncc_score(test_roi, golden_roi), "aspect_dev": aspect_dev,
        "quarter_body_ratios": [round(x,4) for x in brs],
        "quarter_diffs": [round(x,4) for x in diffs],
    }

def roi_structural_defect_hint(metrics):
    """推断结构性缺陷(缺件/立碑) — 移植自 common.py"""
    m = {"roi_body_ratio_max":0.85, "roi_diff_min":0.12, "roi_min_quarter_body_max":0.55, "roi_body_ratio_soft_max":0.95}
    t = {"roi_body_ratio_min":0.85, "roi_diff_min":0.28, "roi_ncc_max":0.12, "roi_shape_min":0.20, "roi_shape_max":1.0}
    if (metrics["body_ratio"]<m["roi_body_ratio_max"] and metrics["diff_score"]>m["roi_diff_min"]) or \
       (metrics["min_quarter_body_ratio"]<m["roi_min_quarter_body_max"] and metrics["body_ratio"]<m["roi_body_ratio_soft_max"]):
        return "缺件"
    if metrics["body_ratio"]>=t["roi_body_ratio_min"] and metrics["diff_score"]>=t["roi_diff_min"] and \
       metrics["ncc"]<=t["roi_ncc_max"] and t["roi_shape_min"]<=metrics["shape_dist"]<=t["roi_shape_max"]:
        return "立碑"
    return None

# ── 特征度量 ──
def ncc_score(roi, tmpl):
    if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
        tmpl = cv2.resize(tmpl, (roi.shape[1], roi.shape[0]))
    rg = to_gray(roi); tg = to_gray(tmpl)
    if rg.shape == tg.shape:
        # 同尺寸全覆盖：直接算归一化互相关标量，避免 matchTemplate 滑窗开销
        a = rg.astype(np.float32); b = tg.astype(np.float32)
        a -= a.mean(); b -= b.mean()
        denom = float(np.sqrt(float((a * a).sum()) * float((b * b).sum())))
        return float((a * b).sum() / denom) if denom > 1e-9 else 0.0
    res = cv2.matchTemplate(rg, tg, cv2.TM_CCOEFF_NORMED)
    return float(res.max())

@_memo1
def hsv_hist(img, bins=(32,32)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, list(bins), [0,180,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX); return hist

def hist_correlation(img_a, img_b):
    return float(cv2.compareHist(hsv_hist(img_a), hsv_hist(img_b), cv2.HISTCMP_CORREL))

_SSIM_MAXSIDE = 160  # SSIM 内部工作分辨率上限；差异图上采样回原尺寸，保证 blob 面积阈值仍按原分辨率
USE_CV2_SSIM = True  # True=OpenCV 高斯 SSIM（快），False=skimage（若可用）


def _ssim_cv2(a, b):
    """OpenCV 高斯加权 SSIM（Wang 2004），比 skimage 快数倍。

    返回 (mean_ssim, ssim_map)，ssim_map 与输入同尺寸、值约在 [-1,1]（结构相似处≈1）。
    C1/C2 取 8bit 动态范围 L=255 的标准常数。
    """
    a = a.astype(np.float32); b = b.astype(np.float32)
    win, sigma = (11, 11), 1.5
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, win, sigma)
    mu_b = cv2.GaussianBlur(b, win, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa2 = cv2.GaussianBlur(a * a, win, sigma) - mu_a2
    sb2 = cv2.GaussianBlur(b * b, win, sigma) - mu_b2
    sab = cv2.GaussianBlur(a * b, win, sigma) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sa2 + sb2 + C2))
    return float(ssim_map.mean()), ssim_map


def ssim_diff(gray_a, gray_b):
    if gray_a.shape != gray_b.shape: gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    H, W = gray_a.shape[:2]
    long_side = max(H, W)
    if long_side > _SSIM_MAXSIDE:
        f = _SSIM_MAXSIDE / float(long_side)
        a = cv2.resize(gray_a, (max(1, int(W * f)), max(1, int(H * f))))
        b = cv2.resize(gray_b, (a.shape[1], a.shape[0]))
    else:
        a, b = gray_a, gray_b
    if USE_CV2_SSIM:
        score, smap = _ssim_cv2(a, b)
        dmap = (1.0 - smap).astype(np.float32)
    elif _HAS_SKIMAGE:
        score, smap = _ssim(a, b, full=True)
        dmap = (1.0 - smap).astype(np.float32)
    else:
        dmap = cv2.absdiff(a, b).astype(np.float32) / 255.0
        score = float(1.0 - dmap.mean())
    if dmap.shape[:2] != (H, W):
        dmap = cv2.resize(dmap, (W, H))
    return float(score), dmap

# ── 元件定位 / 标注框 ──
def smt_dark_body_bbox(img, margin=3, max_area_ratio=0.88, min_area=150):
    """贴片暗色本体（黑电阻/黑 IC）外接框；失败返回 None。"""
    h, w = img.shape[:2]
    gray = to_gray(img)
    if img.ndim == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 暗色低饱和区域 ≈ 黑色元件本体，排除绿色焊盘背景
        dark = ((hsv[:, :, 2] < 95) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    else:
        dark = (gray < 95).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnt = largest_contour(dark)
    if cnt is None or cv2.contourArea(cnt) < min_area:
        return None
    if cv2.contourArea(cnt) >= w * h * max_area_ratio:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return clamp_box((x - margin, y - margin, bw + 2 * margin, bh + 2 * margin), img.shape)


def body_bbox_xywh(img, margin=3):
    """从本体掩膜提取外接矩形 (x,y,w,h)；失败返回整图。"""
    box = smt_dark_body_bbox(img, margin=margin)
    if box is not None:
        return box
    h, w = img.shape[:2]
    mask = body_mask(img)
    cnt = largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 50:
        return 0, 0, w, h
    area_ratio = cv2.contourArea(cnt) / max(1, w * h)
    if area_ratio >= 0.92:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(cnt)
    return clamp_box((x - margin, y - margin, bw + 2 * margin, bh + 2 * margin), img.shape)


@_memo2
def component_body_bbox(golden, test, margin=3):
    """优先用来料图本体框（移位后真实位置），其次金板，最后整图。"""
    h, w = test.shape[:2]
    full_area = float(max(1, w * h))
    for img in (test, golden):
        box = smt_dark_body_bbox(img, margin=margin)
        if box is not None:
            x, y, bw, bh = box
            if bw * bh < full_area * 0.98:
                return float(x), float(y), float(bw), float(bh)
        # 彩色本体（钽电容等）
        if img.ndim == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            color_fg = (sat > 35).astype(np.uint8) * 255
            color_fg = cv2.morphologyEx(color_fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnt = largest_contour(color_fg)
            if cnt is not None:
                area = cv2.contourArea(cnt)
                if 200 <= area < full_area * 0.92:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    bx, by, bw, bh = clamp_box((x - margin, y - margin, bw + 2 * margin, bh + 2 * margin), img.shape)
                    return float(bx), float(by), float(bw), float(bh)
        x, y, bw, bh = body_bbox_xywh(img, margin=margin)
        if bw * bh >= full_area * 0.98:
            continue
        if bw * bh >= 100:
            return float(x), float(y), float(bw), float(bh)
    # 退化为两图本体并集
    xs, ys, xe, ye = w, h, 0, 0
    found = False
    for img in (test, golden):
        x, y, bw, bh = body_bbox_xywh(img, margin=0)
        if bw * bh < 100:
            continue
        found = True
        xs = min(xs, x); ys = min(ys, y)
        xe = max(xe, x + bw); ye = max(ye, y + bh)
    if found:
        bx, by, bw, bh = clamp_box((xs - margin, ys - margin, xe - xs + 2 * margin, ye - ys + 2 * margin), test.shape)
        return float(bx), float(by), float(bw), float(bh)
    return 0.0, 0.0, float(w), float(h)


def find_component_boxes(golden, min_area=400):
    """从金板分离元件区域；优先暗色贴片本体，避免 FOV 整图被当成一块。"""
    h, w = golden.shape[:2]
    full_area = w * h
    boxes = []
    if golden.ndim == 3:
        hsv = cv2.cvtColor(golden, cv2.COLOR_BGR2HSV)
        dark = ((hsv[:, :, 2] < 95) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    else:
        dark = (to_gray(golden) < 95).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area >= full_area * 0.92:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append(clamp_box((x, y, bw, bh), golden.shape))
    if not boxes:
        one = smt_dark_body_bbox(golden, margin=0)
        if one is not None:
            boxes = [one]
    if not boxes:
        mask = body_mask(golden)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area >= full_area * 0.92:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            boxes.append(clamp_box((x, y, bw, bh), golden.shape))
    if not boxes:
        boxes = [(0, 0, w, h)]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _mark_left_rects(gh, gw):
    """本体内「中心字符区(mark)」与「左缘区(left)」的相对矩形 (x,y,w,h)。

    供 polarity_mark_crops 与 polarity_silkscreen_compare 共用，保证两处区域定义一致。
    """
    ly = max(0, int(gh * 0.05))
    lw = max(4, int(gw * 0.58))
    lh = max(4, gh - ly)
    left = (0, ly, lw, lh)
    mx = max(0, int(gw * 0.12))
    my = max(0, int(gh * 0.10))
    mw = min(max(4, int(gw * 0.76)), gw - mx)
    mh = min(max(4, int(gh * 0.80)), gh - my)
    mark = (mx, my, mw, mh)
    return mark, left


@_memo2
def polarity_mark_crops(test, gold):
    """提取本体及丝印/标记区（覆盖左缘色带+字符，不只中心条带）。"""
    bx, by, bw, bh = component_body_bbox(gold, test)
    gh0, gw0 = gold.shape[:2]
    if bw * bh < 0.12 * gw0 * gh0:
        bx, by, bw, bh = 0, 0, gw0, gh0
    test_body = crop(test, (bx, by, bw, bh))
    gold_body = crop(gold, (bx, by, bw, bh))
    gh, gw = gold_body.shape[:2]
    (mx, my, mw, mh), (lx, ly, lw, lh) = _mark_left_rects(gh, gw)
    g_left = gold_body[ly:ly + lh, lx:lx + lw]
    t_left = test_body[ly:ly + lh, lx:lx + lw]
    g_mark = gold_body[my:my + mh, mx:mx + mw]
    t_mark = test_body[my:my + mh, mx:mx + mw]
    return test_body, gold_body, t_mark, g_mark, t_left, g_left, (bx, by, bw, bh)


def _ncc_flip_diff(test_patch, gold_patch):
    """0° 与 180° NCC 差值；patch 过小则返回 None。"""
    if test_patch is None or gold_patch is None:
        return None
    if test_patch.size < 80 or gold_patch.size < 80:
        return None
    front = ncc_score(test_patch, gold_patch)
    flip = ncc_score(test_patch, cv2.rotate(gold_patch, cv2.ROTATE_180))
    return {"front": front, "flip": flip, "diff": flip - front}


def polarity_silkscreen_compare(test, gold, ncc_diff_min=0.06):
    """判据1：丝印/字符图案 0° vs 180° 比对（不 OCR，只比图案）。

    提速：只做 1 次本体位姿对齐，mark/left 子区域直接从对齐后的本体切片复用，
    替代原先「本体/mark/left 各对齐一次」的 3 次对齐。
    """
    test_body, gold_body, t_mark, g_mark, t_left, g_left, _ = polarity_mark_crops(test, gold)
    # 单次本体对齐，子区域复用对齐结果（避免 FOV 轻微偏移影响 NCC）
    try:
        aligned_body = align_by_pose(test_body, gold_body)
    except Exception:
        aligned_body = test_body
    if aligned_body.shape[:2] == gold_body.shape[:2]:
        gh, gw = gold_body.shape[:2]
        (mx, my, mw, mh), (lx, ly, lw, lh) = _mark_left_rects(gh, gw)
        t_mark = aligned_body[my:my + mh, mx:mx + mw]
        t_left = aligned_body[ly:ly + lh, lx:lx + lw]
        test_body = aligned_body
    diffs = []
    fronts, flips = [], []
    for t_patch, g_patch in ((t_mark, g_mark), (t_left, g_left), (test_body, gold_body)):
        r = _ncc_flip_diff(t_patch, g_patch)
        if r is not None:
            diffs.append(r["diff"])
            fronts.append(r["front"])
            flips.append(r["flip"])
    g_edge = cv2.Canny(to_gray(gold_body), 40, 120)
    t_edge = cv2.Canny(to_gray(test_body), 40, 120)
    edge_r = _ncc_flip_diff(t_edge, g_edge)
    if edge_r is not None:
        diffs.append(edge_r["diff"] * 1.1)
        fronts.append(edge_r["front"])
        flips.append(edge_r["flip"])
    silk_diff = max(diffs) if diffs else 0.0
    best_front = max(fronts) if fronts else 0.0
    best_flip = max(flips) if flips else 0.0
    applicable = len(diffs) > 0 and max(max(abs(d) for d in diffs), abs(best_flip - best_front)) > 0.02
    # 必须要求 180° 匹配本身足够好：若 front/flip 都接近 0（光照/模糊导致整体不像），
    # 仅靠相对差值 silk_diff 会把噪声误判为极反。
    flip_ok = best_flip > 0.35
    reversed_by_silkscreen = (
        (silk_diff > ncc_diff_min and flip_ok)
        or (best_flip > best_front + max(0.04, ncc_diff_min * 0.45) and flip_ok)
        or (best_front < 0.55 and best_flip > 0.55 and (best_flip - best_front) > 0.03)
    )
    return {
        "reversed_by_silkscreen": reversed_by_silkscreen,
        "silk_diff": silk_diff,
        "mark_front": best_front,
        "mark_flip": best_flip,
        "applicable": applicable,
    }


@_memo1
def _band_asymmetry(body):
    """判据2辅助：检测本体左/右或上/下较宽色带/极性纹。返回 (side, strength)。
    side: -1=左或上侧重, +1=右或下侧重, 0=无明显色带"""
    if body is None or body.size == 0 or body.ndim != 3:
        return 0.0, 0.0
    h, w = body.shape[:2]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV).astype(np.float32)
    edge = max(2, int(min(h, w) * 0.14))
    cx1, cx2 = int(w * 0.32), max(int(w * 0.32) + 1, int(w * 0.68))
    cy1, cy2 = int(h * 0.32), max(int(h * 0.32) + 1, int(h * 0.68))
    center = hsv[cy1:cy2, cx1:cx2]

    def _region_feat(region):
        if region.size == 0:
            return np.zeros(3, dtype=np.float32)
        return np.array([
            float(np.mean(region[:, :, 0])),
            float(np.mean(region[:, :, 1])),
            float(np.mean(region[:, :, 2])),
        ], dtype=np.float32)

    def _edge_score(region):
        if region.size == 0 or center.size == 0:
            return 0.0
        rf = _region_feat(region)
        cf = _region_feat(center)
        return float(np.abs(rf - cf).sum())

    left = _edge_score(hsv[:, :edge])
    right = _edge_score(hsv[:, w - edge:])
    top = _edge_score(hsv[:edge, :])
    bottom = _edge_score(hsv[h - edge:, :])

    h_diff = abs(left - right)
    v_diff = abs(top - bottom)
    h_denom = max(1.0, (left + right) * 0.5)
    v_denom = max(1.0, (top + bottom) * 0.5)
    h_strength = h_diff / h_denom
    v_strength = v_diff / v_denom

    if h_strength >= v_strength:
        strength = h_strength
        side = (-1.0 if left > right else 1.0) if h_diff > 1e-3 else 0.0
    else:
        strength = v_strength
        side = (-1.0 if top > bottom else 1.0) if v_diff > 1e-3 else 0.0

    peak = max(left, right, top, bottom)
    if strength < 0.08 and peak < 10.0:
        return 0.0, strength
    if strength < 0.06 and peak >= 12.0 and max(h_diff, v_diff) >= 4.0:
        strength = max(strength, 0.10)
    elif strength < 0.06:
        return 0.0, strength
    return side, strength


def polarity_color_band_compare(test, gold, side_tol=0.12):
    """判据2：较宽色带/极性纹侧别对比；装反 180° 时色带跑到对侧。"""
    test_body, gold_body, *_rest = polarity_mark_crops(test, gold)
    g_side, g_strength = _band_asymmetry(gold_body)
    t_side, t_strength = _band_asymmetry(test_body)
    has_band = g_strength >= 0.12 and abs(g_side) > 0.5
    if not has_band:
        return {
            "has_band": False,
            "reversed_by_band": False,
            "gold_band_side": g_side,
            "test_band_side": t_side,
            "band_strength": g_strength,
        }
    # 金板与来料色带侧别相反 → 极反
    opposite = g_side * t_side < 0 and t_strength >= side_tol
    # 来料侧别弱但 180° 翻转后更吻合金板侧别
    if not opposite and t_strength < side_tol:
        t_flip_side, t_flip_str = _band_asymmetry(cv2.rotate(test_body, cv2.ROTATE_180))
        if g_side * t_flip_side > 0 and t_flip_str >= side_tol * 0.8:
            opposite = True
    return {
        "has_band": True,
        "reversed_by_band": opposite,
        "gold_band_side": g_side,
        "test_band_side": t_side,
        "band_strength": g_strength,
    }


@_memo1
def _extract_die_patch(body):
    """提取二极管/LED 灯芯区域小图与相对横向位置 [0,1]。"""
    if body is None or body.size == 0:
        return None, None
    gray = to_gray(body)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gmax = float(np.max(blur))
    gmin = float(np.min(blur))
    span = max(1.0, gmax - gmin)
    # 灯芯与封装有对比：取最亮或最暗小块
    hi = max(gmin + span * 0.35, gmax - max(8.0, span * 0.25))
    lo = min(gmax - span * 0.35, gmin + max(8.0, span * 0.25))
    candidates = []
    for mask in ((blur >= hi).astype(np.uint8) * 255, (blur <= lo).astype(np.uint8) * 255):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnt = largest_contour(mask)
        if cnt is None:
            continue
        area = cv2.contourArea(cnt)
        if area < max(10, h * w * 0.005) or area > h * w * 0.45:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = max(bw, bh) / max(1, min(bw, bh))
        if ar > 6.0:
            continue
        candidates.append((area, cnt, x, y, bw, bh))
    if not candidates:
        return None, None
    _, cnt, x, y, bw, bh = max(candidates, key=lambda t: t[0])
    m = cv2.moments(cnt)
    cx = (m["m10"] / m["m01"]) if m["m01"] else x + bw * 0.5
    cy = (m["m01"] / m["m00"]) if m["m00"] else y + bh * 0.5
    rel_x = float(cx / max(1, w))
    rel_y = float(cy / max(1, h))
    pad = 2
    patch = body[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
    if patch.size < 30:
        return None, None
    return patch, rel_x, rel_y


def _diode_likelihood(body):
    got = _extract_die_patch(body)
    if got[0] is None:
        return 0.0
    patch = got[0]
    ph, pw = patch.shape[:2]
    if patch.size >= 30 and max(ph, pw) >= 4:
        return 0.75
    return 0.4


def polarity_diode_die_compare(test, gold, ncc_diff_min=0.10, side_tol=0.10):
    """判据3：二极管灯芯形状/位置 0° vs 180° 比对。"""
    test_body, gold_body, *_ = polarity_mark_crops(test, gold)
    g_ret = _extract_die_patch(gold_body)
    t_ret = _extract_die_patch(test_body)
    g_patch, g_rel_x = g_ret[0], g_ret[1] if len(g_ret) > 1 else None
    g_rel_y = g_ret[2] if len(g_ret) > 2 else None
    t_patch = t_ret[0]
    t_rel_x = t_ret[1] if len(t_ret) > 1 else None
    t_rel_y = t_ret[2] if len(t_ret) > 2 else None
    is_diode = _diode_likelihood(gold_body) >= 0.4 or _diode_likelihood(test_body) >= 0.4
    if not is_diode or g_patch is None:
        return {"is_diode": False, "reversed_by_die": False}
    reversed_by_die = False
    die_diff = 0.0
    if t_patch is not None:
        r = _ncc_flip_diff(t_patch, g_patch)
        if r is not None:
            die_diff = r["diff"]
            reversed_by_die = die_diff > ncc_diff_min * 0.75
    gh, gw = gold_body.shape[:2]
    long_is_vertical = gh >= gw
    if g_rel_x is not None and t_rel_x is not None and not long_is_vertical:
        if abs(g_rel_x - 0.5) > side_tol and abs(t_rel_x - 0.5) > side_tol * 0.7:
            if (g_rel_x - 0.5) * (t_rel_x - 0.5) < 0:
                reversed_by_die = True
    if g_rel_y is not None and t_rel_y is not None and long_is_vertical:
        if abs(g_rel_y - 0.5) > side_tol and abs(t_rel_y - 0.5) > side_tol * 0.7:
            if (g_rel_y - 0.5) * (t_rel_y - 0.5) < 0:
                reversed_by_die = True
    return {
        "is_diode": True,
        "reversed_by_die": reversed_by_die,
        "die_diff": die_diff,
        "gold_die_rel_x": g_rel_x,
        "test_die_rel_x": t_rel_x,
        "gold_die_rel_y": g_rel_y,
        "test_die_rel_y": t_rel_y,
    }


def polarity_mark_dark_ratios(test, gold, mark_ratio=0.30):
    """极性暗标记（二极管暗点/缺口）侧别辅助；亮色封装跳过。"""
    test_body, gold_body, t_mark, g_mark, _tl, _gl, _ = polarity_mark_crops(test, gold)
    if np.median(to_gray(gold_body)) > 115:
        return 0.0, 0.0, False
    _, g_bin = cv2.threshold(to_gray(g_mark), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, t_bin = cv2.threshold(to_gray(t_mark), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    g_dark = cv2.countNonZero(g_bin) / max(1, g_mark.size)
    t_dark = cv2.countNonZero(t_bin) / max(1, t_mark.size)
    reversed_by_mark = g_dark > mark_ratio and t_dark < mark_ratio * 0.5
    return g_dark, t_dark, reversed_by_mark


_SILK_EMPTY = {"reversed_by_silkscreen": False, "silk_diff": 0.0,
               "mark_front": 0.0, "mark_flip": 0.0, "applicable": False}
_BAND_EMPTY = {"has_band": False, "reversed_by_band": False,
               "gold_band_side": 0.0, "test_band_side": 0.0, "band_strength": 0.0}
_DIODE_EMPTY = {"is_diode": False, "reversed_by_die": False}


def polarity_detect_all(test, gold, ncc_diff_min=0.06, mark_ratio=0.30, band_side_tol=0.12,
                        enable_silkscreen=True, enable_color_band=True,
                        enable_diode=True, enable_dark_mark=True):
    """极反四路联合：丝印图案 / 色带侧别 / 二极管灯芯 / 暗极性标记。

    各路可独立开关（enable_*）。关闭的路按需跳过，返回中性结果，从而按需提速。
    默认四路全开（向后兼容）；调用方（如 reverse_polarity 算法）可只开丝印一路。
    """
    silk = (polarity_silkscreen_compare(test, gold, ncc_diff_min=ncc_diff_min)
            if enable_silkscreen else dict(_SILK_EMPTY))
    band = (polarity_color_band_compare(test, gold, side_tol=band_side_tol)
            if enable_color_band else dict(_BAND_EMPTY))
    diode = (polarity_diode_die_compare(test, gold, ncc_diff_min=ncc_diff_min, side_tol=band_side_tol)
             if enable_diode else dict(_DIODE_EMPTY))
    if enable_dark_mark:
        g_dark, t_dark, reversed_by_dark = polarity_mark_dark_ratios(test, gold, mark_ratio=mark_ratio)
    else:
        g_dark, t_dark, reversed_by_dark = 0.0, 0.0, False

    triggers = []
    if silk.get("reversed_by_silkscreen"):
        triggers.append("silkscreen")
    if band.get("has_band") and band.get("reversed_by_band"):
        triggers.append("color_band")
    if diode.get("is_diode") and diode.get("reversed_by_die"):
        triggers.append("diode_die")
    if reversed_by_dark:
        triggers.append("dark_mark")

    is_reversed = len(triggers) > 0
    score_parts = [
        silk.get("silk_diff", 0.0),
        band.get("band_strength", 0.0) if band.get("reversed_by_band") else 0.0,
        diode.get("die_diff", 0.0),
    ]
    return {
        "is_reversed": is_reversed,
        "triggers": triggers,
        "silkscreen": silk,
        "color_band": band,
        "diode_die": diode,
        "reversed_by_dark": reversed_by_dark,
        "mark_dark_g": g_dark,
        "mark_dark_t": t_dark,
        "score_hint": max(score_parts) if score_parts else 0.0,
        # 兼容旧字段
        "reversed_by_ncc": silk.get("reversed_by_silkscreen", False),
        "mark_diff": silk.get("silk_diff", 0.0),
        "body_diff": silk.get("silk_diff", 0.0),
        "mark_front": 0.0,
        "mark_flip": silk.get("silk_diff", 0.0),
        "body_front": 0.0,
        "body_flip": 0.0,
        "ncc_front": 0.0,
        "ncc_flip": 0.0,
    }


def polarity_ncc_compare(test, gold, ncc_diff_min=0.10):
    """兼容旧接口：返回丝印比对为主的结果 dict。"""
    all_r = polarity_detect_all(test, gold, ncc_diff_min=ncc_diff_min)
    silk = all_r["silkscreen"]
    return {
        "body_front": silk.get("silk_diff", 0.0),
        "body_flip": 0.0,
        "mark_front": 0.0,
        "mark_flip": silk.get("silk_diff", 0.0),
        "mark_diff": silk.get("silk_diff", 0.0),
        "body_diff": silk.get("silk_diff", 0.0),
        "reversed_by_ncc": all_r["is_reversed"],
        "body_box": component_body_bbox(gold, test),
        "ncc_front": 0.0,
        "ncc_flip": 0.0,
        "triggers": all_r.get("triggers", []),
    }


def is_probable_polarity_reversal(test, gold, ncc_diff_min=0.06, **flags):
    """移位前排除 180° 极反，避免把翻转误判为平移。

    flags 透传给 polarity_detect_all 的四路开关（enable_silkscreen / enable_color_band /
    enable_diode / enable_dark_mark）。shift 默认只用轻量「丝印单路」预检以提速。
    """
    all_r = polarity_detect_all(test, gold, ncc_diff_min=ncc_diff_min, **flags)
    return bool(all_r["is_reversed"])


def localize_defect_bboxes(defects, golden, test, offset_x=0, offset_y=0, margin=3):
    """将缺陷框从整图改为元件本体 tight bbox（叠加 offset 用于多元件）。"""
    if not defects:
        return defects
    bx, by, bw, bh = component_body_bbox(golden, test, margin=margin)
    from core.models import BoundingBox
    for d in defects:
        d.bounding_box = BoundingBox(bx + offset_x, by + offset_y, bw, bh)
    return defects


def run_per_component_detect(detect_roi_fn, image, template, min_area=400):
    """按金板上的元件区域逐块检测，仅对 NG 元件输出标注框。

    detect_roi_fn(test_crop, gold_crop) -> AlgorithmResult
    """
    from core.interfaces import AlgorithmResult

    if template is None or template.size == 0:
        return detect_roi_fn(image, template)

    h, w = image.shape[:2]
    full_area = max(1, w * h)
    bx, by, bw, bh = [int(v) for v in component_body_bbox(template, image)]
    body_area = bw * bh
    # FOV 单元件：整图检测 + 本体定位，避免金板连通域碎裂成多块后漏检
    if body_area >= full_area * 0.06 and body_area <= full_area * 0.92:
        result = detect_roi_fn(image, template)
        localize_defect_bboxes(result.defects, template, image)
        result.defects = result.defects[:1]
        return result

    boxes = find_component_boxes(template, min_area=min_area)
    if len(boxes) == 1:
        result = detect_roi_fn(image, template)
        localize_defect_bboxes(result.defects, template, image)
        result.defects = result.defects[:1]
        return result

    all_defects = []
    total_ms = 0.0
    meta_parts = []
    for x, y, bw, bh in boxes:
        gold_crop = crop(template, (x, y, bw, bh))
        if gold_crop.size == 0:
            continue
        sub_img = crop(image, (x, y, bw, bh))
        if sub_img.size == 0:
            continue
        tx, ty, tbw, tbh = component_body_bbox(gold_crop, sub_img)
        test_crop = crop(image, (x + tx, y + ty, tbw, tbh))
        if test_crop.size == 0 or gold_crop.size == 0:
            continue
        sub = detect_roi_fn(test_crop, gold_crop)
        total_ms += float(getattr(sub, "processing_time_ms", 0) or 0)
        if getattr(sub, "metadata", None):
            meta_parts.append({"box": [x, y, bw, bh], **sub.metadata})
        if sub.defects:
            localize_defect_bboxes(sub.defects, gold_crop, test_crop, offset_x=x + tx, offset_y=y + ty)
            all_defects.extend(sub.defects[:1])
            # 同一类型已在此元件上检出，其余元件不再重复检测该类型
            break

    status = "NG" if all_defects else "OK"
    return AlgorithmResult(
        status=status,
        defects=all_defects,
        processing_time_ms=total_ms,
        metadata={"multi_component": True, "component_count": len(boxes), "parts": meta_parts},
    )

# ── 焊盘分析 ──
def solder_area(pad_img, thresh=0):
    gray = to_gray(pad_img)
    if thresh > 0: _, binv = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    else: _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(cv2.countNonZero(binv))

def infer_end_pad_boxes(golden_roi, band_ratio=0.35, end_ratio=0.22):
    h, w = golden_roi.shape[:2]
    if w >= h:
        y0 = int(h*(1.0-band_ratio)); end_w = max(4, int(w*end_ratio))
        return [[0,y0,end_w,h-y0], [w-end_w,y0,end_w,h-y0]]
    x0 = int(w*(1.0-band_ratio)); end_h = max(4, int(h*end_ratio))
    return [[x0,0,w-x0,end_h], [x0,h-end_h,w-x0,end_h]]

def end_pad_change_asymmetry(golden_roi, test_roi, pads, solder_thresh=0):
    changes, areas_g, areas_t = [], [], []
    for box in pads[:2]:
        g_crop = crop(golden_roi, box); t_crop = crop(test_roi, box)
        ag = max(1, solder_area(g_crop, solder_thresh)); at_val = solder_area(t_crop, solder_thresh)
        areas_g.append(ag); areas_t.append(at_val); changes.append(abs(at_val-ag)/ag)
    if len(changes) < 2: return 1.0, areas_g, areas_t
    hi = max(changes[0], changes[1])
    return (min(changes[0],changes[1])/hi if hi>1e-6 else 1.0), areas_g, areas_t
