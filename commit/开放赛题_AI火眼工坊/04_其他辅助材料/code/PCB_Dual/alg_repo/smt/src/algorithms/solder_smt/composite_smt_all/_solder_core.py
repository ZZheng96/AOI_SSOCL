import os
import cv2
import numpy as np
import base64, zlib

# 尝试导入C++加速模块 (pybind11), 失败则回退到Python实现
try:
    from _fast_core import compute_diff_fast, detect_crack_hough_fast, detect_crack_batch_fast
    _USE_CPP = True
except ImportError:
    _USE_CPP = False


# ── HS色板矩阵编解码 ──

def _decode_hs_lut(b64str):
    """解码base64+zlib的HS矩阵 (36x64 -> 180x256)"""
    if not b64str:
        return None
    try:
        packed = np.frombuffer(zlib.decompress(base64.b64decode(b64str)), dtype=np.uint8)
        small = np.unpackbits(packed)[:36*64].reshape(36, 64).astype(bool)
        lut = np.repeat(np.repeat(small, 5, axis=0), 4, axis=1)[:180, :256]
        return lut
    except Exception:
        return None


def _apply_hs_lut_mask(hsv, lut, v_min=0, v_max=255):
    """用HS矩阵查表替代cv2.inRange (仅H/S用矩阵, V仍用范围)

    Args:
        hsv: HxWx3 HSV图像
        lut: 180x256 bool矩阵 (None则返回None)
        v_min, v_max: V通道范围
    Returns:
        mask: HxW uint8 (0/255)
    """
    if lut is None:
        return None
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    hs_mask = lut[h_ch, s_ch]
    v_mask = (v_ch >= v_min) & (v_ch <= v_max)
    return (hs_mask & v_mask).astype(np.uint8) * 255



# 工具: 自动缩放

def auto_resize_image(img, max_size=400, threshold=450):
    """当图片宽或高超过threshold时, 等比缩小到能放入max_size x max_size的矩形

    返回: (resized_img, scale)  scale=1.0表示未缩放
    """
    h, w = img.shape[:2]
    if max(h, w) <= threshold:
        return img, 1.0
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_pad_rects(pad_rects, scale):
    """等比缩放pad矩形坐标"""
    if scale == 1.0:
        return pad_rects
    return [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
            for (x, y, w, h) in pad_rects]


def scale_group_rects(group_rects, scale):
    """等比缩放group矩形坐标"""
    if scale == 1.0 or not group_rects:
        return group_rects
    return [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
            for (x, y, w, h) in group_rects]


# Step1: 干扰去除

def _remove_interference(img, p, loose=False, hsv=None):
    """干扰排除

    Args:
        img: BGR图像
        p: 参数dict
        loose: 宽松模式(用于grow, 不需要pad检测时)
            - 阻焊/丝印HSV范围略微收紧
            - 跳过_keep_large和扩大干扰区的形态学操作
        hsv: 预计算的HSV图像, 避免重复转换
    """
    if hsv is None:
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    margin = p.get("loose_margin", 3)

    if loose:
        # 宽松: 收紧阻焊和丝印范围, 但margin不能导致范围变空集
        # 若原始范围 < 2*margin, 则不加margin (范围太窄时保持原样)
        sm_h_low = p["mask_h_low"] + margin if p["mask_h_high"] - p["mask_h_low"] > 2 * margin else p["mask_h_low"]
        sm_h_high = p["mask_h_high"] - margin if p["mask_h_high"] - p["mask_h_low"] > 2 * margin else p["mask_h_high"]
        sm_s_min = min(p["mask_s_min"] + margin * 2, 255)
        sm_v_min = min(p["mask_v_min"] + margin * 2, 255)
        mask_sm = cv2.inRange(hsv,
            (sm_h_low, sm_s_min, sm_v_min),
            (sm_h_high, 255, p.get("mask_v_max", 255)))
        sk_h_low = p["silk_h_low"] + margin if p["silk_h_high"] - p["silk_h_low"] > 2 * margin else p["silk_h_low"]
        sk_h_high = p["silk_h_high"] - margin if p["silk_h_high"] - p["silk_h_low"] > 2 * margin else p["silk_h_high"]
        sk_s_low = p["silk_s_low"] + margin * 2 if p["silk_s_high"] - p["silk_s_low"] > 2 * margin * 2 else p["silk_s_low"]
        sk_s_high = p["silk_s_high"] - margin * 2 if p["silk_s_high"] - p["silk_s_low"] > 2 * margin * 2 else p["silk_s_high"]
        sk_v_low = p["silk_v_low"] + margin * 2 if p["silk_v_high"] - p["silk_v_low"] > 2 * margin * 2 else p["silk_v_low"]
        sk_v_high = p["silk_v_high"] - margin * 2 if p["silk_v_high"] - p["silk_v_low"] > 2 * margin * 2 else p["silk_v_high"]
        mask_sk = cv2.inRange(hsv,
            (sk_h_low, sk_s_low, sk_v_low),
            (sk_h_high, sk_s_high, sk_v_high))
    else:
        # 严格: 原始范围
        mask_sm = cv2.inRange(hsv, (p["mask_h_low"], p["mask_s_min"], p["mask_v_min"]),
                                   (p["mask_h_high"], 255, p.get("mask_v_max", 255)))
        mask_sk = cv2.inRange(hsv, (p["silk_h_low"], p["silk_s_low"], p["silk_v_low"]),
                                   (p["silk_h_high"], p["silk_s_high"], p["silk_v_high"]))

    comp_h_low = p.get("comp_h_low", 0)
    comp_h_high = p.get("comp_h_high", 179)
    comp_s_min = p.get("comp_s_min", 0)
    comp_s_max = p.get("comp_s_max", 255)
    mask_comp = cv2.inRange(hsv, (comp_h_low, comp_s_min, 0), (comp_h_high, comp_s_max, p["comp_v_max"]))

    if loose:
        # 宽松: 直接合并, 不做_keep_large和扩大干扰区的形态学操作
        inter = cv2.bitwise_or(cv2.bitwise_or(mask_sm, mask_sk), mask_comp)
        return inter

    # 严格: _keep_large + 扩大干扰区
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    def _keep_large(mask, min_a):
        k_close7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close7)
        n, labels, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if int(st[i, cv2.CC_STAT_AREA]) >= min_a: out[labels == i] = 255
        return out
    mask_sm = _keep_large(mask_sm, p["min_interference_area"])
    mask_sk = _keep_large(mask_sk, p["min_interference_area"])
    mask_comp = _keep_large(mask_comp, p["min_interference_area"])
    inter = cv2.bitwise_or(cv2.bitwise_or(mask_sm, mask_sk), mask_comp)
    inter_close_ks = p.get("interference_close_ks", 5)
    k_inter_close = cv2.getStructuringElement(cv2.MORPH_RECT, (inter_close_ks, inter_close_ks))
    inter = cv2.morphologyEx(inter, cv2.MORPH_CLOSE, k_inter_close, iterations=2)  # 闭运算去除裂隙
    inter = cv2.dilate(inter, k5, iterations=1)
    return inter


# Step1+: 焊盘分类 (基础工具 + 聚类 + 完整分类流程)

def _clamp_rect(x, y, w, h, img_w, img_h):
    """框超出图片时整体向内平移, 保持框尺寸"""
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > img_w: x = img_w - w
    if y + h > img_h: y = img_h - h
    return max(0, min(x, img_w - w)), max(0, min(y, img_h - h)), w, h

def _has_overlap(rects):
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            if ix > 0 and iy > 0: return True
    return False

def _contains(outer, inner):
    """outer是否完全包含inner. outer/inner=(x,y,w,h)"""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ox+ow >= ix+iw and oy+oh >= iy+ih

def _bounding_box(rects):
    """取多个矩形的外接矩形"""
    if len(rects) == 0: return (0, 0, 0, 0)
    x = min(r[0] for r in rects)
    y = min(r[1] for r in rects)
    w = max(r[0]+r[2] for r in rects) - x
    h = max(r[1]+r[3] for r in rects) - y
    return (x, y, w, h)

def _merge_overlapping_groups(rects):
    """只合并有重叠的矩形组(并查集), 不重叠的保持不变"""
    if len(rects) == 0: return []
    parent = list(range(len(rects)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            x1,y1,w1,h1 = rects[i]; x2,y2,w2,h2 = rects[j]
            ix = max(0, min(x1+w1,x2+w2)-max(x1,x2))
            iy = max(0, min(y1+h1,y2+h2)-max(y1,y2))
            if ix > 0 and iy > 0: union(i, j)
    groups = {}
    for i in range(len(rects)):
        r = find(i)
        groups.setdefault(r, []).append(rects[i])
    result = []
    for grp in groups.values():
        if len(grp) == 1: result.append(grp[0])
        else: result.append(_bounding_box(grp))
    return result

def _cluster_single_pass(features, n_classes, ar_thresh, area_ratio, w_area=0.5, w_ar=0.5, score_thresh=0.5):
    """单轮聚类: 宽高比+面积线性加权求和作为相似度指标
    features: [(ar, area, x, y, w, h), ...]
    w_area: 面积权重, w_ar: 宽高比权重 (w_area+w_ar=1)
    score_thresh: 相似度阈值, 超过则归入该簇
    返回: (kept_clusters, discarded_clusters) — 每个cluster是features列表
    """
    if len(features) == 0: return [], []
    features_sorted = sorted(features, key=lambda f: f[1], reverse=True)
    clusters = []
    for f in features_sorted:
        ar, area = f[0], f[1]
        placed = False
        for cl in clusters:
            cl_ar = float(np.mean([c[0] for c in cl]))
            cl_area = float(np.median([c[1] for c in cl]))
            # 面积相似度: 1 - |area差|/max(area), 宽高比相似度: 1 - |ar差|/max(ar)
            area_sim = 1.0 - abs(area - cl_area) / max(area, cl_area, 1)
            ar_sim = 1.0 - abs(ar - cl_ar) / max(ar, cl_ar, 1)
            score = w_area * area_sim + w_ar * ar_sim
            if score >= score_thresh:
                cl.append(f); placed = True; break
        if not placed: clusters.append([f])
    clusters.sort(key=lambda cl: float(np.median([c[1] for c in cl])), reverse=True)
    return clusters[:n_classes], clusters[n_classes:]

def _unify_and_clamp(cluster, img_w, img_h):
    """统一尺寸: 组内最大w×h, 中心点不变 → 超边界向内移
    cluster: [(ar, area, x, y, w, h), ...]
    返回: [(unified_x, unified_y, max_w, max_h, orig_x, orig_y, orig_w, orig_h), ...]
          前四个是统一后, 后四个是对应的原始矩形
    """
    if len(cluster) == 0: return []
    max_w = max(c[4] for c in cluster)
    max_h = max(c[5] for c in cluster)
    result = []
    for (ar, area, x, y, w, h) in cluster:
        cx = x + w // 2; cy = y + h // 2
        nx = cx - max_w // 2; ny = cy - max_h // 2
        nx, ny, _, _ = _clamp_rect(nx, ny, max_w, max_h, img_w, img_h)
        result.append((nx, ny, max_w, max_h, x, y, w, h))
    return result

def classify_pads(raw_rects, img_w, img_h, p):
    """完整三轮分类流程

    返回: {
        'steps': [(label, rects_with_class), ...],  # 每步结果用于可视化
        'final': [(x,y,w,h,ci), ...],
        'n_used': int,
    }
    rects_with_class: [(x,y,w,h,ci), ...] 或 [(x,y,w,h), ...]
    """
    ar_t = p["cluster_ar_thresh"]
    ar_r = p["cluster_area_ratio"]
    ar_t2 = p["cluster_ar_thresh_r2"]
    ar_r2 = p["cluster_area_ratio_r2"]
    n_spec = p["n_classes"]
    is_auto = (n_spec == "auto")

    # 提取特征
    features = []
    for (x, y, w, h) in raw_rects:
        features.append((w / max(h, 1), w * h, x, y, w, h))

    steps = []  # [(label, rects)]
    all_discarded = []  # 第一轮丢弃的原始矩形 (x,y,w,h)

    # ── 第一轮 ──
    # 分N+1类, 保留N类, 丢弃碎片
    if is_auto:
        r1_n = 1; r1_cluster_n = 2  # auto: 分2类保留1类
    else:
        r1_n = int(n_spec); r1_cluster_n = int(n_spec) + 1  # 用户指定N: 分N+1类保留N类

    # 第一轮: 侧重面积 (w_area=0.7, w_ar=0.3)
    all_r1, disc_r1 = _cluster_single_pass(features, r1_cluster_n, ar_t, ar_r, w_area=0.7, w_ar=0.3, score_thresh=0.5)
    # 只保留前r1_n类
    kept_r1 = all_r1[:r1_n]
    disc_r1 = all_r1[r1_n:] + disc_r1
    # 第一轮聚类结果(显示用)
    r1_cluster_rects = []
    for ci, cl in enumerate(kept_r1):
        for (ar, area, x, y, w, h) in cl:
            r1_cluster_rects.append((x, y, w, h, ci))
    for cl in disc_r1:
        for (ar, area, x, y, w, h) in cl:
            all_discarded.append((x, y, w, h))
    steps.append(("R1_cluster", r1_cluster_rects, len(all_discarded)))

    # 统一尺寸
    r1_unified = []  # [(ux,uy,uw,uh, ci, orig_x,orig_y,orig_w,orig_h)]
    for ci, cl in enumerate(kept_r1):
        for item in _unify_and_clamp(cl, img_w, img_h):
            r1_unified.append((*item[:4], ci, *item[4:]))
    r1_result = [(r[0], r[1], r[2], r[3], r[4]) for r in r1_unified]
    steps.append(("R1_unified", r1_result, 0))

    # 检查重叠
    r1_pure = [(r[0], r[1], r[2], r[3]) for r in r1_unified]
    if not _has_overlap(r1_pure):
        return {"steps": steps, "final": r1_result, "n_used": r1_n}

    # ── 第二轮 ──
    # 检查第一轮结果的框是否完全包含了一些被舍弃的碎片
    # 如果是, 取该框对应的原始矩形和碎片共同的外接矩形
    r2_input = []  # [(ar, area, x, y, w, h), ...]
    for r in r1_unified:
        ux, uy, uw, uh, ci, ox, oy, ow, oh = r
        contained_fragments = []
        for frag in all_discarded:
            if _contains((ux, uy, uw, uh), frag):
                contained_fragments.append(frag)
        if contained_fragments:
            # 取原始矩形和碎片的外接
            bb = _bounding_box([(ox, oy, ow, oh)] + contained_fragments)
            r2_input.append((bb[2]/max(bb[3],1), bb[2]*bb[3], bb[0], bb[1], bb[2], bb[3]))
        else:
            r2_input.append((ow/max(oh,1), ow*oh, ox, oy, ow, oh))

    # N=1: 吞并碎片后直接返回各矩形, 无需再分类
    if not is_auto and r1_n == 1:
        final = [(r[2], r[3], r[4], r[5], 0) for r in r2_input]
        steps.append(("R2_bbox", final, 0))
        return {"steps": steps, "final": final, "n_used": 1}

    if is_auto:
        r2_n = 2
    else:
        r2_n = int(n_spec)

    # 形状优先分r2_n类, 不舍弃
    # 第二轮: 侧重宽高比 (w_area=0.3, w_ar=0.7)
    all_r2, _ = _cluster_single_pass(r2_input, len(r2_input), ar_t2, ar_r2, w_area=0.3, w_ar=0.7, score_thresh=0.5)
    if len(all_r2) > r2_n:
        kept_r2 = all_r2[:r2_n]
        for cl in all_r2[r2_n:]:
            cl_ar = float(np.mean([c[0] for c in cl]))
            best_idx = 0; best_diff = 999
            for i, kc in enumerate(kept_r2):
                kc_ar = float(np.mean([c[0] for c in kc]))
                d = abs(cl_ar - kc_ar)
                if d < best_diff: best_diff = d; best_idx = i
            kept_r2[best_idx].extend(cl)
    else:
        kept_r2 = all_r2

    # 第二轮聚类结果(显示用)
    r2_cluster_rects = []
    for ci, cl in enumerate(kept_r2):
        for (ar, area, x, y, w, h) in cl:
            r2_cluster_rects.append((x, y, w, h, ci))
    steps.append(("R2_cluster", r2_cluster_rects, 0))

    # 统一尺寸
    r2_unified = []
    for ci, cl in enumerate(kept_r2):
        for item in _unify_and_clamp(cl, img_w, img_h):
            r2_unified.append((*item[:4], ci, *item[4:]))
    r2_result = [(r[0], r[1], r[2], r[3], r[4]) for r in r2_unified]
    steps.append(("R2_unified", r2_result, 0))

    # 检查重叠
    r2_pure = [(r[0], r[1], r[2], r[3]) for r in r2_unified]
    if not _has_overlap(r2_pure):
        return {"steps": steps, "final": r2_result, "n_used": r2_n}

    # 用户指定N: 合并重叠矩形后直接返回, 不进入第三轮
    if not is_auto:
        r2_merged = _merge_overlapping_groups(r2_pure)
        r2_final = [(r[0], r[1], r[2], r[3], 0) for r in r2_merged]
        steps.append(("R2_merged", r2_final, 0))
        return {"steps": steps, "final": r2_final, "n_used": r2_n}

    # ── 第三轮 (仅auto模式) ──
    if is_auto:
        r3_n = 3
        # auto: 先分3类
        all_r3, _ = _cluster_single_pass(r2_input, len(r2_input), ar_t2, ar_r2, w_area=0.3, w_ar=0.7, score_thresh=0.5)
        if len(all_r3) > r3_n:
            kept_r3 = all_r3[:r3_n]
            for cl in all_r3[r3_n:]:
                cl_ar = float(np.mean([c[0] for c in cl]))
                best_idx = 0; best_diff = 999
                for i, kc in enumerate(kept_r3):
                    kc_ar = float(np.mean([c[0] for c in kc]))
                    d = abs(cl_ar - kc_ar)
                    if d < best_diff: best_diff = d; best_idx = i
                kept_r3[best_idx].extend(cl)
        else:
            kept_r3 = all_r3
        r3_cluster_rects = []
        for ci, cl in enumerate(kept_r3):
            for (ar, area, x, y, w, h) in cl:
                r3_cluster_rects.append((x, y, w, h, ci))
        steps.append(("R3_cluster", r3_cluster_rects, 0))

        # 统一后检查重叠
        r3_unified_tmp = []
        for ci, cl in enumerate(kept_r3):
            for item in _unify_and_clamp(cl, img_w, img_h):
                r3_unified_tmp.append((*item[:4], ci))
        r3_pure_tmp = [(r[0], r[1], r[2], r[3]) for r in r3_unified_tmp]

        if _has_overlap(r3_pure_tmp):
            # 仍重叠: 合并原始矩形外接 → 再分3类
            r3_merged = _merge_overlapping_groups(r2_pure)
            r3_features = [(r[2]/max(r[3],1), r[2]*r[3], r[0], r[1], r[2], r[3]) for r in r3_merged]
            all_r3b, _ = _cluster_single_pass(r3_features, len(r3_features), ar_t2, ar_r2, w_area=0.3, w_ar=0.7, score_thresh=0.5)
            if len(all_r3b) > r3_n:
                kept_r3 = all_r3b[:r3_n]
                for cl in all_r3b[r3_n:]:
                    cl_ar = float(np.mean([c[0] for c in cl]))
                    best_idx = 0; best_diff = 999
                    for i, kc in enumerate(kept_r3):
                        kc_ar = float(np.mean([c[0] for c in kc]))
                        d = abs(cl_ar - kc_ar)
                        if d < best_diff: best_diff = d; best_idx = i
                    kept_r3[best_idx].extend(cl)
            else:
                kept_r3 = all_r3b
            r3_cluster_rects2 = []
            for ci, cl in enumerate(kept_r3):
                for (ar, area, x, y, w, h) in cl:
                    r3_cluster_rects2.append((x, y, w, h, ci))
            steps.append(("R3_merge_cluster", r3_cluster_rects2, 0))

    # 第三轮统一尺寸
    r3_unified = []
    for ci, cl in enumerate(kept_r3):
        for item in _unify_and_clamp(cl, img_w, img_h):
            r3_unified.append((*item[:4], ci))
    r3_result = [(r[0], r[1], r[2], r[3], r[4]) for r in r3_unified]
    steps.append(("R3_unified", r3_result, 0))

    return {"steps": steps, "final": r3_result, "n_used": r3_n}


# Step2: 焊锡提取 (H校正W + 分层膨胀 + 局部W距离 + SV约束)

def _build_w_lut(p):
    """H通道非线性校正LUT: 拉伸焊锡段, 压缩干扰段"""
    lut = np.zeros(256, dtype=np.uint8)
    # 段边界(可参数化)
    a_hi = p.get("w_lut_a_hi", 25)      # 红/棕焊锡段上界
    b_hi = p.get("w_lut_b_hi", 35)      # 黄丝印段上界
    c_hi = p.get("w_lut_c_hi", 77)      # 绿阻焊段上界
    d_hi = p.get("w_lut_d_hi", 100)     # 青光段上界
    # 输出分配
    w_a = p.get("w_lut_w_a", 60)        # 红褐焊锡输出宽
    w_b = p.get("w_lut_w_b", 3)         # 黄丝印输出宽
    w_c = p.get("w_lut_w_c", 4)         # 绿阻焊输出宽
    w_d = p.get("w_lut_w_d", 3)         # 青光输出宽
    # w_e = 179 - (w_a+w_b+w_c+w_d)     # 蓝焊锡输出宽(自动)
    for h_val in range(180):
        h = float(h_val)
        if h <= a_hi:
            w = h * w_a / a_hi
        elif h <= b_hi:
            w = w_a + (h - a_hi) * w_b / (b_hi - a_hi)
        elif h <= c_hi:
            w = w_a + w_b + (h - b_hi) * w_c / (c_hi - b_hi)
        elif h <= d_hi:
            w = w_a + w_b + w_c + (h - c_hi) * w_d / (d_hi - c_hi)
        else:
            w = w_a + w_b + w_c + w_d + (h - d_hi) * (179 - w_a - w_b - w_c - w_d) / (179 - d_hi)
        lut[h_val] = np.clip(w, 0, 179)
    return lut

def _w_dist(w1, w2):
    """校正W的环形距离"""
    d = np.abs(w1.astype(np.int16) - w2.astype(np.int16))
    return np.minimum(d, 179 - d)

def _expand_rect(x, y, w, h, ratio, img_w, img_h, short_ratio=None, aspect_thresh=2.0,
                 match_aspect=False):
    """扩展矩形。长边扩展ratio，当长短边比例>aspect_thresh时短边扩展short_ratio。
    若match_aspect=True，扩展后按图像宽高比调整。
    """
    long_side = max(w, h)
    short_side = min(w, h)
    if short_ratio is not None and long_side > short_side * aspect_thresh:
        if w >= h:
            dw = int(w * ratio); dh = int(h * short_ratio)
        else:
            dw = int(w * short_ratio); dh = int(h * ratio)
    else:
        dw = int(w * ratio); dh = int(h * ratio)
    nx = max(0, x - dw); ny = max(0, y - dh)
    nw = min(img_w - nx, w + 2 * dw); nh = min(img_h - ny, h + 2 * dh)
    if match_aspect:
        img_ratio = img_w / img_h
        min_w = w + 2 * dw  # 至少扩展量
        min_h = h + 2 * dh
        if min_w / min_h > img_ratio * 1.01:
            nw, nh = min_w, int(min_w / img_ratio)   # 宽度够, 增加高度
        elif min_w / min_h < img_ratio * 0.99:
            nw, nh = int(min_h * img_ratio), min_h   # 高度够, 增加宽度
        else:
            nw, nh = min_w, min_h
        nx = max(0, x + w // 2 - nw // 2)
        ny = max(0, y + h // 2 - nh // 2)
        nw = min(img_w - nx, nw)
        nh = min(img_h - ny, nh)
    return nx, ny, nw, nh

def _merge_overlapping_rects(rects):
    if len(rects) == 0: return []
    parent = list(range(len(rects)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            x1,y1,w1,h1 = rects[i]; x2,y2,w2,h2 = rects[j]
            ix = max(0, min(x1+w1,x2+w2)-max(x1,x2))
            iy = max(0, min(y1+h1,y2+h2)-max(y1,y2))
            if ix > 0 and iy > 0: union(i, j)
    groups = {}
    for i in range(len(rects)):
        groups.setdefault(find(i), []).append(rects[i])
    result = []
    for grp in groups.values():
        x = min(r[0] for r in grp); y = min(r[1] for r in grp)
        w = max(r[0]+r[2] for r in grp) - x; h = max(r[1]+r[3] for r in grp) - y
        result.append((x, y, w, h))
    return result

def extract_solder(img, interference, pad_rects, group_rects, p, hsv=None, gray=None):
    """H校正W + 分层膨胀 + 局部W距离 + SV约束"""
    if hsv is None:
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]
    h_vals = hsv[:, :, 0]
    s_vals = hsv[:, :, 1]
    v_vals = hsv[:, :, 2]

    # H校正 → W通道
    w_lut = _build_w_lut(p)
    w_vals = w_lut[h_vals]

    # 焊盘组框mask
    region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if group_rects:
        for (gx, gy, gw, gh) in group_rects:
            region_mask[gy:gy+gh, gx:gx+gw] = 255
    else:
        for (px, py, pw, ph) in pad_rects:
            region_mask[py:py+ph, px:px+pw] = 255

    non_inter = cv2.bitwise_not(interference)
    if gray is None:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    # 焊锡生长允许区: 非干扰区 -> 膨胀 -> close+fill补洞
    enable_noninter_process = p.get("solder_enable_noninter_process", True)
    if enable_noninter_process:
        non_inter_dilate_ks = p.get("solder_noninter_dilate_ks", 15)
        k_heavy = cv2.getStructuringElement(cv2.MORPH_RECT, (non_inter_dilate_ks, non_inter_dilate_ks))
        non_inter_heavy = cv2.dilate(non_inter, k_heavy, iterations=1)
        non_inter_heavy = cv2.morphologyEx(non_inter_heavy, cv2.MORPH_CLOSE, k_heavy)
        contours_ni, _ = cv2.findContours(non_inter_heavy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(non_inter_heavy, contours_ni, -1, 255, -1)
        growzone_open_ks = p.get("solder_growzone_open_ks", 5)
        k_gz_open = cv2.getStructuringElement(cv2.MORPH_RECT, (growzone_open_ks, growzone_open_ks))
        non_inter_heavy = cv2.morphologyEx(non_inter_heavy, cv2.MORPH_OPEN, k_gz_open)
    else:
        non_inter_heavy = non_inter

    # ── 1. 蓝色种子 (严格) ∩ 非干扰区 ──
    blue_v_min = p.get("solder_blue_v_min", 150)
    # 优先使用HS色板矩阵
    _solder_lut = _decode_hs_lut(p.get("solder_blue_hs_lut"))
    if _solder_lut is not None:
        mask_blue = _apply_hs_lut_mask(hsv, _solder_lut, v_min=blue_v_min)
    else:
        blue_h_low = p.get("solder_blue_h_low", 85)
        blue_h_high = p.get("solder_blue_h_high", 115)
        blue_s_min = p.get("solder_blue_s_min", 60)
        mask_blue = cv2.inRange(hsv, (blue_h_low, blue_s_min, blue_v_min),
                                      (blue_h_high, 255, 255))
        # V补偿: H接近h_high(最蓝端), 只放低V门槛(捕捉低亮蓝色)
        h_center = blue_h_high
        h_half = max(3, (blue_h_high - blue_h_low) // 4)
        v_comp_min = max(40, blue_v_min // 2)
        mask_blue_comp = cv2.inRange(hsv, (h_center - h_half, blue_s_min, v_comp_min),
                                           (h_center + h_half, 255, 255))
        mask_blue = cv2.bitwise_or(mask_blue, mask_blue_comp)
    mask_blue = cv2.bitwise_and(mask_blue, non_inter)
    if not p.get("seed_outside_region", True):
        mask_blue = cv2.bitwise_and(mask_blue, region_mask)

    mask_current = mask_blue.copy()

    # 生长量限制: 防止少量蓝色种子在铜板上过度生长
    blue_seed_count = np.count_nonzero(mask_blue)
    max_growth_ratio = p.get("solder_max_growth_ratio", 0)  # 0=不限制

    # 按连通域停止: 定期检查蓝色种子占比，过低的连通域停止生长
    growth_check_interval = p.get("solder_growth_check_interval", 0)  # 0=不检查
    min_blue_ratio = p.get("solder_min_blue_ratio", 0.1)
    frozen = np.zeros_like(mask_current)  # 累积冻结区域

    # ── 2. 分层膨胀 + 校正W局部距离 + 自适应V约束 ──
    dilate_ks = p.get("solder_grow_dilate_ks", 3)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ks, dilate_ks))
    max_iters = p.get("solder_grow_max_iters", 0)

    # 自适应S/V阈值: 按W段不同要求不同
    # 焊锡段(蓝W>75, 红W<60): S要求低(焊锡红/蓝反光饱和度不一定高)
    # 干扰段(W60-75): S要求高(排除低饱和丝印)
    v_min_blue = p.get("solder_v_min_blue", 150)    # 蓝焊锡段(W>75): 高亮
    v_min_interf = p.get("solder_v_min_interf", 150) # 干扰段(W60-75): 高亮(排除暗阻焊)
    v_min_red = p.get("solder_v_min_red", 80)       # 红褐焊锡段(W<60): 允许暗(氧化锡)
    s_min_blue = p.get("solder_s_min_blue", 60)     # 蓝焊锡段: S下限
    s_min_interf = p.get("solder_s_min_interf", 30) # 干扰段: S下限降低(允许过渡)
    s_min_red = p.get("solder_s_min_red", 0)        # 红褐焊锡段: S下限(亮红焊锡S低)
    s_max_red = p.get("solder_s_max_red", 100)      # 红褐焊锡段: S上限(排除高S铜板)
    v_min_global = p.get("solder_v_min_global", 110)  # 全局V下限(排除暗区)

    # W+V线性加权和参数
    w_scale = p.get("solder_wv_w_scale", 45.0)   # W距离缩放
    v_scale = p.get("solder_wv_v_scale", 80.0)   # V距离缩放
    wv_thresh = p.get("solder_wv_thresh", 1.0)   # 加权距离阈值(<1允许生长)
    w_weight = p.get("solder_wv_w_weight", 0.6)
    v_weight = p.get("solder_wv_v_weight", 0.4)

    def _adaptive_sv_ok(w):
        """根据W值返回S/V约束: 焊锡段允许低S但限制高S(排除铜板), 干扰段要求高S"""
        v_thresh = np.full_like(w, v_min_blue, dtype=np.uint8)
        v_thresh[w < 75] = v_min_interf
        v_thresh[w < 60] = v_min_red
        s_lo = np.full_like(w, s_min_blue, dtype=np.uint8)
        s_lo[w < 75] = s_min_interf
        s_lo[w < 60] = s_min_red
        # 红褐段加S上限: 排除高S铜板
        s_ok = (s_vals >= s_lo)
        s_ok = s_ok & ~((w < 60) & (s_vals > s_max_red))
        return s_ok & (v_vals >= v_thresh)

    # 自适应S/V约束 + 全局V下限
    enable_sv_ok = p.get("solder_enable_sv_ok", True)
    if enable_sv_ok:
        sv_ok = (_adaptive_sv_ok(w_vals) & (v_vals >= v_min_global)).astype(np.uint8) * 255
    else:
        sv_ok = None

    for layer in range(1, max_iters + 1):
        # 定期检查: 蓝色种子占比过低的连通域冻结(停止生长但保留已有像素)
        if growth_check_interval > 0 and layer % growth_check_interval == 0:
            n_cc_gc, labels_gc = cv2.connectedComponents(mask_current, 8)
            for ci in range(1, n_cc_gc):
                comp = labels_gc == ci
                comp_area = int(comp.sum())
                if comp_area < 20:
                    continue
                blue_in_comp = int(np.count_nonzero(comp & (mask_blue > 0)))
                if blue_in_comp / comp_area < min_blue_ratio:
                    frozen[comp] = 255

        # 膨胀1次 → 新增边界 (冻结区域不参与膨胀)
        growable = cv2.subtract(mask_current, frozen) if np.count_nonzero(frozen) > 0 else mask_current
        dilated = cv2.dilate(growable, k_dil, iterations=1)
        boundary = cv2.subtract(dilated, mask_current)
        # 约束: 允许生长区 + 组框 + S/V约束
        boundary = cv2.bitwise_and(boundary, non_inter_heavy)
        if not p.get("grow_outside_region", True):
            boundary = cv2.bitwise_and(boundary, region_mask)
        if sv_ok is not None:
            boundary = cv2.bitwise_and(boundary, sv_ok)
        if np.count_nonzero(boundary) == 0:
            break

        # W+V线性加权和: V相似可补偿W差异 (冻结区域不参与参考)
        non_ref = (mask_current == 0) | (frozen > 0)
        w_masked = w_vals.copy().astype(np.float32)
        w_masked[non_ref] = -1  # 标记非参考区域
        v_masked = v_vals.copy().astype(np.float32)
        v_masked[non_ref] = -1
        # 膨胀参考值到边界
        w_ref = cv2.dilate(w_masked, k_dil, iterations=1)
        v_ref = cv2.dilate(v_masked, k_dil, iterations=1)
        # W环形距离
        wd = np.abs(w_vals.astype(np.float32) - w_ref)
        wd = np.minimum(wd, 179 - wd)
        # V距离
        vd = np.abs(v_vals.astype(np.float32) - v_ref)
        # 线性加权和: dist = w_weight*(wd/w_scale) + v_weight*(vd/v_scale)
        w_weight = p.get("solder_wv_w_weight", 0.6)
        v_weight = p.get("solder_wv_v_weight", 0.4)
        wv_dist = w_weight * (wd / w_scale) + v_weight * (vd / v_scale)
        wv_ok = (wv_dist < wv_thresh).astype(np.uint8) * 255
        new_pixels = cv2.bitwise_and(boundary, wv_ok)

        added = np.count_nonzero(new_pixels)
        if added == 0:
            break
        mask_current = cv2.bitwise_or(mask_current, new_pixels)
        # 生长量限制: 超过蓝色种子的N倍则停止
        if max_growth_ratio > 0 and blue_seed_count > 0:
            if np.count_nonzero(mask_current) > blue_seed_count * max_growth_ratio:
                break

    # ── 3. 闭运算 + 补洞 ──
    mask_merged = mask_current.copy()
    close_ks = p.get("solder_close_ks", 7)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ks, close_ks))
    mask_merged = cv2.morphologyEx(mask_merged, cv2.MORPH_CLOSE, k_close)
    mask_filled = mask_merged.copy()
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_filled, contours, -1, 255, -1)

    # ── 4. 纹理二次过滤 ──
    enable_texture_filter = p.get("solder_enable_texture_filter", True)
    if enable_texture_filter:
        texture_ksize = p.get("solder_texture_ksize", 7)
        texture_thresh = p.get("solder_texture_thresh", 3.0)
        gray_f = gray.astype(np.float32)
        mean_blur = cv2.blur(gray_f, (texture_ksize, texture_ksize))
        sq_blur = cv2.blur(gray_f ** 2, (texture_ksize, texture_ksize))
        std_map = np.sqrt(np.maximum(sq_blur - mean_blur ** 2, 0))
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, 8)
        mask_texture = np.zeros_like(mask_filled)
        for i in range(1, n_cc):
            if int(stats[i, cv2.CC_STAT_AREA]) < 5: continue
            std_mean = float(np.mean(std_map[labels == i]))
            if std_mean > texture_thresh:
                mask_texture[labels == i] = 255
    else:
        mask_texture = mask_filled
        std_map = np.zeros_like(mask_filled, dtype=np.float32)

    # ── 5. 最终mask ──
    mask_final = mask_texture
    if not p.get("grow_outside_region", True):
        mask_final = cv2.bitwise_and(mask_final, region_mask)

    return (mask_blue, mask_merged, mask_filled,
            mask_texture, mask_final, std_map, non_inter_heavy, w_vals)


# Step3: 缺陷判定 (少锡/多锡/连锡/虚焊)

def _rect_mask(x, y, w, h, img_shape):
    """创建矩形内为255的mask"""
    m = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.rectangle(m, (x, y), (x+w, y+h), 255, -1)
    return m

def _rects_union_mask(rects, img_shape):
    """多个矩形的并集mask"""
    m = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x, y, w, h) in rects:
        cv2.rectangle(m, (x, y), (x+w, y+h), 255, -1)
    return m

def _pads_in_group(group_rect, pad_rects):
    """获取落在group_rect内的焊盘"""
    gx, gy, gw, gh = group_rect
    pads = []
    for (px, py, pw, ph) in pad_rects:
        cx, cy = px + pw//2, py + ph//2
        if gx <= cx < gx+gw and gy <= cy < gy+gh:
            pads.append((px, py, pw, ph))
    return pads

def judge_insufficient(solder_mask, pad_rects, img_shape, p):
    """少锡判断: 焊盘框内锡占比

    返回: [(x,y,w,h, coverage, is_ng), ...]
    """
    thresh = p.get("insufficient_thresh", 0.08)
    results = []
    h_img, w_img = img_shape[:2]
    for i, (x, y, w, h) in enumerate(pad_rects):
        pad_area = w * h
        if pad_area == 0:
            results.append((x, y, w, h, 0.0, False))
            continue
        # ROI切片替代全图mask
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x2 <= x1 or y2 <= y1:
            results.append((x, y, w, h, 0.0, False))
            continue
        solder_in = np.count_nonzero(solder_mask[y1:y2, x1:x2])
        coverage = solder_in / pad_area
        is_ng = coverage < thresh
        results.append((x, y, w, h, coverage, is_ng))
    return results

def judge_excess(solder_mask, pad_rects, group_rects, img_shape, p, diff_mask_full=None):
    """多锡判断: 组框内焊锡 - 焊盘内焊锡 -> (可选)AND diff_mask -> 总面积/中位pad面积 -> 超阈值则框出每个blob

    返回: (excess_mask, is_ng, total_ratio, blobs)
      excess_mask: np.uint8 多余焊锡二值图
      is_ng: bool 整体是否多锡
      total_ratio: 总excess面积 / 中位pad面积
      blobs: [(x,y,w,h), ...] 每个excess blob外接矩形
    """
    thresh = p.get("excess_thresh", 0.08)
    min_area = p.get("excess_min_area", 5)
    h, w = img_shape[:2]

    # 单mask画矩形替代逐矩形创建全图mask
    group_mask = np.zeros((h, w), dtype=np.uint8)
    for (gx, gy, gw, gh) in group_rects:
        cv2.rectangle(group_mask, (int(gx), int(gy)), (int(gx + gw), int(gy + gh)), 255, -1)
    pad_mask = np.zeros((h, w), dtype=np.uint8)
    for (px, py, pw, ph) in pad_rects:
        cv2.rectangle(pad_mask, (int(px), int(py)), (int(px + pw), int(py + ph)), 255, -1)

    excess_mask = cv2.bitwise_and(solder_mask, group_mask)
    excess_mask[pad_mask > 0] = 0

    # diff_grow模式: AND diff_mask, 只保留与模板有差异的区域
    if diff_mask_full is not None:
        excess_mask = cv2.bitwise_and(excess_mask, diff_mask_full)
        # diff模式下用更大的min_area过滤碎片
        min_area = p.get("excess_min_area_diff", 50)

    # 开运算去散点
    kop = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    excess_mask = cv2.morphologyEx(excess_mask, cv2.MORPH_OPEN, kop)

    pad_areas = [pw * ph for (_, _, pw, ph) in pad_rects]
    median_pad_area = float(np.median(pad_areas)) if pad_areas else 1.0

    # 碎片过滤: 先按连通域面积过滤excess_mask, 再计算is_ng
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(excess_mask, 8)
    filtered_mask = np.zeros_like(excess_mask)
    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered_mask[labels == i] = 255
        x, y, bw, bh = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), \
                       int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append((x, y, bw, bh))
    excess_mask = filtered_mask

    # 合并重叠的blob外接矩形（Union-Find，一步完成）
    blobs = _merge_overlap_boxes(blobs)

    # 总excess面积判断 (基于过滤后的mask)
    total_excess = int(np.count_nonzero(excess_mask))
    total_ratio = total_excess / median_pad_area
    is_ng = total_ratio > thresh

    return excess_mask, is_ng, total_ratio, blobs

def judge_bridge(solder_mask, pad_rects, group_rects, img_shape, p):
    """连锡判断: 焊盘组框内相邻焊盘间锡连通

    方法: 在组框内去掉焊盘框区域，检查剩余焊锡是否连通相邻焊盘
    返回: [(group_rect, bridge_pairs, is_ng, bridge_blobs), ...]
    """
    results = []
    for gr in group_rects:
        gx, gy, gw, gh = gr
        pads = _pads_in_group(gr, pad_rects)
        if len(pads) < 2:
            results.append((gr, [], False, []))
            continue

        group_m = _rect_mask(gx, gy, gw, gh, img_shape)

        # 焊盘框mask
        pads_m = _rects_union_mask(pads, img_shape)
        # 组框内焊盘框外的焊锡
        bridge_zone = group_m & (~pads_m)
        bridge_solder = solder_mask & bridge_zone

        if np.count_nonzero(bridge_solder) == 0:
            results.append((gr, [], False, []))
            continue

        # 对bridge_solder做连通域标记(用WithStats一次获取面积和外接矩形)
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(bridge_solder, 8)
        # 对每个焊盘，检查它边缘附近有哪些连通域
        pad_components = []
        for (px, py, pw, ph) in pads:
            # 焊盘边缘膨胀2px
            pad_edge = _rect_mask(px, py, pw, ph, img_shape)
            pad_edge = cv2.dilate(pad_edge, np.ones((3,3), np.uint8), iterations=2)
            pad_edge = pad_edge & bridge_zone  # 只看外围框内
            comps = set(np.unique(labels[pad_edge > 0])) - {0}
            pad_components.append(comps)

        # 检查是否有连通域同时触及两个焊盘
        bridge_pairs = []
        bridge_labels = set()
        for i in range(len(pads)):
            for j in range(i+1, len(pads)):
                common = pad_components[i] & pad_components[j]
                if common:
                    bridge_pairs.append((i, j))
                    bridge_labels.update(common)

        # 连锡连通域外接矩形(从stats直接读取)
        bridge_blobs = []
        for lbl in bridge_labels:
            x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
            y0 = int(stats[lbl, cv2.CC_STAT_TOP])
            w0 = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h0 = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            bridge_blobs.append((x0, y0, w0, h0))

        # 计算连通桥面积(从stats直接读取)
        bridge_area = 0
        for pair in bridge_pairs:
            i, j = pair
            for lbl in pad_components[i] & pad_components[j]:
                bridge_area += int(stats[lbl, cv2.CC_STAT_AREA])

        is_ng = len(bridge_pairs) > 0 and bridge_area > p.get("bridge_min_area", 20)
        results.append((gr, bridge_pairs, is_ng, bridge_blobs))
    return results

def _detect_crack(gray_roi, pad_mask, p=None):
    """pad内裂痕检测 — 局部对比度暗带 + 形态学连接 + 连通域规则过滤

    算法: 21x21高斯模糊得局部均值 → 暗带(比均值暗15%) → 竖向/水平核闭运算连接
          → 连通域过滤(长宽比/方向/边缘/长度)
    Args:
        gray_roi: pad区域的灰度图
        pad_mask: pad区域mask (255=pad内)
    返回: (has_crack, crack_blob) — crack_blob=(x,y,w,h)或None
    """
    h, w = gray_roi.shape
    pad_region = pad_mask > 0
    if pad_region.sum() < 50:
        return False, None

    # 局部对比度暗带: 比周围暗15%
    gray_blur = cv2.GaussianBlur(gray_roi, (21, 21), 0)
    dark_mask = (gray_roi < gray_blur.astype(np.float32) * 0.85) & pad_region
    dark_mask = dark_mask.astype(np.uint8) * 255

    # 边缘区域排除: 在形态学之前移除边缘像素, 避免边缘暗带被连接到内部
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    dark_mask[:margin_y, :] = 0       # 上边缘
    dark_mask[max(h - margin_y, 0):, :] = 0  # 下边缘
    dark_mask[:, :margin_x] = 0       # 左边缘
    dark_mask[:, max(w - margin_x, 0):] = 0  # 右边缘

    # 形态学: 竖向核连接竖向裂痕 + 水平核连接水平裂痕
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_v)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_h)

    combined = dark_mask

    # 连通域分析: 找线状特征 + 方向过滤 + 长度过滤
    pad_is_vertical = h > w
    crack_ratio = (p or {}).get("crack_ratio", 6.0)
    crack_area = (p or {}).get("crack_area", 100)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < crack_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
        if not (aspect > crack_ratio and area > crack_area):
            continue
        # 过滤1: 裂缝方向必须垂直于pad长轴
        crack_is_vertical = ch > cw
        if pad_is_vertical and crack_is_vertical:
            continue
        if not pad_is_vertical and not crack_is_vertical:
            continue
        # 过滤2: 裂缝长度必须超过pad短边的1/3
        crack_length = max(cw, ch)
        pad_short = min(w, h)
        if crack_length < pad_short / 3:
            continue
        return True, (x, y, cw, ch)

    return False, None


def _merge_collinear_lines(lines, angle_tol=15, dist_tol=10):
    """合并角度相近且端点距离相近的线段"""
    if len(lines) <= 1:
        return list(lines)

    merged = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue
        x1, y1, x2, y2 = lines[i]
        used[i] = True
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            x3, y3, x4, y4 = lines[j]
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3))
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > 180:
                angle_diff -= 360
            angle_diff = abs(angle_diff)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            if angle_diff > angle_tol:
                continue

            min_dist = min(
                np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2),
                np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2),
                np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2),
            )
            if min_dist > dist_tol:
                continue

            pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            max_d = 0
            best_pair = (0, 1)
            for a in range(4):
                for b in range(a + 1, 4):
                    d = (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2
                    if d > max_d:
                        max_d = d
                        best_pair = (a, b)
            x1, y1 = pts[best_pair[0]]
            x2, y2 = pts[best_pair[1]]
            used[j] = True

        merged.append((int(x1), int(y1), int(x2), int(y2)))

    return merged


def _detect_crack_hough(gray_roi, pad_mask_roi, p=None):
    """HoughLinesP裂痕检测

    1. 暗带mask (局部均值*0.85)
    2. 边缘排除 (15% margin)
    3. 形态学闭运算 (竖向1x15 + 水平15x1)
    4. HoughLinesP检测线段
    5. 合并共线线段
    6. 过滤: 方向垂直于pad长轴 (tol=15度)
    7. 过滤: 不在pad边缘
    8. 过滤: 合并后线段长度 > pad短边/3

    Returns: (has_crack, crack_lines) where crack_lines is list of (x1,y1,x2,y2)
    """
    h, w = gray_roi.shape
    pad_region = pad_mask_roi > 0
    if pad_region.sum() < 50:
        return False, []

    # 尝试C++加速路径
    if _USE_CPP:
        try:
            return detect_crack_hough_fast(gray_roi, pad_mask_roi, p or {})
        except Exception:
            pass

    # 1. 暗带mask
    gray_blur = cv2.GaussianBlur(gray_roi, (21, 21), 0)
    dark_mask = (gray_roi < gray_blur.astype(np.float32) * 0.85) & pad_region
    dark_mask = dark_mask.astype(np.uint8) * 255

    # 2. 边缘排除
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    dark_mask[:margin_y, :] = 0
    dark_mask[max(h - margin_y, 0):, :] = 0
    dark_mask[:, :margin_x] = 0
    dark_mask[:, max(w - margin_x, 0):] = 0

    # 3. 形态学闭运算
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_v)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_h)

    # 4. HoughLinesP
    pad_short = min(w, h)
    threshold = (p or {}).get('crack_hough_threshold', 15)
    min_length = (p or {}).get('crack_hough_min_length', 0)
    if min_length <= 0:
        min_length = max(pad_short // 3, 5)
    max_gap = (p or {}).get('crack_hough_max_gap', 8)

    lines = cv2.HoughLinesP(dark_mask, rho=1, theta=np.pi / 180,
                            threshold=threshold, minLineLength=min_length, maxLineGap=max_gap)
    if lines is None:
        return False, []

    raw_lines = []
    for ln in lines:
        x1, y1, x2, y2 = ln.reshape(-1)[:4]
        raw_lines.append((x1, y1, x2, y2))

    # 5. 合并共线线段 (角度容差可配)
    angle_tol = (p or {}).get('crack_merge_angle_tol', 5)
    merged = _merge_collinear_lines(raw_lines, angle_tol=angle_tol, dist_tol=10)

    # 6. 方向过滤: 裂痕方向必须垂直于pad长轴 (偏差≤5度)
    pad_is_vertical = h > w
    filtered = []
    for (x1, y1, x2, y2) in merged:
        line_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        line_angle = abs(line_angle)
        if line_angle > 90:
            line_angle = 180 - line_angle
        line_is_vertical = line_angle > 45
        # pad竖直 -> 裂痕应水平(接近0度); pad水平 -> 裂痕应竖直(接近90度)
        if pad_is_vertical:
            if line_angle > 5:  # 偏差不超过5度
                continue
        else:
            if line_angle < 85:  # 偏差不超过5度(90-5)
                continue
        filtered.append((x1, y1, x2, y2))

    # 7. 边缘过滤: 两端点都不在pad边缘
    edge_margin = max(int(pad_short * 0.1), 3)
    edge_filtered = []
    for (x1, y1, x2, y2) in filtered:
        p1_at_edge = (x1 < edge_margin or x1 > w - edge_margin or
                      y1 < edge_margin or y1 > h - edge_margin)
        p2_at_edge = (x2 < edge_margin or x2 > w - edge_margin or
                      y2 < edge_margin or y2 > h - edge_margin)
        if p1_at_edge and p2_at_edge:
            continue
        edge_filtered.append((x1, y1, x2, y2))

    # 8. 长度过滤 (可配参数: crack_min_length_ratio, 默认0.33=pad_short/3)
    min_len_ratio = (p or {}).get('crack_min_length_ratio', 0.5)
    final_lines = []
    for (x1, y1, x2, y2) in edge_filtered:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > pad_short * min_len_ratio:
            final_lines.append((int(x1), int(y1), int(x2), int(y2)))

    has_crack = len(final_lines) > 0
    return has_crack, final_lines


def _compute_blue_solder_mask(img, p, hsv=None):
    """提取蓝色范围焊锡mask (不补洞, 保留虚焊区域空洞)

    用于虚焊检测区(Rule1)的焊锡覆盖率计算.
    """
    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sb_v_min = p.get("solder_blue_v_min", 83)
    # 优先使用HS色板矩阵
    lut = _decode_hs_lut(p.get("solder_blue_hs_lut"))
    if lut is not None:
        mask = _apply_hs_lut_mask(hsv, lut, v_min=sb_v_min)
    else:
        sb_h_low = p.get("solder_blue_h_low", 50)
        sb_h_high = p.get("solder_blue_h_high", 100)
        sb_s_min = p.get("solder_blue_s_min", 56)
        mask = cv2.inRange(hsv, (sb_h_low, sb_s_min, sb_v_min), (sb_h_high, 255, 255))
    _k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _k)
    return mask


def _compute_diff_mask_ratio(diff_mask, pad_rect, img_shape):
    """计算diff_mask在pad区域的覆盖率

    返回: diff_mask_ratio (0.0~1.0)
    """
    px, py, pw, ph = int(pad_rect[0]), int(pad_rect[1]), int(pad_rect[2]), int(pad_rect[3])
    pad_area = pw * ph
    if pad_area == 0:
        return 0.0
    # ROI切片替代全图mask
    h, w = diff_mask.shape[:2]
    x1, y1 = max(0, px), max(0, py)
    x2, y2 = min(w, px + pw), min(h, py + ph)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return np.count_nonzero(diff_mask[y1:y2, x1:x2]) / pad_area


def judge_cold_solder(img, solder_mask, pad_rects, img_shape, p,
                      toe_rects=None, rim_rects=None, diff_mask_full=None,
                      ins_results=None, pad_source="local", hsv=None, gray=None):
    """虚焊判断: 暗比+纹理不均 + crack检测 + 红胶检测

    特征:
      - 焊锡区域内暗像素占比 (V < v_dark)
      - 纹理不均 (V值标准差)
      - crack检测 (局部对比度暗带+连通域规则)
      - 红胶检测 (pad内红色像素占比>1%, terminal专用)

    新增 gull-wing 3优先级规则 (当提供 toe_rects/rim_rects/diff_mask_full 时启用):
      Rule 1: 检测区(rim-toe)内蓝色焊锡覆盖率 < cold_solder_rim_ratio -> 虚焊
      Rule 2: crack检测 AND toe区域金属颜色占比 > toe_metal_ratio_thresh -> 虚焊
      Rule 3: crack检测 AND pad区域diff_mask覆盖率 > cold_diff_ratio_thresh -> 虚焊

    Args:
        toe_rects: dict {pad_index: (x,y,w,h)} 或 None
        rim_rects: dict {pad_index: (x,y,w,h)} 或 None
        diff_mask_full: np.ndarray 或 None (用于Rule3)
        ins_results: list of (x,y,w,h,coverage,is_ng) 少锡检测结果 (用于Rule1)
        pad_source: pad来源 ("ROI"/"template"/"local"), 自动判断不开放给用户
    返回: [(x,y,w,h, cold_score, is_ng, reasons), ...]
        reasons: 触发的路径名列表, 如 ["dark"], ["crack"], ["rule1"], [] 等
    """
    dark_ratio_thresh = p.get("cold_solder_dark_ratio", 0.30)
    v_dark = p.get("cold_solder_v_dark", 80)
    component_type = p.get("pin_type", "gull-wing")
    is_terminal = component_type == "terminal"

    results = []
    if hsv is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    if gray is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = len(pad_rects)

    # ── gull-wing 3优先级规则 (gull-wing类型始终启用, terminal走旧逻辑) ──
    use_new_rules = not is_terminal
    if use_new_rules:
        has_toe_rim = (toe_rects is not None and rim_rects is not None and
                       bool(toe_rects) and bool(rim_rects))
        if not has_toe_rim:
            pass  # 规则1和规则2将跳过

        cold_solder_detect_area_ratio = p.get("cold_solder_rim_ratio", 0.5)
        toe_metal_ratio_thresh = p.get("toe_metal_ratio_thresh", 0.3)
        cold_diff_ratio_thresh = p.get("cold_diff_ratio_thresh", 0.4)

        # 蓝色范围焊锡mask + toe金属mask (仅has_toe_rim时需要, Rule1/Rule2使用)
        if has_toe_rim:
            solder_for_cold = _compute_blue_solder_mask(img, p, hsv=hsv)
            toe_metal_h_low = p.get("toe_metal_h_low", 80)
            toe_metal_h_high = p.get("toe_metal_h_high", 124)
            toe_metal_s_min = p.get("toe_metal_s_min", 39)
            toe_metal_s_max = p.get("toe_metal_s_max", 78)
            toe_metal_v_min = p.get("toe_metal_v_min", 125)
            toe_metal_v_max = p.get("toe_metal_v_max", 222)
            toe_metal_mask = cv2.inRange(hsv,
                (toe_metal_h_low, toe_metal_s_min, toe_metal_v_min),
                (toe_metal_h_high, toe_metal_s_max, toe_metal_v_max))
        else:
            solder_for_cold = None
            toe_metal_mask = None

        # ── 预计算: Rule1 + diff_ratio, 收集需要crack检测的pad ──
        pre_rule1_ng = [False] * len(pad_rects)
        pre_solder_ratio = [0.0] * len(pad_rects)
        pre_diff_ratio = [0.0] * len(pad_rects)
        crack_needed_rois = []  # (pad_index, x, y, w, h)
        for i, (x, y, w, h) in enumerate(pad_rects):
            x1c = max(0, int(x))
            y1c = max(0, int(y))
            x2c = min(gray.shape[1], int(x + w))
            y2c = min(gray.shape[0], int(y + h))
            pw_c = x2c - x1c
            ph_c = y2c - y1c

            # Rule1
            if has_toe_rim and i in rim_rects and i in toe_rects:
                rim_rect = rim_rects[i]
                toe_rect = toe_rects[i]
                rim_mask = _rect_mask(rim_rect[0], rim_rect[1], rim_rect[2], rim_rect[3], img_shape)
                toe_mask = _rect_mask(toe_rect[0], toe_rect[1], toe_rect[2], toe_rect[3], img_shape)
                detect_area_mask = cv2.subtract(rim_mask, toe_mask)
                detect_area_count = int(np.count_nonzero(detect_area_mask))
                if detect_area_count > 0:
                    solder_in_detect = cv2.bitwise_and(solder_for_cold, detect_area_mask)
                    pre_solder_ratio[i] = np.count_nonzero(solder_in_detect) / detect_area_count
                    if pre_solder_ratio[i] < cold_solder_detect_area_ratio:
                        pre_rule1_ng[i] = True
                if ins_results is not None and i < len(ins_results) and ins_results[i][5]:
                    pre_rule1_ng[i] = True
            elif not has_toe_rim and diff_mask_full is not None and pw_c >= 10 and ph_c >= 10:
                pre_diff_ratio[i] = _compute_diff_mask_ratio(diff_mask_full, (x, y, w, h), img_shape)

            # 收集需要crack的pad
            if not pre_rule1_ng[i] and pw_c >= 10 and ph_c >= 10:
                if has_toe_rim:
                    crack_needed_rois.append((i, int(x), int(y), int(w), int(h)))
                elif pre_diff_ratio[i] > cold_diff_ratio_thresh:
                    crack_needed_rois.append((i, int(x), int(y), int(w), int(h)))

        # ── 批量crack检测 ──
        # C++批量检测对小数量pad(<=3)有pybind11开销，用Python更快
        crack_results = {}  # pad_index -> bool
        if crack_needed_rois:
            # C++批量检测对小数量pad有pybind11开销，阈值可配
            crack_cpp_min_pads = (p or {}).get("crack_cpp_min_pads", 4)
            use_cpp_batch = _USE_CPP and len(crack_needed_rois) >= crack_cpp_min_pads
            if use_cpp_batch:
                try:
                    roi_list = [(rx, ry, rw, rh) for (_, rx, ry, rw, rh) in crack_needed_rois]
                    batch_results = detect_crack_batch_fast(gray, roi_list, p)
                    for idx, (pad_i, _, _, _, _) in enumerate(crack_needed_rois):
                        crack_results[pad_i] = bool(batch_results[idx])
                except Exception:
                    use_cpp_batch = False
            if not use_cpp_batch:
                for (pad_i, rx, ry, rw, rh) in crack_needed_rois:
                    x1c = max(0, rx); y1c = max(0, ry)
                    x2c = min(gray.shape[1], rx + rw); y2c = min(gray.shape[0], ry + rh)
                    gray_roi = gray[y1c:y2c, x1c:x2c]
                    pad_mask_roi = np.full((y2c-y1c, x2c-x1c), 255, dtype=np.uint8)
                    crack_results[pad_i], _ = _detect_crack_hough(gray_roi, pad_mask_roi, p)

        for i, (x, y, w, h) in enumerate(pad_rects):
            # 使用预计算结果
            rule1_ng = pre_rule1_ng[i]
            solder_ratio = pre_solder_ratio[i]
            diff_ratio = pre_diff_ratio[i]
            crack_ng = crack_results.get(i, False)

            cold_ratio = 0.0
            if has_toe_rim:
                cold_ratio = solder_ratio
            elif not has_toe_rim:
                cold_ratio = diff_ratio

            # Rule 2: crack + toe金属颜色 (仅Rule1未触发时)
            rule2_ng = False
            toe_metal_ratio = 0.0
            if has_toe_rim and not rule1_ng and i in toe_rects:
                toe_rect = toe_rects[i]
                toe_mask = _rect_mask(toe_rect[0], toe_rect[1], toe_rect[2], toe_rect[3], img_shape)
                toe_area = int(np.count_nonzero(toe_mask))
                if toe_area > 0:
                    metal_in_toe = cv2.bitwise_and(toe_metal_mask, toe_mask)
                    toe_metal_ratio = np.count_nonzero(metal_in_toe) / toe_area
                    if toe_metal_ratio > toe_metal_ratio_thresh and crack_ng:
                        rule2_ng = True
            if rule2_ng:
                cold_ratio = toe_metal_ratio

            # Rule 3: crack + diff_mask覆盖率 (fallback, Rule1&2未触发时)
            rule3_ng = False
            if not has_toe_rim:
                rule3_ng = crack_ng
                if rule3_ng:
                    cold_ratio = diff_ratio
            elif (not rule1_ng and not rule2_ng) and diff_mask_full is not None:
                diff_ratio_r3 = _compute_diff_mask_ratio(diff_mask_full, (x, y, w, h), img_shape)
                if diff_ratio_r3 > cold_diff_ratio_thresh and crack_ng:
                    rule3_ng = True
                    cold_ratio = diff_ratio_r3

            # 最终结果 (8元组, 增加ratio)
            if rule1_ng:
                results.append((x, y, w, h, 1.0, True, ["rule1"], cold_ratio))
            elif rule2_ng:
                results.append((x, y, w, h, 0.9, True, ["rule2"], cold_ratio))
            elif rule3_ng:
                results.append((x, y, w, h, 0.8, True, ["rule3"], cold_ratio))
            else:
                results.append((x, y, w, h, 0.0, False, [], cold_ratio))

        return results

    # ── 既有检测逻辑 (terminal类型 或 向后兼容) ──
    # 第一遍: 计算每个pad的暗比/纹理/焊锡面积
    solder_areas = []
    dark_ratios = []
    cold_scores = []
    for (x, y, w, h) in pad_rects:
        pad_m = _rect_mask(x, y, w, h, img_shape)
        solder_in_pad = solder_mask & pad_m
        solder_area = np.count_nonzero(solder_in_pad)
        solder_areas.append(solder_area)

        if solder_area < 10:
            dark_ratios.append(0.0)
            cold_scores.append(0.0)
            continue

        v_in_solder = v_channel[solder_in_pad > 0]
        dark_ratio = np.count_nonzero(v_in_solder < v_dark) / len(v_in_solder)
        dark_ratios.append(dark_ratio)

        v_std = np.std(v_in_solder) if len(v_in_solder) > 0 else 0
        cold_score = dark_ratio * 0.7 + min(v_std / 80.0, 1.0) * 0.3
        cold_scores.append(cold_score)

    # crack检测 + 最终判定
    for i, (x, y, w, h) in enumerate(pad_rects):
        # crack检测
        crack_ng = False
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(gray.shape[1], x2)
        y2c = min(gray.shape[0], y2)
        pw_c = x2c - x1c
        ph_c = y2c - y1c
        if pw_c >= 10 and ph_c >= 10:
            gray_roi = gray[y1c:y2c, x1c:x2c]
            pad_mask_roi = np.zeros((ph_c, pw_c), dtype=np.uint8)
            pad_mask_roi[:] = 255
            crack_ng, crack_blob = _detect_crack(gray_roi, pad_mask_roi, p)

            # F7红胶检测: pad内红色像素占比>1% -> NG (terminal专用)
            red_glue_ng = False
            if is_terminal:
                hsv_roi = hsv[y1c:y2c, x1c:x2c]
                H_r, S_r, V_r = cv2.split(hsv_roi)
                red_mask = ((H_r < 10) | (H_r > 170)) & (S_r > 100) & (V_r > 30) & (V_r < 140)
                red_ratio = float(red_mask.sum()) / (pw_c * ph_c + 1e-6)
                red_glue_ng = red_ratio > 0.01
        else:
            red_glue_ng = False

        # 暗比判定
        dark_ng = dark_ratios[i] > dark_ratio_thresh

        # 综合判定: 暗比 OR crack OR 红胶
        is_ng = dark_ng or crack_ng or red_glue_ng

        # 路径级诊断
        reasons = []
        if dark_ng:
            reasons.append("dark")
        if crack_ng:
            reasons.append("crack")
        if red_glue_ng:
            reasons.append("red_glue")

        # 综合score
        score = cold_scores[i]
        if crack_ng:
            score = max(score, 1.0)
        if red_glue_ng:
            score = max(score, 0.9)

        # ratio: 暗比(terminal主要指标)
        term_ratio = dark_ratios[i] if dark_ratios[i] > 0 else 0.0
        results.append((x, y, w, h, float(score), is_ng, reasons, term_ratio))
    return results


# Template: 模板对齐 + 映射工具

def find_template(img_path):
    """查找模板图: 同目录下的 _OK.png 或 _template.png"""
    d = os.path.dirname(img_path)
    base = os.path.basename(img_path).replace('.png', '')
    for suffix in ['_OK.png', '_template.png']:
        p = os.path.join(d, base + suffix)
        if os.path.isfile(p):
            return p
    for f in os.listdir(d):
        if f.endswith('_OK.png'):
            return os.path.join(d, f)
    return None

def _mat_to_homogeneous(m2x3):
    """2x3 → 3x3"""
    h = np.eye(3, dtype=np.float64)
    h[:2] = m2x3
    return h

def _homogeneous_to_mat(h3x3):
    """3x3 → 2x3"""
    return h3x3[:2].astype(np.float32)

def align_template(template_img, inspect_img, params=None):
    """模板对齐: 中心对齐 + 等比缩放 + 重叠crop + (可选)ECC EUCLIDEAN

    params:
        ECC_EUCLIDEAN: bool, 默认False, 是否启用ECC精细对齐

    返回 dict:
        aligned_inspect:  对齐后的待检图 (重叠区域, 模板坐标系)
        template_crop:    模板的重叠区域crop
        overlap_rect:     (x, y, w, h) 重叠区域在模板空间的位置
        overlap_ratio:    重叠面积 / 模板面积
        forward:          2x3 正向变换 (原始待检 → 对齐空间)
        inverse:          2x3 逆向变换 (对齐空间 → 原始待检)
        scale:            等比缩放因子
        ecc_cc:           ECC相关系数 (未启用时为0)
        template_size:    (t_w, t_h)
        inspect_size:     (i_w, i_h)
    """
    if params is None:
        params = {}
    use_ecc = params.get("ECC_EUCLIDEAN", False)
    t_h, t_w = template_img.shape[:2]
    i_h, i_w = inspect_img.shape[:2]

    # ── Step 1: 等比缩放因子 (面积匹配) ──
    s = float(np.sqrt(t_w * t_h / (i_w * i_h)))
    new_w = int(round(i_w * s))
    new_h = int(round(i_h * s))

    # ── Step 2: 缩放待检图 ──
    inspect_scaled = cv2.resize(inspect_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ── Step 3: 放到模板尺寸画布上, 中心对齐 ──
    canvas = np.zeros((t_h, t_w, 3), dtype=np.uint8)
    ox = (t_w - new_w) // 2
    oy = (t_h - new_h) // 2

    # 计算实际放置区域 (处理越界)
    cx1, cy1 = max(0, ox), max(0, oy)
    cx2, cy2 = min(t_w, ox + new_w), min(t_h, oy + new_h)
    sx1, sy1 = cx1 - ox, cy1 - oy
    sx2, sy2 = sx1 + (cx2 - cx1), sy1 + (cy2 - cy1)
    canvas[cy1:cy2, cx1:cx2] = inspect_scaled[sy1:sy2, sx1:sx2]

    # ── Step 4: 重叠区域 ──
    overlap_x, overlap_y = cx1, cy1
    overlap_w, overlap_h = cx2 - cx1, cy2 - cy1
    overlap_area = overlap_w * overlap_h
    template_area = t_w * t_h
    overlap_ratio = overlap_area / template_area

    if overlap_w < 20 or overlap_h < 20:
        return None

    # ── Step 5: crop到重叠区域 ──
    template_crop = template_img[overlap_y:overlap_y + overlap_h,
                                  overlap_x:overlap_x + overlap_w].copy()
    inspect_crop = canvas[overlap_y:overlap_y + overlap_h,
                           overlap_x:overlap_x + overlap_w].copy()

    # ── Step 6: (可选) ECC EUCLIDEAN 精细对齐 ──
    cc = 0.0
    warp_euclidean = np.eye(2, 3, dtype=np.float32)
    if use_ecc:
        t_gray = cv2.cvtColor(template_crop, cv2.COLOR_BGR2GRAY)
        i_gray = cv2.cvtColor(inspect_crop, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-5)
        try:
            cc, warp_euclidean = cv2.findTransformECC(
                t_gray, i_gray, warp_euclidean,
                cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        except cv2.error:
            pass

    # ── Step 7: 应用变换 (ECC关闭时跳过warpAffine, inspect_crop已对齐) ──
    if use_ecc:
        aligned_inspect = cv2.warpAffine(
            inspect_crop, warp_euclidean, (overlap_w, overlap_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        aligned_inspect = inspect_crop

    # ── 变换链: 原始待检 → 对齐空间 ──
    # T1: 原始待检坐标 → 画布坐标 (等比缩放 + 中心偏移)
    ox_f = (t_w - new_w) / 2.0  # 使用浮点偏移
    oy_f = (t_h - new_h) / 2.0
    T1 = np.array([[s, 0, ox_f], [0, s, oy_f]], dtype=np.float64)

    # T2: 画布坐标 → 重叠crop坐标 (减去重叠偏移)
    T2 = np.array([[1, 0, -overlap_x], [0, 1, -overlap_y]], dtype=np.float64)

    # T3: 重叠crop坐标 → 对齐坐标 (ECC EUCLIDEAN)
    T3 = warp_euclidean.astype(np.float64)

    # 合成: forward = T3 @ T2 @ T1
    T1_h = _mat_to_homogeneous(T1)
    T2_h = _mat_to_homogeneous(T2)
    T3_h = _mat_to_homogeneous(T3)
    forward_h = T3_h @ T2_h @ T1_h
    inverse_h = np.linalg.inv(forward_h)

    forward = _homogeneous_to_mat(forward_h)
    inverse = _homogeneous_to_mat(inverse_h)

    return {
        'aligned_inspect': aligned_inspect,
        'template_crop': template_crop,
        'overlap_rect': (overlap_x, overlap_y, overlap_w, overlap_h),
        'overlap_ratio': overlap_ratio,
        'forward': forward,
        'inverse': inverse,
        'aligned_matrix': [float(forward[i][j]) for i in range(2) for j in range(3)],
        'scale': s,
        'ecc_cc': cc,
        'template_size': (t_w, t_h),
        'inspect_size': (i_w, i_h),
    }

def map_point_to_original(x, y, info):
    """对齐空间 → 原始待检图空间"""
    inv = info['inverse']
    return float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]), \
           float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2])

def map_point_from_original(x, y, info):
    """原始待检图空间 → 对齐(重叠crop)空间"""
    fwd = info['forward']
    return float(fwd[0, 0] * x + fwd[0, 1] * y + fwd[0, 2]), \
           float(fwd[1, 0] * x + fwd[1, 1] * y + fwd[1, 2])

def map_rect_from_original(rect, info):
    """原始待检图空间矩形 → 重叠crop空间

    rect: (x, y, w, h) in 原始待检图坐标
    返回: (x, y, w, h) in 重叠crop坐标
    """
    x, y, w, h = rect
    corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    mapped = [map_point_from_original(cx, cy, info) for cx, cy in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

def map_rect_to_original(rect, info):
    """模板空间矩形 → 原始待检图空间

    rect: (x, y, w, h) in 模板坐标
    返回: (x, y, w, h) in 原始待检图坐标
    """
    ox, oy, ow, oh = info['overlap_rect']
    # 模板坐标 → 重叠crop坐标
    cx = rect[0] - ox
    cy = rect[1] - oy
    cw = rect[2]
    ch = rect[3]

    # 映射4个角点到原始待检图
    corners = [(cx, cy), (cx + cw, cy), (cx, cy + ch), (cx + cw, cy + ch)]
    mapped = [map_point_to_original(x, y, info) for x, y in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

def map_mask_to_original(mask, info):
    """对齐空间mask → 原始待检图空间"""
    i_w, i_h = info['inspect_size']
    return cv2.warpAffine(mask, info['inverse'], (i_w, i_h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def get_pad_overlap_ratio(rect, info):
    """计算pad框在重叠区域的比例 (方案A: 按可见面积计算)

    rect: (x, y, w, h) in 模板坐标
    返回: 0.0~1.0
    """
    ox, oy, ow, oh = info['overlap_rect']
    rx, ry, rw, rh = rect
    ix1 = max(rx, ox)
    iy1 = max(ry, oy)
    ix2 = min(rx + rw, ox + ow)
    iy2 = min(ry + rh, oy + oh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / (rw * rh)


# Diff: 模板差分计算

def compute_diff(template_img, inspect_img, info=None, params=None):
    """计算模板与待检图的BGR差异 (重叠区域)

    返回: diff_mask, diff_heatmap, info
    """
    if params is None:
        params = {}

    if info is None:
        info = align_template(template_img, inspect_img, params)
        if info is None:
            return None, None, None

    # 尝试C++加速路径 (传入已对齐的crop, 无需warpAffine)
    if _USE_CPP and 'aligned_inspect' in info:
        try:
            return compute_diff_fast(info['template_crop'], info['aligned_inspect'], info, params)
        except Exception:
            pass  # 回退到Python实现

    aligned_inspect = info['aligned_inspect']
    template_crop = info['template_crop']

    # BGR三通道差分取max (uint8直接计算, 单次absdiff)
    t = template_crop[:, :, :3]
    i = aligned_inspect[:, :, :3]
    _diff3 = cv2.absdiff(t, i)  # 3通道同时算
    diff = cv2.max(_diff3[:, :, 0], cv2.max(_diff3[:, :, 1], _diff3[:, :, 2]))

    # 阈值化
    thresh = params.get("diff_thresh", 80)
    _, diff_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # 形态学去噪
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, k)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, k)

    # 过滤小区域
    min_area = params.get("diff_min_area", 10)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, 8)
        diff_mask = np.zeros_like(diff_mask)
        for j in range(1, n):
            if int(stats[j, cv2.CC_STAT_AREA]) >= min_area:
                diff_mask[labels == j] = 255

    # 热力图(可选, 默认跳过以节省时间)
    skip_heatmap = params.get("skip_diff_heatmap", False)
    if skip_heatmap:
        diff_heatmap = None
    else:
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    return diff_mask, diff_heatmap, info


def _merge_overlap_boxes(boxes):
    """合并有重叠的矩形框，一步完成（Union-Find）
    boxes: [(x, y, w, h), ...]
    returns: [(x, y, w, h), ...]
    """
    n = len(boxes)
    if n <= 1:
        return boxes

    rects = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]

    parent = list(range(n))
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def _union(i, j):
        pi, pj = _find(i), _find(j)
        if pi != pj:
            parent[pj] = pi

    for i in range(n):
        x1i, y1i, x2i, y2i = rects[i]
        for j in range(i + 1, n):
            x1j, y1j, x2j, y2j = rects[j]
            ix1 = max(x1i, x1j)
            iy1 = max(y1i, y1j)
            ix2 = min(x2i, x2j)
            iy2 = min(y2i, y2j)
            if ix2 > ix1 and iy2 > iy1:
                _union(i, j)

    groups = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(rects[i])

    merged = []
    for g in groups.values():
        xs = [r[0] for r in g] + [r[2] for r in g]
        ys = [r[1] for r in g] + [r[3] for r in g]
        merged.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))

    return merged
