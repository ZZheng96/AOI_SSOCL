"""分块模块：粒度是配置旋钮（§6.1），不是架构承诺。

mode=single  : 整图 resize 单次前向（demo4 式兜底，~160ms 速度安全网）
mode=tiles9  : 3x3 网格 9 块
mode=tiles36 : tile_size=512 stride=448 全量滑窗（微缺陷最大粒度）
反池化原则：每块独立提特征，融合发生在 tile 分数层，不做跨块特征池化。
"""
import math
import numpy as np


def compute_tiles(image_size, mode, tile_size=512, stride=448):
    """返回 tile 框列表 [(y1,x1,y2,x2)]；single 模式返回整图一个框。"""
    h, w = image_size
    if mode == "single":
        return [(0, 0, h, w)]
    if mode == "tiles9":
        th, tw = math.ceil(h / 3), math.ceil(w / 3)
        # AC-004：小图（h<3 或 ceil 溢出）时过滤空块
        return [t for t in ((r * th, c * tw, min((r + 1) * th, h), min((c + 1) * tw, w))
                            for r in range(3) for c in range(3))
                if t[2] > t[0] and t[3] > t[1]]
    # tiles36：滑窗，边缘块与相邻块对齐到图像边界
    rows = max(1, math.ceil((h - tile_size) / stride) + 1)
    cols = max(1, math.ceil((w - tile_size) / stride) + 1)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y1 = min(r * stride, max(0, h - tile_size))
            x1 = min(c * stride, max(0, w - tile_size))
            tiles.append((y1, x1, min(y1 + tile_size, h), min(x1 + tile_size, w)))
    return tiles


def extract_tiles(img, tiles):
    """img: HxWx3 uint8 ndarray -> list of tile ndarrays"""
    return [img[y1:y2, x1:x2] for y1, x1, y2, x2 in tiles]


def block_aggregate(tile_scores, topk=3):
    """图像分数 = tile 分数 top-k 均值（v2 教训：top-k>>全局平均）。"""
    s = np.sort(np.asarray(tile_scores, dtype=np.float64))[::-1]
    k = max(1, min(topk, len(s)))
    return float(s[:k].mean())


def scoremap_multiscale_aggregate(score_map, scales=(1, 2, 4), weights=(0.6, 0.3, 0.1)):
    """特征图分数层多尺度 block 聚合（demo4 §6.2.1 精髓，零额外 DINO 前向）。

    score_map: (G,G) 每 patch 异常分数图。
    对每个尺度 s：将 (G,G) 划分成 s×s block，block 内先 max（保留局部缺陷峰值）
    再 mean（跨 block 归一），得到该尺度的图像分数；多尺度按 weights 加权。
    s=1 逐 patch（微缺陷），s=2/4 中大缺陷——兼顾不同缺陷尺寸，无额外前向。
    """
    g = np.asarray(score_map, dtype=np.float64)
    out = 0.0
    for s, w in zip(scales, weights):
        bs = max(1, min(g.shape[0], g.shape[1]) // s)
        if bs < 1:
            bs = 1
        h = g.shape[0] - g.shape[0] % bs
        wd = g.shape[1] - g.shape[1] % bs
        block = g[:h, :wd].reshape(h // bs, bs, wd // bs, bs)
        out += w * float(block.max(axis=(1, 3)).mean())
    return out


# ---------------------------------------------------------------------------
# A 块（AC-003/004/005）：确定性单级局部覆盖 + 坐标反解 + 覆盖/标签工具
# ---------------------------------------------------------------------------

def compute_local_tiles(image_size, tile_size=512, stride=448, roi=None):
    """AC-003：单级重叠确定性局部覆盖（与全局分数无关）。

    roi=(y1,x1,y2,x2) 时只覆盖该区域（裁到图像边界），末行/末列回移使最后一块
    贴齐 ROI 边界而非越界或留缝；tile_size 大于 ROI 时收缩为 ROI 本身。
    """
    h, w = image_size
    if roi is None:
        ry1, rx1, ry2, rx2 = 0, 0, h, w
    else:
        ry1, rx1 = max(0, int(roi[0])), max(0, int(roi[1]))
        ry2, rx2 = min(h, int(roi[2])), min(w, int(roi[3]))
    rh, rw = ry2 - ry1, rx2 - rx1
    if rh <= 0 or rw <= 0:
        return []
    stride = max(1, min(int(stride), int(tile_size)))
    th, tw = min(tile_size, rh), min(tile_size, rw)
    rows = max(1, math.ceil((rh - th) / stride) + 1)
    cols = max(1, math.ceil((rw - tw) / stride) + 1)
    tiles = []
    for r in range(rows):
        y1 = ry1 + min(r * stride, rh - th)
        for c in range(cols):
            x1 = rx1 + min(c * stride, rw - tw)
            t = (y1, x1, y1 + th, x1 + tw)
            if not tiles or t != tiles[-1]:
                tiles.append(t)
    return list(dict.fromkeys(tiles))


def tile_to_image_coords(tile, pt_yx, grid_hw):
    """AC-004：tile 内特征网格坐标 (gy,gx) → 原图像素坐标 (y,x)（取 cell 中心）。"""
    y1, x1, y2, x2 = tile
    gh, gw = grid_hw
    gy, gx = pt_yx
    return (y1 + (gy + 0.5) * (y2 - y1) / gh, x1 + (gx + 0.5) * (x2 - x1) / gw)


def tile_box_to_image(tile, box_xyxy, tile_hw):
    """tile 局部像素框 (x1,y1,x2,y2)（在 tile_hw 分辨率上）→ 原图框。"""
    y1, x1, y2, x2 = tile
    th, tw = tile_hw
    sy, sx = (y2 - y1) / th, (x2 - x1) / tw
    bx1, by1, bx2, by2 = box_xyxy
    return (x1 + bx1 * sx, y1 + by1 * sy, x1 + bx2 * sx, y1 + by2 * sy)


def paste_tile_heatmaps(hms, tiles, src_size, dst_size):
    """AC-004：每 tile 热力图按 tile 在原图中的位置贴回 dst 画布（重叠取 max）。

    hms: list[(G,G)]；tiles: 原图坐标 (src_size=(H,W)) 下的框；dst_size=(h,w)。
    """
    import cv2
    H, W = src_size
    h, w = dst_size
    sy, sx = h / max(H, 1), w / max(W, 1)
    canvas = np.zeros((h, w), dtype=np.float32)
    for hm, (y1, x1, y2, x2) in zip(hms, tiles):
        if hm is None:
            continue
        dy1, dx1 = int(round(y1 * sy)), int(round(x1 * sx))
        dy2, dx2 = max(dy1 + 1, int(round(y2 * sy))), max(dx1 + 1, int(round(x2 * sx)))
        dy2, dx2 = min(dy2, h), min(dx2, w)
        if dy2 <= dy1 or dx2 <= dx1:
            continue
        r = cv2.resize(np.asarray(hm, dtype=np.float32), (dx2 - dx1, dy2 - dy1),
                       interpolation=cv2.INTER_LINEAR)
        canvas[dy1:dy2, dx1:dx2] = np.maximum(canvas[dy1:dy2, dx1:dx2], r)
    return canvas


def _inter(t, b):
    y1, x1, y2, x2 = t
    bx1, by1, bx2, by2 = b
    ih = max(0, min(y2, by2) - max(y1, by1))
    iw = max(0, min(x2, bx2) - max(x1, bx1))
    return ih * iw


def coverage_check(tiles, boxes, context_px=0):
    """AC-003/004：缺陷框 (x1,y1,x2,y2) 覆盖检查。

    union_ok：框被 tile 并集完全覆盖（跨块缺陷允许由多块拼合）；
    whole_ok：存在单块完整包含"框 + context_px 上下文"（裁到图像内由调用方保证）。
    返回 list[dict]，每框一项。
    """
    out = []
    for b in boxes:
        bx1, by1, bx2, by2 = [int(round(v)) for v in b]
        bh, bw = max(0, by2 - by1), max(0, bx2 - bx1)
        if bh == 0 or bw == 0:
            out.append({"box": list(b), "union_ok": True, "whole_ok": True, "covered_frac": 1.0})
            continue
        m = np.zeros((bh, bw), dtype=bool)
        whole = False
        for (y1, x1, y2, x2) in tiles:
            iy1, ix1 = max(y1, by1), max(x1, bx1)
            iy2, ix2 = min(y2, by2), min(x2, bx2)
            if iy2 > iy1 and ix2 > ix1:
                m[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1] = True
            if (y1 <= by1 - context_px and x1 <= bx1 - context_px
                    and y2 >= by2 + context_px and x2 >= bx2 + context_px):
                whole = True
        frac = float(m.mean())
        out.append({"box": list(b), "union_ok": frac >= 1.0, "whole_ok": whole,
                    "covered_frac": frac})
    return out


# AC-005：面积占比标签阈值仅为候选值（未冻结），由配置覆盖
TILE_LABEL_POS_CANDIDATE = 0.25
TILE_LABEL_NEG_CANDIDATE = 0.05


def tile_labels_by_area(tiles, boxes, pos=TILE_LABEL_POS_CANDIDATE,
                        neg=TILE_LABEL_NEG_CANDIDATE):
    """AC-005：按缺陷框面积落入 tile 的占比打标签。

    占比 = 交集面积 / 缺陷框面积（取该 tile 内所有框的最大占比）。
    ≥pos → 1（正）；≤neg → 0（负）；中间 → -1（丢弃，不参与训练/校准）。
    """
    labels = []
    for t in tiles:
        best = 0.0
        for b in boxes:
            area = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
            best = max(best, _inter(t, b) / area)
        labels.append(1 if best >= pos else (0 if best <= neg else -1))
    return labels


def context_crop(image_size, box_xyxy, crop_size, context_px=0):
    """AC-005：以缺陷框为中心的裁切框（含上下文），尺寸 ≥ 框+2*context，回移到图内。

    返回 tile 格式 (y1,x1,y2,x2)。
    """
    h, w = image_size
    bx1, by1, bx2, by2 = box_xyxy
    need_h = int(math.ceil(by2 - by1)) + 2 * context_px
    need_w = int(math.ceil(bx2 - bx1)) + 2 * context_px
    ch, cw = min(h, max(crop_size, need_h)), min(w, max(crop_size, need_w))
    cy, cx = (by1 + by2) / 2.0, (bx1 + bx2) / 2.0
    y1 = int(round(cy - ch / 2.0))
    x1 = int(round(cx - cw / 2.0))
    y1 = max(0, min(y1, h - ch))
    x1 = max(0, min(x1, w - cw))
    return (y1, x1, y1 + ch, x1 + cw)


def tile_size_bounds(min_defect_px, grid=32, model_side=448, min_patches=2.0,
                     context_px=0, max_defect_px=None, min_footprint_px=None):
    """AC-004：块尺寸上下界（512 仅为初始候选，不默认 512→256）。

    下限（patch 足迹）：每个 patch 在原图上的足迹 tile/grid 不应小于
        min_footprint_px（缺省 1px=物理下限：再小即 patch 覆盖不足 1 像素、纯插值；
        业务值应由品类档案给出），且 tile 须容纳最大缺陷 + 2*context_px。
    上限（上下文稀释）：最小缺陷在 tile 内须占 ≥min_patches 个 patch，
        tile ≤ min_defect_px * grid / min_patches，再大缺陷被上下文淹没。
    冲突（下限 > 上限）→ global_only，交给全局分支，不强行切块。
    """
    if not min_defect_px or min_defect_px <= 0:
        return {"status": "profile_missing", "lo": None, "hi": None,
                "reason": "min_defect_px 未提供，不编默认值"}
    fp = float(min_footprint_px) if min_footprint_px else 1.0
    hi = int(math.floor(min_defect_px * grid / float(min_patches)))
    big = max_defect_px if max_defect_px else min_defect_px
    lo = int(math.ceil(max(fp * grid, big + 2 * context_px)))
    if lo > hi:
        return {"status": "global_only", "lo": lo, "hi": hi,
                "reason": f"足迹/容纳下界 {lo}px > 上下文稀释上界 {hi}px，交给全局分支"}
    return {"status": "ok", "lo": lo, "hi": hi, "reason": ""}
