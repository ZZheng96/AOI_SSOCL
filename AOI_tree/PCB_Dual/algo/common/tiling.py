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
        return [(r * th, c * tw, min((r + 1) * th, h), min((c + 1) * tw, w))
                for r in range(3) for c in range(3)]
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
        bs = max(1, g.shape[0] // s)
        if bs < 1:
            bs = 1
        h = g.shape[0] - g.shape[0] % bs
        wd = g.shape[1] - g.shape[1] % bs
        block = g[:h, :wd].reshape(h // bs, bs, wd // bs, bs)
        out += w * float(block.max(axis=(1, 3)).mean())
    return out
