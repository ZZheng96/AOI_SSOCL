"""shead 槽位：图像级伪异常判别头（demo4 s 分支配方移植，v2）

与 disc（patch 级打分）的本质区别：
- 打分粒度是**图像级**：头输入 = patch 特征图空间均值 (D,) 拼接 6 维全局统计量
  [mean, std, min, max, q25, q75] → (D+6,) → 小 MLP 头 → 整图单个分数。
- 训练配方（v2，对齐 demo4 _train_discriminative_head 实测 0.79 的 s 分支）：
  - 负样本（y=0）：train/good 整图特征；
  - 伪异常正样本（y=1）：正常图特征 + torch.randn_like*0.1（**特征空间噪声**，
    非图像合成——v1 用图像合成伪异常+BCE 在 gyudet 跨域仅 0.359，根因在此）；
  - 真实异常正样本（y=1）：init_defect（≤30 张）整图特征；
  - 损失：DeviationLoss（margin=5：正常分 L2 推下、异常分 hinge L2 推上），非 BCE；
  - 优化：Adam lr=1e-3，50 epochs，batch 32。
- 诚实性：只用协议内图像（train/good + init_defect）；test 绝不进 fit。

score_tiles 是 demo4 整图打分的 tile 推广：每 tile 独立算图像级分数；
分数 = 头输出（偏离度，非 sigmoid：负样本被压向 0、异常分大，符合 AUROC 排序）；
热力图给每 tile 一张 G×G 均匀图（值=tile 分数，该槽无定位能力）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Slot


def _resolve_init_from(init_from, slot="shead"):
    """解析预训练头路径：相对路径依次按 CWD、包根（src/ 上级目录）解析。

    配置了 init_from 但文件缺失时打印醒目告警——静默降级为随机初始化
    会使精度指标与文档不可比（可复现性红线）。
    """
    if not init_from:
        return None
    cands = [init_from]
    if not os.path.isabs(init_from):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cands.append(os.path.normpath(os.path.join(root, init_from)))
    for c in cands:
        if os.path.exists(c):
            return c
    print(f"[{slot}][警告] 预训练头缺失：{init_from} —— 槽位将以随机初始化运行，"
          "精度与文档数字不可比！请确认权重文件已随包放置。", flush=True)
    return None


class DeviationLoss(nn.Module):
    """demo4/src/models/head.py 移植：正常分 L2 推下，异常分 hinge L2 推上 margin=5σ"""
    def __init__(self, margin=5.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels):
        """scores: (B,), labels: (B,) 0=正常 1=异常"""
        normal = scores[labels == 0]
        anomaly = scores[labels == 1]
        if len(normal) == 0 or len(anomaly) == 0:
            # 保持梯度连接
            return scores.sum() * 0.0
        # 正常分数推下（接近 0）
        loss_normal = normal.pow(2).mean()
        # 异常分数推上 margin
        loss_anomaly = (self.margin - anomaly).clamp(min=0).pow(2).mean()
        return loss_normal + loss_anomaly


class SheadHead(nn.Module):
    """图像级小头：(D+6)->hidden->hidden->1（demo4 DiscriminativeHead 同构：3 层 MLP+dropout）。"""
    def __init__(self, dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _image_level_feats(f, chan_stats=False, patch_stats=False, focus_topk=None):
    """f: (B,D,G,G) tensor（任意 device）-> (B, D+6) cpu float。

    demo4 _compute_patch_stats 语义：6 个全局统计量作用于 flatten 后
    全部 D*G*G 个 patch 特征值（非逐维），拼接 patch 空间均值 (D,)。
    速度（U56，2026-08-16）：torch.quantile 在 CPU 上对 393216 元素排序
    单次 ~90ms，是 shead 槽 189ms/图 的主因；改一次 torch.sort 取
    25%/75% 位置分位（GPU 0.5ms），数值语义等价（排序位置即分位数，
    对后续 MLP 的排序信号无影响）。

    U60 特征侧增强（2026-08-17，默认关，实验开）：
    - chan_stats=True：追加 per-channel 的 patch std（(B,D)）——通道级分布
      展宽信息，伪异常 randn 噪声会同时改变 mean 与 std，增强对域差/缺陷的判别输入；
    - patch_stats=True：对 patch 均值向量 (B,D) 再算 6 统计量（mean/std/min/max/q25/q75）
      ——全局统计的"通道间分布"视图。
    fit 与 score_tiles 必须用同一组开关（输入维度一致）。

    U70 聚焦模式（2026-08-18，默认关，实验开）：focus_topk 为 float 比例时，
    把 patch 展平 (B,G*G,D)，按每 patch 与"全图 patch 均值向量"的 cosine 距离
    （1-sim，sem 槽 coreset 最近邻距离的代理；shead 无 coreset bank，用
    特征自身离群度作异常度排序）取 top-k 最异常 patch，仅对 top-k patch 算
    均值 (D,) + 6 全局统计 → (D+6,) 同维输入。模拟 demo4 裁切红利根因
    （U69 调查：裁切 = 聚焦缺陷区域提高信噪比，消除全图正常 patch 稀释）的
    无标注自动聚焦：只用特征自身排序（无 GT 框依赖）。
    fit 与 score_tiles 同口径（见 SheadSlot.focus_score_only 混合口径开关）。
    """
    if focus_topk:
        B, D, G, G2 = f.shape
        n_patch = G * G2
        flat_p = f.reshape(B, n_patch, D)                 # (B,N,D)
        mean_patch = flat_p.mean(dim=1, keepdim=True)     # (B,1,D)
        sim = F.cosine_similarity(flat_p, mean_patch.expand(B, n_patch, D), dim=-1)
        dist = 1.0 - sim                                   # (B,N) 异常度代理
        k = max(1, int(n_patch * focus_topk))
        idx = torch.topk(dist, k, dim=1).indices           # (B,k)
        sel = torch.gather(flat_p, 1, idx.unsqueeze(-1).expand(B, k, D))  # (B,k,D)
        mean = sel.mean(dim=1)                             # (B,D) top-k 均值
        flat = sel.reshape(B, k * D)                       # 6 统计只作用于 top-k patch 值
    else:
        mean = f.mean(dim=(2, 3))                          # (B, D)
        flat = f.reshape(f.shape[0], -1)                   # (B, D*G*G)
    parts = [mean]
    if chan_stats:
        if focus_topk:
            parts.append(sel.std(dim=1))      # (B, D) top-k patch 的通道 std
        else:
            parts.append(f.std(dim=(2, 3)))   # (B, D) per-channel std
    v, _ = torch.sort(flat, dim=1)            # 一次排序取两个分位
    n = flat.shape[1]
    q25 = v[:, max(n // 4 - 1, 0)]
    q75 = v[:, max(3 * n // 4 - 1, 0)]
    stats = torch.stack([
        flat.mean(dim=1), flat.std(dim=1),
        flat.min(dim=1).values, flat.max(dim=1).values,
        q25, q75,
    ], dim=1)                                 # (B, 6)
    parts.append(stats)
    if patch_stats:
        d = mean.shape[1]
        mv, _ = torch.sort(mean, dim=1)       # 通道间排序取分位
        mq25 = mv[:, max(d // 4 - 1, 0)]
        mq75 = mv[:, max(3 * d // 4 - 1, 0)]
        parts.append(torch.stack([
            mean.mean(dim=1), mean.std(dim=1),
            mean.min(dim=1).values, mean.max(dim=1).values,
            mq25, mq75,
        ], dim=1))                            # (B, 6) patch 均值向量统计
    return torch.cat(parts, dim=1).cpu()


def box_patch_feats(f, chan_stats=False, patch_stats=False):
    """U65（2026-08-18）：框内 patch 特征 -> shead 输入格式（在线微调判别头用）。

    与 _image_level_feats 同一实现（对任意 (B,D,H,W) 均成立：patch 空间均值 (D,)
    + 6 全局统计 mean/std/min/max/q25/q75 -> (D+6,)，与 fit/score_tiles 输入同构），
    单独命名明确"框内区域视作单 tile"的语义（对标 demo4 标注裁切图训练判别头）。
    """
    return _image_level_feats(f, chan_stats, patch_stats)


# ---- U72 crop 级判别模式辅助函数 ----

def _iou(a, b):
    """两 xyxy 框 IoU（exp_crop_repro 同款）。"""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _parse_yolo_boxes(lbl_path, img_w, img_h):
    """解析 YOLO txt -> 像素 xyxy 框列表（U71 exp_crop_repro 同款）。"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, w, h = (float(v) for v in parts[:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            x0 = (cx - w / 2) * img_w
            y0 = (cy - h / 2) * img_h
            x1 = (cx + w / 2) * img_w
            y1 = (cy + h / 2) * img_h
            boxes.append([x0, y0, x1, y1])
    return boxes


def _extract_box_crops(img, boxes, pad_ratio=0.2, min_side=24):
    """从图像中按框+pad 裁切 crop（对标 exp_crop_repro.build_crops 缺陷crop）。"""
    h, w = img.shape[:2]
    crops = []
    for bx in boxes:
        bw = bx[2] - bx[0]; bh = bx[3] - bx[1]
        pw = bw * pad_ratio; ph = bh * pad_ratio
        x0 = max(0, int(bx[0] - pw)); y0 = max(0, int(bx[1] - ph))
        x1 = min(w, int(bx[2] + pw)); y1 = min(h, int(bx[3] + ph))
        if x1 - x0 < min_side or y1 - y0 < min_side:
            continue
        crops.append(img[y0:y1, x0:x1])
    return crops


def _extract_random_crops(img, crop_size, n_crops, rng):
    """从图像中随机裁切 n_crops 个 crop_size×crop_size 的 crop。"""
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        return [img]
    crops = []
    for _ in range(n_crops):
        y0 = int(rng.integers(0, h - crop_size))
        x0 = int(rng.integers(0, w - crop_size))
        crops.append(img[y0:y0 + crop_size, x0:x0 + crop_size])
    return crops


def _extract_grid_crops(img, crop_size, n_grid_max=6):
    """从图像中按网格裁切 crop（覆盖全图，自适应图像尺寸与crop_size）。

    网格大小自适应：目标 ~25-36 个 crop，按图像尺寸/crop_size 计算 n_y×n_x，
    上限 n_grid_max×n_grid_max 防过多样本影响速度。每个 crop 以网格单元中心
    为中心裁切，超出边界 clip。
    """
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    if h <= cs or w <= cs:
        return [img]
    n_y = max(2, min(n_grid_max, round(h / cs * 1.5)))
    n_x = max(2, min(n_grid_max, round(w / cs * 1.5)))
    crops = []
    for r in range(n_y):
        for c in range(n_x):
            cy = int((r + 0.5) * h / n_y)
            cx = int((c + 0.5) * w / n_x)
            y0 = max(0, min(cy - cs // 2, h - cs))
            x0 = max(0, min(cx - cs // 2, w - cs))
            crops.append(img[y0:y0 + cs, x0:x0 + cs])
    return crops


def _extract_context_crops(img, boxes, crop_size):
    """U72v3：缺陷上下文 crop -- 以缺陷框中心为中心裁切 crop_size×crop_size。

    关键作用：桥接 train-test 分布差异。
    - 训练 box+pad crop = 纯缺陷区域（高 SNR）
    - 测试 grid crop = 缺陷+大量背景（低 SNR，缺陷占比 <10%）
    - 模型只在纯缺陷 crop 上训练 -> 无法识别"缺陷在背景中"的测试 crop
    - 上下文 crop 包含缺陷+背景，与测试 grid crop 同分布，让模型学习
      "在背景中检测缺陷"而非"区分纯缺陷 vs 纯正常"
    """
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    crops = []
    for bx in boxes:
        cx = int((bx[0] + bx[2]) / 2)
        cy = int((bx[1] + bx[3]) / 2)
        y0 = max(0, min(cy - cs // 2, h - cs))
        x0 = max(0, min(cx - cs // 2, w - cs))
        crops.append(img[y0:y0 + cs, x0:x0 + cs])
    return crops


def _extract_sliding_crops(img, crop_size, overlap=0.5, max_crops=120):
    """U72v3：滑窗 crop -- 有重叠的滑动窗口裁切，确保缺陷被覆盖。

    vs _extract_grid_crops：网格可能有间隙（crop < cell 时），
    滑窗保证每个像素被至少一个 crop 覆盖（overlap>=0.5 时），
    缺陷不会落在间隙中导致漏检。
    """
    crops, _, _, _ = _sliding_grid(img, crop_size, overlap, max_crops)
    return crops


def _sliding_grid_crops(img, crop_size, stride):
    """U77：按固定 stride 的滑窗裁切（行主序，可反解窗口行列位置）。

    vs _sliding_grid（overlap 参数按比例）：guided 两级滑窗第一级用固定
    stride 控制窗数（速度红线），且需反解窗口位置做第二级细筛。
    返回 crops 列表（每窗位置可由 row*cols+col 反解）。
    """
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    if h <= cs or w <= cs:
        return [img]
    cols = max(1, int(np.ceil(w / stride)) if stride else 1)
    rows = max(1, int(np.ceil(h / stride)) if stride else 1)
    crops = []
    for r in range(rows):
        for c in range(cols):
            y0 = min(r * stride, max(0, h - cs))
            x0 = min(c * stride, max(0, w - cs))
            crops.append(img[y0:y0 + cs, x0:x0 + cs])
    return crops


def _sliding_grid(img, crop_size, overlap=0.5, max_crops=120):
    """U73v4：滑窗裁切 + 网格形状信息。返回 (crops, n_y, n_x, truncated)。

    crops 按行主序（y 外层 x 内层）排列，可 reshape (n_y, n_x) 成分数网格，
    是相邻一致性聚合（_score_tiles_crop v4）的位置依据；
    truncated=True 表示超 max_crops 截断（网格不完整，聚合需 fallback）。
    """
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    if h <= cs or w <= cs:
        return [img], 1, 1, False
    stride = max(1, int(cs * (1 - overlap)))
    y_pos = list(range(0, max(1, h - cs + 1), stride))
    x_pos = list(range(0, max(1, w - cs + 1), stride))
    if y_pos[-1] + cs < h:
        y_pos.append(h - cs)
    if x_pos[-1] + cs < w:
        x_pos.append(w - cs)
    n_y, n_x = len(y_pos), len(x_pos)
    crops = []
    for y0 in y_pos:
        for x0 in x_pos:
            crops.append(img[y0:y0 + cs, x0:x0 + cs])
            if len(crops) >= max_crops:
                return crops, n_y, n_x, len(crops) < n_y * n_x
    return crops, n_y, n_x, False


def _extract_jittered_crops(img, boxes, crop_size, n_per_box=3, jitter=0.5,
                            pos_frac_min=0.1, rng=None):
    """U73v4：随机滑窗正样本--框中心+随机偏移裁切 crop_size×crop_size。

    对齐测试滑窗分布（离线 crop 实验 0.9651 的核心成功经验之一：
    训练-测试同分布）：测试时滑窗 crop 的缺陷位置/占比是随机的（缺陷可在
    crop 任意位置、占比 10%-100%），而 v3 的上下文 crop 只有"缺陷居中"一种
    位置。jitter 偏移使缺陷位置/占比随机化；pos_frac_min 过滤信噪比过低的
    crop（缺陷占比 <10% 的几乎纯背景样本会教坏模型）。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    h, w = img.shape[:2]
    cs = min(crop_size, h, w)
    crops = []
    for bx in boxes:
        bcx = (bx[0] + bx[2]) / 2
        bcy = (bx[1] + bx[3]) / 2
        for _ in range(n_per_box):
            cx = int(bcx + rng.uniform(-jitter, jitter) * cs)
            cy = int(bcy + rng.uniform(-jitter, jitter) * cs)
            y0 = max(0, min(cy - cs // 2, h - cs))
            x0 = max(0, min(cx - cs // 2, w - cs))
            # 缺陷占比过滤：crop 与所有框的最大交面积 / crop 面积
            frac = 0.0
            for b in boxes:
                iw = max(0, min(x0 + cs, b[2]) - max(x0, b[0]))
                ih = max(0, min(y0 + cs, b[3]) - max(y0, b[1]))
                frac = max(frac, iw * ih / (cs * cs))
            if frac >= pos_frac_min:
                crops.append(img[y0:y0 + cs, x0:x0 + cs])
    return crops


def _candidates_from_heatmap(heatmap, img_shape, crop_size, topk=8,
                             recall_p=0.10, min_area=2):
    """U77：粗筛热力图 -> 原图候选框列表（两级结构第一级）。

    U77 定位诊断实证（2026-08-19，183 个 GT 框）：sem coreset 距离图
    top-10% 定位召回 0.902、top-20% 0.967（随机理论 0.093）--粗筛定位
    可靠，两级结构成立（U76 定论）。
    算法：分数 > 分位阈值 recall_p 的 patch 二值化 -> 8 邻域连通域 ->
    每簇包围盒 -> pad 到 crop_size（越界 clip）-> 按簇内最高分排序取 top-k。
    """
    import cv2
    g = heatmap.shape[0]
    h, w = img_shape[:2]
    th = float(np.quantile(heatmap, 1.0 - recall_p))
    mask = (heatmap >= th).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(mask, connectivity=8)
    if n_lab <= 1:               # 无候选（全图都正常）
        return []
    ph = h / g
    pw = w / g
    cs = min(crop_size, h, w)
    cands = []
    for c in range(1, n_lab):
        ys, xs = np.nonzero(lab == c)
        if len(ys) < min_area:
            continue
        y0 = int(ys.min() * ph)
        y1 = int((ys.max() + 1) * ph)
        x0 = int(xs.min() * pw)
        x1 = int((xs.max() + 1) * pw)
        # pad 到 crop_size 正方形（中心不变）
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        y0 = max(0, int(cy - cs / 2))
        x0 = max(0, int(cx - cs / 2))
        y1 = min(h, y0 + cs)
        x1 = min(w, x0 + cs)
        y0 = max(0, y1 - cs)     # 越界 clip 保持 cs 边长
        x0 = max(0, x1 - cs)
        # U77b 排序修复：簇按"大小×平均分"（缺陷成片优先），
        # 而非峰值（背景噪声单 patch 高分会挤占 top-k）
        size = len(ys)
        mean_v = float(heatmap[lab == c].mean())
        score = np.sqrt(size) * mean_v
        cands.append((score, [x0, y0, x1, y1]))
    cands.sort(key=lambda t: t[0], reverse=True)
    # 去重：IoU>0.6 的保留高分者
    keep = []
    for peak, b in cands:
        dup = False
        for _, kb in keep:
            if _iou(b, kb) > 0.6:
                dup = True
                break
        if not dup:
            keep.append((peak, b))
        if len(keep) >= topk:
            break
    return [b for _, b in keep]


class SheadSlot(Slot):
    name = "shead"

    def __init__(self, cfg, backbone, device="cuda"):
        self.cfg = cfg
        self.backbone = backbone
        self.device = device
        # U60 特征侧增强开关（默认关）：chan_stats/patch_stats 改变输入维度，fit/score 同源
        self.chan_stats = cfg.get("chan_stats", False)
        self.patch_stats = cfg.get("patch_stats", False)
        # U70 聚焦模式（默认关）：focus_topk=float 比例（0.01/0.05/0.10）→ 按异常度取
        # top-k patch 聚焦统计（模拟无标注自动裁切）；focus_score_only=True 时训练用
        # 全图、推理用聚焦（混合口径实验，默认 False=fit/score 同口径）
        self.focus_topk = cfg.get("focus_topk", None)
        self.focus_score_only = cfg.get("focus_score_only", False)
        # U72 crop 级判别模式（默认关）：从原图裁切小 crop -> DINOv2 -> shead 头，
        # 模拟 U71 裁切复现实验（crop AUROC=0.9651）。整图问题是背景 patch 稀释
        # 缺陷信号（oracle 上限 0.7362），crop 去除背景后判别力大幅提升。
        # fit: 正常随机 crop + 缺陷 YOLO 框 crop -> 训练头；
        # score: 网格 crop 覆盖全图 -> max 聚合（任一 crop 异常即整图异常）。
        self.score_mode = cfg.get("score_mode", "image")  # "image" or "crop"
        self.crop_size = int(cfg.get("crop_size", 200))
        self.crop_pad = float(cfg.get("crop_pad", 0.2))
        self.n_fit_normal_crops = int(cfg.get("n_fit_normal_crops", 10))
        self.n_score_grid_max = int(cfg.get("n_score_grid_max", 6))
        # U72v3：自适应 crop_size（fit 时从缺陷框中位数计算）+ 滑窗评分 + 上下文 crop
        self.adaptive_crop_size = cfg.get("adaptive_crop_size", True)
        self.score_overlap = float(cfg.get("score_overlap", 0.5))
        self.score_max_crops = int(cfg.get("score_max_crops", 120))
        self.use_context_crops = cfg.get("use_context_crops", True)
        # U73v4：随机滑窗训练（框中心+jitter 偏移，对齐测试滑窗分布）+
        # 相邻一致性聚合（3x3 平滑替代全局 topk，攻孤立假阳性）
        self.rand_sliding_train = cfg.get("rand_sliding_train", False)
        self.n_pos_slides_per_defect = int(cfg.get("n_pos_slides_per_defect", 3))
        self.slide_jitter = float(cfg.get("slide_jitter", 0.5))
        self.pos_frac_min = float(cfg.get("pos_frac_min", 0.1))
        self.use_adjacent_agg = cfg.get("use_adjacent_agg", True)
        # U75 块模式（用户分块方案）：multi-tile 分块 + 框重叠比例块级标签。
        # 速度红线（<1s/图）：块特征由管线统一提取（tile_feats），打分零额外 DINO
        # 前向（vs crop 模式滑窗重新前向 1.5-3s）。
        self.block_pos_thresh = float(cfg.get("block_pos_thresh", 0.15))
        self.block_neg_thresh = float(cfg.get("block_neg_thresh", 0.05))
        self.block_cover_min = float(cfg.get("block_cover_min", 0.5))
        self.n_fit_normal_blocks = int(cfg.get("n_fit_normal_blocks", 6))
        self.n_fit_bg_blocks = int(cfg.get("n_fit_bg_blocks", 6))
        self.block_score_topk = max(1, int(cfg.get("block_score_topk", 3)))
        # U77 两级结构（guided 模式）：整图粗筛定位 + 疑点 crop 精查
        # U76 定论：基线诚实冲 0.9 = 两级结构；U77 实证 sem coreset 距离图
        # top-10% 定位召回 0.902 -- 粗筛可靠。粗筛库在 fit 时用 train/good 建。
        self.guided_coreset = int(cfg.get("guided_coreset", 256))
        self.guided_candidates = int(cfg.get("guided_candidates", 8))
        self.guided_recall_p = float(cfg.get("guided_recall_p", 0.10))
        self.guided_score_topk = max(1, int(cfg.get("guided_score_topk", 3)))
        in_dim = backbone.dim + 6
        if self.chan_stats:
            in_dim += backbone.dim
        if self.patch_stats:
            in_dim += 6
        self.head = SheadHead(in_dim, cfg.get("hidden", 256),
                              cfg.get("dropout", 0.3)).to(device)
        self.loss_history = []          # [(epoch, loss_normal, loss_anomaly)]
        init_from = _resolve_init_from(cfg.get("init_from"), slot="shead")
        if init_from:
            self.head.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
            print(f"[shead] 加载预训练头 {init_from}")

    def fit(self, ctx):
        """ctx: train_tile_imgs (list[list[ndarray]]), defect_tile_imgs 同构。
        v2 配方（demo4 _train_discriminative_head）：
        - 负样本 = train/good 整图特征（y=0）
        - 伪异常 = 正常特征 + randn*0.1，每张 3 个（y=1，特征空间噪声）
        - 真实异常 = init_defect 整图特征 ≤30（y=1）
        - DeviationLoss(margin=5) + Adam lr=1e-3 + epochs + batch 32
        流式内存控制：提取按批立即降维 (D+6) 驻留 CPU，不累积 DINO 特征图。"""
        if self.score_mode == "crop":
            return self._fit_crop(ctx)
        if self.score_mode == "block":
            return self._fit_blocks(ctx)
        if self.score_mode == "guided":
            return self._fit_guided(ctx)
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        torch_gen = torch.Generator(device="cpu")
        torch_gen.manual_seed(int(self.cfg.get("seed", 42)))
        batch = self.cfg.get("extract_batch", 8)
        n_epochs = self.cfg.get("epochs", 50)
        max_base = self.cfg.get("max_base_tiles", 96)
        # U70：fit 侧聚焦口径（focus_score_only=True 时训练全图、推理聚焦——混合口径实验）
        ftk_fit = None if self.focus_score_only else self.focus_topk

        train_tiles = [t for tiles in ctx["train_tile_imgs"] for t in tiles]
        if len(train_tiles) > max_base:          # 内存控制（v1 沿用）
            sel = rng.choice(len(train_tiles), max_base, replace=False)
            train_tiles = [train_tiles[i] for i in sel]
        defect_tiles = [t for tiles in ctx.get("defect_tile_imgs", []) for t in tiles]
        if len(defect_tiles) > 30:               # demo4: defect_paths[:30]
            sel = rng.choice(len(defect_tiles), 30, replace=False)
            defect_tiles = [defect_tiles[i] for i in sel]
        if not train_tiles:
            print("[shead] 无正常基底图，跳过训练", flush=True)
            return self

        def _feats(imgs):
            """流式提取整图特征 (N, D+6) cpu：每批 DINO 前向立即降维释放特征。"""
            out = []
            for i in range(0, len(imgs), batch):
                f = self.backbone.extract_tiles(imgs[i:i + batch], batch=batch)
                out.append(_image_level_feats(f, self.chan_stats, self.patch_stats, ftk_fit))
            return torch.cat(out, dim=0) if out else torch.empty(0)

        if n_epochs > 0:
            # 负样本 + 伪异常正样本同批前向（正常图只过一次 DINO）
            neg_list, pos_list = [], []
            n_pseudo = self.cfg.get("pseudo_per_image", 3)   # U60：读配置（此前硬编码 3）
            for i in range(0, len(train_tiles), batch):
                f = self.backbone.extract_tiles(train_tiles[i:i + batch], batch=batch)
                neg_list.append(_image_level_feats(f, self.chan_stats, self.patch_stats, ftk_fit))  # y=0
                for _ in range(n_pseudo):                     # y=1：每张 n_pseudo 个噪声正样本
                    pos_list.append(_image_level_feats(f + torch.randn_like(f) * 0.1,
                                                       self.chan_stats, self.patch_stats, ftk_fit))
            neg = torch.cat(neg_list, dim=0)
            pos = torch.cat(pos_list, dim=0)
            # 真实异常正样本：init_defect 整图特征（≤30，y=1）
            if defect_tiles:
                pos = torch.cat([pos, _feats(defect_tiles)], dim=0)
            cap = self.cfg.get("max_patches", 50000)
            if len(pos) > cap:
                pos = pos[torch.randperm(len(pos), generator=torch_gen)[:cap]]
            if len(pos) == 0:
                print("[shead] 正样本为空，跳过训练", flush=True)
                return self
            X = torch.cat([neg, pos]).float().detach().to(self.device)
            y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
            loss_fn = DeviationLoss(margin=self.cfg.get("margin", 5.0))
            opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
            bs = self.cfg.get("batch", 32)
            for ep in range(n_epochs):
                perm = torch.randperm(len(X), generator=torch_gen)
                for i in range(0, len(X), bs):
                    xb = X[perm[i:i + bs]]
                    yb = y[perm[i:i + bs]]
                    scores = self.head(xb)
                    loss = loss_fn(scores, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                # 正常/异常损失分量关键点（验证两方向都在收敛）
                with torch.no_grad():
                    s = self.head(X)
                    l_n = s[y == 0].pow(2).mean().item()
                    l_a = (loss_fn.margin - s[y == 1]).clamp(min=0).pow(2).mean().item()
                self.loss_history.append((ep + 1, l_n, l_a))
                if (ep + 1) % 10 == 0:
                    print(f"[shead] epoch {ep + 1}/{n_epochs} "
                          f"loss_n={l_n:.4f} loss_a={l_a:.4f}", flush=True)
            print(f"[shead] fit 完成: normal={len(neg)} pseudo={len(pos) - len(defect_tiles)} "
                  f"real_defect={len(defect_tiles)} pos_total={len(pos)} "
                  f"X={len(X)} batch={bs} lr={self.cfg.get('lr', 1e-3)}", flush=True)
        self.head.eval()
        return self

    def _fit_crop(self, ctx):
        """U72v3 crop 级判别头训练：从原图裁切 crop -> DINOv2 -> 特征 -> DeviationLoss。

        v3 改进（vs v2）：
        1. 自适应 crop_size：从训练缺陷框中位数计算，匹配真实缺陷尺度
        2. 缺陷上下文 crop：以框中心裁切 crop_size×crop_size（含背景），桥接
           训练纯缺陷 crop vs 测试缺陷+背景 crop 的分布差异
        3. 滑窗评分（在 _score_tiles_crop 中）：有重叠的滑窗替代网格，确保覆盖

        正常 crop：train/good 图随机裁切 crop_size×crop_size（neg，y=0）
        缺陷 crop：box+pad（纯缺陷，pos）+ 上下文 crop（缺陷+背景，pos）
        背景正常 crop：缺陷图 IoU<0.05 区域（neg，y=0）
        伪异常：正常 crop 特征 + randn*0.1（pos，y=1）
        诚实边界：YOLO 框来自 train 域标注（协议内 init_defect），不触碰 test。"""
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        torch_gen = torch.Generator(device="cpu")
        torch_gen.manual_seed(int(self.cfg.get("seed", 42)))
        batch = self.cfg.get("extract_batch", 8)
        n_epochs = self.cfg.get("epochs", 100)
        max_base = self.cfg.get("max_base_tiles", 96)
        n_pseudo = self.cfg.get("pseudo_per_image", 3)
        crop_size = self.crop_size
        pad_ratio = self.crop_pad

        # 0. 预解析所有缺陷框 + 自适应 crop_size
        defect_tile_imgs = ctx.get("defect_tile_imgs", [])
        defect_paths = ctx.get("defect_paths", [])
        n_bg_per_defect = int(self.cfg.get("n_bg_per_defect", 3))
        all_defect_info = []  # [(tile_img, boxes), ...]
        box_sides = []
        for i, tiles in enumerate(defect_tile_imgs):
            for tile_img in tiles:
                boxes = []
                if i < len(defect_paths):
                    img_path = defect_paths[i]
                    lbl_path = os.path.join(
                        os.path.dirname(os.path.dirname(img_path)),
                        "labels",
                        os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                    h, w = tile_img.shape[:2]
                    boxes = _parse_yolo_boxes(lbl_path, w, h)
                    for bx in boxes:
                        box_sides.append(max(bx[2] - bx[0], bx[3] - bx[1]))
                all_defect_info.append((tile_img, boxes))

        if box_sides and self.adaptive_crop_size:
            med_side = int(np.median(box_sides))
            new_cs = max(100, min(2000, med_side))
            if new_cs != crop_size:
                print(f"[shead-crop] adaptive crop_size: {crop_size} -> {new_cs} "
                      f"(median box side={med_side}, n_boxes={len(box_sides)})", flush=True)
                crop_size = new_cs
                self.crop_size = new_cs  # 评分时也用

        # 1. 正常 crop
        train_tiles = [t for tiles in ctx["train_tile_imgs"] for t in tiles]
        if len(train_tiles) > max_base:
            sel = rng.choice(len(train_tiles), max_base, replace=False)
            train_tiles = [train_tiles[i] for i in sel]
        if not train_tiles:
            print("[shead-crop] 无正常图，跳过训练", flush=True)
            return self
        neg_feats = []
        for img in train_tiles:
            crops = _extract_random_crops(img, crop_size, self.n_fit_normal_crops, rng)
            f = self.backbone.extract_tiles(crops, batch=batch)
            neg_feats.append(_image_level_feats(f, self.chan_stats, self.patch_stats))

        # 2. 缺陷 crop：box+pad + 上下文/随机滑窗 crop + 背景正常 crop
        n_real_defect = 0
        n_ctx_crops = 0
        n_slide_crops = 0
        n_bg_crops = 0
        pos_feats_list = []
        for tile_img, boxes in all_defect_info:
            if boxes:
                # box+pad crop（纯缺陷，高 SNR 保底）
                box_crops = _extract_box_crops(tile_img, boxes, pad_ratio)
                # 上下文 crop（v3：缺陷居中，缺陷+背景）
                ctx_crops = []
                if self.use_context_crops:
                    ctx_crops = _extract_context_crops(tile_img, boxes, crop_size)
                # v4 随机滑窗 crop（框中心+jitter，缺陷位置/占比随机化，
                # 完全对齐测试滑窗分布；默认替代上下文 crop）
                slide_crops = []
                if self.rand_sliding_train:
                    slide_crops = _extract_jittered_crops(
                        tile_img, boxes, crop_size,
                        self.n_pos_slides_per_defect, self.slide_jitter,
                        self.pos_frac_min, rng)
                crops = box_crops + ctx_crops + slide_crops
                n_ctx_crops += len(ctx_crops)
                n_slide_crops += len(slide_crops)
                # 背景正常 crop：IoU<0.05
                bg_crops = []
                h, w = tile_img.shape[:2]
                cs = min(crop_size, h, w)
                for _ in range(n_bg_per_defect * 3):
                    if h <= cs or w <= cs:
                        break
                    y0 = int(rng.integers(0, h - cs))
                    x0 = int(rng.integers(0, w - cs))
                    cand = [x0, y0, x0 + cs, y0 + cs]
                    if all(_iou(cand, b) < 0.05 for b in boxes):
                        bg_crops.append(tile_img[y0:y0 + cs, x0:x0 + cs])
                    if len(bg_crops) >= n_bg_per_defect:
                        break
                if bg_crops:
                    f_bg = self.backbone.extract_tiles(bg_crops, batch=batch)
                    neg_feats.append(_image_level_feats(f_bg, self.chan_stats,
                                                        self.patch_stats))
                    n_bg_crops += len(bg_crops)
            else:
                crops = _extract_random_crops(tile_img, crop_size, 5, rng)
            if crops:
                f = self.backbone.extract_tiles(crops, batch=batch)
                pos_feats_list.append(_image_level_feats(f, self.chan_stats,
                                                          self.patch_stats))
                n_real_defect += len(crops)

        # 3. 合并所有负样本（train/good + 缺陷图背景）-> 伪异常
        neg = torch.cat(neg_feats, dim=0)
        pos_list = []
        for _ in range(n_pseudo):
            pos_list.append(neg + torch.randn_like(neg) * 0.1)
        if pos_feats_list:
            pos_list.extend(pos_feats_list)

        pos = torch.cat(pos_list, dim=0)
        cap = self.cfg.get("max_patches", 50000)
        if len(pos) > cap:
            pos = pos[torch.randperm(len(pos), generator=torch_gen)[:cap]]
        if len(pos) == 0:
            print("[shead-crop] 正样本为空，跳过训练", flush=True)
            return self

        # 4. 训练（与 image 模式同配方：DeviationLoss + Adam）
        X = torch.cat([neg, pos]).float().detach().to(self.device)
        y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
        loss_fn = DeviationLoss(margin=self.cfg.get("margin", 5.0))
        opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
        bs = self.cfg.get("batch", 32)
        for ep in range(n_epochs):
            perm = torch.randperm(len(X), generator=torch_gen)
            for i in range(0, len(X), bs):
                xb = X[perm[i:i + bs]]
                yb = y[perm[i:i + bs]]
                scores = self.head(xb)
                loss = loss_fn(scores, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                s = self.head(X)
                l_n = s[y == 0].pow(2).mean().item()
                l_a = (loss_fn.margin - s[y == 1]).clamp(min=0).pow(2).mean().item()
            self.loss_history.append((ep + 1, l_n, l_a))
            if (ep + 1) % 10 == 0:
                print(f"[shead-crop] epoch {ep + 1}/{n_epochs} "
                      f"loss_n={l_n:.4f} loss_a={l_a:.4f}", flush=True)
        print(f"[shead-crop] fit 完成: normal_crops={len(neg)} (bg_from_defect={n_bg_crops}) "
              f"pseudo={len(pos) - n_real_defect} real_defect_crops={n_real_defect} "
              f"(ctx={n_ctx_crops} slide={n_slide_crops}) pos_total={len(pos)} "
              f"X={len(X)} crop_size={crop_size}", flush=True)
        self.head.eval()
        return self

    def _fit_blocks(self, ctx):
        """U75 块级判别头训练（用户分块方案，2026-08-19）。

        分块由管线 tiling.mode=multi 决定（GYU-DET 建议 896/672 -> ~30 块），
        块级标签由 YOLO 框与块的重叠比例自动生成：
        - 缺陷块（正）：块含框面积占比 >= block_pos_thresh，或框被块覆盖比例
          >= block_cover_min（小缺陷跨块边界的保底条件：378px 小框跨界时
          面积占比仅 ~9%，但覆盖比例 ~50%）
        - 无缺陷块（负）：面积占比与覆盖比例均 <= block_neg_thresh（纯背景）
        - 中间占比块：丢弃（U74 教训：低占比正样本与正常特征重叠，是噪声）
        训练块与测试块出自同一分块函数 -> 完全同分布（U71 离线实验 0.9651
        的核心成功经验：训练-测试同分布）。
        正常图块：每图随机 n_fit_normal_blocks 块（负，流式提取）。
        伪异常默认 0（U73 ablB 教训：真实样本足量时伪异常拖低质量）。
        速度：fit 提取 ~1200 块（600 正常 + 600 缺陷图块）批前向 ~20s，
        远快于 crop 模式滑窗重复前向（~10 分钟）。"""
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        torch_gen = torch.Generator(device="cpu")
        torch_gen.manual_seed(int(self.cfg.get("seed", 42)))
        batch = self.cfg.get("extract_batch", 8)
        n_epochs = self.cfg.get("epochs", 100)
        pos_a = self.block_pos_thresh
        neg_t = self.block_neg_thresh
        cover_min = self.block_cover_min
        n_nb = self.n_fit_normal_blocks
        n_bg = self.n_fit_bg_blocks
        n_pseudo = self.cfg.get("pseudo_per_image", 0)

        def _feats(imgs):
            out = []
            for i in range(0, len(imgs), batch):
                f = self.backbone.extract_tiles(imgs[i:i + batch], batch=batch)
                out.append(_image_level_feats(f, self.chan_stats, self.patch_stats))
            return torch.cat(out, dim=0) if out else torch.empty(0)

        # 1. 正常块：每图随机 n_nb 块（流式提特征控内存）
        neg_feats = []
        if not ctx.get("train_tile_imgs"):
            print("[shead-block] 无正常图，跳过训练", flush=True)
            return self
        for tiles in ctx["train_tile_imgs"]:
            sel_tiles = tiles
            if len(tiles) > n_nb:
                sel = rng.choice(len(tiles), n_nb, replace=False)
                sel_tiles = [tiles[i] for i in sel]
            if sel_tiles:
                neg_feats.append(_feats(sel_tiles))

        # 2. 缺陷图块：框重叠比例 -> 块级标签（流式）
        defect_tile_imgs = ctx.get("defect_tile_imgs", [])
        defect_tiles = ctx.get("defect_tiles", [])
        defect_shapes = ctx.get("defect_img_shapes", [])
        defect_paths = ctx.get("defect_paths", [])
        pos_feats = []
        n_pos = n_bg_used = n_mid = n_nolbl = 0
        for i, tiles in enumerate(defect_tile_imgs):
            if i >= len(defect_tiles) or i >= len(defect_paths) \
                    or i >= len(defect_shapes):
                continue
            img_h, img_w = defect_shapes[i]
            img_path = defect_paths[i]
            lbl_path = os.path.join(
                os.path.dirname(os.path.dirname(img_path)), "labels",
                os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            boxes = _parse_yolo_boxes(lbl_path, img_w, img_h)
            if not boxes:
                n_nolbl += 1
                continue
            pos_imgs, bg_imgs = [], []
            for tile_img, tb in zip(tiles, defect_tiles[i]):
                y1, x1, y2, x2 = tb
                ta = max(1, (y2 - y1) * (x2 - x1))
                f_area = f_cover = 0.0
                for b in boxes:
                    iw = max(0, min(x2, b[2]) - max(x1, b[0]))
                    ih = max(0, min(y2, b[3]) - max(y1, b[1]))
                    inter = iw * ih
                    f_area = max(f_area, inter / ta)
                    f_cover = max(f_cover,
                                  inter / max(1.0, (b[2] - b[0]) * (b[3] - b[1])))
                if f_area >= pos_a or f_cover >= cover_min:
                    pos_imgs.append(tile_img)
                elif f_area <= neg_t and f_cover <= neg_t:
                    bg_imgs.append(tile_img)
                else:
                    n_mid += 1
            if len(bg_imgs) > n_bg:               # 背景块抽样上限（控提取时间）
                sel = rng.choice(len(bg_imgs), n_bg, replace=False)
                bg_imgs = [bg_imgs[j] for j in sel]
            if pos_imgs:
                pos_feats.append(_feats(pos_imgs))
                n_pos += len(pos_imgs)
            if bg_imgs:
                neg_feats.append(_feats(bg_imgs))
                n_bg_used += len(bg_imgs)

        if not pos_feats:
            print("[shead-block] 缺陷块为空（无框或全部低于阈值），跳过训练",
                  flush=True)
            return self
        neg = torch.cat(neg_feats, dim=0)
        pos_list = []
        for _ in range(n_pseudo):
            pos_list.append(neg + torch.randn_like(neg) * 0.1)
        pos_list.extend(pos_feats)
        pos = torch.cat(pos_list, dim=0)
        cap = self.cfg.get("max_patches", 50000)
        if len(pos) > cap:
            pos = pos[torch.randperm(len(pos), generator=torch_gen)[:cap]]
        if len(pos) == 0:
            print("[shead-block] 正样本为空，跳过训练", flush=True)
            return self
        X = torch.cat([neg, pos]).float().detach().to(self.device)
        y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
        loss_fn = DeviationLoss(margin=self.cfg.get("margin", 5.0))
        opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
        bs = self.cfg.get("batch", 32)
        for ep in range(n_epochs):
            perm = torch.randperm(len(X), generator=torch_gen)
            for i in range(0, len(X), bs):
                xb = X[perm[i:i + bs]]
                yb = y[perm[i:i + bs]]
                scores = self.head(xb)
                loss = loss_fn(scores, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                s = self.head(X)
                l_n = s[y == 0].pow(2).mean().item()
                l_a = (loss_fn.margin - s[y == 1]).clamp(min=0).pow(2).mean().item()
            self.loss_history.append((ep + 1, l_n, l_a))
            if (ep + 1) % 10 == 0:
                print(f"[shead-block] epoch {ep + 1}/{n_epochs} "
                      f"loss_n={l_n:.4f} loss_a={l_a:.4f}", flush=True)
        print(f"[shead-block] fit 完成: neg_blocks={len(neg)} "
              f"(bg_from_defect={n_bg_used}) pos_blocks={len(pos)} "
              f"(中间块丢弃={n_mid} 无标签图={n_nolbl}) X={len(X)}", flush=True)
        self.head.eval()
        return self

    def _fit_guided(self, ctx):
        """U77 两级结构 fit（2026-08-19）：建粗筛 coreset 库 + 训练 crop 判别头。

        两级结构（U76 定论诚实冲 0.9 唯一路径）：
          第一级：整图 patch 级粗筛（coreset 距离热力图）定位疑点区域
          第二级：疑点区域裁高 SNR crop -> 判别头精查
        U77 实证（183 框）：sem coreset 距离 top-10% 定位召回 0.902。

        本 fit 两件事：
        1. 用 train/good patch 建粗筛 coreset 库（贪心 farthest-point，U77 同款）
        2. 训练 crop 判别头（ablB 配方：box+pad pos + 上下文 pos + 背景 neg，
           pseudo=0 -- U74 教训低占比正样本是噪声）
        诚实边界：coreset 库与判别头只用 train 域（train/good + init_defect 框），
        test 绝不进 fit。"""
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        torch_gen = torch.Generator(device="cpu")
        torch_gen.manual_seed(int(self.cfg.get("seed", 42)))
        batch = self.cfg.get("extract_batch", 8)
        n_epochs = self.cfg.get("epochs", 100)
        max_base = self.cfg.get("max_base_tiles", 96)
        n_pseudo = self.cfg.get("pseudo_per_image", 0)
        crop_size = self.crop_size
        pad_ratio = self.crop_pad
        n_bg_per_defect = int(self.cfg.get("n_bg_per_defect", 3))

        # 1. 粗筛 coreset 库（train/good patch 贪心采样）
        train_feats = ctx.get("train_tile_feats", [])
        chunks = []
        for f in train_feats:
            flat = f.flatten(2).permute(0, 2, 1).reshape(-1, f.shape[1]).float().cpu()
            chunks.append(flat)
        if chunks:
            feats = torch.cat(chunks, dim=0)
            feats = F.normalize(feats, dim=1)
            n = feats.shape[0]
            target = min(self.guided_coreset, n)
            if target > 0:
                idx = [int(rng.integers(n))]
                d = 1 - feats @ feats[idx[0]]
                for _ in range(target - 1):
                    i = int(torch.argmax(d).item())
                    idx.append(i)
                    d = torch.minimum(d, 1 - feats @ feats[i])
                self.coreset_bank = feats[idx].cpu().half()
                print(f"[shead-guided] 粗筛 coreset 库: {self.coreset_bank.shape}",
                      flush=True)
        else:
            self.coreset_bank = None
            print("[shead-guided] 警告: 无 train 特征，粗筛库为空", flush=True)

        # 2. 训练 crop 判别头（ablB 配方）
        defect_tile_imgs = ctx.get("defect_tile_imgs", [])
        defect_paths = ctx.get("defect_paths", [])
        all_defect_info = []
        box_sides = []
        for i, tiles in enumerate(defect_tile_imgs):
            for tile_img in tiles:
                boxes = []
                if i < len(defect_paths):
                    img_path = defect_paths[i]
                    lbl_path = os.path.join(
                        os.path.dirname(os.path.dirname(img_path)), "labels",
                        os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                    h, w = tile_img.shape[:2]
                    boxes = _parse_yolo_boxes(lbl_path, w, h)
                    for bx in boxes:
                        box_sides.append(max(bx[2] - bx[0], bx[3] - bx[1]))
                all_defect_info.append((tile_img, boxes))

        if box_sides and self.adaptive_crop_size:
            med_side = int(np.median(box_sides))
            new_cs = max(100, min(2000, med_side))
            if new_cs != crop_size:
                print(f"[shead-guided] adaptive crop_size: {crop_size} -> {new_cs} "
                      f"(median box side={med_side})", flush=True)
                crop_size = new_cs
                self.crop_size = new_cs

        # 正常 crop（neg）
        train_tiles = [t for tiles in ctx["train_tile_imgs"] for t in tiles]
        if len(train_tiles) > max_base:
            sel = rng.choice(len(train_tiles), max_base, replace=False)
            train_tiles = [train_tiles[i] for i in sel]
        if not train_tiles:
            print("[shead-guided] 无正常图，跳过训练", flush=True)
            return self
        neg_feats = []
        for img in train_tiles:
            crops = _extract_random_crops(img, crop_size, self.n_fit_normal_crops, rng)
            f = self.backbone.extract_tiles(crops, batch=batch)
            neg_feats.append(_image_level_feats(f, self.chan_stats, self.patch_stats))

        # 缺陷 crop（pos = box+pad + 上下文；neg = 背景 IoU<0.05）
        n_real_defect = n_ctx_crops = n_bg_crops = 0
        pos_feats_list = []
        for tile_img, boxes in all_defect_info:
            if boxes:
                box_crops = _extract_box_crops(tile_img, boxes, pad_ratio)
                ctx_crops = _extract_context_crops(tile_img, boxes, crop_size)
                crops = box_crops + ctx_crops
                n_ctx_crops += len(ctx_crops)
                bg_crops = []
                h, w = tile_img.shape[:2]
                cs = min(crop_size, h, w)
                for _ in range(n_bg_per_defect * 3):
                    if h <= cs or w <= cs:
                        break
                    y0 = int(rng.integers(0, h - cs))
                    x0 = int(rng.integers(0, w - cs))
                    cand = [x0, y0, x0 + cs, y0 + cs]
                    if all(_iou(cand, b) < 0.05 for b in boxes):
                        bg_crops.append(tile_img[y0:y0 + cs, x0:x0 + cs])
                    if len(bg_crops) >= n_bg_per_defect:
                        break
                if bg_crops:
                    f_bg = self.backbone.extract_tiles(bg_crops, batch=batch)
                    neg_feats.append(_image_level_feats(f_bg, self.chan_stats,
                                                        self.patch_stats))
                    n_bg_crops += len(bg_crops)
            else:
                crops = _extract_random_crops(tile_img, crop_size, 5, rng)
            if crops:
                f = self.backbone.extract_tiles(crops, batch=batch)
                pos_feats_list.append(_image_level_feats(f, self.chan_stats,
                                                          self.patch_stats))
                n_real_defect += len(crops)

        neg = torch.cat(neg_feats, dim=0)
        pos_list = []
        for _ in range(n_pseudo):
            pos_list.append(neg + torch.randn_like(neg) * 0.1)
        if pos_feats_list:
            pos_list.extend(pos_feats_list)
        pos = torch.cat(pos_list, dim=0)
        cap = self.cfg.get("max_patches", 50000)
        if len(pos) > cap:
            pos = pos[torch.randperm(len(pos), generator=torch_gen)[:cap]]
        if len(pos) == 0:
            print("[shead-guided] 正样本为空，跳过训练", flush=True)
            return self

        X = torch.cat([neg, pos]).float().detach().to(self.device)
        y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
        loss_fn = DeviationLoss(margin=self.cfg.get("margin", 5.0))
        opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
        bs = self.cfg.get("batch", 32)
        for ep in range(n_epochs):
            perm = torch.randperm(len(X), generator=torch_gen)
            for i in range(0, len(X), bs):
                xb = X[perm[i:i + bs]]
                yb = y[perm[i:i + bs]]
                scores = self.head(xb)
                loss = loss_fn(scores, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                s = self.head(X)
                l_n = s[y == 0].pow(2).mean().item()
                l_a = (loss_fn.margin - s[y == 1]).clamp(min=0).pow(2).mean().item()
            self.loss_history.append((ep + 1, l_n, l_a))
            if (ep + 1) % 10 == 0:
                print(f"[shead-guided] epoch {ep + 1}/{n_epochs} "
                      f"loss_n={l_n:.4f} loss_a={l_a:.4f}", flush=True)
        print(f"[shead-guided] fit 完成: neg_crops={len(neg)} (bg={n_bg_crops}) "
              f"pos_crops={len(pos)} (real={n_real_defect} ctx={n_ctx_crops}) "
              f"X={len(X)} crop_size={crop_size}", flush=True)
        self.head.eval()
        return self

    @torch.no_grad()
    def score_tiles(self, tile_feats, tile_imgs=None):
        """每 tile 独立图像级打分；分数 = 头输出（偏离度，非 sigmoid：
        负样本分数被压向 0、异常分数大，符合 AUROC 排序方向）。
        热力图给每 tile 一张 G×G 均匀图（值=tile 分数）--该槽无定位能力。"""
        if self.score_mode == "crop" and tile_imgs is not None:
            return self._score_tiles_crop(tile_feats, tile_imgs)
        if self.score_mode == "block":
            return self._score_tiles_block(tile_feats)
        if self.score_mode == "guided":
            return self._score_tiles_guided(tile_feats, tile_imgs)
        T, _, G, _ = tile_feats.shape
        x = _image_level_feats(tile_feats.float().to(self.device), self.chan_stats,
                               self.patch_stats, self.focus_topk)  # U56：GPU 上算统计量；U70：score 恒用聚焦
        p = self.head(x.to(self.device)).reshape(-1)     # (T,)（T=1 时防 0-d）
        tile_scores = p.cpu().numpy().astype(np.float64)  # (T,)
        heatmaps = [np.full((G, G), float(s), dtype=np.float32) for s in tile_scores]
        return tile_scores, heatmaps

    @torch.no_grad()
    def _score_tiles_block(self, tile_feats):
        """U75 块级打分：tile_feats 即块特征（multi-tile 管线统一提取，一次前向），
        逐块过判别头 -> top-k 块均值聚合为整图分。

        速度红线（<1s/图）关键：零额外 DINO 前向（vs crop 模式每图滑窗重新
        前向 1.5-3s）；30 块特征已在管线提取，这里只有统计+MLP（~20ms）。
        topk（默认 3/30 块）抗单块假阳性：要求多个块高分才推高整图分。
        返回单元素（聚合分）--offline 的多尺度聚合对均匀热力图退化为恒等。"""
        T, _, G, _ = tile_feats.shape
        x = _image_level_feats(tile_feats.float().to(self.device), self.chan_stats,
                               self.patch_stats)
        p = self.head(x.to(self.device)).reshape(-1)          # (T,) 块分数
        k = max(1, min(self.block_score_topk, len(p)))
        topk = torch.topk(p, k).values
        agg = float(topk.mean().item())
        return np.array([agg], dtype=np.float64), \
            [np.full((G, G), agg, dtype=np.float32)]

    @torch.no_grad()
    def _score_tiles_guided(self, tile_feats, tile_imgs):
        """U77 两级滑窗打分（2026-08-19 v2）：粗滑窗召回 + 细滑窗精查。

        v1 教训（U77b/c 诊断）：sem coreset 距离粗筛不可用——缺陷框内高分
        patch 仅 median=1 个（稀疏），候选框 IoU 覆盖 0.13，AUROC 0.5994。
        改用判别头两级滑窗：
          第一级：全图粗滑窗（crop 900 stride 700，~20 窗）-> 判别头 -> top-4
          第二级：top-4 窗内 2x2 细滑窗（crop 500）-> 判别头 -> top3 聚合
        判别头在"缺陷 vs 背景"上训练过（比 coreset 域差更懂缺陷），
        第一级召回 + 第二级精查，窗数少（~36）速度红线内。
        速度预估：管线整图 250ms + 36 窗 1 批 250ms + 槽位 50ms ≈ 550ms。"""
        batch = self.cfg.get("extract_batch", 8)
        G = tile_feats.shape[-1] if tile_feats is not None else self.backbone.grid
        c1 = int(self.cfg.get("guided_l1_size", 900))
        c1_stride = int(self.cfg.get("guided_l1_stride", 700))
        c2 = int(self.cfg.get("guided_l2_size", 500))
        n_l1_top = int(self.cfg.get("guided_l1_top", 4))
        n_l2_grid = int(self.cfg.get("guided_l2_grid", 2))
        score_topk = max(1, int(self.cfg.get("guided_score_topk", 3)))
        tile_scores = []
        heatmaps = []
        for t in range(len(tile_imgs)):
            img = tile_imgs[t]
            # 第一级：全图粗滑窗
            l1_crops = _sliding_grid_crops(img, c1, c1_stride)
            if l1_crops:
                f1 = self.backbone.extract_tiles(l1_crops, batch=batch)
                x1 = _image_level_feats(f1.float().to(self.device),
                                        self.chan_stats, self.patch_stats)
                s1 = self.head(x1.to(self.device)).reshape(-1)
            else:
                s1 = torch.empty(0)
            # 第二级：top-k 粗窗内 2x2 细滑窗
            l2_crops = []
            if len(s1) > 0:
                top_idx = torch.topk(s1, min(n_l1_top, len(s1))).indices.cpu().tolist()
                for ti in top_idx:
                    y0 = int((ti // (int(np.ceil(img.shape[1] / c1_stride)) if
                                     c1_stride else 1)) * c1_stride)
                    x0 = int((ti % (int(np.ceil(img.shape[1] / c1_stride)) if
                                    c1_stride else 1)) * c1_stride)
                    win = img[max(0, y0):min(img.shape[0], y0 + c1),
                              max(0, x0):min(img.shape[1], x0 + c1)]
                    wh, ww = win.shape[:2]
                    if wh > c2 and ww > c2:
                        for r in range(n_l2_grid):
                            for c in range(n_l2_grid):
                                yy = int(r * (wh - c2) / max(1, n_l2_grid - 1)) if n_l2_grid > 1 else 0
                                xx = int(c * (ww - c2) / max(1, n_l2_grid - 1)) if n_l2_grid > 1 else 0
                                l2_crops.append(win[yy:yy + c2, xx:xx + c2])
            if l2_crops:
                f2 = self.backbone.extract_tiles(l2_crops, batch=batch)
                x2 = _image_level_feats(f2.float().to(self.device),
                                        self.chan_stats, self.patch_stats)
                s2 = self.head(x2.to(self.device)).reshape(-1)
                all_s = torch.cat([s1, s2]) if len(s1) else s2
            elif len(s1):
                all_s = s1
            else:
                x = _image_level_feats(tile_feats[t:t + 1].float().to(self.device),
                                       self.chan_stats, self.patch_stats)
                all_s = self.head(x.to(self.device)).reshape(-1)
            k = min(score_topk, len(all_s))
            tile_score = float(torch.topk(all_s, k).values.mean().item())
            tile_scores.append(tile_score)
            heatmaps.append(np.full((G, G), tile_score, dtype=np.float32))
        return np.array(tile_scores, dtype=np.float64), heatmaps

    @staticmethod
    def _n_windows(h, w, cs, ov):
        """U86v16：滑窗数（与 _sliding_grid 同构）。"""
        if h <= cs or w <= cs:
            return 1
        stride = max(1, int(cs * (1 - ov)))
        y_pos = list(range(0, max(1, h - cs + 1), stride))
        x_pos = list(range(0, max(1, w - cs + 1), stride))
        if y_pos[-1] + cs < h:
            y_pos.append(h - cs)
        if x_pos[-1] + cs < w:
            x_pos.append(w - cs)
        return len(y_pos) * len(x_pos)

    def _adaptive_overlap(self, img, target=40):
        """U86v16：每图自适应 overlap（目标窗数固定，<1s 前提下覆盖最大化）。"""
        h, w = img.shape[:2]
        cs = min(self.crop_size, h, w)
        lo, hi = 0.1, 0.8
        for _ in range(14):
            mid = (lo + hi) / 2
            if self._n_windows(h, w, cs, mid) > target:
                hi = mid
            else:
                lo = mid
        return lo

    @torch.no_grad()
    def _score_tiles_crop(self, tile_feats, tile_imgs):
        """U73v4 crop 级打分：滑窗网格 -> crop 分数图 -> 3x3 平滑 -> max 聚合。

        v4 相邻一致性聚合（vs v3 全局 topk）：overlap=0.5 滑窗下真缺陷被
        4-9 个相邻 crop 覆盖（空间连续高分簇），孤立假阳性只有单 crop。
        3x3 均值平滑后孤立高分被邻域低分稀释（自占 1/9），连续高分簇保持
        高分 -> max 取最强连续响应。这把检测从"独立 crop 分类"升级为
        "空间结构判别"，直接攻克整图特异度≈单crop特异度^N 的多重比较问题。
        截断（超 max_crops，网格不完整）或开关关闭时 fallback v3 topk。"""
        batch = self.cfg.get("extract_batch", 8)
        score_topk = max(1, int(self.cfg.get("score_topk", 3)))
        topk_ratio = float(self.cfg.get("score_topk_ratio", 0.0))  # >0 时按比例
        # U86v16 集成：adaptive_score=true 时每图控窗（大图减窗保 <1s，中小图保覆盖）
        adaptive_score = self.cfg.get("adaptive_score", False)
        adaptive_target = int(self.cfg.get("adaptive_target", 40))
        G = tile_feats.shape[-1] if tile_feats is not None else self.backbone.grid
        k33 = torch.ones(1, 1, 3, 3, device=self.device) / 9.0
        tile_scores = []
        heatmaps = []
        for t in range(len(tile_imgs)):
            img = tile_imgs[t]
            if adaptive_score:
                ov = self._adaptive_overlap(img, adaptive_target)
            else:
                ov = self.score_overlap
            crops, n_y, n_x, truncated = _sliding_grid(
                img, self.crop_size, ov, self.score_max_crops)
            f = self.backbone.extract_tiles(crops, batch=batch)
            x = _image_level_feats(f.float().to(self.device), self.chan_stats,
                                   self.patch_stats)
            scores = self.head(x.to(self.device)).reshape(-1)
            if self.use_adjacent_agg and not truncated:
                smap = scores.reshape(n_y, n_x).float()[None, None]   # (1,1,ny,nx)
                pad = F.pad(smap, (1, 1, 1, 1), mode="replicate")
                tile_score = float(F.conv2d(pad, k33).max().item())
            else:
                if topk_ratio > 0:
                    k = max(1, int(len(scores) * topk_ratio))
                else:
                    k = min(score_topk, len(scores))
                topk = torch.topk(scores, k).values
                tile_score = float(topk.mean().item())
            tile_scores.append(tile_score)
            heatmaps.append(np.full((G, G), tile_score, dtype=np.float32))
        return np.array(tile_scores, dtype=np.float64), heatmaps
