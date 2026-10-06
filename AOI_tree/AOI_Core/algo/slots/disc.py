"""disc 槽位：图像层伪异常判别头（v2 扶正：唯一泄漏免疫的诚实主力）

关键设计（与 demo4 的本质区别）：
- 伪异常在**图像层**合成（真实缺陷移植/CutPaste/色斑），再提特征——不是特征空间加噪；
- 真实缺陷移植（§15.30，AHL/DRAEM 思路）：按缺陷**位置掩码**从已标注缺陷图抠出缺陷
  区域，随机翻转 + 随机位置贴入正常图，把少量真实缺陷放大为大量训练信号——
  比人工图样更贴近真实异常分布；无位置掩码的图不参与移植（整图/整块贴入无意义）；
- 基底 = 全部协议内图像（train/good 100 + init_defect 30，Q3 结论：全部可用作合成基底）；
- 无异常聚拢项：BCE 把伪异常推离正常，不学异常形态（"避开异常"）；
- 打分在 patch 层：逐 patch 过头，tile 分数=patch top-k 均值，热力图=patch 分数图。
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Slot


_MGRID_CACHE = {}


def _resolve_init_from(init_from, slot="disc"):
    """解析预训练头路径：相对路径依次按 CWD、包根（algo/ 上级目录）解析。

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


def _mgrid(h, w):
    if (h, w) not in _MGRID_CACHE:
        _MGRID_CACHE[(h, w)] = np.mgrid[0:h, 0:w]
    return _MGRID_CACHE[(h, w)]


def load_defect_mask(img_path, shape=None):
    """按命名约定查找缺陷位置掩码（ground truth，§15.30；ingest 只计数不入库）：

    1) 同目录 {stem}_mask{ext} / {stem}_mask.png（通用 folder 导入）；
    2) 兄弟目录 ground_truth/{stem}_mask.png；
    3) MVTec 结构 test/{defect}/xxx.png -> {cat}/ground_truth/xxx_mask.png。

    返回 0/255 uint8 单通道（尺寸对齐 shape）或 None（该图无位置标注）。
    """
    p = Path(img_path)
    stem = p.stem
    cands = [p.with_name(f"{stem}_mask{p.suffix}"),
             p.with_name(f"{stem}_mask.png"),
             p.parent / "ground_truth" / f"{stem}_mask.png"]
    parents = list(p.parents)
    for i in (1, 2):                      # parents[2] = MVTec {cat} 层
        if i < len(parents):
            cands.append(parents[i] / "ground_truth" / f"{stem}_mask.png")
    for c in cands:
        if c.exists():
            m = cv2.imread(str(c), cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if shape is not None and m.shape != tuple(shape):
                m = cv2.resize(m, (int(shape[1]), int(shape[0])),
                               interpolation=cv2.INTER_NEAREST)
            return m
    return None


def synth_pseudo(img, rng, defect_pool=None, pcfg=None):
    """图像层伪异常合成：真实缺陷移植（首选，AHL/DRAEM 思路）/ CutPaste / 色斑。

    img: HxWx3 uint8 RGB -> (img, mask)
    defect_pool: 移植源列表，元素为 (缺陷图/块, 位置掩码) 二元组（掩码 0/255
      uint8 标出缺陷像素）；纯图元素或全 0 掩码视为无位置标注、不参与移植。
    pcfg: augment.pseudo 配置（methods/feather；transplant_scale 保留兼容）。
    按位置掩码外接框抠出缺陷区域，随机翻转 + 随机位置贴入正常图；无有效移植源
    时 defect_transplant 自动回退为 cutpaste（整图/整块贴入无意义，不做）。
    """
    h, w = img.shape[:2]
    out = img.copy()
    mask = np.zeros((h, w), np.float32)
    # 只保留带有效位置掩码的移植源（无位置标注的图不移植）
    pool = [it for it in (defect_pool or [])
            if isinstance(it, (tuple, list)) and it[1] is not None
            and np.count_nonzero(it[1])]
    methods = list((pcfg or {}).get("methods") or
                   ["defect_transplant", "cutpaste", "color_blot"])
    if "defect_transplant" in methods and not pool:
        methods = [m for m in methods if m != "defect_transplant"]
    if not methods:
        methods = ["cutpaste"]
    kind = methods[int(rng.integers(0, len(methods)))]
    if kind == "defect_transplant":
        # 真实缺陷移植（§15.30）：按位置掩码外接框抠缺陷，随机翻转 + 随机位置
        # 贴入（羽化边缘）。把少量已标注缺陷放大为大量训练信号（AHL 核心）。
        d, dm = pool[int(rng.integers(0, len(pool)))]
        dm_bin = (dm > (127 if dm.dtype == np.uint8 else 0.5)).astype(np.uint8)
        ys, xs = np.nonzero(dm_bin)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        d = d[y0:y1, x0:x1]
        alpha = dm_bin[y0:y1, x0:x1].astype(np.float32)
        if rng.random() < 0.5:            # 形态多样：随机水平翻转
            d, alpha = d[:, ::-1], alpha[:, ::-1]
        d = np.ascontiguousarray(d)
        alpha = np.ascontiguousarray(alpha)
        # 统一缩放因子：缺陷块长边 -> 基底短边的 10%~40%（保持纵横比）
        s = rng.uniform(0.10, 0.40)
        k = (min(h, w) * s) / max(d.shape[0], d.shape[1])
        ph = max(8, min(h, int(d.shape[0] * k)))
        pw = max(8, min(w, int(d.shape[1] * k)))
        d = cv2.resize(d, (pw, ph), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (pw, ph), interpolation=cv2.INTER_LINEAR)
        if (pcfg or {}).get("feather", True):
            # 羽化 alpha 蒙版：边缘平滑过渡，避免硬边拼接假影
            alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, min(ph, pw) / 8))
            alpha = np.clip(alpha, 0.0, 1.0)
        y = int(rng.integers(0, h - ph + 1))
        x = int(rng.integers(0, w - pw + 1))
        region = out[y:y + ph, x:x + pw].astype(np.float32)
        out[y:y + ph, x:x + pw] = (d.astype(np.float32) * alpha[..., None]
                                   + region * (1 - alpha[..., None])).astype(np.uint8)
        mask[y:y + ph, x:x + pw] = alpha
    elif kind == "cutpaste":  # CutPaste：裁一块随机翻转后贴回
        ph, pw = rng.integers(h // 12, h // 4, 2)
        y, x = rng.integers(0, h - ph), rng.integers(0, w - pw)
        patch = img[y:y + ph, x:x + pw].copy()
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        y2, x2 = rng.integers(0, h - ph), rng.integers(0, w - pw)
        out[y2:y2 + ph, x2:x2 + pw] = patch
        mask[y2:y2 + ph, x2:x2 + pw] = 1.0
    else:  # color_blot：随机椭圆+颜色偏移（mgrid 按尺寸缓存）
        cy, cx = rng.integers(h // 8, 7 * h // 8, 2)
        ry, rx = rng.integers(h // 24, h // 8, 2)
        yy, xx = _mgrid(h, w)
        region = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1
        shift = rng.uniform(0.5, 1.5, 3)
        out[region] = np.clip(out[region].astype(np.float32) * shift, 0, 255).astype(np.uint8)
        mask[region] = 1.0
    return out, mask


class DiscHead(nn.Module):
    """patch 级小头：384->128->1，~50K 参数（§9.3 预算内）"""
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DiscSlot(Slot):
    name = "disc"

    def __init__(self, cfg, backbone, device="cuda"):
        self.cfg = cfg
        self.backbone = backbone
        self.device = device
        self.head = DiscHead(backbone.dim, cfg.get("hidden", 128)).to(device)
        # 跨品类预训练初始化（§9.1，U11）：init_from=预训练权重路径
        init_from = _resolve_init_from(cfg.get("init_from"), slot="disc")
        if init_from:
            self.head.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
            print(f"[disc] 加载预训练头 {init_from}")

    def fit(self, ctx):
        """ctx: train_tile_imgs (list[list[ndarray]]), defect_tile_imgs 同构；
        defect_tile_masks 同构（tile 级位置掩码，None=该图无位置标注）。
        移植源 = 带有效位置掩码的缺陷 tile（§15.30 按位置抠缺陷）；
        缺陷 tile 同时可作合成基底（标签仍为异常）。"""
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        torch_gen_cpu = torch.Generator(device="cpu")
        torch_gen_cpu.manual_seed(int(self.cfg.get("seed", 42)))
        torch_gen = torch.Generator(device=self.device)
        torch_gen.manual_seed(int(self.cfg.get("seed", 42)))
        n_pseudo = self.cfg.get("pseudo_per_image", 8)
        pcfg = self.cfg.get("pseudo") or {}      # augment.pseudo（factory 合并注入）
        ecfg = self.cfg.get("enhance") or {}     # augment.enhance（factory 合并注入）
        defect_pool = []
        for tiles, ms in zip(ctx.get("defect_tile_imgs", []),
                             ctx.get("defect_tile_masks") or []):
            for t, m in zip(tiles, ms or []):
                if m is not None and np.count_nonzero(m):
                    defect_pool.append((t, m))
        base_tiles = [t for tiles in ctx["train_tile_imgs"] for t in tiles]
        is_defect = [False] * len(base_tiles)
        # 前端反馈 2026-09-13 #2：掩码标出的缺陷内容不作为"正常"负样本参与
        # 训练。缺陷图 tile 默认不进负样本（无标注时缺陷位置未知）；仅"该图有
        # 位置标注且此 tile 掩码全 0"的 tile 属经核验的正常区域，可作负样本。
        # 缺陷 tile 仍作伪异常合成基底/移植源（合成后标签恒为异常），不变。
        neg_ok = [True] * len(base_tiles)
        def_imgs = ctx.get("defect_tile_imgs", [])
        def_masks = ctx.get("defect_tile_masks") or []
        for di, tiles in enumerate(def_imgs):
            ms = def_masks[di] if di < len(def_masks) else None
            base_tiles += tiles              # 缺陷图同样可作合成基底（标签仍为异常）
            is_defect += [True] * len(tiles)
            for ti in range(len(tiles)):
                m = ms[ti] if (ms is not None and ti < len(ms)) else None
                neg_ok.append(m is not None and not np.count_nonzero(m))
        # 成本控制：基底上限采样（避免 GPU OOM，配合分批 CPU 累积）
        max_base = self.cfg.get("max_base_tiles", 96)
        if len(base_tiles) > max_base:
            sel = rng.choice(len(base_tiles), max_base, replace=False)
            base_tiles = [base_tiles[i] for i in sel]
            is_defect = [is_defect[i] for i in sel]
            neg_ok = [neg_ok[i] for i in sel]

        # 训练增强（augment.enhance，§15.29）：只对正常基底扩变体（翻转/旋转90/
        # 亮度对比度），让判别头见过正常外观变化、降低对平移/光照的误报；
        # 缺陷 tile 不增强——保持真实缺陷外观原样（移植源语义）。
        # 在 max_base 截断之后做，峰值内存 ≤基底上限 x4 张小图。
        if ecfg.get("flip") or ecfg.get("rot90") or ecfg.get("brightness"):
            aug_base, aug_neg = [], []
            for t, dfl, nok in zip(base_tiles, is_defect, neg_ok):
                aug_base.append(t)
                aug_neg.append(nok)
                if dfl:
                    continue
                if ecfg.get("flip"):
                    aug_base.append(np.ascontiguousarray(t[:, ::-1]))
                    aug_neg.append(nok)
                if ecfg.get("rot90"):
                    aug_base.append(np.ascontiguousarray(np.rot90(t)))
                    aug_neg.append(nok)
                if ecfg.get("brightness"):
                    aug_base.append(cv2.convertScaleAbs(
                        t, alpha=float(rng.uniform(0.8, 1.2)),
                        beta=float(rng.uniform(-20, 20))))
                    aug_neg.append(nok)
            base_tiles, neg_ok = aug_base, aug_neg

        # 微调轮数：finetune_epochs>0 表示在该轮数内用学习率微调（预训练场景）；
        # 否则用默认 epochs 全量训练；freeze=true 表示冻结（预训练头直接打分）
        n_epochs = self.cfg.get("epochs", 30)
        finetune = self.cfg.get("finetune_epochs")
        if finetune is not None:
            n_epochs = min(finetune, n_epochs)
        if self.cfg.get("freeze"):
            n_epochs = 0

        if n_epochs > 0:
            # 流式提取：按基底图循环，每图生成伪异常后立即提特征并截断，
            # 全程不累积 aug_tiles 列表与全部特征（M0 内存诊断：原版 768 张
            # 伪异常特征 + 图像列表峰值 ~1.7GB CPU）
            cap = self.cfg.get("max_patches", 50000)

            def _extract_flat(imgs, batch=8):
                out = None
                for i in range(0, len(imgs), batch):
                    f = self.backbone.extract_tiles(imgs[i:i + batch], batch=batch).cpu()
                    flat = f.flatten(2).permute(0, 2, 1).reshape(-1, self.backbone.dim)
                    if out is None:
                        out = flat
                    else:
                        out = torch.cat([out, flat])
                    if len(out) > cap:          # 流式截断：保持峰值 ≤cap
                        out = out[torch.randperm(len(out), generator=torch_gen_cpu)[:cap]]
                return out

            def _stream_pseudo(base_tiles, neg_flags):
                """边合成伪异常边提特征，返回 (pos, neg)，内存峰值受控。
                neg=经核验的正常基底特征（掩码缺陷 tile 已剔除，≤96 图批量提取）；
                pos=伪异常（8 倍，流式截断，基底含缺陷 tile 但合成后恒为异常）。"""
                neg_src = [t for t, ok in zip(base_tiles, neg_flags) if ok]
                if not neg_src:      # 兜底：fit 保证 ≥1 正常图，正常不会触发
                    neg_src = base_tiles[:1]
                neg = _extract_flat(neg_src, batch=8)
                pos = None
                for t in base_tiles:
                    augs = [synth_pseudo(t, rng, defect_pool, pcfg)[0]
                            for _ in range(n_pseudo)]
                    p = _extract_flat(augs, batch=8)
                    pos = p if pos is None else torch.cat([pos, p])
                    if len(pos) > cap:
                        pos = pos[torch.randperm(len(pos), generator=torch_gen_cpu)[:cap]]
                return pos, neg

            pos, neg = _stream_pseudo(base_tiles, neg_ok)
            # patch 级子采样：每侧 ≤cap，控制显存与训练时长
            if len(pos) > cap:
                pos = pos[torch.randperm(len(pos), generator=torch_gen_cpu)[:cap]]
            if len(neg) > cap:
                neg = neg[torch.randperm(len(neg), generator=torch_gen_cpu)[:cap]]
            pos, neg = pos[: len(neg) * 4], neg     # 正负比 ≤4:1
            X = torch.cat([neg, pos]).float().to(self.device)
            y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
            opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
            w_pos = len(neg) / max(len(pos), 1)
            for _ in range(n_epochs):
                perm = torch.randperm(len(X), generator=torch_gen, device=self.device)
                for i in range(0, len(X), 4096):
                    xb = X[perm[i:i + 4096]].to(self.device)
                    yb = y[perm[i:i + 4096]]
                    logit = self.head(xb)
                    loss = F.binary_cross_entropy_with_logits(
                        logit, yb, pos_weight=torch.tensor(w_pos, device=self.device))
                    opt.zero_grad(); loss.backward(); opt.step()
        self.head.eval()
        return self

    @torch.no_grad()
    def score_tiles(self, tile_feats, tile_imgs=None):
        T, D, G, _ = tile_feats.shape
        flat = tile_feats.flatten(2).permute(0, 2, 1).reshape(-1, D).float().to(self.device)
        p = torch.sigmoid(self.head(flat.to(self.device))).reshape(T, G * G)
        k = max(1, int(G * G * 0.05))
        tile_scores = p.topk(k, dim=-1).values.mean(dim=-1)
        heatmaps = [p[i].reshape(G, G).cpu().numpy() for i in range(T)]
        return tile_scores.cpu().numpy(), heatmaps
