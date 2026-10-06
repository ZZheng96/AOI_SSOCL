"""disc 槽位：图像层伪异常判别头（v2 扶正：唯一泄漏免疫的诚实主力）

关键设计（与 demo4 的本质区别）：
- 伪异常在**图像层**合成（CutPaste/Perlin 斑块），再提特征——不是特征空间加噪；
- 基底 = 全部协议内图像（train/good 100 + init_defect 30，Q3 结论：全部可用作合成基底）；
- 无异常聚拢项：BCE 把伪异常推离正常，不学异常形态（"避开异常"）；
- 打分在 patch 层：逐 patch 过头，tile 分数=patch top-k 均值，热力图=patch 分数图。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Slot


_MGRID_CACHE = {}


def _resolve_init_from(init_from, slot="disc"):
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


def _mgrid(h, w):
    if (h, w) not in _MGRID_CACHE:
        _MGRID_CACHE[(h, w)] = np.mgrid[0:h, 0:w]
    return _MGRID_CACHE[(h, w)]


def synth_pseudo(img, rng):
    """图像层伪异常合成：CutPaste + 色斑。img: HxWx3 uint8 -> (img, mask)"""
    h, w = img.shape[:2]
    out = img.copy()
    mask = np.zeros((h, w), np.float32)
    kind = rng.integers(0, 2)
    if kind == 0:  # CutPaste：裁一块随机翻转后贴回
        ph, pw = rng.integers(h // 12, h // 4, 2)
        y, x = rng.integers(0, h - ph), rng.integers(0, w - pw)
        patch = img[y:y + ph, x:x + pw].copy()
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        y2, x2 = rng.integers(0, h - ph), rng.integers(0, w - pw)
        out[y2:y2 + ph, x2:x2 + pw] = patch
        mask[y2:y2 + ph, x2:x2 + pw] = 1.0
    else:  # 色斑：随机椭圆+颜色偏移（mgrid 按尺寸缓存）
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
        """ctx: train_tile_imgs (list[list[ndarray]]), defect_tile_imgs 同构（仅作合成基底）"""
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        n_pseudo = self.cfg.get("pseudo_per_image", 8)
        base_tiles = [t for tiles in ctx["train_tile_imgs"] for t in tiles]
        for tiles in ctx.get("defect_tile_imgs", []):
            base_tiles += tiles          # 缺陷图同样可作合成基底（标签仍为异常）
        # 成本控制：基底上限采样（避免 GPU OOM，配合分批 CPU 累积）
        max_base = self.cfg.get("max_base_tiles", 96)
        if len(base_tiles) > max_base:
            sel = rng.choice(len(base_tiles), max_base, replace=False)
            base_tiles = [base_tiles[i] for i in sel]

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
                        out = out[torch.randperm(len(out))[:cap]]
                return out

            def _stream_pseudo(base_tiles):
                """边合成伪异常边提特征，返回 (pos, neg)，内存峰值受控。
                neg=基底特征（≤96 图，批量提取）；pos=伪异常（8 倍，流式截断）。"""
                neg = _extract_flat(base_tiles, batch=8)
                pos = None
                for t in base_tiles:
                    augs = [synth_pseudo(t, rng)[0] for _ in range(n_pseudo)]
                    p = _extract_flat(augs, batch=8)
                    pos = p if pos is None else torch.cat([pos, p])
                    if len(pos) > cap:
                        pos = pos[torch.randperm(len(pos))[:cap]]
                return pos, neg

            pos, neg = _stream_pseudo(base_tiles)
            # patch 级子采样：每侧 ≤cap，控制显存与训练时长
            if len(pos) > cap:
                pos = pos[torch.randperm(len(pos))[:cap]]
            if len(neg) > cap:
                neg = neg[torch.randperm(len(neg))[:cap]]
            pos, neg = pos[: len(neg) * 4], neg     # 正负比 ≤4:1
            X = torch.cat([neg, pos]).float().to(self.device)
            y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(self.device)
            opt = torch.optim.Adam(self.head.parameters(), lr=self.cfg.get("lr", 1e-3))
            w_pos = len(neg) / max(len(pos), 1)
            for _ in range(n_epochs):
                perm = torch.randperm(len(X))
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
