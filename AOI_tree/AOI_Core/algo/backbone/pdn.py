"""M4 PDN 学生主干（U9）：EfficientAD PDN-S 蒸馏到 DINO 特征空间。

- 架构：4 层 CNN（0.7M 参数，stride 4），输入 resize 到 grid*4（G=32 -> 128²）
- 蒸馏：train/good 上 DINO 提特征 (384,32,32)，PDN 输出对齐（patch 级余弦 loss）
- 冻结使用；extract_tiles 接口与 FrozenDINO 完全一致（槽位无感，深插拔边界 §4.4）
- CPU 友好：纯 CNN 无注意力，CPU 推理显著快于 ViT（§6.1 CPU <2s 降配路径）
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PDN(nn.Module):
    """EfficientAD PDN-S：stride-4 CNN，输出 (B,C,H/4,W/4)。"""

    def __init__(self, out_channels=384, mid_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, mid_channels, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-3)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-3)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels, eps=1e-3)
        self.conv4 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class PDNBackbone(nn.Module):
    """PDN 蒸馏学生，接口对齐 FrozenDINO（extract_tiles -> (T,384,G,G)）。"""

    def __init__(self, grid=32, device="cuda", checkpoint=None):
        super().__init__()
        self.grid = grid
        self.model = PDN(out_channels=384)
        self.dim = 384
        self.device_ = device
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if checkpoint and os.path.exists(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(f"[pdn] 加载蒸馏学生 {checkpoint}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,H,W) in [0,1] -> (B,384,G,G)"""
        x = (x - self.mean) / self.std
        target = self.grid * 4
        if x.shape[-1] != target:
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        return self.model(x)

    @torch.no_grad()
    def extract_tiles(self, tile_imgs, batch=32):
        """tile_imgs: list of HxWx3 uint8 -> (T,384,G,G)。CPU 侧 resize 到 128² 再上 GPU。"""
        import cv2
        target = self.grid * 4
        feats = []
        for i in range(0, len(tile_imgs), batch):
            chunk = [cv2.resize(t, (target, target)) if t.shape[:2] != (target, target) else t
                     for t in tile_imgs[i:i + batch]]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) / 255.0)
            feats.append(self.forward(x.permute(0, 3, 1, 2).to(self.device_)))
        return torch.cat(feats, dim=0)


def distill_pdn(student, dino_backbone, tile_imgs, epochs=20, lr=1e-3,
                device="cuda", batch=8, progress_every=5):
    """蒸馏：PDN 输出对齐 DINO 特征（patch 级余弦），train/good tile 上。
    tile_imgs: list[ndarray HxWx3 uint8]（train/good 全量）。返回 (student, info)。"""
    import cv2
    # 蒸馏期解冻 + train 模式（BatchNorm 用 batch 统计）；结束恢复冻结（§4.4 深插拔边界）
    student.model.train()
    for p in student.model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(student.model.parameters(), lr=lr)
    dino_targets = dino_backbone.extract_tiles(tile_imgs).cpu()   # (T,384,32,32)
    n = len(tile_imgs)
    losses = []
    for ep in range(epochs):
        ep_loss = 0.0
        gen = torch.Generator(device=student.model.device if hasattr(student.model, "device") else device)
        gen.manual_seed(42)
        idx = torch.randperm(n, generator=gen, device=device)
        for b0 in range(0, n, batch):
            ids = idx[b0:b0 + batch].tolist()
            chunk = [cv2.resize(tile_imgs[i], (student.grid * 4, student.grid * 4))
                     for i in ids]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2).to(device)
            pdn_f = student.model(x)                                   # (B,384,G,G)
            tgt = dino_targets[ids].to(device)
            # patch 级余弦对齐（尺度无关，适合跨特征空间蒸馏）
            loss = 1.0 - F.cosine_similarity(
                F.normalize(pdn_f, dim=1), F.normalize(tgt, dim=1), dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        losses.append(round(ep_loss / max(1, (n + batch - 1) // batch), 5))
        if progress_every and (ep + 1) % progress_every == 0:
            print(f"  [distill] ep {ep + 1}/{epochs} loss={losses[-1]}", flush=True)
    # 恢复冻结推理状态
    student.model.eval()
    for p in student.model.parameters():
        p.requires_grad = False
    return student, {"losses": losses, "n_tiles": n}
