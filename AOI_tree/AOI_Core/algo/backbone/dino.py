"""DINOv2 ViT-S/14 冻结主干（加载优先级：随包 vendor → 本地 hub 缓存 → 在线下载）。

红线 6：主干永冻。输入 tile 统一 resize 到 grid*14=448，输出 (B,384,G,G)。
支持批量 tile 前向（速度杠杆之一）。

随包离线复现：将 dinov2 仓库代码与权重放在以下任一位置即可完全离线加载——
  <包根>/assets/dinov2/ 或 <包根>/../assets/dinov2/（提交包三组件共享一份），
目录内需含 hubconf.py、dinov2/ 包与 dinov2_vits14_pretrain.pth；
亦可用环境变量 DINOV2_LOCAL_DIR 显式指定该目录。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")


def _vendored_dir():
    """查找随包 vendor 的 dinov2 目录（含 hubconf.py 与模型权重）。"""
    env = os.environ.get("DINOV2_LOCAL_DIR")
    cands = [env] if env else []
    # 包根 = src/ 或 algo/ 的上级目录（各算法工程目录同构）
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands += [os.path.join(root, "assets", "dinov2"),
              os.path.join(os.path.dirname(root), "assets", "dinov2")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "hubconf.py")):
            return d
    return None


def _load_dinov2_from_cache(model_name="vits14"):
    vend = _vendored_dir()
    if vend:
        weights = os.path.join(vend, f"dinov2_{model_name}_pretrain.pth")
        # 2026-10-06 修复：vendor 目录存在但缺权重文件时原实现直接抛错，把可用
        # 的本地 hub 缓存/在线下载一并堵死（实测缓存里已有 dinov2_vits14_
        # pretrain.pth 88MB，却因 assets/dinov2 仅有源码无权重而无法 fit）。
        # 缺权重改为回退到缓存/在线路径；vendor 仅在权重齐备时才作为优先源。
        if not os.path.exists(weights):
            print(f"[dino] 随包 vendor 目录缺权重，回退 hub 缓存/在线: {weights}")
        else:
            if vend not in sys.path:
                sys.path.insert(0, vend)
            import importlib
            backbones = importlib.import_module("dinov2.hub.backbones")
            print(f"[dino] 使用随包 vendor 权重: {weights}")
            return getattr(backbones, f"dinov2_{model_name}")(pretrained=True, weights=weights)
    hubconf_path = os.path.join(DINOV2_CACHE_DIR, "hubconf.py")
    if not os.path.exists(hubconf_path):
        return torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_name}")
    sys.path.insert(0, DINOV2_CACHE_DIR)
    import importlib
    hubconf = importlib.import_module("hubconf")
    return getattr(hubconf, f"dinov2_{model_name}")(pretrained=True)


class FrozenDINO(nn.Module):
    def __init__(self, model_name="vits14", output_layer=9, grid=32, device="cuda"):
        super().__init__()
        self.output_layer = output_layer
        self.grid = grid
        self.model = _load_dinov2_from_cache(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim = self.model.embed_dim  # 384
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
        self.device_ = device

    def _intermediate(self, x):
        x = self.model.prepare_tokens_with_masks(x)
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if (i + 1) == self.output_layer:
                return x[:, 1:]  # 去 CLS
        return x[:, 1:]

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,H,W) float in [0,1] -> (B,384,G,G)"""
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - self.mean) / self.std
        target = self.grid * 14
        if x.shape[-1] != target or x.shape[-2] != target:
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
        # fp16 已回退（2026-08-15 实测）：1660 SUPER 图灵架构 fp16 无速度收益
        # （feats 141→148ms）且疑似损伤 DINO 特征精度（检出 5/20→1/20 退化）
        tok = self._intermediate(x)  # (B,N,D)
        B, N, D = tok.shape
        g = int(N ** 0.5)
        feat = tok.permute(0, 2, 1).reshape(B, D, g, N // g)
        if g != self.grid:
            feat = F.interpolate(feat, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return feat

    @torch.no_grad()
    def extract_tiles(self, tile_imgs, batch=16):
        """tile_imgs: list of HxWx3 uint8 ndarray -> (T,384,G,G) tensor on device。
        性能关键：CPU 侧先 resize 到 448² uint8 再上 GPU（避免 2500² float 插值与拷贝）。"""
        import cv2
        target = self.grid * 14
        feats = []
        for i in range(0, len(tile_imgs), batch):
            chunk = [cv2.resize(t, (target, target)) if t.shape[:2] != (target, target) else t
                     for t in tile_imgs[i:i + batch]]
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) / 255.0)
            feats.append(self.forward(x.permute(0, 3, 1, 2).to(self.device_)))
        return torch.cat(feats, dim=0)
