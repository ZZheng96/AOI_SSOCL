"""sem 槽位：tile 级语义记忆库（PatchCore 式）

设计依据：v2 实测 coreset=512 甜点、top-k 块聚合>>全局平均、差分>>集合 kNN
（差分是 L3 的 tpl 槽位的事，M0 无模板场景用诚实版集合 kNN）。
库 = train/good 全部 tile 的 patch token，贪心 coreset 到 ≤512*8 patch（按 tile 粒度抽样）。
红线 3：fit 只见 train/good。
"""
import numpy as np
import torch
import torch.nn.functional as F
from .base import Slot


class SemSlot(Slot):
    name = "sem"

    def __init__(self, cfg, device="cuda"):
        self.coreset_size = cfg.get("coreset_size", 512)
        self.topk_ratio = cfg.get("tile_topk_ratio", 0.05)
        self.seed = cfg.get("seed", 42)     # U24：固定 coreset 随机源，保证 fit 确定性
        self.device = device
        self.bank = None          # (M,384) fp16 on GPU

    def fit(self, ctx):
        """ctx['train_tile_feats']: list[(T,384,G,G)]，train/good 全部 tile 特征。
        显存控制（6GB 卡）：每 tile 随机子采样 ≤256 patch，总量 ≤20 万再上 GPU。"""
        per_tile = 256
        cap = 200_000
        g = torch.Generator().manual_seed(self.seed)     # U24：固定采样，fit 可复现
        chunks = []
        for f in ctx["train_tile_feats"]:
            flat = f.flatten(2).permute(0, 2, 1).reshape(-1, f.shape[1]).float().cpu()
            if flat.shape[0] > per_tile * f.shape[0]:
                keep = torch.randperm(flat.shape[0], generator=g)[: per_tile * f.shape[0]]
                flat = flat[keep]
            chunks.append(flat)
        feats = torch.cat(chunks, dim=0)
        if feats.shape[0] > cap:
            feats = feats[torch.randperm(feats.shape[0], generator=g)[:cap]]
        feats = F.normalize(feats, dim=1).to(self.device)
        n = feats.shape[0]
        target = min(self.coreset_size * 8, n)
        # 贪心 farthest-point coreset（v2：coreset 512 甜点，patch 级放宽到 8 倍）
        idx = [int(torch.randint(n, (1,), generator=g).item())]
        d = 1 - feats @ feats[idx[0]]
        for _ in range(target - 1):
            i = int(torch.argmax(d).item())
            idx.append(i)
            d = torch.minimum(d, 1 - feats @ feats[i])
        self.bank = feats[idx].half()
        return self

    def score_tiles(self, tile_feats, tile_imgs=None):
        """tile_feats: (T,384,G,G)（CPU 或 GPU）-> tile_scores (T,), heatmaps
        内存控制：sims=(T,G*G,M) 全量可达数百 MB（tiles36×4096 bank），
        按 patch 分块算 max 相似度，峰值降 ~8 倍，结果等价。"""
        T, D, G, _ = tile_feats.shape
        flat = tile_feats.flatten(2).permute(0, 2, 1).to(self.device)  # (T,G*G,384)
        flat = F.normalize(flat.float(), dim=-1)
        bank = self.bank.float()
        sims_max = torch.empty(T, G * G, device=self.device, dtype=torch.float32)
        for p0 in range(0, G * G, 128):                # 每块 128 patch
            s = flat[:, p0:p0 + 128] @ bank.T          # (T,128,M)
            sims_max[:, p0:p0 + 128] = s.max(dim=-1).values
        dist = 1 - sims_max                            # 最近邻距离 (T,G*G)
        k = max(1, int(G * G * self.topk_ratio))
        tile_scores = dist.topk(k, dim=-1).values.mean(dim=-1)  # (T,)
        heatmaps = [dist[i].reshape(G, G).cpu().numpy() for i in range(T)]
        return tile_scores.cpu().numpy(), heatmaps

    def update_add(self, new_tile_feats):
        """在线入库接口（M3 双库制前身）：追加并重新 coreset。
        2026-08-30 修复：bank 在 self.device（fit 时 .to(device)），而反馈路径
        传入的 feats 在 CPU（_get_item .cpu() 驻留）——原实现 cat 设备不一致
        抛异常，被 _sem_add 的 try/except 吞掉（静默失败：回流样本从未真正
        并入 sem 扩展库）。统一对齐 bank.device。"""
        new = new_tile_feats.flatten(2).permute(0, 2, 1).reshape(-1, new_tile_feats.shape[1])
        new = F.normalize(new.float(), dim=1).half().to(self.bank.device)
        self.bank = torch.cat([self.bank, new], dim=0)
