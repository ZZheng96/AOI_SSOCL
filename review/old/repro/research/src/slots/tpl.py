"""tpl 槽位（§4.4 热插拔示例，L3 场景）：模板差分（域不变）

核心（demo4 s_r 的 L3 合法化版本）：
- 用户**显式**提供模板图（L3 场景，红线：任何从文件名推断配对的行为=泄漏，禁用）
- 检测图与模板**同一管线提特征**（同域），逐位置差分 → 域不变
- 差分 = DINO patch 特征 L2 + 像素/色彩差分 加权融合（双通道互补）

demo4 s_r 教训：配对差分本身极强（AUROC 1.0 的域不变分支），但靠文件名隐式配对=作弊；
L3 把配对升级为显式模板契约——保留机理，去除泄漏。
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .base import Slot


class TplSlot(Slot):
    name = "tpl"
    needs_dino = True

    def __init__(self, cfg, backbone, device="cuda"):
        self.backbone = backbone
        self.device = device
        self.templates = []          # list[dict]: {"feat": (384,G,G), "pix": HxWx3 uint8}
        self.feat_w = cfg.get("feat_weight", 0.7)   # DINO 特征差分权重
        self.pix_w = cfg.get("pix_weight", 0.3)     # 像素差分权重
        self.topk_ratio = cfg.get("topk_ratio", 0.1)

    def fit(self, ctx):
        """ctx["templates"]: list[ndarray HxWx3 uint8]（L3 显式模板图）"""
        tpl_paths = ctx.get("templates", [])
        if not tpl_paths:
            return self   # 无模板：L3 未激活，槽位空转（热插拔协议：缺条件退化为 no-op）
        target = self.backbone.grid * 14   # 像素差分在 448 网格缓存（score 零 resize）
        for p in tpl_paths:
            from ..common.io import load_image
            img = load_image(p)
            feats = self.backbone.extract_tiles([img]).cpu()[0]   # (384,G,G)
            pix = cv2.resize(img, (target, target)) \
                if img.shape[:2] != (target, target) else img
            self.templates.append({"feat": feats, "pix": pix})
        return self

    def score_tiles(self, tile_feats, tile_imgs=None):
        """tile_feats: (T,384,G,G)（T=1 单图）-> (tile_scores, heatmaps)
        与模板逐位置差分：取 min 差分（多模板取最近邻）。"""
        if not self.templates:
            return np.array([0.0]), [None]
        f = tile_feats.float().to(self.device)          # (T,384,G,G)
        G = f.shape[-1]
        # DINO 特征差分 (T,G,G)
        best_feat = None
        for t in self.templates:
            tf = t["feat"].float().to(self.device)      # (384,G,G)
            diff = (f - tf.unsqueeze(0)).norm(dim=1)    # (T,G,G) L2 逐位置
            best_feat = diff if best_feat is None else torch.minimum(best_feat, diff)
        best_pix = None
        if tile_imgs and tile_imgs[0] is not None:
            for t in self.templates:
                tp = t["pix"]                                  # fit 已缓存 448（grid*14）
                th, tw = tp.shape[:2]
                ih, iw = tile_imgs[0].shape[:2]
                if (th, tw) != (ih, iw):
                    tp = cv2.resize(tp, (iw, ih))
                pdiff = torch.tensor(
                    np.abs(tile_imgs[0].astype(np.float32) - tp.astype(np.float32)).mean(axis=-1),
                    device=self.device)                # (H,W) -> 下采样到 G
                pdiff = F.interpolate(pdiff.unsqueeze(0).unsqueeze(0), size=(G, G),
                                      mode="bilinear").squeeze()
                best_pix = pdiff if best_pix is None else torch.minimum(best_pix, pdiff)
        if best_pix is None:
            combined = best_feat                              # (T,G,G)
        else:
            # 归一化后加权融合（尺度不同，先各自归一）
            bf = best_feat / (best_feat.max() + 1e-8)
            bp = best_pix / (best_pix.max() + 1e-8)
            combined = self.feat_w * bf + self.pix_w * bp
        # 图像分 = 差分图 top-k 均值（topk_ratio）
        k = max(1, int(G * G * self.topk_ratio))
        scores = combined.flatten(1).topk(k, dim=1).values.mean(dim=1)
        heatmaps = [combined[i].cpu().numpy() for i in range(combined.shape[0])]
        return scores.cpu().numpy(), heatmaps
