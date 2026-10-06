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
        # ② 配准（2026-08-30，P0 完善）：测试图相位相关对齐到参考模板坐标系，
        # 消除产线对位漂移（gold_finger offset 最高 100+px）。默认开；相关性
        # 峰值过低（低纹理/内容差异大）或偏移过小时跳过配准（防误对齐）。
        self.align = bool(cfg.get("align", True))
        self.align_ref = None         # 参考模板灰度（448，配准基准）
        self.align_min_resp = float(cfg.get("align_min_resp", 0.1))
        self.align_min_shift = float(cfg.get("align_min_shift", 1.0))

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
            # 像素差分通道仅在启用时缓存（大图内存 ~384×448×448×3B/张；
            # 2026-08-30 实测：像素差分通道是 tpl 模式 3-7s 超 1s 红线的元凶，
            # 默认关闭走纯 DINO 特征差分——85 模板实测 235ms/图）
            entry = {"feat": feats}
            if self.pix_w > 0:
                pix = cv2.resize(img, (target, target)) \
                    if img.shape[:2] != (target, target) else img
                entry["pix"] = pix
            if self.align:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                entry["gray"] = cv2.resize(g, (target, target))
                # 配准用缩略（448=特征网格分辨率；224 上 1px 误差→原图 23px，
                # 448 上精度 ~0.5px 级，实测校正残差显著改善）
                entry["gray_small"] = cv2.resize(g, (target, target))
            self.templates.append(entry)
        if self.align and self.templates:
            # 配准参考灰度（默认第一张；score 时按相位相关响应动态选）
            self.align_ref = self.templates[0]["gray_small"]
        return self

    def _align_to_ref(self, img, mean_feat=None):
        """相位相关把测试图对齐到"响应最高的模板"坐标系（同产品金样板）。
        mean_feat 保留兼容（DINO 均值选模板对平移不敏感，但不同板卡相似
        时可能选错——改用逐模板相位相关响应 argmax，实测更稳）。
        相关性过低（不同产品/低纹理）或偏移过小则原样返回。"""
        if not self.templates:
            return img
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        small = cv2.resize(g, (448, 448))
        best_idx, best_resp, best_shift = 0, -1.0, (0.0, 0.0)
        for i, t in enumerate(self.templates):
            rg = t.get("gray_small")
            if rg is None:
                continue
            try:
                sh, rs = cv2.phaseCorrelate(
                    rg.astype(np.float32), small.astype(np.float32))
            except cv2.error:
                continue
            if rs > best_resp:
                best_idx, best_resp, best_shift = i, rs, sh
        if best_resp < self.align_min_resp:
            return img
        dx, dy = best_shift
        if abs(dx) < self.align_min_shift and abs(dy) < self.align_min_shift:
            return img
        h, w = img.shape[:2]
        # 缩略(448)上的偏移放大回原图分辨率后取负校正（实测校准 2026-08-30：
        # phaseCorrelate(ref, src) 返回 src 相对 ref 的偏移；同内容偏移下
        # shift=(+dx,+dy)，warp 用 -shift×scale；方向与尺度均已验证）。
        scale_x = w / 448.0
        scale_y = h / 448.0
        M = np.float32([[1, 0, -dx * scale_x], [0, 1, -dy * scale_y]])
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def score_image(self, img):
        """整图打分（配准路径）：相位相关选参考模板 → 配准 → 差分。
        返回 (image_score, tile_scores, heatmaps)；未启用配准/无模板时 None
        （回退 score_tiles 路径，复用管线已提取特征，零额外前向）。"""
        if not self.templates or not self.align:
            return None
        feats0 = self.backbone.extract_tiles([img]).cpu()
        img_a = self._align_to_ref(img)
        feats = feats0 if img_a is img else \
            self.backbone.extract_tiles([img_a]).cpu()
        ts, hms = self.score_tiles(feats, [img_a])
        return float(ts[0]), ts, hms

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
        if self.pix_w > 0 and tile_imgs and tile_imgs[0] is not None:
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
