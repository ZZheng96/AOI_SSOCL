"""tpl 槽位（§4.4 热插拔示例，L3 场景）：模板差分（域不变）

核心（demo4 s_r 的 L3 合法化版本）：
- 用户**显式**提供模板图（L3 场景）。红线（round5 修订）：L3 下由用户/上游**显式**
  提供的修复图配对（pair_template / resolve_pair_template 作为上游配对契约）合法；
  非 L3 场景仍禁止任何从文件名推断配对的行为（=泄漏）。pair_mode 默认关闭。
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
        # I2（2026-10-06）：无显式模板时从 init_normal 自动建模板库（最多 auto_max 张），
        # 测试图按相位相关选最近模板做逐位置差分（位置感知通道）。
        self.auto_from_normal = bool(cfg.get("auto_from_normal", False))
        self.auto_max = int(cfg.get("auto_max", 40))
        # 自匹配排除：与某模板逐位置差分均值 < self_eps（同一张图）时不参与 min，
        # 否则 fit 域正常图在自身模板上≈0，校准分布退化。
        self.self_eps = float(cfg.get("self_eps", 0.05))
        # round5 加速：配准候选预筛（模板特征均值余弦相似 top-k 才做相位相关）。
        # 0=关闭（默认，逐模板相位相关，数值与原路径一致）。差分 min 仍对全部模板。
        self.align_prefilter_k = int(cfg.get("align_prefilter_k", 0))
        # round5 L3 配对模式：score_image 显式给 pair_template 时只与该模板配准+差分
        # （不做全局 min / self_eps 排除）。默认关闭，关闭时行为与数值不变。
        self.pair_mode = bool(cfg.get("pair_mode", False))
        self._pair_cache = {}         # path -> (feat (1,384,G,G) cpu, gray448 f32)
        self._cache_key = None        # 懒缓存键（templates 列表 id + 长度；快照还原后自动重建）
        self._T = None                # (N,384,G,G) device 堆叠模板特征
        self._T_mean = None           # (N,384) 归一化模板均值特征（预筛用）
        self._gray_f32 = None         # list[float32 448 灰度 | None]

    def _ensure_cache(self):
        key = (id(self.templates), len(self.templates))
        if getattr(self, "_cache_key", None) == key:   # 旧快照对象无该属性
            return
        self._T = torch.stack([t["feat"] for t in self.templates]).float().to(self.device)
        self._T_mean = F.normalize(self._T.mean(dim=(2, 3)), dim=1)
        self._gray_f32 = [None if t.get("gray_small") is None
                          else t["gray_small"].astype(np.float32) for t in self.templates]
        self._cache_key = key

    def fit(self, ctx):
        """ctx["templates"]: list[path]（L3 显式模板图）；
        auto_from_normal 时缺省用 ctx["train_imgs"]（init_normal 整图）。"""
        tpl_paths = ctx.get("templates", [])
        imgs = None
        if not tpl_paths and self.auto_from_normal and ctx.get("train_imgs"):
            imgs = list(ctx["train_imgs"])[:self.auto_max]
        if not tpl_paths and not imgs:
            return self   # 无模板：L3 未激活，槽位空转（热插拔协议：缺条件退化为 no-op）
        target = self.backbone.grid * 14   # 像素差分在 448 网格缓存（score 零 resize）
        from ..common.io import load_image
        for p in (imgs if imgs is not None else tpl_paths):
            img = p if isinstance(p, np.ndarray) else load_image(p)
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
        self._ensure_cache()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        small = cv2.resize(g, (448, 448)).astype(np.float32)
        cand = range(len(self.templates))
        k = getattr(self, "align_prefilter_k", 0)
        if k and mean_feat is not None and k < len(self.templates):
            q = F.normalize(mean_feat.float().to(self.device).reshape(1, -1), dim=1)
            sim = (self._T_mean @ q.t()).squeeze(1)
            cand = sorted(sim.topk(k).indices.tolist())
        best_idx, best_resp, best_shift = 0, -1.0, (0.0, 0.0)
        for i in cand:
            rg = self._gray_f32[i]
            if rg is None:
                continue
            try:
                sh, rs = cv2.phaseCorrelate(rg, small)
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

    def _pair_entry(self, tpl):
        """配对模板 → (feat (1,384,G,G) cpu, gray448 float32)；路径按 key 缓存。"""
        key = tpl if isinstance(tpl, str) else None
        cache = getattr(self, "_pair_cache", None)
        if cache is None:
            cache = self._pair_cache = {}
        if key is not None and key in cache:
            return cache[key]
        if isinstance(tpl, np.ndarray):
            img = tpl
        else:
            from ..common.io import load_image
            img = load_image(tpl)
        feat = self.backbone.extract_tiles([img]).cpu()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.resize(g, (448, 448)).astype(np.float32)
        ent = (feat, gray)
        if key is not None:
            if len(cache) > 256:
                cache.clear()
            cache[key] = ent
        return ent

    def _score_pair(self, img, feats, pair_template):
        """配对打分：单模板相位相关配准 → 逐位置特征 L2 差分 → top-k 均值。"""
        tf, tg = self._pair_entry(pair_template)
        img_a = img
        if self.align:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            small = cv2.resize(g, (448, 448)).astype(np.float32)
            try:
                (dx, dy), rs = cv2.phaseCorrelate(tg, small)
            except cv2.error:
                rs, dx, dy = -1.0, 0.0, 0.0
            if rs >= self.align_min_resp and (abs(dx) >= self.align_min_shift
                                              or abs(dy) >= self.align_min_shift):
                h, w = img.shape[:2]
                M = np.float32([[1, 0, -dx * w / 448.0], [0, 1, -dy * h / 448.0]])
                img_a = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        if img_a is img and feats is not None:
            f = feats
        else:
            f = self.backbone.extract_tiles([img_a]).cpu()
        f = f.float().to(self.device)
        d = (f - tf.float().to(self.device)).norm(dim=1)          # (1,G,G)
        G = d.shape[-1]
        k = max(1, int(G * G * self.topk_ratio))
        s = d.flatten(1).topk(k, dim=1).values.mean(dim=1).cpu().numpy()
        return float(s[0]), s, [d[0].cpu().numpy()]

    def score_image(self, img, feats=None, pair_template=None):
        """整图打分（配准路径）：相位相关选参考模板 → 配准 → 差分。
        返回 (image_score, tile_scores, heatmaps)；未启用配准/无模板时 None
        （回退 score_tiles 路径，复用管线已提取特征，零额外前向）。
        feats：调用方已提取的整图特征 (1,384,G,G)（round5：未配准移动时直接复用，
        省去原路径无条件的一次前向；数值与原路径一致）。
        pair_template：L3 显式配对模板（路径/图像）；仅 pair_mode 开启时生效，
        None 时回退原全局模板库路径。"""
        if pair_template is not None and getattr(self, "pair_mode", False):
            return self._score_pair(img, feats, pair_template)
        if not self.templates or not self.align:
            return None
        k = getattr(self, "align_prefilter_k", 0)
        mean_feat = None
        if k and k < len(self.templates):
            if feats is None:
                feats = self.backbone.extract_tiles([img]).cpu()
            mean_feat = feats[0].float().mean(dim=(1, 2))
        img_a = self._align_to_ref(img, mean_feat=mean_feat)
        if img_a is img:
            f = feats if feats is not None else self.backbone.extract_tiles([img]).cpu()
        else:
            f = self.backbone.extract_tiles([img_a]).cpu()
        ts, hms = self.score_tiles(f, [img_a])
        return float(ts[0]), ts, hms

    def score_tiles(self, tile_feats, tile_imgs=None):
        """tile_feats: (T,384,G,G)（T=1 单图）-> (tile_scores, heatmaps)
        与模板逐位置差分：取 min 差分（多模板取最近邻）。"""
        if not self.templates:
            return np.array([0.0]), [None]
        f = tile_feats.float().to(self.device)          # (T,384,G,G)
        G = f.shape[-1]
        # DINO 特征差分 (T,G,G)
        # round5 加速：模板特征懒缓存在 device 上，按模板分块批量差分
        # （逐位置 L2 → 自匹配排除 → 沿模板取 min，与逐模板循环数值一致）
        self._ensure_cache()
        best_feat = None
        chunk = 16
        for s0 in range(0, self._T.shape[0], chunk):
            Tc = self._T[s0:s0 + chunk]                                  # (n,384,G,G)
            diff = (f.unsqueeze(1) - Tc.unsqueeze(0)).norm(dim=2)        # (T,n,G,G)
            same = diff.flatten(2).mean(dim=2) < self.self_eps           # (T,n) 自匹配排除
            if bool(same.any()):
                diff[same] = float("inf")
            m = diff.min(dim=1).values                                   # (T,G,G)
            best_feat = m if best_feat is None else torch.minimum(best_feat, m)
        if torch.isinf(best_feat).any():          # 仅有自身模板时退化为 0
            best_feat = torch.nan_to_num(best_feat, posinf=0.0)
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


_PAIR_EXTS = (".jpg", ".png", ".jpeg", ".bmp")


def resolve_pair_template(path, tpl_dirs=None):
    """L3 上游配对契约：主干 X → X_repaired.*（同目录优先，其次 tpl_dirs 顺序）。
    口径：输入本身是 *_repaired（正常/修复图）→ None（调用方回退原全局模板库逻辑）；
    找不到 → None。仅 L3 显式启用 pair_mode 时使用。"""
    import os
    stem, _ = os.path.splitext(os.path.basename(str(path)))
    if stem.endswith("_repaired"):
        return None
    dirs = [os.path.dirname(str(path))] + [str(d) for d in (tpl_dirs or [])]
    for d in dirs:
        for ext in _PAIR_EXTS:
            c = os.path.join(d, f"{stem}_repaired{ext}")
            if os.path.isfile(c):
                return c
    return None
