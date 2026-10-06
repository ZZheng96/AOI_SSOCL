"""layout 槽位：统计布局模型（尺寸偏差/缺件少件/逻辑顺序，§4.2 第 13 槽位）

无模板场景的诚实版"布局比对"（demo4 pos_bank 是逐网格差分，需要强配准；
layout 不用差分，用连通域统计布局模型——无网络、CPU 原生、天然可解释）。
不依赖 DINO（needs_dino=False）。

fit：对每张 train/good 图提取连通域，统计布局模型：
  - 元件数量分布（缺件/多件）
  - 元件面积分布（尺寸偏差/异物）
  - 元件间距分布（错位/偏移）
  - 左右对称性（逻辑顺序/装配对称性）

score：对测试图提取连通域，四类违例各自打分，图像分数 = 违例最大标准化偏差。
heatmap：每个元件一个"异常强度"，映射到 32x32 网格（供检测框/追溯）。
"""
import numpy as np
import cv2
from .base import Slot


class LayoutSlot(Slot):
    name = "layout"
    needs_dino = False

    def __init__(self, cfg, device="cpu"):
        self.min_area = cfg.get("min_area", 30)          # 连通域最小面积
        self.max_components = cfg.get("max_components", 60)
        self.area_percentile = cfg.get("area_percentile", 0.95)  # 面积违例阈值分位
        # 速度优化（§6.1）：布局统计工作网格（连通域/对称性在缩略图算，
        # 元件相对结构语义保留；min_area 按比例缩放）。layout 是图像级弱槽位
        # （gold_finger AUROC 0.17），提速不改主检。
        self.work_side = cfg.get("work_side", 1024)

    def _prepare(self, img):
        """缩放到 work_side 工作网格，返回 (img_scaled, scale)"""
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > self.work_side:
            scale = self.work_side / max(h, w)
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        return img, scale

    # ── 连通域提取（Otsu 自适应，工作网格） ──────────────────────
    def _components(self, img, scale=1.0):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        # Otsu 二值化：亮元件 vs 暗背景（或反之，取较大连通域集合）
        th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        min_area_eff = max(1.0, self.min_area * scale * scale)   # 面积随缩放平方
        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area_eff:
                comps.append((x, y, w, h, area))
        # 取面积降序前 max_components
        comps.sort(key=lambda c: -c[4])
        return comps[: self.max_components]

    def _spacing(self, comps):
        """相邻元件间距：取排序后相邻质心距离的中位（布局紧凑度）"""
        if len(comps) < 2:
            return 0.0
        pts = np.array([(c[0] + c[2] / 2, c[1] + c[3] / 2) for c in comps])
        d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        return float(np.median(d))

    def _symmetry(self, img):
        """左右对称性：图像左半与镜像右半的差异"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        h, w = gray.shape
        half = w // 2
        left = gray[:, :half]
        right = gray[:, w - half:][:, ::-1]
        return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))

    # ── fit ───────────────────────────────────────────────────────
    def fit(self, ctx):
        counts, areas, spacings, syms = [], [], [], []
        for tiles in ctx["train_tile_imgs"]:
            img = tiles[0]  # single 模式：tile 即整图（图像级槽位）
            img, scale = self._prepare(img)
            comps = self._components(img, scale)
            counts.append(len(comps))
            areas.extend(c[4] for c in comps)
            spacings.append(self._spacing(comps))
            syms.append(self._symmetry(img))
        self.count_median = float(np.median(counts)) if counts else 0.0
        self.count_std = float(np.std(counts)) + 1e-6
        self.area_hi = float(np.percentile(areas, self.area_percentile * 100)) if areas else 1e9
        self.spacing_median = float(np.median(spacings)) if spacings else 0.0
        self.spacing_std = float(np.std(spacings)) + 1e-6
        self.sym_median = float(np.median(syms)) if syms else 0.0
        self.sym_std = float(np.std(syms)) + 1e-6
        return self

    # ── score ─────────────────────────────────────────────────────
    def score_tiles(self, tile_feats=None, tile_imgs=None):
        """四类违例 → 图像分数（tile 分数路径，heatmap=None——layout 是图像级槽位，
        热力图语义是"元件异常强度"而非 patch 分数图，不进 multiscale 聚合）。"""
        img, scale = self._prepare(tile_imgs[0])
        comps = self._components(img, scale)
        n = len(comps)
        s_count = abs(n - self.count_median) / self.count_std        # 缺件/多件
        s_area = max((c[4] / max(self.area_hi, 1.0) - 1.0) for c in comps) \
            if comps and self.area_hi > 0 else 0.0                    # 超大元件/异物
        s_area = max(s_area, 0.0)
        sp = self._spacing(comps)
        s_space = abs(sp - self.spacing_median) / self.spacing_std if n > 1 else 0.0
        sy = self._symmetry(img)
        s_sym = abs(sy - self.sym_median) / self.sym_std
        # U94（2026-08-24）：记录主导违例类型——缺件少件（数量）/ 尺寸偏差（面积）/
        # 逻辑错误（间距/错位 + 对称）。归因层（hierarchical.attribute_type）读此字段
        # 替代掩码启发式（U93 缺件归因 0% 的根因：layout 内部有缺件信号但没传出去）。
        viol = {"缺件少件": s_count, "尺寸偏差": s_area, "逻辑错误": max(s_space, s_sym)}
        self._last_violation = max(viol, key=viol.get) if max(viol.values()) > 0.0 \
            else "尺寸偏差"
        score = max(s_count, s_area, s_space, s_sym)
        return np.asarray([score]), [None]
