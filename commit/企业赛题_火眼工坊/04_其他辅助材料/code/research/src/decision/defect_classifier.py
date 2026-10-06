"""有监督缺陷类型判别器（L3 类型归因，U96，2026-08-24）

U93/U94 归因失败的根因：用"无监督槽位签名 → 规则映射"，槽位签名对五类无判别性
（所有槽位对任何缺陷都饱和）。正确思路：init_defect 缺陷图有类型标签（路径文件夹
名 → 赛题五类），红线 3 允许用 init_defect 训练 → 有监督类型判别器。

验证（scripts/exp_defect_classifier.py）：颜色(HSV 直方图+矩) + 几何(连通域) +
纹理(DoG) + DINO 全局特征(patch 均值) + 随机森林，MVTec 全品类 5 类归因 top-1
0.801（缺件 recall 0.75 / 色彩 0.80 / 外观 0.84），远高于 U93 的无监督映射
（缺件 0 / 色彩 0）。

特征复用：predict 时 DINO feats 由 Pipeline 提供（item["feats"]），零额外前向。
"""
import os
import numpy as np
import cv2

# 赛题五类
FIVE = ["尺寸偏差", "缺件少件", "逻辑错误", "色彩变化", "常见外观缺陷"]

# 缺陷类型（文件夹名，MVTec + data_local）→ 赛题五类（显式归类，其余=外观缺陷）
FIVE_MAP = {
    # 尺寸偏差：几何形变
    "bent": "尺寸偏差", "bent_lead": "尺寸偏差", "bent_wire": "尺寸偏差",
    "squeeze": "尺寸偏差", "squeezed_teeth": "尺寸偏差", "fold": "尺寸偏差",
    "flip": "尺寸偏差", "rough": "尺寸偏差",
    # 缺件少件：缺失/剪断/断齿/多余/墓碑
    "missing_cable": "缺件少件", "missing_wire": "缺件少件",
    "broken_teeth": "缺件少件", "cut_lead": "缺件少件",
    "cut_inner_insulation": "缺件少件", "cut_outer_insulation": "缺件少件",
    "poke_insulation": "缺件少件",
    "missing": "缺件少件", "tombstone": "缺件少件", "extra": "缺件少件",
    # 逻辑错误：顺序/错位
    "cable_swap": "逻辑错误", "misplaced": "逻辑错误", "shift": "逻辑错误",
    # 色彩变化
    "color": "色彩变化",
    # 外观缺陷（显式列出 data_local 焊接/损坏类，语义清晰）
    "damage": "常见外观缺陷", "defect": "常见外观缺陷",
    "bridge": "常见外观缺陷", "cold_solder": "常见外观缺陷",
    "excess": "常见外观缺陷", "insufficient": "常见外观缺陷",
}


def type_from_path(path, category=None, dataset="mvtec"):
    """缺陷图路径 → 赛题五类标签（文件夹名 → FIVE_MAP，未知归外观缺陷）。"""
    folder = os.path.basename(os.path.dirname(str(path)))
    if folder in FIVE_MAP:
        return FIVE_MAP[folder]
    return "常见外观缺陷"


def extract_features(img, feats=None):
    """判别五类缺陷的特征向量：颜色(HSV 直方图+矩) + 几何(连通域) + 纹理(DoG) + DINO。
    feats: DINO patch 特征 (T,384,G,G) tensor，predict 时复用（None 则跳过 DINO 维）。"""
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    # 1. 颜色：HSV H(18)+S(16) 直方图 + H/S/V 矩
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    color = np.concatenate([h_hist, s_hist])
    color = color / (color.sum() + 1e-6)
    hm, sm, vm = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()
    hs, ss, vs = hsv[:, :, 0].std(), hsv[:, :, 1].std(), hsv[:, :, 2].std()
    color_mom = np.array([hm, sm, vm, hs, ss, vs], dtype=np.float32) / 255.0
    # 2. 几何：连通域数量 + 面积统计（归一化）
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    areas = stats[1:, 4].astype(np.float32)
    if len(areas) == 0:
        areas = np.array([0.0], dtype=np.float32)
    total = float(bw.shape[0] * bw.shape[1])
    geom = np.array([len(areas), areas.mean() / total,
                     areas.max() / total, areas.sum() / total], dtype=np.float32)
    # 3. 纹理：DoG 响应统计
    g = gray.astype(np.float32) / 255.0
    dog = np.maximum(
        np.abs(cv2.GaussianBlur(g, (0, 0), 1) - cv2.GaussianBlur(g, (0, 0), 2)),
        np.abs(cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 4)))
    text = np.array([float(dog.mean()), float(dog.max())], dtype=np.float32)
    # 4. DINO 全局特征：patch token 空间均值（384 维）
    dino_feat = np.zeros(0, dtype=np.float32)
    if feats is not None:
        import torch
        if isinstance(feats, torch.Tensor):
            dino_feat = feats.float().mean(dim=(0, 2, 3)).cpu().numpy()
        else:
            dino_feat = np.asarray(feats, dtype=np.float32).reshape(-1)
    return np.concatenate([color, color_mom, geom, text, dino_feat]).astype(np.float32)


class DefectClassifier:
    """有监督类型判别器：init_defect 类型标签训练随机森林，predict 输出五类 top-1。"""

    def __init__(self, seed=42, n_estimators=300):
        self.seed = seed
        self.n_estimators = n_estimators
        self.clf = None
        self.fitted = False

    def fit(self, paths, imgs, feats):
        """paths: init_defect 路径；imgs: 对应 RGB 图；feats: DINO patch 特征 list。
        类型标签由 type_from_path（文件夹名 → FIVE_MAP）自动推断。"""
        from sklearn.ensemble import RandomForestClassifier
        types = [type_from_path(p) for p in paths]
        if len(set(types)) < 2:
            return self           # 单类无法训练（如 BTAD/MPDD 全外观）
        X = np.stack([extract_features(im, f) for im, f in zip(imgs, feats)])
        y = np.array([FIVE.index(t) for t in types])
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators, class_weight="balanced",
            random_state=self.seed, n_jobs=-1)
        self.clf.fit(X, y)
        self.fitted = True
        self._labels = sorted(set(types))
        return self

    def predict(self, img, feats=None):
        """返回 (type, confidence)。未 fit 时返回 None。"""
        if not self.fitted:
            return None
        x = extract_features(img, feats)
        # 维度守卫（2026-08-31 评审修复）：全局判别器含 DINO 384 维（430），
        # CPU 降配模式无主干只能给出图像侧 46 维——回退 None（调用方走规则
        # 映射），不崩溃、不用错维向量乱判。
        n_exp = getattr(self.clf, "n_features_in_", None)
        if n_exp is not None and x.shape[0] != n_exp:
            return None
        proba = self.clf.predict_proba(x[None, :])[0]
        top = int(np.argmax(proba))
        return FIVE[top], float(proba[top])
