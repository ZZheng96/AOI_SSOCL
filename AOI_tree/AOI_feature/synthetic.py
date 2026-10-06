"""合成数据集兜底：无真实数据时验证评估管线端到端可跑（主干 tests 同款范式）

正常图 = 模糊噪声底 + 网格纹理；缺陷图注入三类异常之一：
  blob    亮斑（对应 LoG/SURF/反光组）
  scratch 暗色划痕（对应 Gabor/边缘组）
  tint    色偏（对应色彩组）
生成图带真实分离度但不平凡（噪声幅度与缺陷强度相近）。
"""
import cv2
import numpy as np

IMG_EXT = (".png",)


def _normal_img(rng, size=128):
    base = cv2.GaussianBlur(rng.normal(128, 30, (size, size)).astype(np.float32),
                            (0, 0), 4)
    grid = np.zeros((size, size), np.float32)
    grid[::16, :] = 18
    grid[:, ::16] = 18
    g = base + grid + rng.normal(0, 12, (size, size))
    gray = np.clip(g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _inject(img, rng, kind=None):
    out = img.copy()
    h, w = out.shape[:2]
    kind = kind or ("blob", "scratch", "tint")[int(rng.integers(0, 3))]
    if kind == "blob":
        cx, cy = int(rng.integers(16, w - 16)), int(rng.integers(16, h - 16))
        r = int(rng.integers(3, 7))
        cv2.circle(out, (cx, cy), r, (185, 185, 185), -1)
    elif kind == "scratch":
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(out, p1, p2, (85, 85, 85), 1)
    else:
        out = out.astype(np.float32)
        out[..., 0] += 22
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def make_synthetic(root, category="synthetic", n_train=60, n_test=60,
                   defect_ratio=0.5, seed=42) -> str:
    """在 root/{category}/{train/good,test/{good,defect_synth}} 生成数据，返回品类目录"""
    import os
    rng = np.random.default_rng(seed)
    cat = os.path.join(root, category)
    d_tr = os.path.join(cat, "train", "good")
    d_te_g = os.path.join(cat, "test", "good")
    d_te_d = os.path.join(cat, "test", "defect_synth")
    for d in (d_tr, d_te_g, d_te_d):
        os.makedirs(d, exist_ok=True)
    for i in range(n_train):
        cv2.imwrite(os.path.join(d_tr, f"{i:04d}.png"), _normal_img(rng))
    n_def = int(n_test * defect_ratio)
    for i in range(n_test - n_def):
        cv2.imwrite(os.path.join(d_te_g, f"{i:04d}.png"), _normal_img(rng))
    for i in range(n_def):
        cv2.imwrite(os.path.join(d_te_d, f"{i:04d}.png"),
                    _inject(_normal_img(rng), rng))
    return cat
