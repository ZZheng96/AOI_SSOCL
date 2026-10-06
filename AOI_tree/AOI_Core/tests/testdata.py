"""回归测试数据供给：不依赖特定数据集。

约定目录结构（MVTec-like）：
    {品类根}/train/good/*.png        正常图（>=10 张）
    {品类根}/test/good/*.png         测试正常图（>=2 张）
    {品类根}/test/<缺陷类型>/*.png   缺陷图（合计 >=3 张）

选取顺序：
1. 环境变量 AOI_TEST_DATA（品类目录或数据集根）
2. {AOI_DATA_HOME 或 <仓库>/data} 下第一个满足结构的品类（含 data_origin 等
   下级数据集目录，逐层探测）
3. 仓库同级目录 data_local 下第一个满足结构的品类
4. 兜底：临时目录生成合成数据集（任何机器都可跑回归）

用法：
    from tests.testdata import pick_category, first_normal_image
    cat_dir, cat_name, synthetic = pick_category()
    img = first_normal_image()
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg")


def list_imgs(d, n=None):
    try:
        items = sorted(os.path.join(d, x) for x in os.listdir(d)
                       if x.lower().endswith(IMG_EXTS))
    except OSError:
        return []
    return items[:n] if n else items


def _is_valid_category(cat_dir) -> bool:
    """结构校验：train/good>=10 张、test/good>=2 张、缺陷图合计>=3 张。"""
    train_good = os.path.join(cat_dir, "train", "good")
    test_dir = os.path.join(cat_dir, "test")
    if len(list_imgs(train_good, 10)) < 10:
        return False
    if not os.path.isdir(test_dir) or len(list_imgs(os.path.join(test_dir, "good"), 2)) < 2:
        return False
    n_defect = 0
    for sub in sorted(os.listdir(test_dir)):
        dd = os.path.join(test_dir, sub)
        if os.path.isdir(dd) and sub != "good":
            n_defect += len(list_imgs(dd, 3))
    return n_defect >= 3


def _candidate_roots():
    env = os.environ.get("AOI_TEST_DATA")
    if env:
        yield Path(env)
    data_home = Path(os.environ.get("AOI_DATA_HOME") or (REPO_ROOT / "data"))
    # data_home 下任意数据集目录（data_origin/mvtec、data_origin/BTAD、…）
    origin = data_home / "data_origin"
    if origin.is_dir():
        for sub in sorted(os.listdir(origin)):
            sd = origin / sub
            if sd.is_dir():
                yield sd
    yield REPO_ROOT.parent / "data_local"
    yield data_home


def pick_category():
    """返回 (品类目录, 品类名, 是否合成数据)。优先真实数据集，兜底合成。"""
    for root in _candidate_roots():
        if not root.is_dir():
            continue
        if _is_valid_category(root):
            return str(root), root.name, False
        for cat in sorted(os.listdir(root)):
            cd = root / cat
            if cd.is_dir() and _is_valid_category(cd):
                return str(cd), cat, False
    return _make_synthetic()


def first_normal_image() -> str:
    """任意一张可用的正常图路径（当前探测品类的 train/good 第一张）。"""
    cat_dir, _, _ = pick_category()
    return list_imgs(os.path.join(cat_dir, "train", "good"), 1)[0]


# ── 合成数据集（兜底，保证回归无数据集也可跑）──────────────────
def _texture(seed: int):
    """平滑底纹 + 微噪声，不同 seed 结构一致 → 同分布正常图。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 2 * np.pi, 256)
    yy, xx = np.meshgrid(xs, xs)
    img = np.stack([
        150 + 50 * np.sin(1.5 * yy + 0.3 * seed),
        130 + 45 * np.cos(1.2 * xx + 0.5 * seed),
        120 + 40 * np.sin(0.9 * (yy + xx) + 0.7 * seed),
    ], axis=-1).astype(np.float32)
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_defect(img, seed: int):
    """在底纹上叠加显著异物（暗斑 + 亮线），与正常图强可分。"""
    import numpy as np

    out = img.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    h, w = out.shape[:2]
    # 暗斑
    cy, cx = rng.integers(60, 190, 2)
    r = int(rng.integers(16, 26))
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    out[mask] *= 0.35
    # 亮线
    x0 = int(rng.integers(0, w - 60))
    y0 = int(rng.integers(0, h))
    out[y0:y0 + 3, x0:x0 + 50] = np.clip(out[y0:y0 + 3, x0:x0 + 50] + 70, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_synthetic():
    import cv2

    root = Path(tempfile.mkdtemp(prefix="aoi_synth_")) / "synth"
    for sub in ("train/good", "test/good", "test/stain", "test/scratch"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i in range(10):
        cv2.imwrite(str(root / "train" / "good" / f"syn_{i:02d}.png"), _texture(i))
    for i in range(3):
        cv2.imwrite(str(root / "test" / "good" / f"syn_t{i:02d}.png"), _texture(100 + i))
    for kind, seeds in (("stain", (200, 201)), ("scratch", (300, 301))):
        for i, seed in enumerate(seeds):
            cv2.imwrite(str(root / "test" / kind / f"syn_d{i:02d}.png"),
                        _add_defect(_texture(seed), seed))
    print("[testdata] 未找到可用数据集，已生成合成数据（MVTec-like 结构）"
          f"：{root}", flush=True)
    return str(root), "synth", True


if __name__ == "__main__":
    d, c, s = pick_category()
    print(f"category_dir={d}\ncategory={c}\nsynthetic={s}")
    print("n_train_good =", len(list_imgs(os.path.join(d, "train", "good"))))
