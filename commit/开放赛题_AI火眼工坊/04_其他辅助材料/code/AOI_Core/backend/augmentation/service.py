"""伪异常生成 + 数据预处理/增强服务（赛题问题二：少样本启动的支撑模块）。

生成方法（全部基于正常图，无需真实缺陷）：
  - cutpaste    : 从图内随机裁剪块旋转/缩放后粘贴回随机位置（复用 demo1 思想，本模块自实现）
  - scar        : 随机细长划痕 + 颜色扰动（DRAEM 风格）
  - noise       : 局部高斯/椒盐噪声块
  - color_shift : 局部色彩/亮度偏移（模拟色彩变化类缺陷）

输出统一写入 storage/pseudo/ 并登记 PseudoAnomaly 表 + Image 表(split='pseudo', label='anomaly')。
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import get_settings
from ..core.datasets import create_dataset, refresh_dataset_count
from ..core.imaging import imread, imwrite
from ..core.ingest import register_image
from ..db.database import session_scope
from ..db.models import Image as ImageRow
from ..db.models import PseudoAnomaly

METHODS = ["cutpaste", "scar", "noise", "color_shift"]


def _ts() -> str:
    """时间戳串（用于输出文件名）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── 伪异常生成原语（返回 (异常图BGR, 二值掩码)）────────────────────
def _rand_rect(rng: np.random.Generator, h: int, w: int,
               area_range: Tuple[float, float]) -> Tuple[int, int, int, int]:
    """按面积比例范围随机一个矩形，返回 (x, y, rw, rh)。"""
    ratio = rng.uniform(area_range[0], area_range[1])
    area = h * w * ratio
    rh = max(4, int(math.sqrt(area * h / w)))
    rw = max(4, int(area / rh))
    rh = min(rh, h - 1)
    rw = min(rw, w - 1)
    x = int(rng.integers(0, w - rw + 1))
    y = int(rng.integers(0, h - rh + 1))
    return x, y, rw, rh


def _gen_cutpaste(img: np.ndarray, rng: np.random.Generator,
                  params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """CutPaste：图内随机裁块 → 旋转/缩放/翻转/颜色抖动 → 贴回随机位置。"""
    h, w = img.shape[:2]
    area_range = params.get("area_range", (0.03, 0.15))  # 面积占比3%~15%
    sx, sy, pw, ph = _rand_rect(rng, h, w, area_range)
    patch = img[sy:sy + ph, sx:sx + pw].copy()
    pmask = np.full((ph, pw), 255, np.uint8)

    # 随机旋转±30° + 缩放0.8~1.2（反射填充避免黑边）
    scale = float(rng.uniform(0.8, 1.2))
    angle = float(rng.uniform(-30, 30))
    M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, scale)
    patch = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_REFLECT)
    pmask = cv2.warpAffine(pmask, M, (pw, ph), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)

    # 随机翻转
    flip = int(rng.integers(-1, 2))  # -1 不翻, 0 垂直, 1 水平
    if flip >= 0:
        patch = cv2.flip(patch, flip)
        pmask = cv2.flip(pmask, flip)

    # 颜色抖动，使贴块与背景存在反差
    jitter = rng.uniform(0.8, 1.2, size=(1, 1, 3)).astype(np.float32)
    offset = float(rng.uniform(-15, 15))
    patch = np.clip(patch.astype(np.float32) * jitter + offset, 0, 255).astype(np.uint8)

    # 羽化边缘，避免贴块边界生硬
    pmask = cv2.GaussianBlur(pmask, (5, 5), 0)

    out = img.copy()
    mask = np.zeros((h, w), np.uint8)
    dx = int(rng.integers(0, w - pw + 1))
    dy = int(rng.integers(0, h - ph + 1))
    roi = out[dy:dy + ph, dx:dx + pw]
    alpha = (pmask.astype(np.float32) / 255.0)[..., None]
    roi[:] = np.clip(roi.astype(np.float32) * (1 - alpha)
                     + patch.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    mask[dy:dy + ph, dx:dx + pw] = pmask
    return out, mask


def _gen_scar(img: np.ndarray, rng: np.random.Generator,
              params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """划痕：随机游走生成细长曲线，灰度与周围形成反差并叠加轻微噪声。"""
    h, w = img.shape[:2]
    cfg = get_settings().section("pseudo_anomaly")
    length_range = params.get("scar_length", cfg.get("scar_length", [10, 100]))
    width_range = params.get("scar_width", cfg.get("scar_width", [3, 15]))
    length = int(rng.integers(int(length_range[0]), int(length_range[1]) + 1))
    width = int(rng.integers(int(width_range[0]), int(width_range[1]) + 1))

    # 随机游走轨迹（步长随图尺寸缩放，方向带扰动）
    x = float(rng.uniform(w * 0.2, w * 0.8))
    y = float(rng.uniform(h * 0.2, h * 0.8))
    angle = float(rng.uniform(0, 2 * math.pi))
    step = max(2.0, min(h, w) / max(length, 1) * 1.2)
    pts = []
    for _ in range(length):
        pts.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
        angle += float(rng.normal(0, 0.35))
        x += math.cos(angle) * step * float(rng.uniform(0.6, 1.4))
        y += math.sin(angle) * step * float(rng.uniform(0.6, 1.4))

    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, width,
                  lineType=cv2.LINE_AA)
    mask = (mask > 0).astype(np.uint8) * 255

    # 划痕颜色：取区域均值后叠加亮度±40的随机扰动，形成反差
    mean_bgr = np.array(cv2.mean(img, mask=mask)[:3], np.float32)
    color = mean_bgr + rng.uniform(-40, 40, size=3).astype(np.float32)
    noise = rng.normal(0, 8, img.shape).astype(np.float32)  # 轻微噪声
    scar_layer = np.clip(color.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)

    out = img.copy()
    sel = mask > 0
    out[sel] = scar_layer[sel]
    # 边缘轻微羽化，避免生硬边界
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return out, mask


def _gen_noise(img: np.ndarray, rng: np.random.Generator,
               params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """局部噪声块：随机矩形内加高斯或椒盐噪声。"""
    h, w = img.shape[:2]
    x, y, rw, rh = _rand_rect(rng, h, w, params.get("area_range", (0.02, 0.10)))
    out = img.copy()
    roi = out[y:y + rh, x:x + rw]

    if rng.random() < 0.5:
        # 高斯噪声（std 20~50）
        sigma = float(params.get("gauss_sigma", rng.uniform(20, 50)))
        noisy = roi.astype(np.float32) + rng.normal(0, sigma, roi.shape)
        out[y:y + rh, x:x + rw] = np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        # 椒盐噪声
        prob = float(params.get("sp_prob", rng.uniform(0.05, 0.15)))
        rnd = rng.random(roi.shape[:2])
        roi[rnd < prob / 2] = 0
        roi[rnd > 1 - prob / 2] = 255
        out[y:y + rh, x:x + rw] = roi

    mask = np.zeros((h, w), np.uint8)
    mask[y:y + rh, x:x + rw] = 255
    return out, mask


def _gen_color_shift(img: np.ndarray, rng: np.random.Generator,
                     params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """色彩偏移：随机矩形/自由形状区域内做 HSV 色相/饱和度或亮度偏移。"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if rng.random() < 0.5:
        # 矩形区域
        x, y, rw, rh = _rand_rect(rng, h, w, params.get("area_range", (0.02, 0.12)))
        mask[y:y + rh, x:x + rw] = 255
    else:
        # 椭圆区域（随机中心/轴长/旋转角）
        cx = int(rng.uniform(0.25, 0.75) * w)
        cy = int(rng.uniform(0.25, 0.75) * h)
        ax = int(rng.uniform(0.05, 0.2) * w)
        ay = int(rng.uniform(0.05, 0.2) * h)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, (cx, cy), (max(ax, 4), max(ay, 4)), angle, 0, 360,
                    255, thickness=-1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    mode = int(rng.integers(0, 3))
    if mode == 0:
        # 色相偏移 ±20（OpenCV 色相范围 0~179）
        hsv[..., 0] = (hsv[..., 0] + float(rng.uniform(-20, 20))) % 180
    elif mode == 1:
        # 饱和度偏移 ±40%
        hsv[..., 1] = hsv[..., 1] * float(rng.uniform(0.6, 1.4))
    else:
        # 亮度偏移 ±30%
        hsv[..., 2] = hsv[..., 2] * float(rng.uniform(0.7, 1.3))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    out = img.copy()
    sel = mask > 0
    out[sel] = shifted[sel]
    return out, mask


_GENERATORS = {
    "cutpaste": _gen_cutpaste,
    "scar": _gen_scar,
    "noise": _gen_noise,
    "color_shift": _gen_color_shift,
}


def _generate_one(img: np.ndarray, method: str, params: Dict,
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """按方法生成单张伪异常（内部统一入口）。"""
    if method not in _GENERATORS:
        raise ValueError(f"未知伪异常方法: {method}，可选: {METHODS}")
    rng = np.random.default_rng(seed)
    return _GENERATORS[method](img, rng, params)


def _ensure_base_image(path: Path, category: str) -> int:
    """确保底图已登记到 Image 表（缺失则按 train/normal 登记），返回 image_id。"""
    with session_scope() as s:
        row = s.query(ImageRow).filter(ImageRow.path == str(path)).first()
        if row is not None:
            return row.id
    return register_image(path, category=category, split="train",
                          label="normal", source="manual", copy_to_storage=False)


def generate_pseudo_anomalies(
    base_image_paths: List[str | Path],
    method: str = "cutpaste",
    count_per_image: int = 4,
    params: Optional[Dict] = None,
    category: str = "default",
    register_db: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    """对每张正常图生成 count_per_image 张伪异常。

    Returns: [{"image_path","mask_path","method","params","base_image_path"}...]
    """
    params = params or {}
    out_dir = get_settings().storage("pseudo")
    results: List[Dict] = []
    total = len(base_image_paths) * count_per_image
    done = 0
    # M8a：本次生成挂一个 pseudo 批次（params 存生成参数）
    dataset_id = None
    if register_db:
        dataset_id = create_dataset(
            category, "pseudo",
            params={"method": method, "count_per_image": count_per_image,
                    "gen_params": params})

    for base_path in base_image_paths:
        base_path = Path(base_path)
        img = imread(base_path)
        if img is None:
            raise ValueError(f"无法读取底图: {base_path}")
        base_id = _ensure_base_image(base_path, category) if register_db else None

        ts = _ts()  # 文件名含方法与时间戳，避免覆盖
        for i in range(count_per_image):
            aug, mask = _generate_one(img, method, params)
            img_path = out_dir / f"{base_path.stem}_{method}_{ts}_{i}.png"
            mask_path = out_dir / f"{base_path.stem}_{method}_{ts}_{i}_mask.png"
            imwrite(img_path, aug)
            imwrite(mask_path, mask)

            if register_db:
                with session_scope() as s:
                    s.add(PseudoAnomaly(base_image_id=base_id, method=method,
                                        params=params, image_path=str(img_path),
                                        mask_path=str(mask_path)))
                register_image(img_path, category=category, split="pseudo",
                               label="anomaly", defect_type=method,
                               source="pseudo", copy_to_storage=False,
                               dataset_id=dataset_id)

            results.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "method": method,
                "params": params,
                "base_image_path": str(base_path),
            })
            done += 1
            if progress_cb:
                progress_cb(done, total)
    refresh_dataset_count(dataset_id)
    return results


def preview_pseudo(image_path: str | Path, method: str = "cutpaste",
                   params: Optional[Dict] = None) -> Dict:
    """单张预览：返回 {"image_path","mask_path"}（写入 storage/pseudo/preview_）。"""
    p = Path(image_path)
    img = imread(p)
    if img is None:
        raise ValueError(f"无法读取图像: {p}")
    aug, mask = _generate_one(img, method, params or {})
    out_dir = get_settings().storage("pseudo")
    img_path = out_dir / f"preview_{p.stem}_{method}.png"
    mask_path = out_dir / f"preview_{p.stem}_{method}_mask.png"
    imwrite(img_path, aug)
    imwrite(mask_path, mask)
    return {"image_path": str(img_path), "mask_path": str(mask_path)}


# ── 数据预处理 / 增强 ───────────────────────────────────────
def get_augmentation_pipeline(level: str = "light"):
    """albumentations 组合：light（几何+亮度）/ medium（+噪声模糊）/ heavy（+弹性形变）。"""
    import albumentations as A

    if level == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.7),
        ])
    if level == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                     rotate=(-30, 30), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MotionBlur(blur_limit=(3, 5), p=0.3),
        ])
    if level == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.08),
                     rotate=(-45, 45), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.GaussNoise(std_range=(0.03, 0.12), p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=60, sigma=6, p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        ])
    raise ValueError(f"未知增强档位: {level}，可选: light/medium/heavy")


def _original_meta(path: Path) -> Tuple[str, str]:
    """查 Image 表获取原图 (label, category)（未登记则 unknown/default）。"""
    with session_scope() as s:
        row = s.query(ImageRow).filter(ImageRow.path == str(path)).first()
        if row is None:
            return "unknown", "default"
        return row.label, row.category


def augment_images(image_paths: List[str | Path], level: str = "light",
                   n_aug: int = 2, out_split: str = "train") -> List[str]:
    """批量增强并入库（split=out_split）。返回新图路径列表。

    M8a：按品类各建一个 augment 批次（params 存增强参数）。"""
    pipeline = get_augmentation_pipeline(level)
    out_dir = get_settings().storage("raw_images")
    out_paths: List[str] = []
    ds_by_cat: Dict[str, int] = {}

    for p in image_paths:
        p = Path(p)
        img = imread(p)
        if img is None:
            raise ValueError(f"无法读取图像: {p}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label, category = _original_meta(p)
        if category not in ds_by_cat:
            ds_by_cat[category] = create_dataset(
                category, "augment",
                params={"level": level, "n_aug": n_aug,
                        "out_split": out_split})
        ts = _ts()
        for i in range(n_aug):
            aug_rgb = pipeline(image=rgb)["image"]
            out_path = out_dir / f"aug_{p.stem}_{ts}_{i}.png"
            imwrite(out_path, cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
            register_image(out_path, category=category, split=out_split,
                           label=label, source="augmented",
                           copy_to_storage=False,
                           dataset_id=ds_by_cat[category])
            out_paths.append(str(out_path))
    for ds_id in ds_by_cat.values():
        refresh_dataset_count(ds_id)
    return out_paths


def preview_augmentation(image_path: str | Path, level: str = "light",
                         n: int = 4) -> List[str]:
    """预览 n 种增强结果，返回临时文件路径列表。"""
    p = Path(image_path)
    img = imread(p)
    if img is None:
        raise ValueError(f"无法读取图像: {p}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pipeline = get_augmentation_pipeline(level)
    out_dir = get_settings().storage("uploads")
    ts = _ts()
    out_paths: List[str] = []
    for i in range(n):
        aug_rgb = pipeline(image=rgb)["image"]
        out_path = out_dir / f"preview_aug_{p.stem}_{level}_{ts}_{i}.png"
        imwrite(out_path, cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
        out_paths.append(str(out_path))
    return out_paths


def preprocess_image(image_path: str | Path, target_size: int = 2500,
                     denoise: bool = False, clahe: bool = False) -> str:
    """标准化预处理（尺寸规整/去噪/对比度增强），返回处理后路径。"""
    p = Path(image_path)
    img = imread(p)
    if img is None:
        raise ValueError(f"无法读取图像: {p}")

    # 尺寸规整：长边超过 target_size 时等比缩小，再居中 padding 成正方形
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge > target_size:
        scale = target_size / long_edge
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        top = (target_size - h) // 2
        left = (target_size - w) // 2
        img = cv2.copyMakeBorder(img, top, target_size - h - top,
                                 left, target_size - w - left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 可选非局部均值去噪
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # 可选对比度增强：LAB 空间对 L 通道做 CLAHE
    if clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = cv2.cvtColor(cv2.merge([op.apply(l_ch), a_ch, b_ch]),
                           cv2.COLOR_LAB2BGR)

    out_dir = get_settings().storage("uploads")
    out_path = out_dir / f"preproc_{p.stem}_{_ts()}.png"
    imwrite(out_path, img)
    return str(out_path)
