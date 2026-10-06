"""标准图向测试图平移配准 (测试原图作画布, 仅 warp 标准侧)"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from . import config as C
from .segment import Masks


@dataclass
class AlignMeta:
    dx: float = 0.0
    dy: float = 0.0
    std_dx: float = 0.0
    std_dy: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def map_bbox_to_original(self, bbox: tuple) -> tuple:
        """工作分辨率测试坐标 -> 原始测试图像素坐标 (仅缩放, 无逆平移)。"""
        x, y, w, h = bbox
        sx = self.scale_x if self.scale_x > 1e-6 else 1.0
        sy = self.scale_y if self.scale_y > 1e-6 else 1.0
        return (
            int(round(x * sx)),
            int(round(y * sy)),
            int(round(w * sx)),
            int(round(h * sy)),
        )


def _gray_u8(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def extract_channel(bgr: np.ndarray, channel: str = "gray") -> np.ndarray:
    """按配准比较通道提取单通道图：gray | v(HSV-V) | l(Lab-L)。"""
    ch = (channel or "gray").lower()
    if ch == "v":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    if ch == "l":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    return _gray_u8(bgr)


def _affine_matrix(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)


def scale_rect(rect: tuple, sx: float, sy: float) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )


def scale_bgr_to_canvas(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    if bgr.shape[1] == tw and bgr.shape[0] == th:
        return bgr
    return cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)


def scale_masks_to_canvas(masks: Masks, tw: int, th: int) -> Masks:
    if masks.solder.shape[1] == tw and masks.solder.shape[0] == th:
        return masks
    hsv = cv2.resize(masks.hsv, (tw, th), interpolation=cv2.INTER_LINEAR)
    solder = cv2.resize(masks.solder, (tw, th), interpolation=cv2.INTER_NEAREST)
    red = cv2.resize(masks.red, (tw, th), interpolation=cv2.INTER_NEAREST)
    bright = cv2.resize(masks.bright, (tw, th), interpolation=cv2.INTER_NEAREST)
    return Masks(hsv=hsv, solder=solder, red=red, bright=bright)


def warp_std_layer(layer: np.ndarray, dx: float, dy: float,
                   tw: int, th: int, *, is_mask: bool = False) -> np.ndarray:
    """标准侧图层 warp 到测试画布坐标系。"""
    M = _affine_matrix(dx, dy)
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border = 0 if is_mask else (0, 0, 0)
    return cv2.warpAffine(
        layer, M, (tw, th), flags=flags,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def warp_std_masks(masks: Masks, dx: float, dy: float, tw: int, th: int) -> Masks:
    return Masks(
        hsv=warp_std_layer(masks.hsv, dx, dy, tw, th),
        solder=warp_std_layer(masks.solder, dx, dy, tw, th, is_mask=True),
        red=warp_std_layer(masks.red, dx, dy, tw, th, is_mask=True),
        bright=warp_std_layer(masks.bright, dx, dy, tw, th, is_mask=True),
    )


def warp_std_valid(sw: int, sh: int, dx: float, dy: float,
                   tw: int, th: int) -> np.ndarray:
    """标准 warp 后在测试画布上的有效覆盖区 (标准侧越界处为 0)。"""
    full = np.full((sh, sw), 255, np.uint8)
    return warp_std_layer(full, dx, dy, tw, th, is_mask=True)


def std_shift_to_test(std_dx: float, std_dy: float,
                      sx: float, sy: float) -> tuple[float, float]:
    """相位/NCC 在标准分辨率下得到的偏移 -> 测试画布像素偏移。"""
    return std_dx * sx, std_dy * sy


def _pin_core_mask(h: int, w: int, frac: float) -> np.ndarray:
    """ROI 中心椭圆 (True=引脚核)，用于从配准特征中剔除。"""
    frac = float(max(0.05, min(0.49, frac)))
    mask = np.zeros((h, w), np.uint8)
    rx = max(1, int(round(w * frac)))
    ry = max(1, int(round(h * frac)))
    cv2.ellipse(mask, (w // 2, h // 2), (rx, ry), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def _pad_ring_weight(solder_roi: np.ndarray, pin_frac: float) -> np.ndarray:
    """锡面环权重：锡面内且非引脚核 = 1。多连通域时按各分量中心打孔。"""
    binary = (solder_roi > 0).astype(np.uint8)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 2:
        h, w = solder_roi.shape[:2]
        ring = (solder_roi > 0) & (~_pin_core_mask(h, w, pin_frac))
        return ring.astype(np.float32)

    keep = np.zeros(solder_roi.shape[:2], dtype=np.float32)
    pin_frac = float(max(0.05, min(0.49, pin_frac)))
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < 2 or h < 2:
            continue
        comp = labels == i
        local_pin = _pin_core_mask(h, w, pin_frac)
        pin_full = np.zeros_like(comp)
        pin_full[y:y + h, x:x + w] = local_pin
        keep[comp & (~pin_full)] = 1.0
    return keep


def _mean_fill_masked(gray_f32: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """保留 keep 区域纹理，其余填均值，避免相位相关被中心亮斑/边界伪影带偏。"""
    out = gray_f32.astype(np.float32, copy=True)
    m = keep > 0.5
    fill = float(out[m].mean()) if m.any() else float(out.mean())
    out[~m] = fill
    return out


def _solder_template(std_bgr: np.ndarray, std_solder: np.ndarray, channel: str = "gray",
                     *, punch_pin: bool = False, pin_frac: float = 0.30):
    margin = C.ALIGN["template_margin"]
    ys, xs = np.where(std_solder > 0)
    if len(xs) == 0:
        return None, 0, 0
    gh, gw = std_bgr.shape[:2]
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(gw, int(xs.max()) + margin + 1)
    y1 = min(gh, int(ys.max()) + margin + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, x0, y0
    gray = extract_channel(std_bgr, channel)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    weight = std_solder[y0:y1, x0:x1].astype(np.float32) / 255.0
    if punch_pin:
        full_keep = _pad_ring_weight(std_solder, pin_frac)
        weight = weight * full_keep[y0:y1, x0:x1]
    patch = (patch * (0.25 + 0.75 * weight)).astype(np.uint8)
    return patch, x0, y0


def _match_template_shift(test_gray: np.ndarray, template: np.ndarray,
                          anchor_x: int, anchor_y: int, search_r: int):
    th, tw = template.shape[:2]
    pad = search_r
    padded = cv2.copyMakeBorder(test_gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    axp = anchor_x + pad
    ayp = anchor_y + pad
    sx0 = axp - search_r
    sy0 = ayp - search_r
    sx1 = axp + tw + search_r
    sy1 = ayp + th + search_r
    search = padded[sy0:sy1, sx0:sx1]
    if search.shape[0] < th or search.shape[1] < tw:
        return 0.0, 0.0, 0.0
    res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, ncc, _, max_loc = cv2.minMaxLoc(res)
    px = sx0 + max_loc[0]
    py = sy0 + max_loc[1]
    return float(px - axp), float(py - ayp), float(ncc)


@dataclass
class AlignCache:
    """标准图离线缓存：焊盘 ROI 相位灰度 + NCC 模板 + prior。

    随 `TemplateModel` 一并缓存，供同一模板下所有测试图复用；配准过程中保持
    只读，测试图灰度图始终作为局部变量传递，不写入本对象字段。
    """
    roi_xy: tuple[int, int, int, int]
    roi_gray_f32: np.ndarray
    roi_hann: np.ndarray
    prior_roi: np.ndarray
    phase_gray_f32: np.ndarray
    phase_hann: np.ndarray
    phase_scale: float
    ncc_template: np.ndarray | None = None
    ncc_anchor: tuple[int, int] = (0, 0)
    channel: str = "gray"
    align_keep: np.ndarray | None = None  # pad_ring 锡面环掩膜；full 时为 None
    feature: str = "full"


def _phase_pyramid(roi_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scale = float(C.ALIGN.get("phase_downsample", 0.5))
    if scale >= 1.0:
        hann = cv2.createHanningWindow((roi_gray.shape[1], roi_gray.shape[0]), cv2.CV_32F)
        return roi_gray, hann, 1.0
    pw = max(8, int(round(roi_gray.shape[1] * scale)))
    ph = max(8, int(round(roi_gray.shape[0] * scale)))
    small = cv2.resize(roi_gray, (pw, ph), interpolation=cv2.INTER_AREA)
    hann = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
    return small, hann, pw / roi_gray.shape[1]


def build_align_cache(std_bgr: np.ndarray, std_solder: np.ndarray,
                      roi_rect: tuple, *, feature: str | None = None) -> AlignCache:
    channel = str(C.ALIGN.get("channel", "gray"))
    feature = str(
        feature if feature is not None else C.ALIGN.get("feature", "pad_ring")
        or "pad_ring"
    ).lower()
    pin_frac = float(C.ALIGN.get("pin_suppress_frac", 0.30))
    x, y, w, h = roi_rect
    gray = extract_channel(std_bgr, channel).astype(np.float32)
    roi_gray = gray[y:y + h, x:x + w]
    prior = std_solder[y:y + h, x:x + w]
    align_keep = None
    phase_src = roi_gray
    if feature == "pad_ring":
        align_keep = _pad_ring_weight(prior, pin_frac)
        # 环带像素过少时回退整 ROI，避免相位相关无纹理可锁。
        if int(align_keep.sum()) >= 64:
            phase_src = _mean_fill_masked(roi_gray, align_keep)
        else:
            align_keep = None
            feature = "full"
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    pg, ph, pscale = _phase_pyramid(phase_src)
    tmpl, ax, ay = _solder_template(
        std_bgr, std_solder, channel=channel,
        punch_pin=(feature == "pad_ring"), pin_frac=pin_frac)
    return AlignCache(
        (x, y, w, h), phase_src, hann, prior, pg, ph, pscale, tmpl, (ax, ay),
        channel=channel, align_keep=align_keep, feature=feature)


def test_gray_for_cache(test_bgr: np.ndarray, cache: AlignCache) -> np.ndarray:
    """按缓存约定的比较通道提取测试图灰度图 (纯函数，不写入 cache)。"""
    return extract_channel(test_bgr, cache.channel)


def _roi_gray_f32(cache: AlignCache, test_gray: np.ndarray) -> np.ndarray:
    x, y, w, h = cache.roi_xy
    roi = test_gray[y:y + h, x:x + w].astype(np.float32)
    if cache.align_keep is not None and cache.align_keep.shape == roi.shape:
        return _mean_fill_masked(roi, cache.align_keep)
    return roi


def phase_correlate_shift(cache: AlignCache, test_gray: np.ndarray) -> tuple[float, float, float]:
    tg = _roi_gray_f32(cache, test_gray)
    if cache.phase_scale < 1.0:
        pw, ph = cache.phase_gray_f32.shape[1], cache.phase_gray_f32.shape[0]
        tg = cv2.resize(tg, (pw, ph), interpolation=cv2.INTER_AREA)
        shift, response = cv2.phaseCorrelate(
            cache.phase_gray_f32, tg, cache.phase_hann)
        inv = 1.0 / cache.phase_scale
        dx, dy = float(shift[0]) * inv, float(shift[1]) * inv
    else:
        shift, response = cv2.phaseCorrelate(
            cache.roi_gray_f32, tg, cache.roi_hann)
        dx, dy = float(shift[0]), float(shift[1])
    # 不按 max_shift_px 裁剪，越界由 check_reliability 判定
    return dx, dy, float(response)


def ncc_shift(cache: AlignCache, test_gray: np.ndarray) -> tuple[float, float, float]:
    if cache.ncc_template is None:
        return 0.0, 0.0, 0.0
    ax, ay = cache.ncc_anchor
    return _match_template_shift(
        test_gray, cache.ncc_template, ax, ay, C.ALIGN["search_radius"])


def align_phase_roi(test_bgr: np.ndarray, cache: AlignCache,
                    work_scale_x: float = 1.0, work_scale_y: float = 1.0):
    """估计配准位移，写入 ``meta.dx/dy = -est``。"""
    test_gray = test_gray_for_cache(test_bgr, cache)
    est_dx, est_dy, resp = phase_correlate_shift(cache, test_gray)
    used = "相位"
    min_resp = float(C.ALIGN.get("phase_min_response", 0.12))
    if resp < min_resp:
        fdx, fdy, ncc = ncc_shift(cache, test_gray)
        if ncc >= float(C.ALIGN["ncc_min"]):
            est_dx, est_dy, resp = fdx, fdy, ncc
            used = "NCC"
        else:
            used = "相位低置信"
    dx, dy = -est_dx, -est_dy
    meta = AlignMeta(dx=dx, dy=dy, scale_x=work_scale_x, scale_y=work_scale_y)
    feat = cache.feature or "full"
    if abs(dx) > 0.5 or abs(dy) > 0.5:
        method = f"标准{used}/{feat} warp dx={dx:.1f} dy={dy:.1f} (resp={resp:.3f})"
    else:
        method = f"零位{used}/{feat} resp={resp:.3f}"
    return dx, dy, meta, method, resp


def select_primary_pad(joints: list, shape_hw: tuple[int, int]):
    """选主焊盘：优先 circle，否则面积最大。"""
    from .bridge_joints import pad_joints

    pads = pad_joints(joints)
    best = None
    best_mask = None
    best_key = None
    for j in pads:
        mask = j.mask(shape_hw)
        area = int(cv2.countNonZero(mask))
        if area < 16:
            continue
        key = (1 if j.type == "circle" else 0, area)
        if best_key is None or key > best_key:
            best_key = key
            best = j
            best_mask = mask
    return best, best_mask


def align_phase_primary_pad(
    test_bgr: np.ndarray,
    std_bgr: np.ndarray,
    joints: list,
    work_scale_x: float = 1.0,
    work_scale_y: float = 1.0,
):
    """连锡多框：主焊盘上跑 ``align_phase_roi``（与单焊盘同款），位移共用。"""
    from .segment import roi_from_mask

    sh, sw = std_bgr.shape[:2]
    primary, mask = select_primary_pad(joints, (sh, sw))
    if primary is None or mask is None:
        meta = AlignMeta(scale_x=work_scale_x, scale_y=work_scale_y)
        return 0.0, 0.0, meta, "多焊点配准无主焊盘", 0.0, None

    roi = roi_from_mask(mask)
    cache = build_align_cache(std_bgr, mask, roi.rect)
    dx, dy, meta, method, resp = align_phase_roi(
        test_bgr, cache, work_scale_x, work_scale_y)
    method = f"{method} [主焊盘#{primary.id}/{primary.type}]"
    return dx, dy, meta, method, resp, cache


# 近圆轴比阈值：预览输出 circle 而非 ellipse
_PREVIEW_CIRCLE_AXIS_RATIO = 0.92


def solder_mask_to_preview_roi(mask: np.ndarray) -> dict:
    """锡面 Mask → 预览几何（近圆 circle / 椭圆 ellipse / 失败 rect）。坐标相对 mask 分辨率。"""
    if mask is None or mask.size == 0:
        return {"shape": "rect", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    bin_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"shape": "rect", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 16:
        x, y, w, h = cv2.boundingRect(cnt)
        return {"shape": "rect", "x": float(x), "y": float(y), "w": float(max(1, w)),
                "h": float(max(1, h))}

    if len(cnt) >= 5:
        (cx, cy), (axis_w, axis_h), angle = cv2.fitEllipse(cnt)
        major = float(max(axis_w, axis_h))
        minor = float(min(axis_w, axis_h))
        if major > 1e-6 and (minor / major) >= _PREVIEW_CIRCLE_AXIS_RATIO:
            return {"shape": "circle", "cx": float(cx), "cy": float(cy),
                    "r": 0.25 * (axis_w + axis_h)}
        return {
            "shape": "ellipse",
            "cx": float(cx), "cy": float(cy),
            "axes_w": float(axis_w), "axes_h": float(axis_h),
            "angle": float(angle),
        }

    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    return {"shape": "circle", "cx": float(cx), "cy": float(cy), "r": float(radius)}


def scale_preview_roi(roi: dict, sx: float, sy: float) -> dict:
    """预览 ROI 分辨率缩放（工作分辨率 → 原始标准图等）。"""
    shape = str(roi.get("shape", "rect") or "rect").lower()
    if shape == "circle":
        return {
            "shape": "circle",
            "cx": float(roi["cx"]) * sx,
            "cy": float(roi["cy"]) * sy,
            "r": float(roi["r"]) * (sx + sy) / 2.0,
        }
    if shape == "ellipse":
        return {
            "shape": "ellipse",
            "cx": float(roi["cx"]) * sx,
            "cy": float(roi["cy"]) * sy,
            "axes_w": float(roi["axes_w"]) * sx,
            "axes_h": float(roi["axes_h"]) * sy,
            "angle": float(roi.get("angle", 0.0)),
        }
    return {
        "shape": "rect",
        "x": float(roi["x"]) * sx,
        "y": float(roi["y"]) * sy,
        "w": float(roi["w"]) * sx,
        "h": float(roi["h"]) * sy,
    }


def map_roi_to_test(roi: dict, std_wh: tuple[int, int], test_work_wh: tuple[int, int],
                    meta: "AlignMeta") -> dict:
    """标准图几何 → 测试图像素坐标（缩放 + 平移 ``-dx/-dy`` + 还原测试分辨率）。

    roi: ``rect`` / ``circle`` / ``ellipse``（``axes_w/axes_h`` 为 OpenCV 全轴长）。
    std_wh / test_work_wh: 原始标准分辨率与配准工作画布尺寸。
    """
    gw, gh = std_wh
    tw, th = test_work_wh
    sx = tw / max(1, gw)
    sy = th / max(1, gh)
    scale_x = meta.scale_x if meta.scale_x > 1e-6 else 1.0
    scale_y = meta.scale_y if meta.scale_y > 1e-6 else 1.0

    def _map_point(x: float, y: float) -> tuple[float, float]:
        wx = x * sx - meta.dx
        wy = y * sy - meta.dy
        return wx * scale_x, wy * scale_y

    shape = str(roi.get("shape", "rect") or "rect").lower()
    if shape == "circle":
        cx, cy = _map_point(float(roi["cx"]), float(roi["cy"]))
        r_scale = (sx * scale_x + sy * scale_y) / 2.0
        return {"shape": "circle", "cx": cx, "cy": cy, "r": float(roi["r"]) * r_scale}
    if shape == "ellipse":
        cx, cy = _map_point(float(roi["cx"]), float(roi["cy"]))
        return {
            "shape": "ellipse",
            "cx": cx, "cy": cy,
            "axes_w": float(roi["axes_w"]) * sx * scale_x,
            "axes_h": float(roi["axes_h"]) * sy * scale_y,
            "angle": float(roi.get("angle", 0.0)),
        }

    x0, y0 = _map_point(float(roi["x"]), float(roi["y"]))
    x1, y1 = _map_point(float(roi["x"]) + float(roi["w"]), float(roi["y"]) + float(roi["h"]))
    return {
        "shape": "rect",
        "x": min(x0, x1), "y": min(y0, y1),
        "w": abs(x1 - x0), "h": abs(y1 - y0),
    }
