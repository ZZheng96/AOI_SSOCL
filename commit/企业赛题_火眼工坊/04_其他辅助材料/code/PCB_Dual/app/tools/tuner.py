"""调参辅助核心（复现 reference5/适配/shift2 的 tuner 效果）。

包含四类能力（对应上位机调参控件）：
1. 自动抽色：ROI 内自动提取 HSV 颜色范围（Otsu 正负样本 + V→S→H 三级
   precision/recall 搜索，或 HSV 3D 联合搜索 + Youden's J 门控）——来自
   shift2 color_extractor
2. 手动色板取色：点选位置邻域统计生成色块（多段 HSV 范围，处理红色跨边界）
3. 图像预处理：auto_gamma 亮度归一化（shift2 gamma）+ 灰度/二值化/通道视图
4. 辅助视图：HSV 范围掩膜 / 通道分离 / 差分图

输出统一为 HSV 六参 (h_lo,h_hi,s_lo,s_hi,v_lo,v_hi)，可直接填入算法参数
（如 SMT 的 solder_blue_h_low/high 等）。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────
# 1. 自动抽色（shift2 color_extractor）
# ────────────────────────────────────────────────────────────
_H_STEP = 3
_H_MIN_RECALL = 0.6
_H_J_GATE = 0.7
_H_FLAT_TOL = 0.03


def auto_gamma(image: np.ndarray, target: float = 0.35) -> np.ndarray:
    """自动 gamma 校正：归一化平均亮度到 target（shift2 gamma）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean() / 255.0
    if mean_val < 0.01 or mean_val > 0.99:
        return image
    gamma = np.clip(np.log(target) / np.log(mean_val), 0.3, 3.0)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def _search_precision_with_recall(ch_flat, pos_flat, neg_flat, n_bins,
                                  min_recall=0.3, fixed_lo=-1, step=3):
    if len(ch_flat) == 0 or pos_flat.sum() == 0:
        lo = fixed_lo if fixed_lo >= 0 else 0
        return lo, n_bins - 1
    hp = np.bincount(ch_flat[pos_flat > 0], minlength=n_bins).astype(np.float64)
    hn = np.bincount(ch_flat[neg_flat > 0], minlength=n_bins).astype(np.float64)
    cp, cn = np.cumsum(hp), np.cumsum(hn)
    tp_total = cp[-1]

    def rs(c, lo, hi):
        return c[hi] - (c[lo - 1] if lo > 0 else 0)

    best_p, best_lo, best_hi = -1.0, (fixed_lo if fixed_lo >= 0 else 0), n_bins - 1
    if fixed_lo >= 0:
        lo = fixed_lo
        for hi in range(lo, n_bins, step):
            tp, fp = rs(cp, lo, hi), rs(cn, lo, hi)
            r = tp / tp_total if tp_total > 0 else 0
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            if r >= min_recall and p > best_p:
                best_p, best_lo, best_hi = p, lo, hi
    else:
        for lo in range(0, n_bins, step):
            for hi in range(lo, n_bins, step):
                tp, fp = rs(cp, lo, hi), rs(cn, lo, hi)
                r = tp / tp_total if tp_total > 0 else 0
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                if r >= min_recall and p > best_p:
                    best_p, best_lo, best_hi = p, lo, hi
    return best_lo, best_hi


def _search_h_band(hf, pf, nf, vs_in, min_recall=_H_MIN_RECALL,
                   step=_H_STEP, flat_tol=_H_FLAT_TOL):
    hp = np.bincount(hf[vs_in & (pf > 0)], minlength=180).astype(np.float64)
    hn = np.bincount(hf[vs_in & (nf > 0)], minlength=180).astype(np.float64)
    tp_total, tn_total = hp.sum(), hn.sum()
    if tp_total == 0:
        return 0, 179, 0.0
    cp = np.concatenate([[0.0], np.cumsum(hp)])
    cn = np.concatenate([[0.0], np.cumsum(hn)])

    def seg(lo, hi):
        return cp[hi + 1] - cp[lo], cn[hi + 1] - cn[lo]

    idx = np.arange(0, 180, step)
    lo_arr, hi_arr = idx[:, None], idx[None, :]
    tp_m = cp[hi_arr + 1] - cp[lo_arr]
    fp_m = cn[hi_arr + 1] - cn[lo_arr]
    valid = np.triu(np.ones((len(idx), len(idx)), dtype=bool))
    j_m = np.where(valid, tp_m / tp_total - (fp_m / tn_total if tn_total else 0.0), -1.0)
    j_m = np.where(valid & (tp_m >= tp_total * min_recall), j_m, -1.0)
    flat = int(np.argmax(j_m))
    j_max = j_m.ravel()[flat]
    if j_max < 0:
        return 0, 179, 0.0
    lo_i, hi_i = np.unravel_index(flat, j_m.shape)
    lo, hi = int(idx[lo_i]), int(idx[hi_i])

    def _j_of(tp, fp):
        return tp / tp_total - (fp / tn_total if tn_total else 0.0)

    j_cur = j_max
    while hi < 179:
        tp, fp = seg(lo, hi + 1)
        j2 = _j_of(tp, fp)
        if j2 >= j_cur - flat_tol and tp >= tp_total * min_recall:
            hi += 1
            if j2 > j_cur:
                j_cur = j2
        else:
            break
    return lo, hi, _j_of(*seg(lo, hi))


def extract_hsv_range(img: np.ndarray, roi_box: tuple[int, int, int, int],
                      use_autogamma: bool = False,
                      autogamma_target: float = 0.35) -> dict:
    """自动抽色：ROI（x,y,w,h）内提取 HSV 颜色范围。

    返回 {"h_lo","h_hi","s_lo","s_hi","v_lo","v_hi","method","recall_note"}
    """
    work = auto_gamma(img, autogamma_target) if use_autogamma else img
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    rx, ry, rw, rh = roi_box
    rx2, ry2 = min(w, rx + rw), min(h, ry + rh)
    rx, ry = max(0, rx), max(0, ry)
    roi_v = hsv[ry:ry2, rx:rx2, 2]
    if roi_v.size < 10:
        return {"h_lo": 0, "h_hi": 179, "s_lo": 0, "s_hi": 255,
                "v_lo": 0, "v_hi": 255, "method": "empty"}
    _, roi_base = cv2.threshold(roi_v, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pos_mask = np.zeros((h, w), dtype=np.uint8)
    pos_mask[ry:ry2, rx:rx2] = roi_base
    neg_mask = np.ones((h, w), dtype=np.uint8)
    neg_mask[ry:ry2, rx:rx2] = 0
    hf, sf, vf = hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()
    pf, nf = pos_mask.ravel(), neg_mask.ravel()

    v_lo, v_hi = _search_precision_with_recall(vf, pf, nf, 256, min_recall=0.3, fixed_lo=0)
    v_in = (vf >= v_lo) & (vf <= v_hi)
    s_lo, s_hi = _search_precision_with_recall(sf[v_in], pf[v_in], nf[v_in],
                                               256, min_recall=0.5)
    vs_in = v_in & (sf >= s_lo) & (sf <= s_hi)
    h_lo, h_hi, j_h = _search_h_band(hf, pf, nf, vs_in)
    if j_h < _H_J_GATE:
        h_lo, h_hi = 0, 179
    return {"h_lo": int(h_lo), "h_hi": int(h_hi), "s_lo": int(s_lo),
            "s_hi": int(s_hi), "v_lo": int(v_lo), "v_hi": int(v_hi),
            "method": "auto_extract", "h_j": round(float(j_h), 3)}


# ────────────────────────────────────────────────────────────
# 2. 手动色板取色（点选邻域统计，处理红色跨 H=0/180 边界）
# ────────────────────────────────────────────────────────────
def color_swatch_at(img: np.ndarray, x: int, y: int,
                    patch: int = 15, tol: int = 12) -> dict:
    """在 (x,y) 邻域统计生成一个色块（HSV 范围）。

    patch=邻域半宽，tol=容差（每通道 ±tol）。返回单段范围与色块代表色。
    """
    h, w = img.shape[:2]
    x0, y0 = max(0, x - patch), max(0, y - patch)
    x1, y1 = min(w, x + patch + 1), min(h, y + patch + 1)
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h_med = int(np.median(hh))
    s_med, v_med = int(np.median(ss)), int(np.median(vv))
    h_lo, h_hi = max(0, h_med - tol), min(179, h_med + tol)
    return {
        "x": x, "y": y,
        "h_lo": h_lo, "h_hi": h_hi,
        "s_lo": max(0, s_med - tol), "s_hi": min(255, s_med + tol),
        "v_lo": max(0, v_med - tol), "v_hi": min(255, v_med + tol),
        "rgb": [int(v) for v in img[y, x]],
        "bgr_rep": [int(v) for v in img[y, x]],
    }


def multi_swatch_ranges(img: np.ndarray, points: list[tuple[int, int]],
                        tol: int = 12) -> list[dict]:
    """多取色点 → 多段 HSV 范围（红色跨边界时拆成两段：0-低 & 高-179）。"""
    swatches = [color_swatch_at(img, x, y, tol=tol) for x, y in points]
    ranges = []
    for sw in swatches:
        h_lo, h_hi = sw["h_lo"], sw["h_hi"]
        if h_lo > 179 - tol and h_hi < tol:  # 跨边界（红）
            ranges.append((0, h_hi, sw["s_lo"], sw["s_hi"], sw["v_lo"], sw["v_hi"]))
            ranges.append((h_lo, 179, sw["s_lo"], sw["s_hi"], sw["v_lo"], sw["v_hi"]))
        else:
            ranges.append((h_lo, h_hi, sw["s_lo"], sw["s_hi"], sw["v_lo"], sw["v_hi"]))
    return ranges


# ────────────────────────────────────────────────────────────
# 3. 辅助视图（掩膜 / 通道 / 差分）
# ────────────────────────────────────────────────────────────
def hsv_mask(img: np.ndarray, hsv_range: dict) -> np.ndarray:
    """按 HSV 六参生成掩膜（单段；跨边界自动补第二段）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_lo, h_hi = int(hsv_range["h_lo"]), int(hsv_range["h_hi"])
    s_lo, s_hi = int(hsv_range["s_lo"]), int(hsv_range["s_hi"])
    v_lo, v_hi = int(hsv_range["v_lo"]), int(hsv_range["v_hi"])
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))
    else:  # 跨 0/180
        mask1 = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, s_hi, v_hi))
        mask2 = cv2.inRange(hsv, (0, s_lo, v_lo), (h_hi, s_hi, v_hi))
        mask = cv2.bitwise_or(mask1, mask2)
    return mask


def channel_view(img: np.ndarray, space: str = "hsv") -> dict[str, np.ndarray]:
    """通道分离视图（RGB/HSV 三通道）。"""
    out = {}
    if space == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out["H"] = hsv[:, :, 0]
        out["S"] = hsv[:, :, 1]
        out["V"] = hsv[:, :, 2]
    else:
        out["B"], out["G"], out["R"] = cv2.split(img)
    return out


def preprocess_views(img: np.ndarray) -> dict[str, np.ndarray]:
    """预处理辅助视图：灰度 / Otsu 二值 / 自适应二值 / 自动gamma。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 5)
    return {"灰度": gray, "Otsu": otsu, "自适应二值": adapt,
            "自动Gamma": auto_gamma(img, 0.35)}


# ────────────────────────────────────────────────────────────
# 4. 抽色对象注册（复现 zz pcba_tuner 的 HSV_OBJECTS 设计）
# ────────────────────────────────────────────────────────────
# 一个"抽色对象"= 某类物质（焊锡/阻焊/丝印/元件/toe）的 HSV 六参，
# 对应算法参数中的一组键（如 solder_blue_h_low 等）。调参辅助对话框
# 从已勾选算法的 param_specs 自动发现这些组，多算法可注册。
import re
from dataclasses import dataclass


@dataclass
class HsvObjectSpec:
    """一个可抽色的 HSV 对象。

    keys 为 6 槽 → 算法参数 key 的映射；某槽为 None 表示算法不暴露该键
    （应用时跳过，缺省值交给算法 default_config）。
    """

    id: str                      # 前缀，如 "solder_blue" / "mask" / "silk"
    label: str                   # 显示名，如 "焊锡"
    color: tuple[int, int, int]  # 高亮色（BGR），画布叠加用
    keys: dict[str, str | None]  # {"h_lo","h_hi","s_lo","s_hi","v_lo","v_hi"}

    def resolved_keys(self) -> dict[str, str]:
        return {slot: k for slot, k in self.keys.items() if k}


# zz pcba_tuner HSV_OBJECTS 的预置对象模板：按参数前缀识别。
_PRESET_OBJECTS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "solder": ("焊锡", (180, 180, 180)),
    "mask": ("阻焊", (0, 200, 0)),
    "silk": ("丝印", (255, 255, 255)),
    "component": ("元件", (180, 0, 255)),
    "comp": ("元件", (180, 0, 255)),
    "toe": ("toe", (0, 215, 255)),
    "toe_metal": ("toe金属", (0, 215, 255)),
    "feature": ("元件底座", (255, 180, 0)),
}

# 参数 key → HSV 槽：h_low/h_lo/h_min → h_lo；h_high/h_hi/h_max → h_hi；s/v 同理
_HSV_KEY_RE = re.compile(
    r"^(?P<prefix>.*)_(?P<ch>h|s|v)_(?P<side>low|high|lo|hi|min|max)$")
_SIDE_TO_SLOT = {"low": "lo", "lo": "lo", "min": "lo",
                 "high": "hi", "hi": "hi", "max": "hi"}


def discover_hsv_objects(algorithm_ids: list[str]) -> list[HsvObjectSpec]:
    """扫描算法的 param_specs，发现可抽色的 HSV 参数组（多算法可注册）。

    除各算法自身的参数外，还扫描其"共用参数"层（Catalog.shared_param_id，
    如 SMT 的 solder_blue_* 挂在 "solder_共用参数" 上）——这类颜色参数对
    同组所有缺陷统一生效，是调参的主要入口。

    判定规则：同一前缀下存在 h 对（h_lo+h_hi）且 s/v 至少各有下界 → 构成对象。
    缺 max 键时该槽为 None（写回时跳过，255 交给算法默认）。
    """
    from app.detect.catalog import get_catalog
    from app.detect.registry import get_adapter_for

    catalog = get_catalog()
    groups: dict[str, dict[str, str]] = {}
    for alg_id in algorithm_ids:
        adapter = get_adapter_for(alg_id)
        if adapter is None:
            continue
        shared = catalog.shared_param_id(alg_id)
        scan_ids = [alg_id]
        if shared and shared != alg_id:
            scan_ids.append(shared)
        for aid in scan_ids:
            try:
                specs = list(adapter.param_specs(aid) or [])
            except Exception:  # noqa: BLE001 - 外部仓库规格异常不影响调参工具
                continue
            for spec in specs:
                m = _HSV_KEY_RE.match(str(spec.key))
                if not m:
                    continue
                prefix, ch, side = m.group("prefix"), m.group("ch"), m.group("side")
                slot = f"{ch}_{_SIDE_TO_SLOT[side]}"
                groups.setdefault(prefix, {})[slot] = str(spec.key)

    out: list[HsvObjectSpec] = []
    for prefix in sorted(groups):
        slots = groups[prefix]
        if "h_lo" not in slots or "h_hi" not in slots:
            continue
        if "s_lo" not in slots and "v_lo" not in slots:
            continue
        base = prefix.split("_")[0]
        label, color = _PRESET_OBJECTS.get(
            base, (prefix, (200, 200, 200)))
        out.append(HsvObjectSpec(
            id=prefix, label=label, color=color,
            keys={slot: slots.get(slot) for slot in
                  ("h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi")},
        ))
    return out


def obj_hsv_from_params(spec: HsvObjectSpec, params: dict) -> dict:
    """从算法参数读出该对象的 HSV 六值（缺省值用 zz 默认：s/v 全开）。"""
    keys = spec.resolved_keys()
    h_lo = int(params.get(keys.get("h_lo", ""), 0) if keys.get("h_lo") else 0)
    h_hi = int(params.get(keys.get("h_hi", ""), 179) if keys.get("h_hi") else 179)
    s_lo = int(params.get(keys.get("s_lo", ""), 0) if keys.get("s_lo") else 0)
    s_hi = int(params.get(keys.get("s_hi", ""), 255) if keys.get("s_hi") else 255)
    v_lo = int(params.get(keys.get("v_lo", ""), 0) if keys.get("v_lo") else 0)
    v_hi = int(params.get(keys.get("v_hi", ""), 255) if keys.get("v_hi") else 255)
    return {"h_lo": h_lo, "h_hi": h_hi, "s_lo": s_lo, "s_hi": s_hi,
            "v_lo": v_lo, "v_hi": v_hi}


def hsv_to_params(spec: HsvObjectSpec, hsv: dict) -> dict[str, int]:
    """把 HSV 六值写回算法参数（仅写算法暴露的键）。"""
    out: dict[str, int] = {}
    for slot, key in spec.resolved_keys().items():
        if slot in hsv:
            out[key] = int(hsv[slot])
    return out
