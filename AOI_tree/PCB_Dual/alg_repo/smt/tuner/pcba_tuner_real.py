# -*- coding: utf-8 -*-
"""
PCBA 检测调参工具
==================
基于 hsv_viewer.py 扩展，用于 AOI 焊锡缺陷检测的参数调试。

功能：
  1. ROI 绘制：pad框(焊盘) / toe框(引脚) / rim框(虚焊检测边缘)，矩形为主
     - 编辑模式：选中 → 移动 / 拖角缩放 / Delete 删除
  2. 4 类 HSV 抽色对象：阻焊 / 丝印 / 元件 / 焊锡
     - 下拉切换当前对象，6 条滑条 + Mask 预览 + H/S/V 通道随之切换
     - 每类独立保存 HSV 范围
  3. 实时判定：
     - 少锡：pad框内 焊锡占比 < 少锡阈值  -> pad框 红，否则 绿
     - 虚焊：(rim框 − toe框) 剩余区域 焊锡占比 < 虚焊阈值 -> rim框 红，否则 青
  4. 持久化：保存/加载 params.json(HSV+阈值) 与 LabelMe 风格 ROI JSON

快捷键：q 退出 | 1/2/3 画 pad/toe/rim | e 编辑 | s 采样 | Tab 切换对象
        Delete 删除选中 | 点击小图切换主视图
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import math
import sys
import os
import json
import copy
import platform
import base64
import zlib
from PIL import Image as PILImage, ImageTk as PILImageTk

# ─────────────────────────────────────────────
# 导入真实算法函数
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_CONTRACT = os.path.join(_HERE, "..", "contract_reference")
for _p in (_SRC, _CONTRACT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from algorithms.solder_smt.composite_smt_all._solder_core import (
    _remove_interference, extract_solder, judge_insufficient, judge_excess,
    judge_bridge, judge_cold_solder, align_template, map_rect_to_original,
    map_mask_to_original, compute_diff, auto_resize_image, scale_pad_rects,
    _merge_overlapping_rects, _expand_rect, _detect_crack_hough,
)
from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg

# ─────────────────────────────────────────────
# 4 类 HSV 抽色对象配置
#   keys: 内部 h_min..v_max → params.json 的键名（与生产代码对齐）
#   defaults: 缺省值（对齐 pair4/params.json）
#   color: 该对象在主图 Mask 高亮时的代表色 (BGR)
# ─────────────────────────────────────────────
HSV_OBJECTS = [
    {
        "name": "阻焊", "label": "solder_mask", "color": (0, 200, 0),
        "keys": {
            "h_min": "mask_h_low", "h_max": "mask_h_high",
            "s_min": "mask_s_min", "s_max": "mask_s_max",
            "v_min": "mask_v_min", "v_max": "mask_v_max",
        },
        "defaults": {"h_min": 60, "h_max": 95, "s_min": 80, "s_max": 255, "v_min": 30, "v_max": 255},
    },
    {
        "name": "丝印", "label": "silk", "color": (255, 255, 255),
        "keys": {
            "h_min": "silk_h_low", "h_max": "silk_h_high",
            "s_min": "silk_s_low", "s_max": "silk_s_high",
            "v_min": "silk_v_low", "v_max": "silk_v_high",
        },
        "defaults": {"h_min": 97, "h_max": 101, "s_min": 109, "s_max": 133, "v_min": 141, "v_max": 207},
    },
    {
        "name": "元件", "label": "component", "color": (180, 0, 255),
        "keys": {
            "h_min": "comp_h_low", "h_max": "comp_h_high",
            "s_min": "comp_s_min", "s_max": "comp_s_max",
            "v_min": "comp_v_min", "v_max": "comp_v_max",
        },
        "defaults": {"h_min": 0, "h_max": 179, "s_min": 0, "s_max": 255, "v_min": 0, "v_max": 60},
    },
    {
        "name": "toe", "label": "toe_metal", "color": (0, 215, 255),
        "keys": {
            "h_min": "toe_metal_h_low", "h_max": "toe_metal_h_high",
            "s_min": "toe_metal_s_min", "s_max": "toe_metal_s_max",
            "v_min": "toe_metal_v_min", "v_max": "toe_metal_v_max",
        },
        "defaults": {"h_min": 80, "h_max": 124, "s_min": 39, "s_max": 78, "v_min": 125, "v_max": 222},
    },
    {
        "name": "焊锡", "label": "solder", "color": (180, 180, 180),
        "keys": {
            "h_min": "solder_blue_h_low", "h_max": "solder_blue_h_high",
            "s_min": "solder_blue_s_min", "s_max": "solder_blue_s_max",
            "v_min": "solder_blue_v_min", "v_max": "solder_blue_v_max",
        },
        "defaults": {"h_min": 50, "h_max": 100, "s_min": 56, "s_max": 255, "v_min": 83, "v_max": 255},
    },
]
OBJ_NAMES = [o["name"] for o in HSV_OBJECTS]
OBJ_BY_NAME = {o["name"]: o for o in HSV_OBJECTS}

# ROI 三类
ROI_TYPES = ["pad", "toe", "rim"]
# toe 框恒定颜色（无判定状态）；pad/rim 颜色由判定结果决定
TOE_COLOR = (0, 255, 255)        # 黄
PAD_OK = (0, 255, 0)             # 绿
NG_COLOR = (0, 0, 255)           # 红
EXCESS_OK = (90, 90, 90)         # 多锡外扩框OK灰
EXCESS_NG = (0, 0, 255)          # 多锡外扩框NG红
BRIDGE_NG = (0, 0, 220)          # 连锡NG红
RIM_OK = (255, 255, 0)           # 青
DETECT_FILL = (255, 0, 255)      # rim-toe 检测区填充（品红，半透明）
SEL_COLOR = (255, 255, 255)      # 选中框

# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
state = {
    "img": None, "hsv": None, "path": "",
    "mouse_x": 0, "mouse_y": 0,
    "lock_points": [],            # 采样点 [(x,y,h,s,v),...]
    "show_overlay": True,         # 主图是否叠加当前对象 Mask
    "show_detect": True,          # 是否显示 rim-toe 检测区填充
    "main_view_idx": 0,           # 0=image, 1=mask, 2=H, 3=S, 4=V
    # ROI：三类独立列表，每项 {"x","y","w","h"}
    "rois": {"pad": [], "toe": [], "rim": []},
    # 4 类 HSV 参数
    "hsv_params": {o["name"]: dict(o["defaults"]) for o in HSV_OBJECTS},
    "cur_obj": "焊锡",            # 当前调参对象
    # 阈值（对齐 solder_smt_all_alg.json）
    "thresh": {
        "insufficient": 0.08,        # 少锡：pad内锡占比 <
        "excess": 0.08,              # 多锡：pad外扩环锡占比 >
        "bridge_min_area": 20,       # 连锡：连通域面积(px) ≥
        "cold_rim": 0.5,             # 虚焊 Rule1: rim-toe剩余区锡占比 <
        "toe_metal_ratio": 0.3,      # 虚焊 Rule2: toe框内金属占比 >
        "cold_diff_ratio": 0.38,     # 虚焊 Rule3: diff覆盖率 >
        "crack_min_length_ratio": 0.5, # crack: 线段长度 ≥ pad短边 × 此比例
        "cold_dark_ratio": 0.30,     # terminal旧逻辑: 暗区比例 >
        "cold_v_dark": 80,           # terminal旧逻辑: 暗区V值 <
        "crack_cpp_min_pads": 4,    # C++批量crack最小pad数(0=始终Python)
    },
    "pin_type": "gull-wing",       # 引脚类型: gull-wing | terminal
    # pad 外扩比例（多锡判定用，固定值，不在GUI暴露）
    "pad_expand_ratio": 0.30,
    # 交互模式
    "mode": "edit",               # sample | draw_pad | draw_toe | draw_rim | edit
    "drawing": None,              # 正在绘制 {"type","x0","y0","x1","y1"}
    "sel": None,                  # 选中 {"type","idx"}
    "drag": None,                 # 拖拽 {"kind","ox","oy","orig","pushed"}
    "heavy_dirty": True,          # 重计算(检测/mask/ROI绘制)需重建缓存
    "light_dirty": True,          # 轻量叠加(鼠标HSV/绘制中矩形)需刷新
    "locks_dirty": True,          # 采样点Tk面板需刷新
    "undo_stack": [],             # 撤销栈(存rois深拷贝)
    "cache": {"base": None, "mask_vis": None, "ch": {}},  # 重计算结果缓存
    # 多图支持
    "pair_dir": "",
    "pair_images": [],       # [(display_name, filepath), ...] 含模板图
    "pair_img_idx": 0,
    "template_img": None,    # 模板图(np.ndarray)
    "orig_rois": None,       # pad.json原始ROI(模板坐标系)
    "rois_src_size": None,   # ROI来源图片的(w, h)
    "zoom": 1.0,
    "palette_points": {name: [] for name in OBJ_NAMES},  # polygon control points per object: [[h, s], ...]
    "hs_lut": {name: None for name in OBJ_NAMES},  # 180x256 bool numpy array per object
    "use_hs_lut": False,  # 是否启用HS色板矩阵替代1D范围
}

VIEW_LABELS = ["Image", "Mask", "H", "S", "V", "Template", "DiffMask", "Crack"]

sliders = {}   # name -> IntVar (绑定当前对象)
_suppress_trace = False  # 程序化设置滑条时抑制trace回调

# ─────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────
def _rect_mask(x, y, w, h, img_shape):
    """矩形内填 255 的 mask（对齐生产代码 _rect_mask）"""
    m = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.rectangle(m, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
    return m

def norm_rect(x0, y0, x1, y1):
    """两点 → (x,y,w,h) 规范化"""
    x = int(round(min(x0, x1)))
    y = int(round(min(y0, y1)))
    w = int(round(abs(x1 - x0)))
    h = int(round(abs(y1 - y0)))
    return {"x": x, "y": y, "w": max(1, w), "h": max(1, h)}

def get_obj_hsv(obj_name):
    p = state["hsv_params"][obj_name]
    lower = np.array([p["h_min"], p["s_min"], p["v_min"]], dtype=np.uint8)
    upper = np.array([p["h_max"], p["s_max"], p["v_max"]], dtype=np.uint8)
    return lower, upper

def compute_mask_for(obj_name):
    if state["hsv"] is None:
        return None
    lower, upper = get_obj_hsv(obj_name)
    return cv2.inRange(state["hsv"], lower, upper)

# ─────────────────────────────────────────────
# 实时检测：少锡 + 虚焊
# ─────────────────────────────────────────────
def compute_detection():
    """返回 (pad_results, rim_results, excess_results, bridge_results, solder_mask)
    pad_results:    [{"x","y","w","h","cov","ng"}, ...]               少锡
    rim_results:    [{"x","y","w","h","ratio","ng","darea"}, ...]     虚焊 Rule1 rim-toe
    excess_results: [{"x","y","w","h","ex","ng"}, ...]                多锡（pad外扩环）
    bridge_results: [{"x","y","w","h","area","ng"}, ...]              连锡（连通域bbox）
    """
    if state["img"] is None:
        return [], [], [], [], None
    shape = state["img"].shape
    solder_mask = compute_mask_for("焊锡")
    t = state["thresh"]
    ins_t = t["insufficient"]
    exc_t = t["excess"]
    bridge_t = t["bridge_min_area"]
    cold_t = t["cold_rim"]

    # 少锡：每个 pad 内焊锡占比
    pad_results = []
    pad_union = np.zeros(shape[:2], dtype=np.uint8)
    for r in state["rois"]["pad"]:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        pad_m = _rect_mask(x, y, w, h, shape)
        pad_union |= pad_m
        area = w * h
        sin = int(np.count_nonzero(solder_mask & pad_m))
        cov = sin / area if area > 0 else 0.0
        pad_results.append(dict(r, cov=cov, ng=cov < ins_t))

    # 多锡：每个 pad 外扩框 - pad框 = 外扩环，环内焊锡占比 > excess_thresh → NG
    excess_results = []
    er = state.get("pad_expand_ratio", 0.30)
    for r in state["rois"]["pad"]:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        dx, dy = int(w * er), int(h * er)
        ex0 = max(0, x - dx); ey0 = max(0, y - dy)
        ex1 = min(shape[1], x + w + dx); ey1 = min(shape[0], y + h + dy)
        ext_m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.rectangle(ext_m, (ex0, ey0), (ex1, ey1), 255, -1)
        ring = cv2.subtract(ext_m, _rect_mask(x, y, w, h, shape))  # 外扩环
        rarea = int(np.count_nonzero(ring))
        rsin = int(np.count_nonzero(solder_mask & ring))
        ex_ratio = rsin / rarea if rarea > 0 else 0.0
        excess_results.append(dict(r, ex=ex_ratio, ng=ex_ratio > exc_t,
                                   ext=(ex0, ey0, ex1 - ex0, ey1 - ey0)))

    # toe 并集 mask
    toe_union = np.zeros(shape[:2], dtype=np.uint8)
    for r in state["rois"]["toe"]:
        toe_union |= _rect_mask(r["x"], r["y"], r["w"], r["h"], shape)

    # 虚焊 Rule1：每个 rim 减去 toe 并集 → 剩余区域焊锡占比
    rim_results = []
    for r in state["rois"]["rim"]:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        rim_m = _rect_mask(x, y, w, h, shape)
        detect = cv2.subtract(rim_m, toe_union)   # rim - toe
        darea = int(np.count_nonzero(detect))
        sin = int(np.count_nonzero(solder_mask & detect))
        ratio = sin / darea if darea > 0 else 0.0
        rim_results.append(dict(r, ratio=ratio, ng=ratio < cold_t, darea=darea))

    # 连锡：pad 外扩框并集 - pad 并集 = 外扩环并集，焊锡连通域接触 ≥ 2 个 pad 且面积 ≥ bridge_min_area → NG
    bridge_results = []
    if state["rois"]["pad"]:
        # 计算外扩框并集
        er = state.get("pad_expand_ratio", 0.30)
        ext_union = np.zeros(shape[:2], dtype=np.uint8)
        for r in state["rois"]["pad"]:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            dx, dy = int(w * er), int(h * er)
            ex0 = max(0, x - dx); ey0 = max(0, y - dy)
            ex1 = min(shape[1], x + w + dx); ey1 = min(shape[0], y + h + dy)
            cv2.rectangle(ext_union, (ex0, ey0), (ex1, ey1), 255, -1)
        between = cv2.subtract(ext_union, pad_union)            # 外扩环并集
        blue_between = cv2.bitwise_and(solder_mask, between)
        k_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blue_between = cv2.morphologyEx(blue_between, cv2.MORPH_OPEN, k_noise)
        if int(np.count_nonzero(blue_between)) > 0:
            pad_dilated = cv2.dilate(pad_union, k_noise)
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(blue_between, connectivity=8)
            for i in range(1, n_lbl):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < bridge_t:
                    continue
                comp_mask = np.zeros(shape[:2], dtype=np.uint8)
                comp_mask[labels == i] = 255
                comp_dilated = cv2.dilate(comp_mask, k_noise)
                # 检查接触的 pad 数量
                pad_count = 0
                for r in state["rois"]["pad"]:
                    pm = _rect_mask(r["x"], r["y"], r["w"], r["h"], shape)
                    if np.any(cv2.bitwise_and(comp_dilated, pm)):
                        pad_count += 1
                if pad_count >= 2:
                    bx = int(stats[i, cv2.CC_STAT_LEFT])
                    by = int(stats[i, cv2.CC_STAT_TOP])
                    bw = int(stats[i, cv2.CC_STAT_WIDTH])
                    bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    bridge_results.append({"x": bx, "y": by, "w": bw, "h": bh,
                                           "area": area, "ng": True, "pads": pad_count})

    return pad_results, rim_results, excess_results, bridge_results, solder_mask

# ─────────────────────────────────────────────
# 真实算法检测：调用 _solder_core 函数
# ─────────────────────────────────────────────
def _build_cfg_from_state():
    """从 tuner 的 state 构建 _solder_core 所需的 cfg dict"""
    # 以算法默认配置为基础，确保所有参数一致
    cfg = SolderSmtAllAlg().default_config()

    # 先应用 params.json 的原始参数 (如 template_mode, crack_min_length_ratio 等)
    # HSV和阈值参数会在后续被 state 覆盖，确保用户通过滑块修改的值优先
    for k, v in state.get("_raw_params", {}).items():
        cfg[k] = v

    # 从 tuner HSV 参数覆盖
    hp = state["hsv_params"]
    cfg["mask_h_low"] = hp["阻焊"]["h_min"]
    cfg["mask_h_high"] = hp["阻焊"]["h_max"]
    cfg["mask_s_min"] = hp["阻焊"]["s_min"]
    cfg["mask_v_min"] = hp["阻焊"]["v_min"]
    cfg["mask_v_max"] = hp["阻焊"]["v_max"]

    cfg["silk_h_low"] = hp["丝印"]["h_min"]
    cfg["silk_h_high"] = hp["丝印"]["h_max"]
    cfg["silk_s_low"] = hp["丝印"]["s_min"]
    cfg["silk_s_high"] = hp["丝印"]["s_max"]
    cfg["silk_v_low"] = hp["丝印"]["v_min"]
    cfg["silk_v_high"] = hp["丝印"]["v_max"]

    cfg["comp_h_low"] = hp["元件"]["h_min"]
    cfg["comp_h_high"] = hp["元件"]["h_max"]
    cfg["comp_s_min"] = hp["元件"]["s_min"]
    cfg["comp_s_max"] = hp["元件"]["s_max"]
    cfg["comp_v_max"] = hp["元件"]["v_max"]

    cfg["toe_metal_h_low"] = hp["toe"]["h_min"]
    cfg["toe_metal_h_high"] = hp["toe"]["h_max"]
    cfg["toe_metal_s_min"] = hp["toe"]["s_min"]
    cfg["toe_metal_s_max"] = hp["toe"]["s_max"]
    cfg["toe_metal_v_min"] = hp["toe"]["v_min"]
    cfg["toe_metal_v_max"] = hp["toe"]["v_max"]

    cfg["solder_blue_h_low"] = hp["焊锡"]["h_min"]
    cfg["solder_blue_h_high"] = hp["焊锡"]["h_max"]
    cfg["solder_blue_s_min"] = hp["焊锡"]["s_min"]
    cfg["solder_blue_v_min"] = hp["焊锡"]["v_min"]

    # 从 tuner 阈值覆盖
    t = state["thresh"]
    cfg["insufficient_thresh"] = t["insufficient"]
    cfg["excess_thresh"] = t["excess"]
    cfg["bridge_min_area"] = t["bridge_min_area"]
    cfg["cold_solder_rim_ratio"] = t["cold_rim"]
    cfg["toe_metal_ratio_thresh"] = t["toe_metal_ratio"]
    cfg["cold_diff_ratio_thresh"] = t["cold_diff_ratio"]
    cfg["cold_solder_dark_ratio"] = t["cold_dark_ratio"]
    cfg["cold_solder_v_dark"] = t["cold_v_dark"]
    cfg["crack_min_length_ratio"] = t["crack_min_length_ratio"]
    cfg["crack_cpp_min_pads"] = t["crack_cpp_min_pads"]

    cfg["pin_type"] = state["pin_type"]

    # 传递HS色板矩阵给算法 (仅在开关开启且有矩阵时)
    if state.get("use_hs_lut", False):
        for o in HSV_OBJECTS:
            name = o["name"]
            lut = state["hs_lut"].get(name)
            if lut is not None and lut.any():
                lut_key = o["keys"]["h_min"].replace("_h_low", "_hs_lut")
                cfg[lut_key] = _encode_hs_lut(lut)

    return cfg


def _compute_outers_real(pad_rects, img_w, img_h, cfg):
    """计算焊盘外扩框 (对齐 SolderSmtAllAlg._compute_outers)"""
    expand_ratio = float(cfg.get("pad_expand_ratio", 0.2))
    short_ratio = float(cfg.get("pad_expand_short_ratio", 0.30))
    aspect_thresh = float(cfg.get("pad_expand_aspect_thresh", 2.0))
    match_aspect = bool(cfg.get("pad_expand_match_aspect", False))
    return [_expand_rect(x, y, w, h, expand_ratio, img_w, img_h,
                         short_ratio, aspect_thresh, match_aspect)
            for (x, y, w, h) in pad_rects]


def _match_to_pads(pad_rects, ann_rects):
    """将标注矩形按中心点匹配到焊盘 (对齐 _match_annotations_to_pads)
    返回: {pad_index: (x,y,w,h)} 或 {}
    """
    if not ann_rects:
        return {}
    result = {}
    for ann_rect in ann_rects:
        ax, ay, aw, ah = ann_rect
        acx = ax + aw / 2.0
        acy = ay + ah / 2.0
        best_pad = -1
        best_overlap = 0
        for pi, (px, py, pw, ph) in enumerate(pad_rects):
            if px <= acx <= px + pw and py <= acy <= py + ph:
                ix1 = max(ax, int(px))
                iy1 = max(ay, int(py))
                ix2 = min(ax + aw, int(px + pw))
                iy2 = min(ay + ah, int(py + ph))
                overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pad = pi
        if best_pad >= 0:
            result[best_pad] = ann_rect
    return result


def compute_detection_real():
    """调用真实算法函数 (_solder_core) 进行检测。
    返回格式与 compute_detection() 兼容:
    (pad_results, rim_results, excess_results, bridge_results, solder_mask)
    """
    if state["img"] is None:
        return [], [], [], [], None
    if not state["rois"]["pad"]:
        return [], [], [], [], None

    try:
        image = state["img"]
        shape = image.shape
        h_img, w_img = shape[:2]

        # 1. 构建 cfg
        cfg = _build_cfg_from_state()

        # 2. pad_rects (已在待检图坐标系)
        pad_rects_orig = [(r["x"], r["y"], r["w"], r["h"]) for r in state["rois"]["pad"]]

        # 2b. auto_resize (对齐生产代码: 缩小图片检测, 结果坐标还原)
        _resize_scale = 1.0
        if cfg.get("enable_auto_resize", True):
            _rs_max = cfg.get("auto_resize_max_size", 400)
            _rs_thresh = cfg.get("auto_resize_threshold", 450)
            _resized, _resize_scale = auto_resize_image(image, _rs_max, _rs_thresh)
            if _resize_scale < 1.0:
                image = _resized
                pad_rects_orig = scale_pad_rects(pad_rects_orig, _resize_scale)
        state["cache"]["detection_img"] = image
        pad_rects = list(pad_rects_orig)

        # 3. 模板对齐 (用于 compute_diff)
        has_tpl = state["template_img"] is not None
        tpl = None
        tpl_info = None
        if has_tpl:
            tpl = state["template_img"].copy()
            if tpl.ndim == 2:
                tpl = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
            elif tpl.ndim == 3 and tpl.shape[2] == 4:
                tpl = tpl[:, :, :3]
            # 同步缩放模板
            if _resize_scale < 1.0:
                tpl, _ = auto_resize_image(tpl, cfg.get("auto_resize_max_size", 400),
                                           cfg.get("auto_resize_threshold", 450))
            tpl_info = align_template(tpl, image, cfg)

        # 3b. 模板对齐映射: 将模板坐标系的ROI映射到待检图坐标系 (对齐算法run())
        if has_tpl and tpl_info is not None:
            pad_rects = [tuple(int(v) for v in map_rect_to_original(r, tpl_info))
                         for r in pad_rects]
            pad_rects = [(x, y, w, h) for (x, y, w, h) in pad_rects if w > 5 and h > 5]
            # 存储映射后的坐标供显示使用
            state["cache"]["aligned_pad_rects"] = [(x, y, w, h) for (x, y, w, h) in pad_rects]
        else:
            state["cache"]["aligned_pad_rects"] = list(pad_rects)

        # 4. outer_rects + group_rects
        outer_rects = _compute_outers_real(pad_rects, w_img, h_img, cfg)
        group_rects = _merge_overlapping_rects(outer_rects)

        # 5. 干扰排除 (预计算HSV供后续复用, loose=True对齐生产代码)
        hsv_shared = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        interference = _remove_interference(image, cfg, loose=True, hsv=hsv_shared)

        # 6. 焊锡提取
        gray_shared = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        solder_result = extract_solder(image, interference, pad_rects, group_rects,
                                       cfg, hsv=hsv_shared, gray=gray_shared)
        solder_mask = solder_result[4]  # mask_final

        # 7. 模板差分 (用于 excess 和 cold_solder)
        diff_mask_exc = None
        if has_tpl and tpl_info is not None:
            try:
                _diff_p = dict(cfg)
                _diff_p["diff_thresh"] = cfg.get("diff_thresh_exc", 30)
                _dm_low, _, _ = compute_diff(tpl, image, tpl_info, _diff_p)
                diff_mask_exc = map_mask_to_original(_dm_low, tpl_info) if _dm_low is not None else None
            except Exception:
                diff_mask_exc = None

        # merge_compute_diff=True -> cold_solder 复用 diff_mask_exc
        diff_mask_cold = diff_mask_exc if cfg.get("merge_compute_diff", True) else None
        # 存储diff_mask用于辅助视图显示
        state["cache"]["diff_mask_raw"] = diff_mask_exc

        # 8. toe/rim 匹配到 pad (toe/rim也需要缩放+对齐映射)
        toe_list = [(r["x"], r["y"], r["w"], r["h"]) for r in state["rois"]["toe"]]
        rim_list = [(r["x"], r["y"], r["w"], r["h"]) for r in state["rois"]["rim"]]
        if _resize_scale < 1.0:
            toe_list = scale_pad_rects(toe_list, _resize_scale)
            rim_list = scale_pad_rects(rim_list, _resize_scale)
        # 模板对齐映射 (toe/rim从模板坐标映射到待检图坐标)
        if has_tpl and tpl_info is not None:
            toe_list = [tuple(int(v) for v in map_rect_to_original(r, tpl_info)) for r in toe_list]
            rim_list = [tuple(int(v) for v in map_rect_to_original(r, tpl_info)) for r in rim_list]
        # 存储映射后的toe/rim坐标(还原到原图尺寸)供显示使用
        if _resize_scale < 1.0:
            _inv = 1.0 / _resize_scale
            state["cache"]["aligned_toe_rects"] = [(int(x*_inv), int(y*_inv), int(w*_inv), int(h*_inv)) for (x,y,w,h) in toe_list]
            state["cache"]["aligned_rim_rects"] = [(int(x*_inv), int(y*_inv), int(w*_inv), int(h*_inv)) for (x,y,w,h) in rim_list]
        else:
            state["cache"]["aligned_toe_rects"] = list(toe_list)
            state["cache"]["aligned_rim_rects"] = list(rim_list)
        toe_rects = _match_to_pads(pad_rects, toe_list)
        rim_rects = _match_to_pads(pad_rects, rim_list)
        # judge_cold_solder 期望 None (非空 dict) 表示无 toe/rim
        toe_rects = toe_rects if toe_rects else None
        rim_rects = rim_rects if rim_rects else None

        # 9. 缺陷判定
        # skip_rb_score=True -> combined_solder = solder_mask
        combined_solder = solder_mask

        pad_res_raw = judge_insufficient(solder_mask, pad_rects, image.shape, cfg)

        exc_mask, exc_is_ng, exc_total_ratio, exc_blobs = judge_excess(
            combined_solder, pad_rects, group_rects, image.shape, cfg,
            diff_mask_full=diff_mask_exc)

        brg_res_raw = judge_bridge(combined_solder, pad_rects, group_rects, image.shape, cfg)

        cold_res_raw = judge_cold_solder(
            image, solder_mask, pad_rects, image.shape, cfg,
            toe_rects=toe_rects, rim_rects=rim_rects,
            diff_mask_full=diff_mask_cold,
            ins_results=pad_res_raw,
            pad_source="ROI",
            hsv=hsv_shared, gray=gray_shared)

        # 10. 转换为兼容格式
        pad_results = [{"x": x, "y": y, "w": w, "h": h, "cov": cov, "ng": ng}
                       for (x, y, w, h, cov, ng) in pad_res_raw]

        rim_results = []
        for i, (x, y, w, h, score, ng, reasons, ratio) in enumerate(cold_res_raw):
            # 使用 rim_rect 坐标（如有），避免与 pad 框重叠
            if rim_rects and i in rim_rects:
                rx, ry, rw, rh = rim_rects[i]
            else:
                rx, ry, rw, rh = x, y, w, h
            rule_str = ",".join(reasons) if reasons else ""
            rim_results.append({"x": rx, "y": ry, "w": rw, "h": rh,
                                "ratio": ratio, "ng": ng, "darea": 0, "rule": rule_str,
                                "has_rim": bool(rim_rects and i in rim_rects)})

        excess_results = []
        for (bx, by, bw, bh) in exc_blobs:
            excess_results.append({"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
                                   "ex": exc_total_ratio, "ng": exc_is_ng,
                                   "ext": (int(bx), int(by), int(bw), int(bh))})

        bridge_results = []
        for (gr, bridge_pairs, is_ng, bridge_blobs) in brg_res_raw:
            if is_ng and bridge_blobs:
                for (bx, by, bw, bh) in bridge_blobs:
                    bridge_results.append({"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
                                           "area": int(bw * bh), "ng": True,
                                           "pads": len(bridge_pairs)})

        # 10b. auto_resize坐标还原: 缩放结果和mask回原图尺寸
        if _resize_scale < 1.0:
            _inv = 1.0 / _resize_scale
            # 还原pad_results
            pad_results = [{"x": int(p["x"]*_inv), "y": int(p["y"]*_inv),
                            "w": int(p["w"]*_inv), "h": int(p["h"]*_inv),
                            "cov": p["cov"], "ng": p["ng"]} for p in pad_results]
            # 还原rim_results (保留has_rim字段)
            rim_results = [{"x": int(r["x"]*_inv), "y": int(r["y"]*_inv),
                            "w": int(r["w"]*_inv), "h": int(r["h"]*_inv),
                            "ratio": r["ratio"], "ng": r["ng"], "darea": r["darea"],
                            "rule": r["rule"], "has_rim": r.get("has_rim", False)} for r in rim_results]
            # 还原excess_results
            excess_results = [{"x": int(e["x"]*_inv), "y": int(e["y"]*_inv),
                               "w": int(e["w"]*_inv), "h": int(e["h"]*_inv),
                               "ex": e["ex"], "ng": e["ng"],
                               "ext": (int(e["x"]*_inv), int(e["y"]*_inv),
                                       int(e["w"]*_inv), int(e["h"]*_inv))} for e in excess_results]
            # 还原bridge_results
            bridge_results = [{"x": int(b["x"]*_inv), "y": int(b["y"]*_inv),
                               "w": int(b["w"]*_inv), "h": int(b["h"]*_inv),
                               "area": int(b["w"]*_inv*b["h"]*_inv), "ng": b["ng"],
                               "pads": b["pads"]} for b in bridge_results]
            # 还原solder_mask到原图尺寸
            solder_mask = cv2.resize(solder_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        return pad_results, rim_results, excess_results, bridge_results, solder_mask

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], [], [], [], None

# ─────────────────────────────────────────────
# 通道范围条（沿用 hsv_viewer）
# ─────────────────────────────────────────────
def _add_range_bar(ch, name, lo, hi, max_val):
    h, w = ch.shape[:2]
    bar_h = 28
    bar = np.zeros((bar_h, w), dtype=np.uint8)
    for i in range(w):
        bar[:, i] = int(i / w * max_val)
    if name == "H":
        bar_col = np.zeros((bar_h, w, 3), dtype=np.uint8)
        for i in range(w):
            bar_col[:, i] = cv2.cvtColor(
                np.uint8([[[int(i / w * 179), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    elif name == "S":
        bar_col = np.zeros((bar_h, w, 3), dtype=np.uint8)
        for i in range(w):
            bar_col[:, i] = cv2.cvtColor(
                np.uint8([[[90, int(i / w * 255), 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    else:
        bar_col = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
    x1 = int(max(0, lo) / max_val * w)
    x2 = int(min(max_val, hi) / max_val * w)
    if x2 <= x1:
        x2 = x1 + 1
    bar_col[:, x1:x2] = (0, 255, 100)
    ch_gray = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR) if len(ch.shape) == 2 else ch
    disp = np.vstack([ch_gray, bar_col])
    cv2.rectangle(disp, (0, h), (w, h + bar_h), (20, 20, 20), -1)
    txt = f"{name} range: [{lo}, {hi}]  (max {max_val})"
    cv2.putText(disp, txt, (8, h + bar_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return disp

# ─────────────────────────────────────────────
# 选中 / 命中测试
# ─────────────────────────────────────────────
HANDLE_R = 7

def rect_corners(r):
    x, y, w, h = r["x"], r["y"], r["w"], r["h"]
    return {
        "tl": (x, y), "tr": (x + w, y),
        "bl": (x, y + h), "br": (x + w, y + h),
    }

def hit_handle(r, mx, my):
    """返回命中的角标 tl/tr/bl/bl，否则 None"""
    for name, (hx, hy) in rect_corners(r).items():
        if abs(mx - hx) <= HANDLE_R and abs(my - hy) <= HANDLE_R:
            return name
    return None

def hit_roi(mx, my):
    """返回点击命中的 {"type","idx","handle"}。
    把手优先(仅选中框)；否则在所有包含该点的ROI中选面积最小的(内层优先,如toe在rim内)"""
    # 先看选中框的把手
    if state["sel"]:
        t, i = state["sel"]["type"], state["sel"]["idx"]
        if 0 <= i < len(state["rois"][t]):
            h = hit_handle(state["rois"][t][i], mx, my)
            if h:
                return {"type": t, "idx": i, "handle": h}
    best = None
    best_area = None
    for t in ROI_TYPES:
        for i, r in enumerate(state["rois"][t]):
            if r["x"] <= mx <= r["x"] + r["w"] and r["y"] <= my <= r["y"] + r["h"]:
                area = r["w"] * r["h"]
                if best_area is None or area < best_area:
                    best_area = area
                    best = {"type": t, "idx": i, "handle": None}
    return best

def clamp_xy(x, y):
    if state["img"] is None:
        return x, y
    h_img, w_img = state["img"].shape[:2]
    return max(0, min(x, w_img - 1)), max(0, min(y, h_img - 1))

def mark_dirty():
    """重计算+轻量叠加都需刷新(状态变化:ROI/参数/阈值/对象/开关)"""
    state["heavy_dirty"] = True
    state["light_dirty"] = True

def mark_light():
    """仅轻量叠加需刷新(鼠标移动:HSV读数/绘制中矩形)"""
    state["light_dirty"] = True

# ─────────────────────────────────────────────
# 撤销
# ─────────────────────────────────────────────
UNDO_LIMIT = 60

def push_undo():
    state["undo_stack"].append(copy.deepcopy(state["rois"]))
    if len(state["undo_stack"]) > UNDO_LIMIT:
        state["undo_stack"].pop(0)

def undo():
    if not state["undo_stack"]:
        return
    state["rois"] = state["undo_stack"].pop()
    state["sel"] = None
    mark_dirty()

# ─────────────────────────────────────────────
# 鼠标回调（按模式分发）
# ─────────────────────────────────────────────
def handle_mouse(etype, x, y):
    """统一鼠标处理。etype: down/up/move/rdown。x,y 为图像坐标。"""
    s = state
    if s["img"] is None:
        return
    if s["main_view_idx"] != 0:
        return  # 非主图视图不响应鼠标
    x, y = clamp_xy(x, y)
    s["mouse_x"], s["mouse_y"] = x, y
    mode = s["mode"]
    if etype == "move":
        mark_light()

    # ── 采样模式：左键采样，右键清空 ──
    if mode == "sample":
        if etype == "down" and s["hsv"] is not None:
            h, sv, v = s["hsv"][y, x]
            s["lock_points"].append((x, y, int(h), int(sv), int(v)))
            if len(s["lock_points"]) > 12:
                s["lock_points"].pop(0)
            s["locks_dirty"] = True
            mark_dirty()
        elif etype == "rdown":
            s["lock_points"].clear()
            s["locks_dirty"] = True
            mark_dirty()
        return

    # ── 绘制模式：左键拖拽画矩形 ──
    if mode in ("draw_pad", "draw_toe", "draw_rim"):
        rtype = mode.split("_")[1]
        if etype == "down":
            s["drawing"] = {"type": rtype, "x0": x, "y0": y, "x1": x, "y1": y}
            mark_light()
        elif etype == "move" and s["drawing"]:
            s["drawing"]["x1"], s["drawing"]["y1"] = x, y
            mark_light()
        elif etype == "up" and s["drawing"]:
            d = s["drawing"]
            rect = norm_rect(d["x0"], d["y0"], d["x1"], d["y1"])
            if rect["w"] >= 3 and rect["h"] >= 3:
                push_undo()
                s["rois"][rtype].append(rect)
                s["sel"] = {"type": rtype, "idx": len(s["rois"][rtype]) - 1}
            s["drawing"] = None
            mark_dirty()
        elif etype == "rdown":
            s["drawing"] = None
            mark_dirty()
        return

    # ── 编辑模式：选中/移动/缩放/删除 ──
    if mode == "edit":
        if etype == "down":
            hit = hit_roi(x, y)
            if hit:
                s["sel"] = {"type": hit["type"], "idx": hit["idx"]}
                r = s["rois"][hit["type"]][hit["idx"]]
                if hit["handle"]:
                    s["drag"] = {"kind": "resize-" + hit["handle"], "ox": x, "oy": y, "orig": dict(r), "pushed": False}
                else:
                    s["drag"] = {"kind": "move", "ox": x, "oy": y, "orig": dict(r), "pushed": False}
            else:
                if s["sel"] is not None:
                    s["sel"] = None
                else:
                    return   # 空白处点击且本就无选中:不变,不触发重算
            mark_dirty()
        elif etype == "move" and s["drag"] and s["sel"]:
            t, i = s["sel"]["type"], s["sel"]["idx"]
            if not (0 <= i < len(s["rois"][t])):
                s["drag"] = None
            else:
                r = s["rois"][t][i]
                dx, dy = x - s["drag"]["ox"], y - s["drag"]["oy"]
                if not s["drag"]["pushed"] and (abs(dx) > 0 or abs(dy) > 0):
                    push_undo()
                    s["drag"]["pushed"] = True
                orig = s["drag"]["orig"]
                if s["drag"]["kind"] == "move":
                    r["x"] = max(0, orig["x"] + dx)
                    r["y"] = max(0, orig["y"] + dy)
                else:
                    corner = s["drag"]["kind"].split("-")[1]
                    x0, y0 = orig["x"], orig["y"]
                    x1, y1 = orig["x"] + orig["w"], orig["y"] + orig["h"]
                    if corner == "tl":
                        x0, y0 = orig["x"] + dx, orig["y"] + dy
                    elif corner == "tr":
                        x1, y0 = orig["x"] + orig["w"] + dx, orig["y"] + dy
                    elif corner == "bl":
                        x0, y1 = orig["x"] + dx, orig["y"] + orig["h"] + dy
                    elif corner == "br":
                        x1, y1 = orig["x"] + orig["w"] + dx, orig["y"] + orig["h"] + dy
                    nx, ny = min(x0, x1), min(y0, y1)
                    nw, nh = abs(x1 - x0), abs(y1 - y0)
                    r["x"], r["y"], r["w"], r["h"] = max(0, nx), max(0, ny), max(1, nw), max(1, nh)
            mark_dirty()
        elif etype == "up":
            s["drag"] = None
        elif etype == "rdown":
            s["sel"] = None
            mark_dirty()
        return

# ─────────────────────────────────────────────
# 渲染主帧
# ─────────────────────────────────────────────
COLORS_BGR = [
    (0, 255, 0), (0, 200, 255), (255, 100, 0), (180, 0, 255),
    (0, 255, 200), (255, 180, 0), (0, 0, 255), (255, 0, 180),
]

# 上下信息栏高度（图片不被遮挡，文字画在黑底填充区）
TOP_BAR_H = 48
BOT_BAR_H = 36

def _build_crack_vis(img, pad_rects):
    """生成crack可视化图: 灰度图 + 暗带mask(白) + Hough线(红/绿)"""
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h_img, w_img = img.shape[:2]

    for (x, y, w, h) in pad_rects:
        x1c = max(0, int(x)); y1c = max(0, int(y))
        x2c = min(w_img, int(x + w)); y2c = min(h_img, int(y + h))
        pw, ph = x2c - x1c, y2c - y1c
        if pw < 10 or ph < 10:
            continue
        gray_roi = gray[y1c:y2c, x1c:x2c]
        pad_mask_roi = np.full((ph, pw), 255, dtype=np.uint8)
        has_crack, lines = _detect_crack_hough(gray_roi, pad_mask_roi, _build_cfg_from_state())

        # 暗带mask叠白
        gray_blur = cv2.GaussianBlur(gray_roi, (21, 21), 0)
        dark_mask = (gray_roi < (gray_blur * 0.85)).astype(np.uint8) * 255
        vis_roi = vis[y1c:y2c, x1c:x2c]
        vis_roi[dark_mask > 0] = (200, 200, 200)

        # pad框: 红=有crack, 绿=无crack
        color = (0, 0, 255) if has_crack else (0, 200, 0)
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), color, 1)

        # Hough线
        for ln in lines:
            x1, y1, x2, y2 = ln
            cv2.line(vis, (x1c + int(x1), y1c + int(y1)), (x1c + int(x2), y1c + int(y2)), (0, 0, 255), 2)

    return vis

def build_heavy():
    """重计算层：图+mask叠加+检测框+采样点+底栏+顶栏背景/行1。
    不含鼠标相关元素(顶栏行2的HSV读数、绘制中矩形)。结果缓存。"""
    s = state
    if s["img"] is None:
        return None, None, None
    frame = s["img"].copy()
    h_img, w_img = frame.shape[:2]

    cur = s["cur_obj"]

    # 调用真实算法检测
    pad_res, rim_res, exc_res, brg_res, solder_mask = compute_detection_real()

    # 当前对象为"焊锡"时，优先使用真实算法的 solder_mask，无pad框时回退到HSV
    if cur == "焊锡":
        cur_mask = solder_mask if solder_mask is not None else compute_mask_for(cur)
    else:
        cur_mask = compute_mask_for(cur)

    if s["show_overlay"] and cur_mask is not None:
        highlight = frame.copy()
        highlight[cur_mask > 0] = OBJ_BY_NAME[cur]["color"]
        frame = cv2.addWeighted(frame, 0.6, highlight, 0.4, 0)

    # 标签缓存: 记录每个框的标签信息，选中时重绘到最上层
    _label_cache = {}  # {(type, idx): (text, x, y, color)}

    # 获取对齐映射后的toe/rim坐标 (无模板时回退到原始ROI)
    _aligned_toe = s["cache"].get("aligned_toe_rects")
    _aligned_rim = s["cache"].get("aligned_rim_rects")
    _toe_rects_display = _aligned_toe if _aligned_toe else [(r["x"], r["y"], r["w"], r["h"]) for r in s["rois"]["toe"]]
    _rim_rects_display = _aligned_rim if _aligned_rim else [(r["x"], r["y"], r["w"], r["h"]) for r in s["rois"]["rim"]]

    if s["show_detect"] and _rim_rects_display:
        shape = s["img"].shape
        toe_union = np.zeros(shape[:2], dtype=np.uint8)
        for (tx, ty, tw, th) in _toe_rects_display:
            toe_union |= _rect_mask(tx, ty, tw, th, shape)
        det_layer = np.zeros_like(frame)
        for (rx, ry, rw, rh) in _rim_rects_display:
            rim_m = _rect_mask(rx, ry, rw, rh, shape)
            det = cv2.subtract(rim_m, toe_union)
            det_layer[det > 0] = DETECT_FILL
        frame = cv2.addWeighted(frame, 1.0, det_layer, 0.25, 0)

    # toe 框（黄色细线 + 金属占比）
    _toe_metal_mask = None
    if _toe_rects_display and s["img"] is not None:
        _toe_hsv = cv2.cvtColor(s["img"][:, :, :3], cv2.COLOR_BGR2HSV)
        hp = s["hsv_params"]["toe"]
        _toe_metal_mask = cv2.inRange(_toe_hsv,
            (hp["h_min"], hp["s_min"], hp["v_min"]),
            (hp["h_max"], hp["s_max"], hp["v_max"]))
    for i, (tx, ty, tw, th) in enumerate(_toe_rects_display):
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), TOE_COLOR, 1)
        if _toe_metal_mask is not None and tw * th > 0:
            roi_mask = np.zeros_like(_toe_metal_mask)
            cv2.rectangle(roi_mask, (tx, ty), (tx+tw, ty+th), 255, -1)
            metal_ratio = np.count_nonzero(_toe_metal_mask & roi_mask) / (tw * th)
            _ltxt = f"toe{i} metal {metal_ratio:.0%}"
            _label(frame, _ltxt, tx, ty, TOE_COLOR)
            _label_cache[("toe", i)] = (_ltxt, tx, ty, TOE_COLOR)
        else:
            _ltxt = f"toe{i}"
            _label(frame, _ltxt, tx, ty, TOE_COLOR)
            _label_cache[("toe", i)] = (_ltxt, tx, ty, TOE_COLOR)

    # 多锡外扩框（先画外扩框，避免被pad框盖住）
    for i, r in enumerate(exc_res):
        ex0, ey0, ew, eh = r["ext"]
        col = EXCESS_NG if r["ng"] else EXCESS_OK
        cv2.rectangle(frame, (ex0, ey0), (ex0 + ew, ey0 + eh), col, 1)
        if r["ng"]:
            _label(frame, f"excess{i} {r['ex']:.0%} NG", ex0, ey0 - 2, col)

    # pad 框（少锡判定）
    ins_idx = 0  # 少锡NG计数器
    for i, r in enumerate(pad_res):
        col = NG_COLOR if r["ng"] else PAD_OK
        cv2.rectangle(frame, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), col, 2)
        if r["ng"]:
            _ltxt = f"pad{i} solder {r['cov']:.0%} NG insufficient{ins_idx}"
            _label(frame, _ltxt, r["x"], r["y"], col)
            _label_cache[("pad", i)] = (_ltxt, r["x"], r["y"], col)
            ins_idx += 1
        else:
            _ltxt = f"pad{i} solder {r['cov']:.0%}"
            _label(frame, _ltxt, r["x"], r["y"], col)
            _label_cache[("pad", i)] = (_ltxt, r["x"], r["y"], col)

    # 虚焊框（内收避免与pad框重合）
    cold_idx = 0  # 虚焊NG计数器
    for i, r in enumerate(rim_res):
        ng = r["ng"]
        has_rim = r.get("has_rim", False)
        # 无rim/toe时: 只有NG才显示
        if not has_rim and not ng:
            continue
        col = NG_COLOR if ng else RIM_OK
        rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
        # 有rim/toe: 与rim框重合(不内收); 无rim/toe: 内收与pad框区分
        if has_rim:
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), col, 2)
            lbl_x, lbl_y = rx, ry
        else:
            inset = max(4, min(rw, rh) // 8)
            cv2.rectangle(frame, (rx + inset, ry + inset),
                          (rx + rw - inset, ry + rh - inset), col, 2)
            lbl_x, lbl_y = rx + inset, ry + inset
        rule_txt = f" [{r['rule']}]" if r.get("rule") else ""
        ratio_pct = r['ratio']
        if has_rim:
            # 有rim/toe: rim{i} ratio% NG cold{j} [rule{k}]
            if ng:
                _ltxt = f"rim{i} {ratio_pct:.0%} NG cold{cold_idx}{rule_txt}"
                _label(frame, _ltxt, lbl_x, lbl_y, col, y_offset=14)
                _label_cache[("rim", i)] = (_ltxt, lbl_x, lbl_y, col)
                cold_idx += 1
            else:
                _ltxt = f"rim{i} {ratio_pct:.0%}"
                _label(frame, _ltxt, lbl_x, lbl_y, col, y_offset=14)
                _label_cache[("rim", i)] = (_ltxt, lbl_x, lbl_y, col)
        else:
            # 无rim/toe: cold{j} ratio% NG [rule{k}]
            _ltxt = f"cold{cold_idx} {ratio_pct:.0%} NG{rule_txt}"
            _label(frame, _ltxt, lbl_x, lbl_y, col, y_offset=14)
            _label_cache[("rim", i)] = (_ltxt, lbl_x, lbl_y, col)
            cold_idx += 1

    # 连锡框（连通域bbox，红色粗框+面积+pad数）
    for i, r in enumerate(brg_res):
        cv2.rectangle(frame, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]),
                      BRIDGE_NG, 2)
        _label(frame, f"bridge{i} a={r['area']} pads={r.get('pads', '?')}", r["x"], r["y"] - 2, BRIDGE_NG)

    if s["sel"]:
        t, i = s["sel"]["type"], s["sel"]["idx"]
        if 0 <= i < len(s["rois"][t]):
            r = s["rois"][t][i]
            cv2.rectangle(frame, (r["x"] - 1, r["y"] - 1),
                          (r["x"] + r["w"] + 1, r["y"] + r["h"] + 1), SEL_COLOR, 1)
            for (hx, hy) in rect_corners(r).values():
                cv2.rectangle(frame, (hx - 3, hy - 3), (hx + 3, hy + 3), SEL_COLOR, -1)
            # 重绘选中框的原始标签到最上层(白色高亮)
            cached = _label_cache.get((t, i))
            if cached:
                _txt, _lx, _ly, _lc = cached
                _label(frame, _txt, _lx, _ly, SEL_COLOR, y_offset=14 if t == "rim" else 0)

    for i, (lx, ly, lh, ls, lv) in enumerate(s["lock_points"]):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        cv2.circle(frame, (lx, ly), 7, color, 2)
        cv2.circle(frame, (lx, ly), 2, color, -1)
        _label(frame, f"#{i + 1} H{lh} S{ls} V{lv}", lx + 8, ly - 8, color)

    # Mask 预览
    cur_label = OBJ_BY_NAME[cur]["label"]
    mask_vis = None
    if cur_mask is not None:
        mask_vis = cv2.cvtColor(cur_mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(cur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_vis, contours, -1, (0, 200, 80), 1)
        pct = int(np.count_nonzero(cur_mask) / cur_mask.size * 100)
        cv2.rectangle(mask_vis, (0, 0), (mask_vis.shape[1], 28), (20, 20, 20), -1)
        cv2.putText(mask_vis, f"[{cur_label}] Mask cov={pct}% n={len(contours)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 255), 1, cv2.LINE_AA)

    # 通道（始终生成）
    p = s["hsv_params"][cur]
    H, S, V_ = cv2.split(s["hsv"])
    ch = {
        "H": _add_range_bar(H, "H", p["h_min"], p["h_max"], 179),
        "S": _add_range_bar(S, "S", p["s_min"], p["s_max"], 255),
        "V": _add_range_bar(V_, "V", p["v_min"], p["v_max"], 255),
    }
    # 在图片上下添加黑底信息栏（不遮挡图片）
    ng_ins = sum(1 for r in pad_res if r["ng"])
    ng_exc = sum(1 for r in exc_res if r["ng"])
    ng_cold = sum(1 for r in rim_res if r["ng"])
    ng_brg = len(brg_res)
    mode_txt = {"sample": "SAMPLE", "draw_pad": "DRAW pad",
                "draw_toe": "DRAW toe", "draw_rim": "DRAW rim",
                "edit": "EDIT"}[s["mode"]]
    padded = np.full((h_img + TOP_BAR_H + BOT_BAR_H, w_img, 3), 20, dtype=np.uint8)
    padded[TOP_BAR_H:TOP_BAR_H + h_img, :] = frame
    fs = 0.46 if w_img >= 300 else max(0.28, 0.46 * w_img / 300)
    cv2.putText(padded,
        f"[{mode_txt}] OBJ:{cur_label} pad:{len(pad_res)} cold:{len(rim_res)} toe:{len(s['rois']['toe'])}",
        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(padded,
        f"NG: ins<{ng_ins} exc<{ng_exc} cold<{ng_cold} brg<{ng_brg}",
        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (255, 200, 200), 1, cv2.LINE_AA)
    bar_y = h_img + TOP_BAR_H
    th = s["thresh"]
    if s["pin_type"] == "gull-wing":
        cold_txt = f"R1<{th['cold_rim']:.0%} R2>{th['toe_metal_ratio']:.0%} R3>{th['cold_diff_ratio']:.0%}"
    else:
        cold_txt = f"dark>{th['cold_dark_ratio']:.0%} V<{th['cold_v_dark']}"
    cv2.putText(padded,
        f"{cur_label} H[{p['h_min']},{p['h_max']}] S[{p['s_min']},{p['s_max']}] V[{p['v_min']},{p['v_max']}]",
        (8, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(padded,
        f"[{s['pin_type']}] ins<{th['insufficient']:.0%} exc>{th['excess']:.0%} brg>={th['bridge_min_area']}px {cold_txt}",
        (8, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 120, 120), 1, cv2.LINE_AA)

    # 生成模板和diff_mask预览（用于辅助视图）
    tpl_vis = None
    if state["template_img"] is not None:
        tpl_vis = state["template_img"].copy()
        if tpl_vis.ndim == 2:
            tpl_vis = cv2.cvtColor(tpl_vis, cv2.COLOR_GRAY2BGR)
    state["cache"]["template_vis"] = tpl_vis

    diff_vis = None
    dm = state["cache"].get("diff_mask_raw")
    if dm is not None:
        diff_vis = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)
        # 差异区域用红色高亮
        diff_vis[dm > 0] = (0, 0, 255)
    state["cache"]["diffmask_vis"] = diff_vis

    # crack可视化: 每个pad的暗带mask + Hough线
    crack_vis = None
    try:
        aligned_pads = state["cache"].get("aligned_pad_rects")
        det_img = state["cache"].get("detection_img")
        if aligned_pads and det_img is not None:
            crack_vis = _build_crack_vis(det_img, aligned_pads)
    except Exception:
        pass
    state["cache"]["crack_vis"] = crack_vis

    return padded, mask_vis, ch

def zoom_to(new_zoom):
    """设置缩放比例并重新渲染"""
    state["zoom"] = max(0.1, min(10.0, round(new_zoom, 2)))
    zoom_label.config(text=f"{int(state['zoom']*100)}%")
    # 重新渲染主图
    render_base_to_canvas()
    # 更新 scrollregion
    if state["img"] is not None:
        h, w = state["img"].shape[:2]
        zoom = state["zoom"]
        canvas.config(scrollregion=(0, 0, int(w * zoom), int((h + TOP_BAR_H + BOT_BAR_H) * zoom)))
    # 刷新轻量层（绘制中的矩形位置）
    state["light_dirty"] = True

def zoom_in():
    zoom_to(state.get("zoom", 1.0) * 1.25)

def zoom_out():
    zoom_to(state.get("zoom", 1.0) / 1.25)

def zoom_reset():
    zoom_to(1.0)

def _on_ctrl_wheel(event):
    """Ctrl+滚轮缩放"""
    # 跨平台: Windows用event.delta, Linux用event.num
    if hasattr(event, 'num') and event.num > 0:
        delta = 1.1 if event.num == 4 else (1/1.1)  # 4=上滚(放大), 5=下滚(缩小)
    else:
        delta = 1.1 if event.delta > 0 else (1/1.1)
    zoom_to(state.get("zoom", 1.0) * delta)

def _on_canvas_wheel(event):
    """滚轮: Ctrl+滚轮缩放, 普通滚轮滚动 (跨平台)"""
    if event.state & 0x4:  # Ctrl 键按下
        _on_ctrl_wheel(event)
    else:
        delta = event.delta if event.num == 0 else event.delta if hasattr(event, 'delta') else 0
        if event.num == 4:  # Linux上滚
            canvas.yview_scroll(-3, "units")
        elif event.num == 5:  # Linux下滚
            canvas.yview_scroll(3, "units")
        else:  # Windows/Mac
            canvas.yview_scroll(int(-event.delta / 120), "units")

def render_base_to_canvas():
    """重计算结果(基础帧)-> Tk Canvas 图片项。仅 heavy 变化时调用。"""
    base = state["cache"]["base"]
    if base is None or canvas is None:
        return
    zoom = state.get("zoom", 1.0)
    if zoom != 1.0:
        h, w = base.shape[:2]
        new_w, new_h = int(w * zoom), int(h * zoom)
        base = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    photo = PILImageTk.PhotoImage(pil)
    canvas.itemconfig(canvas_img_id, image=photo)
    canvas.photo = photo   # 防 GC

def update_ng_info():
    """更新右侧 NG 统计信息栏（heavy 重算后调用）"""
    if state["img"] is None:
        ng_info_var.set("未加载图片")
        return
    pad_res, rim_res, exc_res, brg_res, _ = compute_detection_real()
    ng_ins = sum(1 for r in pad_res if r["ng"])
    ng_exc = sum(1 for r in exc_res if r["ng"])
    ng_cold = sum(1 for r in rim_res if r["ng"])
    ng_brg = len(brg_res)
    t = state["thresh"]
    pt = state["pin_type"]
    # 显示实际触发的规则 + 参数值
    rule_params = {
        "rule1": f"rim-toe<{t['cold_rim']:.0%}",
        "rule2": f"toe金属>{t['toe_metal_ratio']:.0%}",
        "rule3": f"diff>{t['cold_diff_ratio']:.0%}",
        "dark": f"暗区>{t['cold_dark_ratio']:.0%}",
        "crack": "crack检测",
        "red_glue": "红胶",
    }
    rules_set = set()
    for r in rim_res:
        if r["ng"] and r.get("rule"):
            rules_set.update(r["rule"].split(","))
    if rules_set:
        parts = []
        for rname in sorted(rules_set):
            pval = rule_params.get(rname, rname)
            parts.append(f"{rname}({pval})")
        rules_str = " ".join(parts)
    else:
        rules_str = "无"
    cold_line = f"虚焊 NG: {ng_cold}/{len(rim_res)}  [{rules_str}]"
    txt = (f"[{pt}] pad:{len(pad_res)} toe:{len(state['rois']['toe'])} rim:{len(rim_res)}\n"
           f"少锡 NG: {ng_ins}/{len(pad_res)}  (占比<{t['insufficient']:.0%})\n"
           f"多锡 NG: {ng_exc}/{len(exc_res)}  (外扩>{t['excess']:.0%})\n"
           f"{cold_line}\n"
           f"连锡 NG: {ng_brg}  (面积≥{t['bridge_min_area']}px)")
    ng_info_var.set(txt)

def update_light_items():
    """轻量层: 更新 Tk Canvas 上的 HSV读数文本 + 绘制中矩形(不重算不转图)。"""
    s = state
    if s["img"] is None or canvas is None:
        return
    mx, my = s["mouse_x"], s["mouse_y"]
    # bounds check: 防止切换图片后鼠标坐标越界
    h_img, w_img = s["img"].shape[:2]
    mx = max(0, min(mx, w_img - 1))
    my = max(0, min(my, h_img - 1))
    hv, sv_val, vv = s["hsv"][my, mx]
    hsv_readout_var.set(f"X:{mx:4d} Y:{my:4d}  H:{hv:3d} S:{sv_val:3d} V:{vv:3d}")
    if s["drawing"]:
        d = s["drawing"]
        zoom = s.get("zoom", 1.0)
        canvas.coords(canvas_draw_id, d["x0"] * zoom, (d["y0"] + TOP_BAR_H) * zoom,
                      d["x1"] * zoom, (d["y1"] + TOP_BAR_H) * zoom)
        canvas.itemconfig(canvas_draw_id, state="normal")
    else:
        canvas.itemconfig(canvas_draw_id, state="hidden")

def _label(frame, txt, x, y, color, y_offset=0):
    """在指定位置写标签（带黑底），y_offset向下偏移防止重叠"""
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    lx = min(int(x), frame.shape[1] - tw - 4)
    ly = max(int(y) - 4 + y_offset, th + 2)
    cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (20, 20, 20), -1)
    cv2.putText(frame, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
# 刷新循环(主图走Tk Canvas, mask/通道走OpenCV)
# ─────────────────────────────────────────────
def cv_update():
    try:
        if state.get("locks_dirty"):
            state["locks_dirty"] = False
            try:
                update_lock_panel()
            except Exception:
                pass

        if state["img"] is not None:
            cache = state["cache"]
            if state["heavy_dirty"]:
                state["heavy_dirty"] = False
                try:
                    base, mask_vis, ch = build_heavy()
                    cache["base"], cache["mask_vis"], cache["ch"] = base, mask_vis, ch
                except Exception:
                    pass
                try:
                    update_ng_info()
                except Exception:
                    pass
                # 渲染主视图和辅助视图
                _render_main_view()
                render_aux_canvases()
                state["light_dirty"] = True
            if state["light_dirty"]:
                state["light_dirty"] = False
                try:
                    update_light_items()
                except Exception:
                    pass
    except Exception:
        pass

    root.after(30, cv_update)

# ─────────────────────────────────────────────
# ROI缩放 (图片尺寸与pad.json来源不同时)
# ─────────────────────────────────────────────
def _scale_rois(orig_rois, src_w, src_h, dst_w, dst_h):
    """按比例缩放ROI坐标"""
    if src_w == dst_w and src_h == dst_h:
        return copy.deepcopy(orig_rois)
    sx = dst_w / src_w
    sy = dst_h / src_h
    rois = {"pad": [], "toe": [], "rim": []}
    for t in ("pad", "toe", "rim"):
        for r in orig_rois.get(t, []):
            rois[t].append({
                "x": max(0, int(round(r["x"] * sx))),
                "y": max(0, int(round(r["y"] * sy))),
                "w": max(1, int(round(r["w"] * sx))),
                "h": max(1, int(round(r["h"] * sy))),
            })
    return rois

def _apply_rois_for_current_image():
    """根据当前图片尺寸，从orig_rois缩放生成当前rois"""
    s = state
    if s["orig_rois"] is None or s["img"] is None or s["rois_src_size"] is None:
        return
    src_w, src_h = s["rois_src_size"]
    dst_h, dst_w = s["img"].shape[:2]
    s["rois"] = _scale_rois(s["orig_rois"], src_w, src_h, dst_w, dst_h)
    s["sel"] = None
    s["undo_stack"].clear()
    mark_dirty()

def switch_pair_image(idx):
    """切换到pair中的第idx张图"""
    s = state
    if not s["pair_images"] or idx < 0 or idx >= len(s["pair_images"]):
        return
    # 先保存当前ROI回orig（如果当前图就是模板尺寸）
    if s["orig_rois"] is not None and s["rois_src_size"] is not None:
        src_w, src_h = s["rois_src_size"]
        cur_h, cur_w = s["img"].shape[:2] if s["img"] is not None else (0, 0)
        if cur_w == src_w and cur_h == src_h:
            s["orig_rois"] = copy.deepcopy(s["rois"])
    
    name, path = s["pair_images"][idx]
    s["pair_img_idx"] = idx
    _load_image_internal(path)
    _apply_rois_for_current_image()
    _update_image_nav_label()

def next_pair_image():
    s = state
    if s["pair_images"]:
        switch_pair_image((s["pair_img_idx"] + 1) % len(s["pair_images"]))

def prev_pair_image():
    s = state
    if s["pair_images"]:
        switch_pair_image((s["pair_img_idx"] - 1) % len(s["pair_images"]))

def _update_image_nav_label():
    s = state
    if not s["pair_images"]:
        img_nav_var.set("")
        return
    name, path = s["pair_images"][s["pair_img_idx"]]
    total = len(s["pair_images"])
    img_nav_var.set(f"[{s['pair_img_idx']+1}/{total}] {name}")

# ─────────────────────────────────────────────
# 加载图片
# ─────────────────────────────────────────────
def _load_image_internal(path):
    """内部加载图片（不弹文件选择框，不重置窗口位置）"""
    path = path.strip()
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        messagebox.showerror("Error", f"Cannot read image:\n{path}")
        return
    state["img"] = img
    state["hsv"] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    state["path"] = path
    state["lock_points"].clear()
    state["sel"] = None
    state["drawing"] = None
    state["undo_stack"].clear()
    # 重置鼠标坐标到安全位置（防止切换到更小的图时越界）
    h, w = img.shape[:2]
    state["mouse_x"] = min(state["mouse_x"], w - 1)
    state["mouse_y"] = min(state["mouse_y"], h - 1)
    state["heavy_dirty"] = True
    state["light_dirty"] = True
    state["locks_dirty"] = True

    h, w = img.shape[:2]
    setup_canvas(w, h)

    root.title(f"PCBA Tuner - {os.path.basename(path)}")
    lbl_file.config(text=f"[Img] {os.path.basename(path)}  ({w}x{h})")

def load_image(path=None):
    if path is None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
    if not path:
        return
    # 单独打开图片时清空pair状态
    state["pair_images"] = []
    state["pair_img_idx"] = 0
    state["orig_rois"] = None
    state["rois_src_size"] = None
    _load_image_internal(path)

# ─────────────────────────────────────────────
# 模式 / 对象切换
# ─────────────────────────────────────────────
def set_mode(m):
    state["mode"] = m
    state["drawing"] = None
    state["drag"] = None
    _sync_mode_buttons()
    mark_dirty()

def _sync_mode_buttons():
    for m, btn in mode_buttons.items():
        if m == state["mode"]:
            btn.config(bg=ACCENT, fg="#1e1e2e")
        else:
            btn.config(bg=BG2, fg=FG)

def sync_sliders_to_obj():
    """当前滑条值 → 写入当前对象参数"""
    name = state["cur_obj"]
    for k in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max"):
        state["hsv_params"][name][k] = sliders[k].get()

def load_obj_to_sliders():
    """当前对象参数 -> 滑条"""
    global _suppress_trace
    name = state["cur_obj"]
    p = state["hsv_params"][name]
    _suppress_trace = True
    for k in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max"):
        sliders[k].set(p[k])
    _suppress_trace = False
    lbl_obj.config(text=f"当前对象: {name}")
    _update_hs_lut(name)
    _draw_palette()

def switch_object(name):
    sync_sliders_to_obj()
    state["cur_obj"] = name
    load_obj_to_sliders()
    var_obj.set(name)

def cycle_object():
    idx = OBJ_NAMES.index(state["cur_obj"])
    switch_object(OBJ_NAMES[(idx + 1) % len(OBJ_NAMES)])

def on_obj_changed(*_):
    sync_sliders_to_obj()
    state["cur_obj"] = var_obj.get()
    load_obj_to_sliders()
    mark_dirty()

# ─────────────────────────────────────────────
# 阈值 / 滑条联动
# ─────────────────────────────────────────────
def on_slider_changed(*_):
    if _suppress_trace:
        return
    sync_sliders_to_obj()
    _update_hs_lut(state["cur_obj"])
    _draw_palette()
    mark_dirty()

def on_thresh_changed(*_):
    if _suppress_trace:
        return
    state["thresh"]["insufficient"] = var_ins.get() / 100.0
    state["thresh"]["excess"] = var_exc.get() / 100.0
    state["thresh"]["bridge_min_area"] = var_brg.get()
    state["thresh"]["cold_rim"] = var_cold.get() / 100.0
    state["thresh"]["toe_metal_ratio"] = var_toe_metal.get() / 100.0
    state["thresh"]["cold_dark_ratio"] = var_dark.get() / 100.0
    state["thresh"]["cold_v_dark"] = var_vdark.get()
    state["thresh"]["cold_diff_ratio"] = var_diff.get() / 100.0
    state["thresh"]["crack_min_length_ratio"] = var_crack_len.get() / 100.0
    state["thresh"]["crack_cpp_min_pads"] = var_cpp_min_pads.get()
    mark_dirty()

# ─────────────────────────────────────────────
# HS 色板 (circular palette)
# ─────────────────────────────────────────────
_palette_bg_cache = None

def _gen_palette_bg(size=200):
    """Generate circular HSV palette background image"""
    global _palette_bg_cache
    if _palette_bg_cache is not None:
        return _palette_bg_cache
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (20, 20, 20))
    px = img.load()
    cx, cy = size // 2, size // 2
    r_max = size // 2 - 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= r_max:
                h = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)  # 0-180
                s = int(min(dist / r_max, 1.0) * 255)  # 0-255
                v = 255
                bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
                px[x, y] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    _palette_bg_cache = img
    return img

PALETTE_SIZE = 200
PALETTE_CENTER = PALETTE_SIZE // 2
PALETTE_RADIUS = PALETTE_CENTER - 2

def _hs_to_canvas(h, s):
    """H(0-180), S(0-255) -> canvas (x, y)"""
    angle = math.radians(h * 2)  # H*2 to get 0-360 degrees
    r = s / 255.0 * PALETTE_RADIUS
    x = PALETTE_CENTER + r * math.cos(angle)
    y = PALETTE_CENTER + r * math.sin(angle)
    return int(x), int(y)

def _canvas_to_hs(x, y):
    """canvas (x, y) -> H(0-180), S(0-255)"""
    dx, dy = x - PALETTE_CENTER, y - PALETTE_CENTER
    dist = (dx * dx + dy * dy) ** 0.5
    h = int((math.degrees(math.atan2(dy, dx)) % 360) / 2)
    s = int(min(dist / PALETTE_RADIUS, 1.0) * 255)
    return h, s

def _update_hs_lut(obj_name):
    """Update HS LUT from polygon points + slider sector"""
    p = state["hsv_params"][obj_name]
    h_lo, h_hi = int(p["h_min"]), int(p["h_max"])
    s_lo, s_hi = int(p["s_min"]), int(p["s_max"])

    lut = np.zeros((180, 256), dtype=bool)
    # Sector
    if h_lo <= h_hi:
        lut[h_lo:h_hi+1, s_lo:s_hi+1] = True
    else:
        lut[h_lo:, s_lo:s_hi+1] = True
        lut[:h_hi+1, s_lo:s_hi+1] = True

    # Polygon intersection - 在画布(极坐标)空间画多边形, 再映射回LUT
    pts = state["palette_points"].get(obj_name, [])
    if len(pts) >= 3:
        # 1. 在画布上画多边形
        canvas_mask = np.zeros((PALETTE_SIZE, PALETTE_SIZE), dtype=np.uint8)
        canvas_pts = np.array([_hs_to_canvas(h, s) for h, s in pts], dtype=np.int32)
        cv2.fillPoly(canvas_mask, [canvas_pts], 1)

        # 2. 对LUT中每个(h,s), 转换到画布坐标查mask
        h_all = np.arange(180).reshape(-1, 1)   # (180,1)
        s_all = np.arange(256).reshape(1, -1)    # (1,256)
        angles = np.radians(h_all * 2.0)          # (180,1)
        radii = s_all / 255.0 * PALETTE_RADIUS    # (1,256)
        cx = (PALETTE_CENTER + radii * np.cos(angles)).astype(np.int32)  # (180,256)
        cy = (PALETTE_CENTER + radii * np.sin(angles)).astype(np.int32)
        cx = np.clip(cx, 0, PALETTE_SIZE - 1)
        cy = np.clip(cy, 0, PALETTE_SIZE - 1)
        poly_mask = canvas_mask[cy, cx] > 0       # (180,256) bool
        lut &= poly_mask

    state["hs_lut"][obj_name] = lut

def _encode_hs_lut(lut):
    """Encode 180x256 bool LUT to base64 string (downsampled 36x64 + zlib)"""
    if lut is None:
        return None
    # Downsample: 180->36 (every 5), 256->64 (every 4)
    small = lut[::5, ::4]  # 36x64
    packed = np.packbits(small.flatten())  # 288 bytes
    return base64.b64encode(zlib.compress(packed.tobytes())).decode('ascii')

def _decode_hs_lut(b64str):
    """Decode base64 string to 180x256 bool LUT (upsampled)"""
    if not b64str:
        return None
    packed = np.frombuffer(zlib.decompress(base64.b64decode(b64str)), dtype=np.uint8)
    small = np.unpackbits(packed)[:36*64].reshape(36, 64).astype(bool)
    # Upsample: 36->180 (repeat 5x), 64->256 (repeat 4x)
    lut = np.repeat(np.repeat(small, 5, axis=0), 4, axis=1)[:180, :256]
    return lut

_palette_drag_mode = None  # None, "h_min", "h_max", "s_min", "s_max", "polygon"
_palette_hover_boundary = None
_palette_drag_pt_idx = -1

def _draw_palette():
    """Redraw the HS palette canvas"""
    cv = palette_canvas
    cv.delete("all")

    # Background
    bg_img = _gen_palette_bg(PALETTE_SIZE)
    cv._bg_photo = PILImageTk.PhotoImage(bg_img)
    cv.create_image(0, 0, anchor="nw", image=cv._bg_photo)

    obj = state["cur_obj"]
    p = state["hsv_params"][obj]
    pts = state["palette_points"].get(obj, [])

    # Sector boundary (dashed)
    h_lo, h_hi = int(p["h_min"]), int(p["h_max"])
    s_lo, s_hi = int(p["s_min"]), int(p["s_max"])

    # Draw sector fill (semi-transparent stipple)
    sector_pts = []
    for h in range(h_lo, h_hi + 1 if h_lo <= h_hi else 180):
        x, y = _hs_to_canvas(h, s_hi)
        sector_pts.extend([x, y])
    for h in range(h_hi if h_lo <= h_hi else 179, h_lo - 1 if h_lo <= h_hi else -1, -1):
        x, y = _hs_to_canvas(h, s_lo)
        sector_pts.extend([x, y])
    if len(sector_pts) >= 6:
        cv.create_polygon(sector_pts, fill="", outline="", stipple="", tags="sector")

    # Sector boundary lines (dashed)
    # H boundaries (radial lines)
    for label, h_val, tag in [("H_min", h_lo, "h_min"), ("H_max", h_hi, "h_max")]:
        x1, y1 = _hs_to_canvas(h_val, s_lo)
        x2, y2 = _hs_to_canvas(h_val, s_hi)
        dash = () if _palette_hover_boundary == tag else (4, 2)
        color = "#ff6655" if _palette_hover_boundary == tag else "#ffffff"
        cv.create_line(x1, y1, x2, y2, fill=color, dash=dash, width=2, tags=tag)

    # S boundaries (arcs)
    for label, s_val, tag in [("S_min", s_lo, "s_min"), ("S_max", s_hi, "s_max")]:
        r = s_val / 255.0 * PALETTE_RADIUS
        dash = () if _palette_hover_boundary == tag else (4, 2)
        color = "#55ff77" if _palette_hover_boundary == tag else "#ffffff"
        # Draw arc as series of line segments
        arc_pts = []
        h_start = min(h_lo, h_hi)
        h_end = max(h_lo, h_hi)
        for h in range(h_start, h_end + 1):
            x, y = _hs_to_canvas(h, s_val)
            arc_pts.extend([x, y])
        if len(arc_pts) >= 4:
            cv.create_line(arc_pts, fill=color, dash=dash, width=2, tags=tag)

    # Intersection fill (highlighted)
    _draw_intersection(cv, obj, p, pts)

    # Polygon (solid line)
    if len(pts) >= 2:
        poly_xy = []
        for h, s in pts:
            x, y = _hs_to_canvas(h, s)
            poly_xy.extend([x, y])
        if len(pts) >= 3:
            cv.create_polygon(poly_xy, fill="", outline="#ffff00", width=2, tags="polygon")
        else:
            cv.create_line(poly_xy, fill="#ffff00", width=2, tags="polygon")

    # Control points
    for i, (h, s) in enumerate(pts):
        x, y = _hs_to_canvas(h, s)
        cv.create_oval(x-4, y-4, x+4, y+4, fill="#ffff00", outline="#000", tags=f"pt_{i}")

def _draw_intersection(cv, obj, p, pts):
    """Draw the intersection of polygon and sector as darkened overlay"""
    # 没有控制点时不画交集
    if len(pts) < 3:
        return
    lut = state["hs_lut"].get(obj)
    if lut is None:
        return
    # 用半透明暗色小矩形覆盖交集区域
    for h in range(0, 180, 2):
        for s in range(0, 256, 6):
            if lut[h, s]:
                x, y = _hs_to_canvas(h, s)
                cv.create_rectangle(x-1, y-1, x+1, y+1, fill="#404040",
                                    outline="", stipple="gray50", tags="intersection")

def _on_palette_motion(event):
    """Handle mouse motion on palette - hover detection and dragging"""
    global _palette_hover_boundary, _palette_drag_mode, _palette_drag_pt_idx, _suppress_trace
    x, y = event.x, event.y

    if _palette_drag_mode:
        # Currently dragging
        if _palette_drag_mode in ("h_min", "h_max"):
            h, s = _canvas_to_hs(x, y)
            obj = state["cur_obj"]
            p = state["hsv_params"][obj]
            if _palette_drag_mode == "h_min":
                p["h_min"] = max(0, min(179, h))
                _suppress_trace = True
                sliders["h_min"].set(p["h_min"])
                _suppress_trace = False
            else:
                p["h_max"] = max(0, min(179, h))
                _suppress_trace = True
                sliders["h_max"].set(p["h_max"])
                _suppress_trace = False
            _update_hs_lut(obj)
            _draw_palette()
            mark_dirty()
        elif _palette_drag_mode in ("s_min", "s_max"):
            h, s = _canvas_to_hs(x, y)
            obj = state["cur_obj"]
            p = state["hsv_params"][obj]
            if _palette_drag_mode == "s_min":
                p["s_min"] = max(0, min(255, s))
                _suppress_trace = True
                sliders["s_min"].set(p["s_min"])
                _suppress_trace = False
            else:
                p["s_max"] = max(0, min(255, s))
                _suppress_trace = True
                sliders["s_max"].set(p["s_max"])
                _suppress_trace = False
            _update_hs_lut(obj)
            _draw_palette()
            mark_dirty()
        elif _palette_drag_mode == "polygon" and _palette_drag_pt_idx >= 0:
            h, s = _canvas_to_hs(x, y)
            obj = state["cur_obj"]
            state["palette_points"][obj][_palette_drag_pt_idx] = [h, s]
            _update_hs_lut(obj)
            _draw_palette()
            mark_dirty()
        return

    # Hover detection: check distance to sector boundaries
    obj = state["cur_obj"]
    p = state["hsv_params"][obj]
    h_lo, h_hi = int(p["h_min"]), int(p["h_max"])
    s_lo, s_hi = int(p["s_min"]), int(p["s_max"])

    new_hover = None
    # Check polygon points first
    pts = state["palette_points"].get(obj, [])
    for i, (ph, ps) in enumerate(pts):
        px, py = _hs_to_canvas(ph, ps)
        if (x - px)**2 + (y - py)**2 <= 36:  # 6px radius
            new_hover = f"pt_{i}"
            break

    if new_hover is None:
        # Check sector boundaries
        mh, ms = _canvas_to_hs(x, y)
        # H boundaries: check if angle is close to h_lo or h_hi
        h_tol = 3
        if abs(mh - h_lo) <= h_tol or abs(mh - h_lo - 180) <= h_tol or abs(mh - h_lo + 180) <= h_tol:
            if s_lo <= ms <= s_hi:
                new_hover = "h_min"
        elif abs(mh - h_hi) <= h_tol or abs(mh - h_hi - 180) <= h_tol or abs(mh - h_hi + 180) <= h_tol:
            if s_lo <= ms <= s_hi:
                new_hover = "h_max"
        # S boundaries
        s_tol = 8
        if abs(ms - s_lo) <= s_tol and (h_lo <= mh <= h_hi if h_lo <= h_hi else (mh >= h_lo or mh <= h_hi)):
            new_hover = "s_min"
        elif abs(ms - s_hi) <= s_tol and (h_lo <= mh <= h_hi if h_lo <= h_hi else (mh >= h_lo or mh <= h_hi)):
            new_hover = "s_max"

    if new_hover != _palette_hover_boundary:
        _palette_hover_boundary = new_hover
        _draw_palette()
        if new_hover and new_hover.startswith("h_") or new_hover and new_hover.startswith("s_"):
            palette_canvas.config(cursor="hand2")
        else:
            palette_canvas.config(cursor="")

def _on_palette_click(event):
    """Handle click on palette - start dragging"""
    global _palette_drag_mode, _palette_drag_pt_idx

    if _palette_hover_boundary:
        if _palette_hover_boundary.startswith("pt_"):
            _palette_drag_mode = "polygon"
            _palette_drag_pt_idx = int(_palette_hover_boundary.split("_")[1])
        else:
            _palette_drag_mode = _palette_hover_boundary
            _palette_drag_pt_idx = -1

def _on_palette_release(event):
    """Handle mouse release - stop dragging"""
    global _palette_drag_mode, _palette_drag_pt_idx
    _palette_drag_mode = None
    _palette_drag_pt_idx = -1

def _on_palette_dblclick(event):
    """Double click to add polygon control point"""
    x, y = event.x, event.y
    dx, dy = x - PALETTE_CENTER, y - PALETTE_CENTER
    if dx*dx + dy*dy > PALETTE_RADIUS * PALETTE_RADIUS:
        return
    h, s = _canvas_to_hs(x, y)
    obj = state["cur_obj"]
    pts = state["palette_points"].setdefault(obj, [])
    pts.append([h, s])
    _update_hs_lut(obj)
    _draw_palette()
    mark_dirty()

def _on_palette_rclick(event):
    """Right click to delete nearest polygon point"""
    global _palette_hover_boundary
    if _palette_hover_boundary and _palette_hover_boundary.startswith("pt_"):
        idx = int(_palette_hover_boundary.split("_")[1])
        obj = state["cur_obj"]
        pts = state["palette_points"].get(obj, [])
        if 0 <= idx < len(pts):
            pts.pop(idx)
            _update_hs_lut(obj)
            _draw_palette()
            mark_dirty()

def _on_palette_leave(event):
    """Mouse leaves palette - clear hover"""
    global _palette_hover_boundary
    if _palette_drag_mode is None:
        _palette_hover_boundary = None
        _draw_palette()
        palette_canvas.config(cursor="")

# ─────────────────────────────────────────────
# 采样点 / 自动填充
# ─────────────────────────────────────────────
def hsv_to_hex(h, s, v):
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
    return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

def circular_mean(angles, modulus):
    sin_sum = sum(math.sin(2 * math.pi * a / modulus) for a in angles)
    cos_sum = sum(math.cos(2 * math.pi * a / modulus) for a in angles)
    return int(math.atan2(sin_sum, cos_sum) * modulus / (2 * math.pi)) % modulus

def auto_fill_from_locks():
    pts = state["lock_points"]
    if not pts:
        messagebox.showinfo("Info", "Sample at least 1 point first (switch to SAMPLE mode, left-click on image)")
        return
    margin_h = int(entry_margin_h.get() or 10)
    margin_s = int(entry_margin_s.get() or 40)
    margin_v = int(entry_margin_v.get() or 40)
    hs = [p[2] for p in pts]; ss = [p[3] for p in pts]; vs = [p[4] for p in pts]
    h_mean = circular_mean(hs, 180); s_mean = int(np.mean(ss)); v_mean = int(np.mean(vs))
    sync_sliders_to_obj()
    name = state["cur_obj"]
    p = state["hsv_params"][name]
    p["h_min"] = max(0, h_mean - margin_h); p["h_max"] = min(179, h_mean + margin_h)
    p["s_min"] = max(0, s_mean - margin_s); p["s_max"] = min(255, s_mean + margin_s)
    p["v_min"] = max(0, v_mean - margin_v); p["v_max"] = min(255, v_mean + margin_v)
    load_obj_to_sliders()
    mark_dirty()

def update_lock_panel():
    for w in lock_frame.winfo_children():
        w.destroy()
    pts = state["lock_points"]
    if not pts:
        tk.Label(lock_frame, text="（无采样点：切到「采样」模式左键点击）", fg="#888", bg=BG,
                 font=FONT_SMALL).pack(anchor="w", padx=4)
        return
    for i, (x, y, h, s, v) in enumerate(pts):
        color_hex = hsv_to_hex(h, s, v)
        row = tk.Frame(lock_frame, bg=BG)
        row.pack(fill="x", padx=4, pady=1)
        tk.Label(row, text="  ", bg=color_hex, width=2).pack(side="left")
        tk.Label(row, text=f"#{i + 1} ({x},{y}) H{h:3d} S{s:3d} V{v:3d}",
                 fg="#dde", bg=BG, font=FONT_SMALL).pack(side="left", padx=4)

# ─────────────────────────────────────────────
# ROI 操作
# ─────────────────────────────────────────────
def delete_selected():
    if not state["sel"]:
        return
    t, i = state["sel"]["type"], state["sel"]["idx"]
    if 0 <= i < len(state["rois"][t]):
        push_undo()
        state["rois"][t].pop(i)
    state["sel"] = None
    mark_dirty()

def clear_rois():
    if not any(state["rois"].values()):
        return
    push_undo()
    state["rois"] = {"pad": [], "toe": [], "rim": []}
    state["sel"] = None
    mark_dirty()

# ─────────────────────────────────────────────
# 持久化：params.json + LabelMe ROI JSON
# ─────────────────────────────────────────────
def _set_status(msg):
    """在主窗口信息栏显示状态（3秒后恢复）"""
    lbl_file.config(text=msg)
    if state["path"] and state["img"] is not None:
        h, w = state["img"].shape[:2]
        root.after(3000, lambda: lbl_file.config(text=f"[Img] {os.path.basename(state['path'])}  ({w}x{h})"))
    else:
        root.after(3000, lambda: lbl_file.config(text="未加载图片"))

def _apply_params_dict(data):
    """从参数字典应用到 state (HSV + 阈值)"""
    for o in HSV_OBJECTS:
        p = state["hsv_params"][o["name"]]
        for ik, pk in o["keys"].items():
            if pk in data:
                p[ik] = int(data[pk])
        for k in ("s_max", "v_max"):
            if p.get(k) is None:
                p[k] = 255
    # Load HS LUT for each object
    for o in HSV_OBJECTS:
        name = o["name"]
        lut_key = o["keys"]["h_min"].replace("_h_low", "_hs_lut")
        if lut_key in data:
            state["hs_lut"][name] = _decode_hs_lut(data[lut_key])
    if "insufficient_thresh" in data:
        state["thresh"]["insufficient"] = float(data["insufficient_thresh"])
    if "excess_thresh" in data:
        state["thresh"]["excess"] = float(data["excess_thresh"])
    if "bridge_min_area" in data:
        state["thresh"]["bridge_min_area"] = int(data["bridge_min_area"])
    if "cold_solder_rim_ratio" in data:
        state["thresh"]["cold_rim"] = float(data["cold_solder_rim_ratio"])
    if "toe_metal_ratio_thresh" in data:
        state["thresh"]["toe_metal_ratio"] = float(data["toe_metal_ratio_thresh"])
    if "cold_solder_dark_ratio" in data:
        state["thresh"]["cold_dark_ratio"] = float(data["cold_solder_dark_ratio"])
    if "cold_solder_v_dark" in data:
        state["thresh"]["cold_v_dark"] = int(data["cold_solder_v_dark"])
    if "cold_diff_ratio_thresh" in data:
        state["thresh"]["cold_diff_ratio"] = float(data["cold_diff_ratio_thresh"])
    if "crack_min_length_ratio" in data:
        state["thresh"]["crack_min_length_ratio"] = float(data["crack_min_length_ratio"])
    if "crack_cpp_min_pads" in data:
        state["thresh"]["crack_cpp_min_pads"] = int(data["crack_cpp_min_pads"])
    if "pin_type" in data:
        state["pin_type"] = str(data["pin_type"])
    if "use_hs_lut" in data:
        state["use_hs_lut"] = bool(data["use_hs_lut"])
        var_use_hs_lut.set(state["use_hs_lut"])

def _sync_ui_from_state():
    """从 state 同步到 UI 控件"""
    global _suppress_trace
    load_obj_to_sliders()
    _suppress_trace = True  # load_obj_to_sliders 会重置为 False，这里重新设置
    var_ins.set(int(state["thresh"]["insufficient"] * 100))
    var_exc.set(int(state["thresh"]["excess"] * 100))
    var_brg.set(int(state["thresh"]["bridge_min_area"]))
    var_cold.set(int(state["thresh"]["cold_rim"] * 100))
    var_toe_metal.set(int(state["thresh"]["toe_metal_ratio"] * 100))
    var_dark.set(int(state["thresh"]["cold_dark_ratio"] * 100))
    var_vdark.set(int(state["thresh"]["cold_v_dark"]))
    var_diff.set(int(state["thresh"]["cold_diff_ratio"] * 100))
    var_crack_len.set(int(state["thresh"]["crack_min_length_ratio"] * 100))
    var_cpp_min_pads.set(state["thresh"]["crack_cpp_min_pads"])
    _suppress_trace = False
    if hasattr(state, 'get') and state.get("pin_type"):
        var_pin.set(state["pin_type"])
        on_pin_type_changed()

def _load_preset_params():
    """加载预设参数 (solder_smt_all_alg.json 的 defaultParam)"""
    _json_path = os.path.join(_HERE, "..", "src", "resources", "config", "alg", "solder_smt", "solder_smt_all_alg.json")
    if not os.path.isfile(_json_path):
        return
    with open(_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dp = data.get("defaultParam", data)
    _apply_params_dict(dp)
    state["_raw_params"] = dp
    _sync_ui_from_state()
    mark_dirty()

def load_params(path=None):
    if path is None:
        path = filedialog.askopenfilename(title="Load params.json",
                                          filetypes=[("JSON", "*.json")])
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _apply_params_dict(data)
    state["_raw_params"] = data
    _sync_ui_from_state()
    mark_dirty()
    _set_status(f"[OK] params: {os.path.basename(path)}")

def save_params(path=None):
    sync_sliders_to_obj()
    data = dict(state.get("_raw_params", {}))
    for o in HSV_OBJECTS:
        p = state["hsv_params"][o["name"]]
        for ik, pk in o["keys"].items():
            data[pk] = p[ik]
    # Save HS LUT for each object
    for o in HSV_OBJECTS:
        name = o["name"]
        lut_key = o["keys"]["h_min"].replace("_h_low", "_hs_lut")  # e.g. solder_blue_hs_lut
        lut = state["hs_lut"].get(name)
        if lut is not None and lut.any():
            data[lut_key] = _encode_hs_lut(lut)
    data["insufficient_thresh"] = state["thresh"]["insufficient"]
    data["excess_thresh"] = state["thresh"]["excess"]
    data["bridge_min_area"] = state["thresh"]["bridge_min_area"]
    data["cold_solder_rim_ratio"] = state["thresh"]["cold_rim"]
    data["toe_metal_ratio_thresh"] = state["thresh"]["toe_metal_ratio"]
    data["cold_solder_dark_ratio"] = state["thresh"]["cold_dark_ratio"]
    data["cold_solder_v_dark"] = state["thresh"]["cold_v_dark"]
    data["cold_diff_ratio_thresh"] = state["thresh"]["cold_diff_ratio"]
    data["crack_min_length_ratio"] = state["thresh"]["crack_min_length_ratio"]
    data["crack_cpp_min_pads"] = state["thresh"]["crack_cpp_min_pads"]
    data["pin_type"] = state["pin_type"]
    data["use_hs_lut"] = state.get("use_hs_lut", False)
    if path is None:
        path = filedialog.asksaveasfilename(title="Save params.json",
                                            defaultextension=".json",
                                            initialfile="params.json",
                                            filetypes=[("JSON", "*.json")])
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _set_status(f"[OK] 已保存: {os.path.basename(path)}")

def _rois_from_labelme(data):
    rois = {"pad": [], "toe": [], "rim": []}
    for s in data.get("shapes", []):
        lab = s.get("label", "")
        if lab in rois and s.get("shape_type") == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            rois[lab].append({"x": int(round(min(x1, x2))), "y": int(round(min(y1, y2))),
                              "w": max(1, int(round(abs(x2 - x1)))),
                              "h": max(1, int(round(abs(y2 - y1))))})
    return rois

def load_rois(path=None):
    if path is None:
        path = filedialog.askopenfilename(title="Load ROI JSON (LabelMe)",
                                          filetypes=[("JSON", "*.json")])
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    push_undo()
    state["rois"] = _rois_from_labelme(data)
    state["orig_rois"] = copy.deepcopy(state["rois"])
    # ROI来源尺寸: 优先JSON里的imageWidth/Height，其次当前图片尺寸
    rw = data.get("imageWidth", 0)
    rh = data.get("imageHeight", 0)
    if rw > 0 and rh > 0:
        state["rois_src_size"] = (rw, rh)
    elif state["img"] is not None:
        ih, iw = state["img"].shape[:2]
        state["rois_src_size"] = (iw, ih)
    state["sel"] = None
    state["undo_stack"].clear()   # 加载后清空撤销栈
    mark_dirty()
    _set_status(f"[OK] ROI: pad={len(state['rois']['pad'])} toe={len(state['rois']['toe'])} rim={len(state['rois']['rim'])}")

def save_rois(path=None):
    if path is None:
        path = filedialog.asksaveasfilename(title="Save ROI JSON (LabelMe)",
                                            defaultextension=".json",
                                            initialfile="pad.json",
                                            filetypes=[("JSON", "*.json")])
    if not path:
        return
    # 将ROI坐标还原到来源坐标系(rois_src_size)，保持pad.json坐标系一致
    cur_h, cur_w = state["img"].shape[:2] if state["img"] is not None else (0, 0)
    src_w, src_h = state.get("rois_src_size", (cur_w, cur_h))
    sx = src_w / cur_w if cur_w > 0 else 1.0
    sy = src_h / cur_h if cur_h > 0 else 1.0
    shapes = []
    for t in ROI_TYPES:
        for r in state["rois"][t]:
            ox = int(round(r["x"] * sx))
            oy = int(round(r["y"] * sy))
            ow = max(1, int(round(r["w"] * sx)))
            oh = max(1, int(round(r["h"] * sy)))
            shapes.append({
                "label": t,
                "points": [[ox, oy], [ox + ow, oy + oh]],
                "group_id": None, "description": "", "shape_type": "rectangle",
                "flags": {}, "mask": None,
            })
    data = {
        "version": "6.3.1", "flags": {}, "shapes": shapes,
        "imagePath": "",
        "imageData": None, "imageHeight": src_h, "imageWidth": src_w,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _set_status(f"[OK] 已保存ROI: {os.path.basename(path)}")

def _load_pair_internal(d):
    """内部: 加载pair目录（所有图片+模板+params+ROI），不弹对话框"""
    s = state
    s["pair_dir"] = d
    # 扫描所有图片（含模板_OK），排除result图
    all_imgs = []
    test_imgs = []
    ok_imgs = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            if "result" in f.lower():
                continue
            fp = os.path.join(d, f)
            if "_ok" in f.lower():
                ok_imgs.append((f, fp))
            else:
                test_imgs.append((f, fp))
    # 图片列表: 待检图在前，模板图在后
    all_imgs = test_imgs + ok_imgs
    s["pair_images"] = all_imgs
    s["pair_img_idx"] = 0

    # 加载模板图（用于获取ROI来源尺寸）
    if ok_imgs:
        tpl_path = ok_imgs[0][1]
        tpl = cv2.imdecode(np.fromfile(tpl_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        s["template_img"] = tpl
    else:
        s["template_img"] = None

    # 加载第一张待检图
    if all_imgs:
        _load_image_internal(all_imgs[0][1])

    # 先加载预设参数，再加载pair的params覆盖
    _load_preset_params()
    pj = os.path.join(d, "params.json")
    if os.path.isfile(pj):
        load_params(pj)

    # 加载ROI: 只加载 pad.json (不自动加载 *_OK.json 等其他标注)
    rj = None
    if os.path.isfile(os.path.join(d, "pad.json")):
        rj = os.path.join(d, "pad.json")
    if rj:
        with open(rj, "r", encoding="utf-8") as f:
            data = json.load(f)
        s["orig_rois"] = _rois_from_labelme(data)
        # ROI来源尺寸: 优先用JSON里的imageWidth/Height，其次用模板图尺寸
        # pad.json通常来自模板图，坐标系与模板图一致
        rw = data.get("imageWidth", 0)
        rh = data.get("imageHeight", 0)
        if rw > 0 and rh > 0:
            s["rois_src_size"] = (rw, rh)
        elif s["template_img"] is not None:
            th, tw = s["template_img"].shape[:2]
            s["rois_src_size"] = (tw, th)
        elif s["img"] is not None:
            ih, iw = s["img"].shape[:2]
            s["rois_src_size"] = (iw, ih)
        # 按当前图片尺寸缩放ROI
        _apply_rois_for_current_image()

    _update_image_nav_label()

def load_pair_folder():
    """一键加载某目录：图片 + params.json + ROI(*_OK.json 或 pad.json)"""
    d = filedialog.askdirectory(title="Select pair folder (image + params.json + ROI json)")
    if not d:
        return
    try:
        _load_pair_internal(d)
    except Exception as e:
        import traceback
        messagebox.showerror("Load Pair Error", f"{e}\n\n{traceback.format_exc()}")

# ─────────────────────────────────────────────
# 复制 / 开关
# ─────────────────────────────────────────────
def copy_range():
    sync_sliders_to_obj()
    p = state["hsv_params"][state["cur_obj"]]
    txt = f"cv2.inRange(hsv, ({p['h_min']},{p['s_min']},{p['v_min']}), ({p['h_max']},{p['s_max']},{p['v_max']}))"
    root.clipboard_clear(); root.clipboard_append(txt)
    btn_copy.config(text="[OK] 已复制！")
    root.after(1500, lambda: btn_copy.config(text="[Copy] 复制 inRange"))

def toggle_overlay():
    state["show_overlay"] = var_overlay.get()
    mark_dirty()

def toggle_detect():
    state["show_detect"] = var_detect.get()
    mark_dirty()

def on_close():
    root.destroy()

# ─────────────────────────────────────────────
# 主图 Tk 显示窗口 (替代 OpenCV 主窗口, 原生可拖动/缩放)
# ─────────────────────────────────────────────
canvas = None
canvas_img_id = None
canvas_draw_id = None
aux_canvases = []      # 4个小画布
aux_canvas_views = []  # 每个小画布当前显示的视图索引

def _canvas_img_xy(event):
    """Tk 鼠标事件 -> 图像坐标(考虑 Canvas 滚动偏移 + 顶栏高度 + 缩放)"""
    if canvas is None:
        return 0, 0
    zoom = state.get("zoom", 1.0)
    x = int(canvas.canvasx(event.x) / zoom)
    y = int(canvas.canvasy(event.y) / zoom) - TOP_BAR_H
    return x, y

def _on_mouse_move(event):
    handle_mouse("move", *_canvas_img_xy(event))

def _on_mouse_down(event):
    handle_mouse("down", *_canvas_img_xy(event))

def _on_mouse_up(event):
    handle_mouse("up", *_canvas_img_xy(event))

def _on_mouse_rdown(event):
    handle_mouse("rdown", *_canvas_img_xy(event))

def _on_key(event):
    k = event.keysym
    if k == "q":
        on_close()
    elif k == "1": set_mode("draw_pad")
    elif k == "2": set_mode("draw_toe")
    elif k == "3": set_mode("draw_rim")
    elif k == "e": set_mode("edit")
    elif k == "s": set_mode("sample")
    elif k == "z": undo()
    elif k == "Tab":
        cycle_object(); return "break"
    elif k in ("Delete", "BackSpace"):
        delete_selected()
    elif k in ("Left", "Prior"):
        prev_pair_image()
    elif k in ("Right", "Next"):
        next_pair_image()

def setup_canvas(w, h):
    """图片加载后:设置 scrollregion（含上下信息栏高度 + 缩放）"""
    zoom = state.get("zoom", 1.0)
    canvas.config(scrollregion=(0, 0, int(w * zoom), int((h + TOP_BAR_H + BOT_BAR_H) * zoom)))
    canvas.itemconfig(canvas_draw_id, state="hidden")

# ─────────────────────────────────────────────
# 辅助画布渲染
# ─────────────────────────────────────────────
def _render_to_canvas(canvas_widget, img, label=""):
    """将图像渲染到Tk Canvas，自适应缩放。img为None时显示提示文字"""
    if canvas_widget is None:
        return
    canvas_widget.update_idletasks()
    cw = max(canvas_widget.winfo_width(), 50)
    ch = max(canvas_widget.winfo_height(), 50)
    canvas_widget.delete("all")
    if img is None:
        canvas_widget.create_text(cw // 2, ch // 2 - 8,
                                   text=label, fill="#7af0ff", font=_mono_font(10))
        canvas_widget.create_text(cw // 2, ch // 2 + 10,
                                   text="(无)", fill="#666", font=_mono_font(9))
        return
    ih, iw = img.shape[:2]
    scale = min(cw / iw, ch / ih, 1.0)  # 不放大，只缩小
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (new_w, new_h))
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    photo = PILImageTk.PhotoImage(pil)
    canvas_widget.create_image(cw // 2, ch // 2, anchor="center", image=photo)
    canvas_widget.photo = photo  # 防 GC
    if label:
        canvas_widget.create_text(5, 5, anchor="nw", text=label,
                                   fill="#7af0ff", font=_mono_font(9))

def render_aux_canvases():
    """渲染6个辅助画布"""
    global aux_canvas_views
    cache = state["cache"]
    main_idx = state["main_view_idx"]
    views = [
        cache.get("base"),
        cache.get("mask_vis"),
        cache.get("ch", {}).get("H"),
        cache.get("ch", {}).get("S"),
        cache.get("ch", {}).get("V"),
        cache.get("template_vis"),
        cache.get("diffmask_vis"),
        cache.get("crack_vis"),
    ]
    aux_canvas_views = []
    aux_idx = 0
    for i in range(8):
        if i == main_idx:
            continue
        v = views[i] if i < len(views) else None
        if aux_idx < len(aux_canvases):
            _render_to_canvas(aux_canvases[aux_idx], v, label=VIEW_LABELS[i])
            aux_canvas_views.append(i)
            aux_idx += 1

def swap_main_view(idx):
    """切换主视图: 点击小图与主图互换"""
    if idx == state["main_view_idx"]:
        # 点击的是当前主视图对应的小图，切回image
        state["main_view_idx"] = 0
    else:
        state["main_view_idx"] = idx
    # 重新渲染
    _render_main_view()
    render_aux_canvases()

def _render_main_view():
    """渲染主画布"""
    main_idx = state["main_view_idx"]
    if main_idx == 0:
        # 显示主图（带叠加），使用已有的 render_base_to_canvas
        render_base_to_canvas()
    else:
        # 显示通道/模板/diff图，全尺寸显示在主画布
        cache = state["cache"]
        views = [cache.get("base"), cache.get("mask_vis"),
                 cache.get("ch", {}).get("H"),
                 cache.get("ch", {}).get("S"),
                 cache.get("ch", {}).get("V"),
                 cache.get("template_vis"),
                 cache.get("diffmask_vis"),
                 cache.get("crack_vis")]
        v = views[main_idx] if main_idx < len(views) else None
        if v is not None:
            zoom = state.get("zoom", 1.0)
            if len(v.shape) == 2:
                v_bgr = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            else:
                v_bgr = v
            if zoom != 1.0:
                h, w = v_bgr.shape[:2]
                v_bgr = cv2.resize(v_bgr, (int(w * zoom), int(h * zoom)), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(v_bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            photo = PILImageTk.PhotoImage(pil)
            canvas.itemconfig(canvas_img_id, image=photo)
            canvas.photo = photo
            h, w = v_bgr.shape[:2]
            canvas.config(scrollregion=(0, 0, w, h))
    # 更新鼠标事件状态
    _update_mouse_state()

def _update_mouse_state():
    """根据当前主视图更新鼠标事件绑定"""
    if state["main_view_idx"] == 0:
        # 主图模式：启用鼠标事件
        canvas.config(cursor="crosshair")
    else:
        # 通道模式：禁用鼠标事件
        canvas.config(cursor="arrow")

def _on_aux_click(aux_idx):
    """点击辅助画布，与主视图互换"""
    if aux_idx < len(aux_canvas_views):
        view_idx = aux_canvas_views[aux_idx]
        swap_main_view(view_idx)

# ─────────────────────────────────────────────
# Tk UI
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("PCBA Tuner")
root.configure(bg="#1e1e2e")
root.protocol("WM_DELETE_WINDOW", on_close)
root.resizable(True, True)
# 高DPI适配 (Linux/Windows)
try:
    root.tk.call('tk', 'scaling', 1.3)
except Exception:
    pass

# 设置全局默认字体 (影响文件对话框等所有Tk组件)
import tkinter.font as tkfont
def _detect_cjk_font():
    """检测系统可用的中文字体"""
    available = set(tkfont.families())

    # 优先用 fc-match 找系统字体, 但必须验证tkinter能识别
    try:
        import subprocess
        out = subprocess.check_output(
            ["fc-match", "-f", "%{family[0]}", "sans-serif"],
            stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
        if out:
            name = out.split(",")[0].strip()
            if name and name in available:
                return name
    except Exception:
        pass

    sys_name = platform.system()
    if sys_name == "Windows":
        candidates = ["微软雅黑", "SimHei", "Microsoft YaHei"]
    elif sys_name == "Darwin":
        candidates = ["PingFang SC", "Heiti SC", "STHeiti"]
    else:
        candidates = ["wenquanyi bitmap song", "Noto Sans CJK SC", "Noto Sans CJK",
                      "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                      "Droid Sans Fallback", "DejaVu Sans",
                      "song ti", "fangsong ti", "Sans Serif"]
    for c in candidates:
        if c in available:
            return c
    return "Sans Serif"

_global_font_name = _detect_cjk_font()
for fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
           "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
    try:
        tkfont.nametofont(fn).configure(family=_global_font_name)
    except Exception:
        pass

def _font(size, bold=False):
    """跨平台字体"""
    return (_global_font_name, size, "bold") if bold else (_global_font_name, size)

def _mono_font(size):
    """跨平台等宽字体 (Linux回退到CJK字体避免中文乱码)"""
    sys_name = platform.system()
    if sys_name == "Windows":
        return ("Consolas", size)
    elif sys_name == "Darwin":
        return ("Menlo", size)
    else:
        return (_global_font_name, size)

FONT_LABEL = _font(10)
FONT_SMALL = _mono_font(9)
BG = "#1e1e2e"; BG2 = "#2a2a3e"; FG = "#cdd6f4"
ACCENT = "#89b4fa"; SLIDER_TRO = "#313244"
root.option_add("*Background", BG)
root.option_add("*Foreground", FG)

# ── 顶部工具栏 ──
top = tk.Frame(root, bg=BG, pady=6)
top.pack(fill="x", padx=10)
tk.Button(top, text="[Open] 打开图片", command=load_image, font=FONT_LABEL,
          bg=ACCENT, fg="#1e1e2e", relief="flat", padx=8, pady=3).pack(side="left", padx=(0, 6))
tk.Button(top, text="[Folder] 加载pair目录", command=load_pair_folder, font=FONT_LABEL,
          bg="#74c7ec", fg="#1e1e2e", relief="flat", padx=8, pady=3).pack(side="left", padx=(0, 6))
tk.Button(top, text="[Cfg]加载params", command=lambda: load_params(), font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=3).pack(side="left", padx=(0, 4))
tk.Button(top, text="[ROI]加载ROI", command=lambda: load_rois(), font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=3).pack(side="left", padx=(0, 4))
tk.Button(top, text="[Save]保存params", command=lambda: save_params(), font=FONT_SMALL,
          bg="#a6e3a1", fg="#1e1e2e", relief="flat", padx=6, pady=3).pack(side="left", padx=(0, 4))
tk.Button(top, text="[Save]保存ROI", command=lambda: save_rois(), font=FONT_SMALL,
          bg="#a6e3a1", fg="#1e1e2e", relief="flat", padx=6, pady=3).pack(side="left", padx=(0, 4))
lbl_file = tk.Label(top, text="未加载图片", fg="#888", bg=BG, font=FONT_SMALL)
lbl_file.pack(side="left", padx=8)

# 图片导航（多图支持）
img_nav_frame = tk.Frame(top, bg=BG)
img_nav_frame.pack(side="left", padx=(4, 0))
tk.Button(img_nav_frame, text="<", command=prev_pair_image, font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=2, width=3).pack(side="left", padx=1)
img_nav_var = tk.StringVar(value="")
tk.Label(img_nav_frame, textvariable=img_nav_var, fg="#fab387", bg=BG, font=FONT_SMALL,
         width=30, anchor="w").pack(side="left", padx=2)
tk.Button(img_nav_frame, text=">", command=next_pair_image, font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=2, width=3).pack(side="left", padx=1)

# 缩放控制
zoom_frame = tk.Frame(top, bg=BG)
zoom_frame.pack(side="left", padx=(4, 0))
tk.Button(zoom_frame, text="[-]", command=zoom_out, font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=2, width=3).pack(side="left", padx=1)
tk.Button(zoom_frame, text="1:1", command=zoom_reset, font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=2, width=4).pack(side="left", padx=1)
tk.Button(zoom_frame, text="+", command=zoom_in, font=FONT_SMALL,
          bg=BG2, fg=FG, relief="flat", padx=6, pady=2, width=3).pack(side="left", padx=1)
zoom_label = tk.Label(zoom_frame, text="100%", fg="#fab387", bg=BG, font=FONT_SMALL, width=6)
zoom_label.pack(side="left", padx=2)

tk.Frame(root, bg="#444", height=1).pack(fill="x", padx=10)

body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=10, pady=6)

# ── 左侧控制面板 ──
left = tk.Frame(body, bg=BG)
left.pack(side="left", fill="y", padx=(0, 10))

# 模式
tk.Label(left, text="操作模式", font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w")
mode_row = tk.Frame(left, bg=BG); mode_row.pack(fill="x", pady=2)
mode_buttons = {}
for m, txt in [("draw_pad", "①画pad"), ("draw_toe", "②画toe"),
               ("draw_rim", "③画rim"), ("edit", "[Edit]编辑"), ("sample", "[Aim]采样")]:
    b = tk.Button(mode_row, text=txt, command=lambda mm=m: set_mode(mm),
                  font=FONT_SMALL, bg=BG2, fg=FG, relief="flat", padx=4, pady=2, width=7)
    b.pack(side="left", padx=1)
    mode_buttons[m] = b
roi_btn_row = tk.Frame(left, bg=BG); roi_btn_row.pack(fill="x", pady=(2, 0))
tk.Button(roi_btn_row, text="[Undo] (Ctrl+Z)", command=undo,
          font=FONT_SMALL, bg="#45475a", fg=FG, relief="flat", padx=4, pady=2).pack(side="left", fill="x", expand=True, padx=(0, 2))
tk.Button(roi_btn_row, text="[Del] sel", command=delete_selected,
          font=FONT_SMALL, bg="#6a475a", fg=FG, relief="flat", padx=4, pady=2).pack(side="left", fill="x", expand=True, padx=2)
tk.Button(roi_btn_row, text="[Clear] all", command=clear_rois,
          font=FONT_SMALL, bg="#6a475a", fg=FG, relief="flat", padx=4, pady=2).pack(side="left", fill="x", expand=True, padx=(2, 0))

tk.Frame(left, bg="#444", height=1).pack(fill="x", pady=6)

# HSV 对象选择
tk.Label(left, text="HSV 抽色对象", font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w")
var_obj = tk.StringVar(value=state["cur_obj"])
var_obj.trace_add("write", on_obj_changed)
obj_row = tk.Frame(left, bg=BG); obj_row.pack(fill="x", pady=2)
for nm in OBJ_NAMES:
    tk.Radiobutton(obj_row, text=nm, variable=var_obj, value=nm,
                   font=FONT_SMALL, bg=BG, fg=FG, selectcolor=BG2,
                   activebackground=BG, activeforeground=FG,
                   highlightthickness=0, command=on_obj_changed).pack(side="left")
lbl_obj = tk.Label(left, text=f"当前对象: {state['cur_obj']}", fg="#fab387", bg=BG, font=FONT_SMALL)
lbl_obj.pack(anchor="w")

# HSV 滑条
slider_defs = [
    ("h_min", "H_min", 0,   0, 179, "#f38ba8"),
    ("h_max", "H_max", 179, 0, 179, "#f38ba8"),
    ("s_min", "S_min", 40,  0, 255, "#a6e3a1"),
    ("s_max", "S_max", 255, 0, 255, "#a6e3a1"),
    ("v_min", "V_min", 40,  0, 255, "#fab387"),
    ("v_max", "V_max", 255, 0, 255, "#fab387"),
]
for key, label, default, lo, hi, color in slider_defs:
    row = tk.Frame(left, bg=BG); row.pack(fill="x", pady=1)
    tk.Label(row, text=label, width=6, anchor="w", font=_font(9, bold=True),
             fg=color, bg=BG).pack(side="left")
    var = tk.IntVar(value=default)
    sliders[key] = var
    var.trace_add("write", on_slider_changed)
    tk.Label(row, textvariable=var, width=4, font=_mono_font(9), fg=FG, bg=BG).pack(side="right")
    tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
             length=190, showvalue=False, bg=BG2, fg=color, troughcolor=SLIDER_TRO,
             highlightthickness=0, activebackground=color, bd=0).pack(side="left", fill="x", expand=True)

# margin + 自动填充
tk.Frame(left, bg="#444", height=1).pack(fill="x", pady=6)
mrow = tk.Frame(left, bg=BG); mrow.pack(fill="x", pady=2)
tk.Label(mrow, text="自动填充 margin:", font=FONT_SMALL, fg="#888", bg=BG).pack(side="left")
for lab, df in [("H±", "12"), ("S±", "45"), ("V±", "50")]:
    tk.Label(mrow, text=lab, font=FONT_SMALL, fg="#aaa", bg=BG).pack(side="left", padx=(6, 0))
    e = tk.Entry(mrow, width=4, font=FONT_SMALL, bg=BG2, fg=FG, insertbackground=FG,
                 relief="flat", bd=1, highlightthickness=1, highlightcolor=ACCENT)
    e.insert(0, df); e.pack(side="left")
    if lab == "H±":
        entry_margin_h = e
    elif lab == "S±":
        entry_margin_s = e
    else:
        entry_margin_v = e
tk.Button(left, text="[Aim] 从采样点自动填充当前对象", command=auto_fill_from_locks,
          font=FONT_SMALL, bg="#45475a", fg=FG, relief="flat", padx=6, pady=3).pack(fill="x", pady=(4, 0))
btn_copy = tk.Button(left, text="[Copy] 复制 inRange", command=copy_range,
                     font=FONT_SMALL, bg=ACCENT, fg="#1e1e2e", relief="flat", padx=6, pady=3)
btn_copy.pack(fill="x", pady=2)

# 阈值（按规则分组）
tk.Frame(left, bg="#444", height=1).pack(fill="x", pady=6)
tk.Label(left, text="判定阈值", font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w")

# pin_type 选择器
pin_row = tk.Frame(left, bg=BG); pin_row.pack(fill="x", pady=(2, 4))
tk.Label(pin_row, text="引脚类型:", font=FONT_SMALL, fg=FG, bg=BG).pack(side="left")
var_pin = tk.StringVar(value=state["pin_type"])
for val, lab in [("gull-wing", "gull-wing"), ("terminal", "terminal")]:
    tk.Radiobutton(pin_row, text=lab, variable=var_pin, value=val,
                   font=FONT_SMALL, fg=FG, bg=BG, selectcolor=BG2,
                   activebackground=BG, activeforeground=FG,
                   highlightthickness=0,
                   command=lambda: on_pin_type_changed()).pack(side="left", padx=4)

var_ins = tk.IntVar(value=int(state["thresh"]["insufficient"] * 100))
var_exc = tk.IntVar(value=int(state["thresh"]["excess"] * 100))
var_brg = tk.IntVar(value=int(state["thresh"]["bridge_min_area"]))
var_cold = tk.IntVar(value=int(state["thresh"]["cold_rim"] * 100))
var_toe_metal = tk.IntVar(value=int(state["thresh"]["toe_metal_ratio"] * 100))
var_dark = tk.IntVar(value=int(state["thresh"]["cold_dark_ratio"] * 100))
var_vdark = tk.IntVar(value=int(state["thresh"]["cold_v_dark"]))
var_diff = tk.IntVar(value=int(state["thresh"]["cold_diff_ratio"] * 100))
var_crack_len = tk.IntVar(value=int(state["thresh"]["crack_min_length_ratio"] * 100))
var_cpp_min_pads = tk.IntVar(value=state["thresh"]["crack_cpp_min_pads"])
for v in (var_ins, var_exc, var_brg, var_cold, var_toe_metal, var_dark, var_vdark, var_diff, var_crack_len):
    v.trace_add("write", on_thresh_changed)

def _pct_slider(parent, var, label, color):
    r = tk.Frame(parent, bg=BG); r.pack(fill="x", pady=1)
    tk.Label(r, text=label, font=FONT_SMALL, fg=color, bg=BG).pack(side="left")
    tk.Scale(r, from_=0, to=100, orient="horizontal", variable=var, length=110,
             showvalue=False, bg=BG2, fg=color, troughcolor=SLIDER_TRO,
             highlightthickness=0, bd=0).pack(side="left", fill="x", expand=True)
    tk.Label(r, textvariable=var, width=3, font=_mono_font(9), fg=FG, bg=BG).pack(side="left")
    tk.Label(r, text="%", font=FONT_SMALL, fg="#888", bg=BG).pack(side="left")

def _int_slider(parent, var, label, color, hi=500, unit=""):
    r = tk.Frame(parent, bg=BG); r.pack(fill="x", pady=1)
    tk.Label(r, text=label, font=FONT_SMALL, fg=color, bg=BG).pack(side="left")
    tk.Scale(r, from_=0, to=hi, orient="horizontal", variable=var, length=110,
             showvalue=False, bg=BG2, fg=color, troughcolor=SLIDER_TRO,
             highlightthickness=0, bd=0).pack(side="left", fill="x", expand=True)
    tk.Label(r, textvariable=var, width=4, font=_mono_font(9), fg=FG, bg=BG).pack(side="left")
    if unit:
        tk.Label(r, text=unit, font=FONT_SMALL, fg="#888", bg=BG).pack(side="left")

# 少锡/多锡/连锡
tk.Label(left, text="少锡/多锡/连锡", font=FONT_SMALL, fg="#888", bg=BG).pack(anchor="w", pady=(2, 0))
_pct_slider(left, var_ins, "少锡 pad占比<", "#f38ba8")
_pct_slider(left, var_exc, "多锡 外扩环占比>", "#f38ba8")
_int_slider(left, var_brg, "连锡 最小面积≥", "#fab387", hi=500, unit="px")

# 虚焊阈值组容器（根据 pin_type 切换显示）
cold_frame = tk.Frame(left, bg=BG)
cold_frame.pack(fill="x", pady=(4, 0))

# gull-wing 组：Rule1/Rule2/Rule3
gw_frame = tk.Frame(cold_frame, bg=BG)
tk.Label(gw_frame, text="虚焊 Rule1 (rim-toe焊锡覆盖率)", font=FONT_SMALL, fg="#888", bg=BG).pack(anchor="w")
_pct_slider(gw_frame, var_cold, "rim-toe占比<", "#f9e2af")
tk.Label(gw_frame, text="虚焊 Rule2 (crack+toe金属占比)", font=FONT_SMALL, fg="#888", bg=BG).pack(anchor="w", pady=(4, 0))
_pct_slider(gw_frame, var_toe_metal, "toe金属占比>", "#a6e3a1")
tk.Label(gw_frame, text="虚焊 Rule3 (crack+diff覆盖率)", font=FONT_SMALL, fg="#888", bg=BG).pack(anchor="w", pady=(4, 0))
_pct_slider(gw_frame, var_diff, "diff占比>", "#cba6f7")
_pct_slider(gw_frame, var_crack_len, "crack长度>", "#f38ba8")

# C++加速阈值输入框
cpp_frame = tk.Frame(gw_frame, bg=BG)
cpp_frame.pack(fill="x", pady=(4, 0))
tk.Label(cpp_frame, text="C++加速(≥pad数):", font=FONT_SMALL, fg="#888", bg=BG).pack(side="left")
tk.Entry(cpp_frame, textvariable=var_cpp_min_pads, font=FONT_SMALL, width=4, bg=BG2, fg=FG,
         insertbackground=FG, bd=0).pack(side="left", padx=4)
tk.Label(cpp_frame, text="(0=纯Python)", font=FONT_SMALL, fg="#666", bg=BG).pack(side="left")
var_cpp_min_pads.trace_add("write", on_thresh_changed)

# terminal 组：暗区比例 + 暗区V值
term_frame = tk.Frame(cold_frame, bg=BG)
tk.Label(term_frame, text="terminal暗区检测", font=FONT_SMALL, fg="#888", bg=BG).pack(anchor="w")
_pct_slider(term_frame, var_dark, "暗区比例>", "#a6e3a1")
_int_slider(term_frame, var_vdark, "暗区V值<", "#a6e3a1", hi=255, unit="")

def on_pin_type_changed():
    state["pin_type"] = var_pin.get()
    if state["pin_type"] == "gull-wing":
        term_frame.pack_forget()
        gw_frame.pack(fill="x")
    else:
        gw_frame.pack_forget()
        term_frame.pack(fill="x")
    mark_dirty()

# 初始显示
on_pin_type_changed()


# 开关
tk.Frame(left, bg="#444", height=1).pack(fill="x", pady=6)
var_overlay = tk.BooleanVar(value=True)
var_detect = tk.BooleanVar(value=True)
for var, txt, cmd in [(var_overlay, "主图Mask叠加", toggle_overlay),
                      (var_detect, "rim-toe检测区", toggle_detect)]:
    tk.Checkbutton(left, text=txt, variable=var, command=cmd, font=FONT_SMALL,
                   bg=BG, fg=FG, selectcolor=BG2, activebackground=BG,
                   activeforeground=FG, highlightthickness=0).pack(anchor="w")

# ── 中间主画布区域 ──
center_frame = tk.Frame(body, bg=BG)
center_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
sy = tk.Scrollbar(center_frame, orient="vertical")
sx = tk.Scrollbar(center_frame, orient="horizontal")
canvas = tk.Canvas(center_frame, bg="#000", highlightthickness=0, takefocus=True,
                   yscrollcommand=sy.set, xscrollcommand=sx.set)
sy.config(command=canvas.yview); sx.config(command=canvas.xview)
sy.pack(side="right", fill="y"); sx.pack(side="bottom", fill="x")
canvas.pack(side="left", fill="both", expand=True)
canvas_img_id = canvas.create_image(0, 0, anchor="nw")
canvas_draw_id = canvas.create_rectangle(0, 0, 0, 0, outline="#b4b4b4", state="hidden")
canvas.bind("<Motion>", _on_mouse_move)
canvas.bind("<Button-1>", _on_mouse_down)
canvas.bind("<ButtonRelease-1>", _on_mouse_up)
canvas.bind("<B1-Motion>", _on_mouse_move)
canvas.bind("<Button-3>", _on_mouse_rdown)
canvas.bind("<Enter>", lambda e: canvas.bind_all(
    "<MouseWheel>", _on_canvas_wheel))
canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
# Linux滚轮事件
canvas.bind("<Button-4>", _on_canvas_wheel)
canvas.bind("<Button-5>", _on_canvas_wheel)
canvas.bind("<Key>", _on_key)

# ── 右侧面板：NG统计 + 采样点 + 辅助画布 ──
right_panel = tk.Frame(body, bg=BG)
right_panel.pack(side="left", fill="y")

# NG 统计信息栏（heavy 重算后更新）
tk.Label(right_panel, text="NG 统计", font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w")
ng_info_var = tk.StringVar(value="未加载图片")
ng_info_label = tk.Label(right_panel, textvariable=ng_info_var, font=_mono_font(9),
                         fg=FG, bg=BG2, anchor="w", justify="left", relief="flat",
                         padx=8, pady=6, height=6)
ng_info_label.pack(fill="x", pady=(0, 6))

tk.Label(right_panel, text="HSV 读数", font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w", pady=(6, 2))
hsv_readout_var = tk.StringVar(value="X:--- Y:--- H:--- S:--- V:---")
tk.Label(right_panel, textvariable=hsv_readout_var, font=_mono_font(10), fg="#7af0ff", bg=BG2,
         anchor="w", padx=8, pady=4).pack(fill="x")

tk.Label(right_panel, text="采样点 (采样模式下左键点击 / 右键清空)",
         font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w", pady=(0, 4))
lock_frame = tk.Frame(right_panel, bg=BG, relief="flat", bd=1,
                      highlightthickness=1, highlightcolor="#444")
lock_frame.pack(fill="x")
update_lock_panel()

# 辅助画布 4x2 网格（Image/Mask/H/S/V/Template/DiffMask/Crack，点击切换主视图）
tk.Label(right_panel, text="辅助视图 (点击切换主视图)",
         font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w", pady=(6, 4))
aux_grid = tk.Frame(right_panel, bg=BG)
aux_grid.pack(fill="x")
aux_canvases.clear()
for i in range(7):
    r, c = divmod(i, 2)
    ac = tk.Canvas(aux_grid, bg="#111", highlightthickness=1,
                   highlightbackground="#444", width=160, height=100)
    ac.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
    ac.bind("<Button-1>", lambda e, idx=i: _on_aux_click(idx))
    aux_canvases.append(ac)
aux_grid.grid_columnconfigure(0, weight=1)
aux_grid.grid_columnconfigure(1, weight=1)

# HS色板
tk.Label(right_panel, text="HS 色板 (双击添加点/右键删除/拖动边界)",
         font=_font(9, bold=True), fg=ACCENT, bg=BG).pack(anchor="w", pady=(6, 4))

# 色板开关
var_use_hs_lut = tk.BooleanVar(value=state.get("use_hs_lut", False))
def _on_toggle_hs_lut():
    state["use_hs_lut"] = var_use_hs_lut.get()
    mark_dirty()
tk.Checkbutton(right_panel, text="启用HS色板矩阵 (替代H,S一维范围)", variable=var_use_hs_lut,
               command=_on_toggle_hs_lut, font=FONT_SMALL,
               bg=BG, fg=FG, selectcolor=BG2, activebackground=BG,
               activeforeground=FG, highlightthickness=0).pack(anchor="w", pady=(0, 2))

palette_canvas = tk.Canvas(right_panel, width=PALETTE_SIZE, height=PALETTE_SIZE,
                           bg="#111", highlightthickness=1, highlightbackground="#444")
palette_canvas.pack(pady=2)
palette_canvas.bind("<Motion>", _on_palette_motion)
palette_canvas.bind("<Button-1>", _on_palette_click)
palette_canvas.bind("<ButtonRelease-1>", _on_palette_release)
palette_canvas.bind("<Double-Button-1>", _on_palette_dblclick)
palette_canvas.bind("<Button-3>", _on_palette_rclick)
palette_canvas.bind("<Leave>", _on_palette_leave)

# 底部提示
tk.Frame(root, bg="#444", height=1).pack(fill="x", padx=10)
tk.Label(root, text="快捷键: q退出 1/2/3画pad/toe/rim e编辑 s采样 z/Ctrl+Z撤销 Tab切对象 Del删除 点击小图切换主视图 Ctrl+滚轮=缩放 | 框色: pad绿=少锡OK 红=NG  rim青=虚焊OK 红=NG  外扩框灰=多锡OK 红=NG  连锡红框=NG",
        fg="#555", bg=BG, font=_font(8)).pack(pady=4)

# 初始化
load_obj_to_sliders()
_sync_mode_buttons()
set_mode("edit")

# 命令行参数：图片 或 pair目录
def _load_dir_arg(d):
    _load_pair_internal(d)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            root.after(100, lambda: _load_dir_arg(arg))
        else:
            root.after(100, lambda: load_image(arg))

    root.after(50, cv_update)
    root.mainloop()
