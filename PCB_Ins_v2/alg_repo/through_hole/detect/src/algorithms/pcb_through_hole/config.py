"""插件焊点检测集中配置：HSV 分割、校准、五类缺陷阈值"""
from __future__ import annotations

# 缺陷类别 (孔洞 / 少锡 / 多锡 / 连锡 / 不出脚)
DEFECT_NAMES = {
    0: "OK",
    12: "孔洞",
    13: "少锡",
    14: "多锡(包锡)",
    15: "连锡",
    16: "不出脚",
}

DEFECT_PRIORITY = {
    13: 70,
    12: 68,
    15: 65,
    14: 60,
    16: 55,
}

DEFECT_COLORS = {
    12: (0, 0, 255),
    13: (0, 165, 255),
    14: (0, 255, 255),
    15: (255, 128, 0),
    16: (255, 255, 0),
}

DEFECT_ID_VOID = 12

# 少锡预设挡位 LOW / MED / HIGH（在 config.yaml 中设置 insuf_preset）
INSUF_PRESET = "MED"

INSUF_PRESETS = {
    "LOW": {
        "insuf_area_ratio_max": 0.78,
        "insuf_red_ratio_min": 0.08,
        "insuf_ring_red_min": 0.14,
        "insuf_thin_area_max": 0.82,
        "insuf_exposed_core_min": 0.12,
    },
    "MED": {
        "insuf_area_ratio_max": 0.85,
        "insuf_red_ratio_min": 0.06,
        "insuf_ring_red_min": 0.10,
        "insuf_thin_area_max": 0.90,
        "insuf_exposed_core_min": 0.08,
    },
    "HIGH": {
        "insuf_area_ratio_max": 0.92,
        "insuf_red_ratio_min": 0.04,
        "insuf_ring_red_min": 0.08,
        "insuf_thin_area_max": 0.98,
        "insuf_exposed_core_min": 0.08,
    },
}


# 对少锡阈值的显式手动覆盖（优先级高于挡位表，见 insuf_th()）。
# 少锡的多数关键阈值只存在于 INSUF_PRESETS 里、不在 TH 里，写 TH 不会生效。
INSUF_OVERRIDES: dict = {}


def insuf_th(key: str) -> float:
    """读取当前预设挡位下的少锡阈值；用户显式覆盖优先于挡位表。"""
    if key in INSUF_OVERRIDES:
        return INSUF_OVERRIDES[key]
    preset = INSUF_PRESETS.get(INSUF_PRESET, INSUF_PRESETS["MED"])
    if key in preset:
        return preset[key]
    return TH[key]


# 校准
ENABLE_CALIBRATION = True

# 标准图长边降采样上限；None 表示不限制
WORK_MAX_DIM = 256

ALIGN = {
    "template_margin": 12,
    "search_radius": 18,
    "max_shift_px": 18,
    "ncc_min": 0.22,
    "phase_min_response": 0.25,
    "phase_downsample": 0.5,
    # 配准比较通道: gray|v|l
    "channel": "gray",
    # pad_ring=锡面环(压引脚核) | full=整 ROI
    "feature": "pad_ring",
    "pin_suppress_frac": 0.30,
}

PROFILE = {
    "hsv_margin_h": 10,
    "hsv_margin_sv": 35,
    "ring_inner_frac": 0.26,
    "ring_outer_frac": 0.88,
    "prior_dilate": 17,
}

# HSV 分割
SOLDER_HSV_LOW = (82, 38, 42)
SOLDER_HSV_HIGH = (132, 255, 255)

SOLDER_USE_STANDARD_PRIOR = True
SOLDER_PUNCH_CENTER = False
SOLDER_WIDE_HSV_LOW = (75, 22, 38)
SOLDER_WIDE_HSV_HIGH = (135, 255, 255)

MORPH_KERNEL = 3
SOLDER_MORPH_CLOSE = 5
SOLDER_MORPH_OPEN = 0
SOLDER_MEDIAN = 0
SOLDER_KEEP_LARGEST = True
SOLDER_MIN_AREA_FRAC = 0.02
SOLDER_EXCLUDE_RED = True
SOLDER_RED_BAND = 13

RED1_HSV_LOW = (0, 50, 60)
RED1_HSV_HIGH = (12, 255, 255)
RED2_HSV_LOW = (168, 50, 60)
RED2_HSV_HIGH = (179, 255, 255)

VOID_V_MAX = 85
VOID_MORPH_KERNEL = 3
SOLDER_VOID_STRIP_V_MAX = 85
SOLDER_VOID_STRIP_S_MAX = 130

HIGHLIGHT_V_MIN = 235
HIGHLIGHT_S_MAX = 60

EDGE_RING_INNER = 0.38
EDGE_RING_OUTER = 0.48

# 缺陷判定阈值 (相对标准)
TH = {
    "border_reject_margin": 6,

    # --- 标准/测试锡面尺寸公差圈 (掩膜差边缘细环过滤) ---
    "edge_align_tol_px": 4,
    "edge_align_erode_frac": 0.035,
    "edge_artifact_ring_min_frac": 0.75,
    "edge_artifact_missing_min_frac": 0.85,
    "edge_artifact_near_test_min_frac": 0.55,
    "edge_artifact_in_test_max_frac": 0.12,
    "edge_artifact_max_dark_frac": 0.88,

    # --- 12 孔洞 (相对锡面统计) ---
    "void_v_rel_margin": 3.0,
    "void_v_rel_median_k": 0.70,
    "void_v_abs_floor": 60,
    "void_s_rel_margin": 50.0,
    "void_min_area": 35,
    "void_min_area_ratio": 0.005,
    # --- 孔洞简化判定（面积 + 暗度）；旧多路径键仍保留兼容 YAML ---
    "void_decide_min_area": 180,
    "void_decide_area_frac": 0.0035,
    "void_decide_min_dark": 0.90,
    "void_decide_min_miss": 0.90,
    "void_enc_min": 0.28,
    "void_enc_max": 0.52,
    "void_path_a_area_frac": 0.0048,
    "void_dark_path_area_frac": 0.016,
    "void_dark_compact_area_frac": 0.0103,
    "void_solid_dark_area_frac": 0.008,
    "void_path_b_min_miss": 0.35,
    "void_path_b_min_enc": 0.12,
    "void_min_circularity": 0.28,
    "void_max_aspect_ratio": 4.0,
    "void_max_count": 12,
    "void_use_mask_diff": True,
    "void_dark_s_max": 130,
    "void_desat_s_rel_margin": 40.0,
    "void_desat_s_floor": 40,
    "void_min_desat_frac": 0.75,
    "void_min_enclosure": 0.35,
    "void_min_miss_frac": 0.98,
    "void_path_a_min_area": 300,
    "void_path_a_min_dark": 0.32,
    "void_path_a_max_enc": 0.40,
    "void_border_margin_px": 3,
    "void_dark_min_frac": 0.84,
    "void_core_min_frac": 0.68,
    "void_dark_path_min_area": 1000,
    "void_dark_path_min_enclosure": 0.25,
    "void_dark_path_miss_frac": 0.94,
    "void_dark_compact_min_area": 650,
    "void_dark_compact_min_enclosure": 0.28,
    "void_solid_dark_min_area": 500,
    "void_solid_dark_min_enclosure": 0.30,
    "void_solid_dark_min_miss": 0.95,

    # --- 路径E: 标准锡面内高暗占比孔洞 ---
    "void_type2_min_area": 300,
    "void_type2_area_frac": 0.006,
    "void_type2_min_dark": 0.94,
    "void_type2_min_miss": 0.94,
    "void_type2_min_enc": 0.28,
    "void_type2_max_enc": 0.38,
    "void_type2_min_core": 0.65,
    "void_core_compact_min_area": 200,
    "void_core_compact_area_frac": 0.004,
    "void_core_compact_min_dark": 0.98,
    "void_core_compact_min_miss": 0.97,
    "void_core_compact_min_core": 0.95,
    "void_core_compact_max_enc": 0.50,
    "void_edge_shadow_max_enc": 0.32,
    "void_edge_shadow_min_dark": 0.92,
    "void_edge_shadow_border_px": 8,

    # --- 13 少锡 ---
    "insuf_area_ratio_min": 0.20,
    "insuf_exposed_v_min": 90,
    "insuf_exposed_s_max": 110,
    "insuf_exposed_v_rel_drop": 40.0,
    "insuf_exposed_s_rel_margin": 30.0,
    "insuf_core_erode_frac": 0.12,
    "insuf_exposed_hard_min": 0.06,
    "insuf_exposed_hard_roi_min": 0.04,
    "insuf_exposed_hard_red_min": 0.015,
    "insuf_copper_tex_max": 0.25,
    "insuf_thin_over_exposed_tex": 0.35,
    # 分型：边缘环带红达到此值优先薄锡；核心裸露达到此值且环带红很少优先露铜
    "insuf_thin_ring_prefer": 0.03,
    "insuf_copper_exposed_prefer": 0.10,
    "insuf_thin_support_ar_max": 0.72,
    "insuf_thin_support_exposed_min": 0.03,
    "insuf_thin_support_ring_min": 0.02,
    "insuf_nolead_center_ar_min": 0.82,
    "insuf_nolead_center_red_max": 0.04,
    "insuf_nolead_center_ring_max": 0.06,
    "insuf_nolead_center_diff_min": 2.0,
    "insuf_shift_nolead_ar_max": 0.65,
    "insuf_shift_nolead_exposed_min": 0.12,
    "insuf_shift_nolead_ring_min": 0.12,

    # --- 14 多锡(包锡) ---
    "excess_area_ratio_min": 1.15,
    "excess_extra_solder_frac_min": 0.025,
    "highlight_v_rel_delta": 85.0,
    "highlight_v_rel_p25_delta": 60.0,
    "highlight_s_rel_margin": 20.0,
    "excess_highlight_ratio_min": 0.035,
    "excess_circularity_min": 0.55,
    "excess_solidity_min": 0.82,
    "excess_bright_closure_min": 0.60,
    "pin_top_frac": 0.18,
    "pin_top_dark_tol": 15.0,
    "pin_top_core_std_min": 200,
    "pin_top_core_roi_frac": 0.008,
    "pin_top_core_std_frac": 0.35,
    "excess_bulge_pin_ret_min": 0.82,
    "excess_bulge_pin_ret_max": 0.96,
    "excess_bulge_extra_cyan_min": 0.032,
    "excess_bulge_diff_min": 1.15,
    "excess_bulge_diff_max": 1.45,
    "excess_bulge_ncc_max": -0.20,
    "excess_bulge_area_min": 0.92,
    "excess_bulge_area_max": 1.05,

    # --- 16 不出脚 ---
    "nolead_cyan_h_low": 82,
    "nolead_cyan_h_high": 104,
    "nolead_cyan_s_min": 90,
    "nolead_cyan_v_min": 160,
    "nolead_center_frac": 0.30,
    "nolead_center_cyan_abs": 0.22,
    "nolead_std_cyan_min": 0.10,
    "nolead_center_cyan_delta": 0.08,
    "nolead_std_v_peak_min": 130,
    "nolead_std_grad_min": 18.0,
    "nolead_patch_ncc_max": -0.15,
    "nolead_center_diff_min": 1.00,
    "nolead_center_diff_soft": 0.90,
    "nolead_peak_ratio_max": 0.50,
    "nolead_diff_ok_max": 1.15,
    "nolead_cyan_match_max": 0.12,
    "nolead_pin_visible_peak_min": 1.05,
    "nolead_pin_likely_peak_min": 0.98,
    "nolead_pin_visible_diff_max": 1.60,
    "nolead_pin_high_diff_min": 1.60,
    "nolead_cyan_override_delta": 0.25,
    "nolead_transition_ring_inner": 0.12,
    "nolead_transition_ring_outer": 0.22,
    "nolead_ring_v_delta_min": 30.0,
    "nolead_ring_contrast_drop_min": 28.0,
    "nolead_min_votes": 1,
    "nolead_ring_vote_weight": 2,

    # --- 15 连锡  ---
    "bridge_erode_frac": 0.18,
    "bridge_erode_min_px": 4,
    "bridge_search_margin": 12,
    "bridge_exclude_margin": 2,
    "bridge_weights": (0.4, 1.0, 1.0),
    "bridge_std_floor": (4.0, 3.0, 3.0),
    "bridge_std_cap": (16.0, 12.0, 12.0),
    "bridge_k_sigma": 2.2,
    "bridge_change_weights": (0.5, 1.0, 1.0),
    "bridge_change_tol": (10.0, 6.0, 6.0),
    "bridge_change_scale_clip": (0.5, 2.0),
    "bridge_change_k": 1.0,
    "bridge_change_blur_ksize": 5,
    "bridge_open_ksize": 3,
    "bridge_min_bridge_pixels": 6,
    "bridge_clearance_bg_percentile": 70.0,
    "bridge_clearance_bg_k": 3.0,
    "bridge_min_clearance_ratio": 0.35,
    "bridge_min_clearance_px": 2,
}

# 模板模型配置 (TemplateModel: 缺陷类型开关 / 覆盖率早停)

# 启用的缺陷类型 (缺陷 ID 参见上方 DEFECT_NAMES)
ENABLED_DEFECTS = [12, 13, 14, 15, 16]

# 按标准图 (templ_N) 覆盖 ENABLED_DEFECTS，未列出的模板使用全局 ENABLED_DEFECTS
TEMPLATE_OVERRIDES: dict = {}

# 覆盖率早停 (配准后校验测试锡面与模板锡面的重合度，不达标转入待复检)
COVERAGE = {
    "ok_min": 0.97,
    "review_min": 0.85,
}
