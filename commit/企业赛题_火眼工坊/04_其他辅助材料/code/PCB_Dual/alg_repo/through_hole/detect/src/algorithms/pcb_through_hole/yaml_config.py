"""从 YAML 加载检测参数并写入本包 config 模块。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import config as C
from .tuning import normalize_preset_level

try:
    import yaml
except ImportError as exc:
    yaml = None  # type: ignore
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


# segment.* YAML 键 (snake_case) -> config 模块常量名映射，供 apply_yaml_data
# 逐键 setattr；模块级常量，避免每次加载配置时在循环内重复构造同一字典。
_SEG_NAME_MAP: dict[str, str] = {
    "morph_kernel": "MORPH_KERNEL",
    "solder_morph_close": "SOLDER_MORPH_CLOSE",
    "solder_morph_open": "SOLDER_MORPH_OPEN",
    "solder_median": "SOLDER_MEDIAN",
    "solder_min_area_frac": "SOLDER_MIN_AREA_FRAC",
    "solder_red_band": "SOLDER_RED_BAND",
    "solder_use_standard_prior": "SOLDER_USE_STANDARD_PRIOR",
    "solder_punch_center": "SOLDER_PUNCH_CENTER",
    "solder_keep_largest": "SOLDER_KEEP_LARGEST",
    "solder_exclude_red": "SOLDER_EXCLUDE_RED",
    "void_morph_kernel": "VOID_MORPH_KERNEL",
    "void_v_max": "VOID_V_MAX",
    "solder_void_strip_v_max": "SOLDER_VOID_STRIP_V_MAX",
    "solder_void_strip_s_max": "SOLDER_VOID_STRIP_S_MAX",
    "highlight_v_min": "HIGHLIGHT_V_MIN",
    "highlight_s_max": "HIGHLIGHT_S_MAX",
    "edge_ring_inner": "EDGE_RING_INNER",
    "edge_ring_outer": "EDGE_RING_OUTER",
}


def _as_tuple(val: Any) -> tuple:
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    raise TypeError(f"期望 HSV 三元组 list，收到: {val!r}")


def _merge_dict(target: dict, src: dict) -> None:
    for k, v in src.items():
        target[k] = v


def _normalize_preset_dict_keys(table: dict) -> dict:
    """将预设表键名统一为 LOW / MED / HIGH。"""
    out: dict = {}
    for name, preset in table.items():
        if not isinstance(preset, dict):
            continue
        out[normalize_preset_level(str(name))] = preset
    return out


# 少锡挡位表覆盖的全部键名；显式 thresholds 覆盖同时写入 C.INSUF_OVERRIDES，
# 否则这些键只存在于挡位表里，写 C.TH 不会被 insuf_th() 读到 (见 config.insuf_th)。
_INSUF_PRESET_KEYS: frozenset = frozenset(C.INSUF_PRESETS.get("MED", {}).keys())


def apply_yaml_data(data: dict[str, Any]) -> None:
    """将解析后的 YAML 字典应用到 config 模块。"""
    if not data:
        return

    if "enable_calibration" in data:
        C.ENABLE_CALIBRATION = bool(data["enable_calibration"])
    if "work_max_dim" in data:
        v = data["work_max_dim"]
        C.WORK_MAX_DIM = None if v is None else int(v)
    if "insuf_preset" in data:
        C.INSUF_PRESET = normalize_preset_level(str(data["insuf_preset"]), kind="insuf")
    elif "insuf_sensitivity" in data:
        C.INSUF_PRESET = normalize_preset_level(str(data["insuf_sensitivity"]), kind="insuf")

    if "void_v_max" in data:
        C.VOID_V_MAX = int(data["void_v_max"])

    seg = data.get("segment") or {}
    if seg:
        if "solder_hsv_low" in seg:
            C.SOLDER_HSV_LOW = _as_tuple(seg["solder_hsv_low"])
        if "solder_hsv_high" in seg:
            C.SOLDER_HSV_HIGH = _as_tuple(seg["solder_hsv_high"])
        if "solder_wide_hsv_low" in seg:
            C.SOLDER_WIDE_HSV_LOW = _as_tuple(seg["solder_wide_hsv_low"])
        if "solder_wide_hsv_high" in seg:
            C.SOLDER_WIDE_HSV_HIGH = _as_tuple(seg["solder_wide_hsv_high"])
        for key in _SEG_NAME_MAP:
            if key in seg:
                setattr(C, _SEG_NAME_MAP[key], seg[key])
        if "red1_hsv_low" in seg:
            C.RED1_HSV_LOW = _as_tuple(seg["red1_hsv_low"])
        if "red1_hsv_high" in seg:
            C.RED1_HSV_HIGH = _as_tuple(seg["red1_hsv_high"])
        if "red2_hsv_low" in seg:
            C.RED2_HSV_LOW = _as_tuple(seg["red2_hsv_low"])
        if "red2_hsv_high" in seg:
            C.RED2_HSV_HIGH = _as_tuple(seg["red2_hsv_high"])

    align = data.get("align") or {}
    if align:
        _merge_dict(C.ALIGN, align)

    profile = data.get("profile") or {}
    if profile:
        _merge_dict(C.PROFILE, profile)

    edge = data.get("edge_align") or {}
    if edge:
        if "tol_px" in edge:
            C.TH["edge_align_tol_px"] = int(edge["tol_px"])
        if "erode_frac" in edge:
            C.TH["edge_align_erode_frac"] = float(edge["erode_frac"])
        if "artifact_ring_min_frac" in edge:
            C.TH["edge_artifact_ring_min_frac"] = float(edge["artifact_ring_min_frac"])
        _edge_map = {
            "missing_min_frac": "edge_artifact_missing_min_frac",
            "near_test_min_frac": "edge_artifact_near_test_min_frac",
            "in_test_max_frac": "edge_artifact_in_test_max_frac",
            "max_dark_frac": "edge_artifact_max_dark_frac",
        }
        for yaml_k, th_k in _edge_map.items():
            if yaml_k in edge:
                C.TH[th_k] = float(edge[yaml_k])

    insuf_presets = data.get("insuf_presets") or {}
    for name, preset in _normalize_preset_dict_keys(insuf_presets).items():
        if name in C.INSUF_PRESETS and isinstance(preset, dict):
            C.INSUF_PRESETS[name].update(preset)

    th = data.get("thresholds") or {}

    void_presets = data.get("void_presets") or data.get("void_strictness_presets") or {}
    void_level = data.get("void_preset") or data.get("void_strictness")
    if void_presets or void_level:
        from . import tuning as T
        for level, preset in _normalize_preset_dict_keys(void_presets).items():
            if level in T.VOID_PRESETS and isinstance(preset, dict):
                T.VOID_PRESETS[level].update(preset)
        level = str(void_level) if void_level else "MED"
        if th or void_presets or void_level:
            T.apply_void_preset(level)

    # 显式 thresholds 放在挡位应用之后，避免被 apply_void_preset() 的整体重置冲掉；
    # 少锡的挡位管理键额外写入 C.INSUF_OVERRIDES (insuf_th() 只认这个覆盖表)。
    if th:
        for k, v in th.items():
            C.TH[k] = v
            if k in _INSUF_PRESET_KEYS:
                C.INSUF_OVERRIDES[k] = v

    if "enabled_defects" in data:
        C.ENABLED_DEFECTS = list(data["enabled_defects"])

    template_overrides = data.get("template_overrides") or {}
    if template_overrides:
        _merge_dict(C.TEMPLATE_OVERRIDES, {
            str(k): list(v) for k, v in template_overrides.items()
        })

    coverage = data.get("coverage") or {}
    if coverage:
        _merge_dict(C.COVERAGE, coverage)


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "需要 PyYAML: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def apply_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """加载 YAML 并写入 config，返回解析后的字典。"""
    data = load_yaml(path)
    apply_yaml_data(data)
    return data


def default_config_path() -> Path:
    """仓库根目录 (detect/) 下的 config.yaml；自本文件向上查找。"""
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        cand = p / "config.yaml"
        if cand.is_file():
            return cand
    # yaml_config → pcb_through_hole → algorithms → src → detect
    return Path(__file__).resolve().parents[3] / "config.yaml"


def load_if_exists(path: str | os.PathLike | None = None) -> dict[str, Any] | None:
    """若 path 或默认 config.yaml 存在则加载，否则跳过。"""
    p = Path(path) if path else default_config_path()
    if not p.is_file():
        return None
    return apply_yaml(p)
