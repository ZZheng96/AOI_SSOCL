"""孔洞 / 少锡 统一预设挡位 LOW | MED | HIGH，及孔洞检测预设挡位应用逻辑"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import config as C

PRESET_LEVELS = ("LOW", "MED", "HIGH")

# 旧键名 → 统一挡位 (兼容历史 config / 脚本)
_VOID_ALIASES = {
    "low": "LOW", "lo": "LOW",
    "medium": "MED", "med": "MED", "standard": "MED", "mid": "MED",
    "high": "HIGH", "hi": "HIGH", "strict": "HIGH",
}
_INSUF_ALIASES = {
    "low": "LOW", "lo": "LOW", "loose": "LOW",
    "medium": "MED", "med": "MED", "standard": "MED", "mid": "MED",
    "high": "HIGH", "hi": "HIGH", "strict": "HIGH",
}
_PRESET_ALIASES = {
    "low": "LOW", "lo": "LOW", "loose": "LOW",
    "medium": "MED", "med": "MED", "standard": "MED", "mid": "MED",
    "high": "HIGH", "hi": "HIGH", "strict": "HIGH",
}


def normalize_preset_level(level: str, *, kind: str = "preset") -> str:
    """将挡位名规范为 LOW / MED / HIGH。"""
    key = str(level).strip()
    if key in PRESET_LEVELS:
        return key
    lower = key.lower()
    table = _INSUF_ALIASES if kind == "insuf" else (
        _VOID_ALIASES if kind == "void" else _PRESET_ALIASES)
    if lower in table:
        return table[lower]
    if key.upper() in PRESET_LEVELS:
        return key.upper()
    opts = ", ".join(PRESET_LEVELS)
    raise ValueError(f"未知挡位 {level!r}，可选: {opts}")


# 挡位：在 config.TH / VOID_V_MAX 基准上的覆盖值（MED 为空表示用 config 默认）。
# 简化后主要调节面积门槛与暗度敏感度（VOID_V_MAX / decide_min_dark）。
VOID_PRESETS: dict[str, dict[str, Any]] = {
    "LOW": {
        "VOID_V_MAX": 78,
        "void_min_area": 45,
        "void_min_area_ratio": 0.008,
        "void_min_circularity": 0.32,
        "void_dark_s_max": 115,
        "void_decide_min_area": 250,
        "void_decide_area_frac": 0.005,
        "void_decide_min_dark": 0.94,
        "void_decide_min_miss": 0.95,
        "void_enc_min": 0.32,
        "void_enc_max": 0.48,
    },
    "MED": {},
    "HIGH": {
        "VOID_V_MAX": 92,
        "void_min_area": 30,
        "void_min_area_ratio": 0.004,
        "void_min_circularity": 0.22,
        "void_dark_s_max": 140,
        "void_decide_min_area": 150,
        "void_decide_area_frac": 0.0025,
        "void_decide_min_dark": 0.86,
        "void_decide_min_miss": 0.86,
        "void_enc_min": 0.25,
        "void_enc_max": 0.55,
    },
}

_BASELINE_TH: dict[str, Any] | None = None
_BASELINE_VOID_V_MAX: int | None = None
_ACTIVE_VOID_PRESET: str = "MED"


def _snapshot_baseline() -> None:
    global _BASELINE_TH, _BASELINE_VOID_V_MAX
    if _BASELINE_TH is None:
        _BASELINE_TH = deepcopy(C.TH)
        _BASELINE_VOID_V_MAX = int(C.VOID_V_MAX)


def apply_void_preset(level: str = "MED") -> str:
    """应用孔洞预设挡位，返回实际使用的挡位名 (LOW/MED/HIGH)。"""
    norm = normalize_preset_level(level, kind="void")
    _snapshot_baseline()
    if norm not in VOID_PRESETS:
        raise ValueError(f"未知孔洞挡位 {level!r}，可选: {', '.join(PRESET_LEVELS)}")

    C.TH.clear()
    C.TH.update(deepcopy(_BASELINE_TH))
    C.VOID_V_MAX = _BASELINE_VOID_V_MAX

    overrides = VOID_PRESETS[norm]
    for key, val in overrides.items():
        if key == "VOID_V_MAX":
            C.VOID_V_MAX = int(val)
        else:
            C.TH[key] = val

    global _ACTIVE_VOID_PRESET
    _ACTIVE_VOID_PRESET = norm
    return norm


def active_void_preset() -> str:
    return _ACTIVE_VOID_PRESET
