"""插件焊点缺陷检测：孔洞 / 少锡 / 多锡 / 连锡 / 不出脚"""

from .detection import Detection
from .pipeline import InspectResult, inspect
from .template import TemplateModel
from . import config
from .tuning import (
    PRESET_LEVELS,
    apply_void_preset,
    active_void_preset,
    VOID_PRESETS,
)

__all__ = [
    "TemplateModel",
    "InspectResult",
    "inspect",
    "Detection",
    "config",
    "PRESET_LEVELS",
    "apply_void_preset",
    "active_void_preset",
    "VOID_PRESETS",
]
