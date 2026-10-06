"""浅色现代主题（兼容入口）。

主题的唯一实现已迁移至 ui.theme（LIGHT_QSS / apply_light_theme），
本模块仅做转发，保持旧代码 `from ui.style import QSS, apply_theme` 可用。
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from ui.theme import (  # noqa: F401
    BG,
    CARD_BORDER,
    DANGER,
    LIGHT_QSS,
    PRIMARY,
    PRIMARY_DARK,
    SUCCESS,
    TEXT,
    TEXT_SUB,
    WARNING,
    apply_light_theme,
)

QSS = LIGHT_QSS


def apply_theme(app: QApplication) -> None:
    """应用浅色现代主题（等价于 ui.theme.apply_light_theme）。"""
    apply_light_theme(app)
