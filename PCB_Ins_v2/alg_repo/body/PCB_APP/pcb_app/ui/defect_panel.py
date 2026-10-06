"""左侧：瑕疵类型清单（勾选激活 + 点击查看/编辑其参数）。

数据完全来自 ``metadata.DEFECT_METAS``，新增算法自动出现。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel,
    QScrollArea, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, ColorDot
from ..metadata import DEFECT_METAS, DefectMeta


class DefectRow(QFrame):
    toggled = Signal(str, bool)
    clicked = Signal(str)

    def __init__(self, meta: DefectMeta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self.setObjectName("DefectRow")
        self._selected = False
        self._apply_style(False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)

        self.chk = QCheckBox()
        self.chk.toggled.connect(lambda v: self.toggled.emit(self.meta.code, v))
        lay.addWidget(self.chk)

        lay.addWidget(ColorDot(meta.color, 12))

        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(meta.name)
        name.setStyleSheet("font-weight: 600; font-size: 13px;")
        summ = QLabel(meta.summary)
        summ.setStyleSheet(f"color: {theme.TEXT_MUTE}; font-size: 11px;")
        summ.setWordWrap(True)
        text.addWidget(name)
        text.addWidget(summ)
        lay.addLayout(text, 1)

    def _apply_style(self, selected: bool):
        border = theme.ACCENT if selected else theme.BORDER
        bg = theme.BG_2 if selected else theme.BG_1
        self.setStyleSheet(
            f"#DefectRow {{ background: {bg}; border: 1px solid {border};"
            f" border-radius: 10px; }}"
            f"#DefectRow:hover {{ border-color: {theme.ACCENT}; }}"
        )

    def set_selected(self, sel: bool):
        self._selected = sel
        self._apply_style(sel)

    def is_checked(self) -> bool:
        return self.chk.isChecked()

    def set_checked(self, v: bool):
        self.chk.setChecked(v)

    def mousePressEvent(self, e):
        self.clicked.emit(self.meta.code)
        super().mousePressEvent(e)


class DefectPanel(Card):
    selection_changed = Signal(set)     # 当前激活的 code 集合
    current_changed = Signal(str)       # 当前查看的 code

    def __init__(self, parent=None):
        super().__init__("瑕疵类型", parent)
        hint = QLabel("勾选要检测的类型；点击行编辑其判据与参数")
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        self.add(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        holder.setObjectName("ScrollHost")
        self._vl = QVBoxLayout(holder)
        self._vl.setContentsMargins(0, 0, 6, 0)
        self._vl.setSpacing(8)

        self.rows: Dict[str, DefectRow] = {}
        for meta in DEFECT_METAS:
            row = DefectRow(meta)
            row.toggled.connect(self._on_toggle)
            row.clicked.connect(self._on_click)
            self.rows[meta.code] = row
            self._vl.addWidget(row)
        self._vl.addStretch(1)
        scroll.setWidget(holder)
        self.add(scroll)

        self._current: Optional[str] = None

    def _on_toggle(self, code: str, checked: bool):
        if checked:
            self._select(code)
        self.selection_changed.emit(self.active_codes())

    def _on_click(self, code: str):
        self._select(code)

    def _select(self, code: str):
        self._current = code
        for c, row in self.rows.items():
            row.set_selected(c == code)
        self.current_changed.emit(code)

    def active_codes(self) -> Set[str]:
        return {c for c, r in self.rows.items() if r.is_checked()}

    def current(self) -> Optional[str]:
        return self._current

    def set_checked(self, code: str, v: bool):
        if code in self.rows:
            self.rows[code].set_checked(v)
