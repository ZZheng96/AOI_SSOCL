"""左下：缺陷类型清单（勾选启用 + 点击选中联动右侧参数面板）"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from . import theme
from .widgets import Card, ColorDot
from ..metadata import (
    DEFECT_METAS,
    DEFAULT_ENABLED_DEFECTS,
    DefectMeta,
    defects_for_pad_count,
)


class DefectRow(QFrame):
    toggled = Signal(int, bool)
    clicked = Signal(int)

    def __init__(self, meta: DefectMeta, parent=None):
        super().__init__(parent)
        self.meta = meta
        self._selected = False
        self.setObjectName("DefectRow")
        self._apply_style()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(10)

        self.chk = QCheckBox()
        self.chk.setChecked(True)
        self.chk.toggled.connect(lambda v: self.toggled.emit(self.meta.defect_id, v))
        lay.addWidget(self.chk)

        lay.addWidget(ColorDot(meta.color, 12))

        name = QLabel(meta.name)
        name.setStyleSheet("font-weight: 600; font-size: 13px;")
        lay.addWidget(name, 1)

    def _apply_style(self):
        border = theme.ACCENT if self._selected else theme.BORDER
        bg = theme.BG_2 if self._selected else theme.BG_1
        self.setStyleSheet(
            f"#DefectRow {{ background: {bg}; border: 1px solid {border};"
            f" border-radius: 10px; }}"
        )

    def is_checked(self) -> bool:
        return self.chk.isChecked()

    def set_checked(self, v: bool):
        """静默设置勾选（不触发 toggled）。"""
        self.chk.blockSignals(True)
        self.chk.setChecked(v)
        self.chk.blockSignals(False)

    def set_enabled(self, v: bool):
        """当前框数模式下是否可勾选。"""
        self.chk.setEnabled(v)

    def set_selected(self, v: bool):
        if v == self._selected:
            return
        self._selected = v
        self._apply_style()

    def mousePressEvent(self, event):
        self.clicked.emit(self.meta.defect_id)
        super().mousePressEvent(event)


class DefectPanel(Card):
    changed = Signal()
    defect_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__("缺陷类型", parent)

        actions = QFrame()
        al = QHBoxLayout(actions)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(8)
        self.btn_select_all = QPushButton("全选")
        self.btn_clear_all = QPushButton("取消全选")
        self.btn_select_all.clicked.connect(self.select_all)
        self.btn_clear_all.clicked.connect(self.clear_all)
        al.addWidget(self.btn_select_all)
        al.addWidget(self.btn_clear_all)
        al.addStretch(1)
        self.add(actions)

        self.rows: Dict[int, DefectRow] = {}
        self._selected_id: Optional[int] = None
        self._allowed: Set[int] = set(DEFAULT_ENABLED_DEFECTS)
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {theme.TEXT_MUTE}; font-size: 12px;")
        self.add(self._hint)
        for meta in DEFECT_METAS:
            row = DefectRow(meta)
            row.toggled.connect(lambda *_: self.changed.emit())
            row.clicked.connect(self._on_row_clicked)
            self.rows[meta.defect_id] = row
            self.add(row)

        if DEFECT_METAS:
            self._on_row_clicked(DEFECT_METAS[0].defect_id)

    # ---- API ----
    def enabled_defects(self) -> List[int]:
        return [i for i, r in self.rows.items() if r.is_checked() and i in self._allowed]

    def set_enabled_defects(self, ids: List[int]):
        ids_set = set(ids)
        for i, r in self.rows.items():
            r.set_checked(i in ids_set)

    def apply_pad_count_mode(self, n_pads: int, *, emit: bool = True):
        """按框数切换可勾选缺陷。"""
        allowed = defects_for_pad_count(n_pads)
        self._allowed = set(allowed)
        if n_pads >= 2:
            self._hint.setText("多焊点：仅检连锡")
        elif n_pads == 1:
            self._hint.setText("单焊点：仅检盘内四种缺陷")
        else:
            self._hint.setText("自动锡面：仅检盘内四种缺陷")
        for i, r in self.rows.items():
            on = i in self._allowed
            r.set_enabled(on)
            r.set_checked(on)
        if allowed:
            self._on_row_clicked(allowed[0])
        if emit:
            self.changed.emit()

    def selected_defect_id(self) -> Optional[int]:
        return self._selected_id

    def select(self, defect_id: int):
        self._on_row_clicked(defect_id)

    def select_all(self):
        for i, r in self.rows.items():
            if i in self._allowed:
                r.set_checked(True)
        self.changed.emit()

    def clear_all(self):
        for i, r in self.rows.items():
            if i in self._allowed:
                r.set_checked(False)
        self.changed.emit()

    # ---- internal ----
    def _on_row_clicked(self, defect_id: int):
        if defect_id not in self.rows:
            return
        self._selected_id = defect_id
        for i, r in self.rows.items():
            r.set_selected(i == defect_id)
        self.defect_selected.emit(defect_id)
