"""左下：ROI 列表（重命名/改绑瑕疵/删除/选中定位）。

改绑控件支持勾选多种瑕疵类型：同一个 ROI 可以同时绑定多种缺陷检测算法，
检测时该 ROI 范围会分别被每一种已绑定的算法检测一次。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea,
    QToolButton, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, ColorDot
from ..core.models import ROI
from ..metadata import DEFECT_METAS, DEFECT_BY_CODE


class ROIRow(QFrame):
    selected = Signal(int)
    rebind = Signal(int, list)
    remove = Signal(int)

    def __init__(self, roi: ROI, parent=None):
        super().__init__(parent)
        self.roi = roi
        self.setObjectName("ROIRow")
        self._apply(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        self.dot = ColorDot(self._color(), 11)
        lay.addWidget(self.dot)

        shape_txt = "○" if roi.shape == "circle" else "▭"
        self.name = QLabel(f"{shape_txt} {roi.name}")
        self.name.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.name)
        lay.addStretch(1)

        # 多选下拉：点击弹出可勾选的瑕疵类型菜单，支持一次绑定多种
        self.btn_bind = QToolButton()
        self.btn_bind.setFixedWidth(110)
        self.btn_bind.setPopupMode(QToolButton.InstantPopup)
        self.btn_bind.setToolTip("勾选该 ROI 要检测的瑕疵类型（可多选）")
        self.menu = QMenu(self.btn_bind)
        self._actions: Dict[str, QAction] = {}
        for m in DEFECT_METAS:
            act = QAction(m.name, self.menu)
            act.setCheckable(True)
            act.setChecked(m.code in self.roi.defect_codes)
            act.toggled.connect(lambda checked, code=m.code: self._on_toggle(code, checked))
            self.menu.addAction(act)
            self._actions[m.code] = act
        self.btn_bind.setMenu(self.menu)
        self._refresh_btn_text()
        lay.addWidget(self.btn_bind)

        btn = QPushButton("✕")
        btn.setFixedSize(24, 24)
        btn.clicked.connect(lambda: self.remove.emit(self.roi.rid))
        lay.addWidget(btn)

    def _color(self) -> str:
        m = DEFECT_BY_CODE.get(self.roi.defect_code or "")
        return m.color if m else theme.ACCENT

    def _refresh_btn_text(self):
        codes = self.roi.defect_codes
        if not codes:
            self.btn_bind.setText("未绑定")
        elif len(codes) == 1:
            m = DEFECT_BY_CODE.get(codes[0])
            self.btn_bind.setText(m.name if m else codes[0])
        else:
            names = "、".join(DEFECT_BY_CODE[c].name for c in codes if c in DEFECT_BY_CODE)
            self.btn_bind.setText(f"{len(codes)}种瑕疵")
            self.btn_bind.setToolTip(f"已绑定：{names}")

    def _on_toggle(self, code: str, checked: bool):
        codes = set(self.roi.defect_codes)
        if checked:
            codes.add(code)
        else:
            codes.discard(code)
        self.roi.defect_codes = sorted(codes)
        self._refresh_btn_text()
        self.dot.set_color(self._color())
        self.rebind.emit(self.roi.rid, list(self.roi.defect_codes))

    def _apply(self, sel: bool):
        border = theme.ACCENT if sel else theme.BORDER
        self.setStyleSheet(
            f"#ROIRow {{ background: {theme.BG_1 if not sel else theme.BG_2};"
            f" border: 1px solid {border}; border-radius: 9px; }}"
            f"#ROIRow:hover {{ border-color: {theme.ACCENT}; }}"
        )

    def set_selected(self, sel: bool):
        self._apply(sel)

    def mousePressEvent(self, e):
        self.selected.emit(self.roi.rid)
        super().mousePressEvent(e)


class ROIPanel(Card):
    roi_selected = Signal(int)
    roi_rebound = Signal()
    roi_removed = Signal(int)

    def __init__(self, parent=None):
        super().__init__("ROI 重点区域", parent)
        self.hint = QLabel("在模板图上用矩形/圆形工具框选；待检图自动同位置对应。"
                           "每个 ROI 可勾选绑定多种瑕疵类型，同一区域可被多种算法分别检测")
        self.hint.setProperty("class", "Hint")
        self.hint.setWordWrap(True)
        self.add(self.hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(120)
        holder = QWidget()
        holder.setObjectName("ScrollHost")
        self._vl = QVBoxLayout(holder)
        self._vl.setContentsMargins(0, 0, 6, 0)
        self._vl.setSpacing(6)
        self._vl.addStretch(1)
        scroll.setWidget(holder)
        self.add(scroll)

        self.rows: Dict[int, ROIRow] = {}
        self._empty = QLabel("暂无 ROI（将对整图检测）")
        self._empty.setProperty("class", "Hint")
        self._empty.setAlignment(Qt.AlignCenter)
        self._vl.insertWidget(0, self._empty)

    def rebuild(self, rois: List[ROI], selected_rid: Optional[int] = None):
        for r in self.rows.values():
            r.setParent(None)
        self.rows.clear()
        self._empty.setVisible(len(rois) == 0)
        for roi in rois:
            row = ROIRow(roi)
            row.selected.connect(self.roi_selected.emit)
            row.rebind.connect(lambda *_: self.roi_rebound.emit())
            row.remove.connect(self.roi_removed.emit)
            row.set_selected(roi.rid == selected_rid)
            self.rows[roi.rid] = row
            self._vl.insertWidget(self._vl.count() - 1, row)

    def set_selected(self, rid: Optional[int]):
        for r_id, row in self.rows.items():
            row.set_selected(r_id == rid)
