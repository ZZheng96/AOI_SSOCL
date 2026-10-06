"""左上：锡面区域 — 自动(圆形/轮廓) 或 人工多焊点框"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QLabel, QPushButton, QRadioButton, QVBoxLayout, QWidget

from . import theme
from .widgets import Card, apply_tooltip

_MODE_TEXT = {
    "ellipse": "自动 · 圆形拟合",
    "contour": "自动 · 不规则轮廓分割",
    "roi": "人工框选焊点",
}


class SolderAreaPanel(Card):
    shape_choice_changed = Signal(str)  # "ellipse" | "contour"
    clear_manual_roi_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("锡面区域", parent)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.radio_ellipse = QRadioButton("圆形")
        self.radio_contour = QRadioButton("不规则轮廓")
        apply_tooltip(self.radio_ellipse, "无人工框时：焊点锡面接近圆/椭圆，用几何拟合")
        apply_tooltip(self.radio_contour, "无人工框时：不规则轮廓自动分割")
        self.radio_ellipse.setChecked(True)
        self._group.addButton(self.radio_ellipse)
        self._group.addButton(self.radio_contour)
        self.radio_ellipse.toggled.connect(self._on_radio_toggled)
        self.radio_contour.toggled.connect(self._on_radio_toggled)

        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(self.radio_ellipse)
        rl.addWidget(self.radio_contour)
        self.add(row)

        self.status_lbl = QLabel()
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(f"color: {theme.ACCENT_Hi}; font-weight: 600;")
        self.add(self.status_lbl)

        self.hint_lbl = QLabel()
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12px;")
        self.add(self.hint_lbl)

        self.btn_clear_roi = QPushButton("清除人工框，改回自动方式")
        self.btn_clear_roi.setVisible(False)
        apply_tooltip(
            self.btn_clear_roi,
            "删除全部人工焊点框后，重新用上方圆形/不规则轮廓自动生成锡面")
        self.btn_clear_roi.clicked.connect(self.clear_manual_roi_requested.emit)
        self.add(self.btn_clear_roi)

        self._building = False
        self.set_pad_status(0, auto_mode="ellipse")

    def shape_choice(self) -> str:
        return "contour" if self.radio_contour.isChecked() else "ellipse"

    def set_shape_choice(self, mode: str):
        self._building = True
        if mode == "contour":
            self.radio_contour.setChecked(True)
        else:
            self.radio_ellipse.setChecked(True)
        self._building = False

    def set_effective_mode(self, mode: str):
        if mode == "roi":
            self.radio_ellipse.setEnabled(False)
            self.radio_contour.setEnabled(False)
            self.btn_clear_roi.setVisible(True)
        else:
            self.set_pad_status(0, auto_mode=mode)

    def set_pad_status(self, count: int, auto_mode: str = "ellipse"):
        """count>0：人工框；count==0：自动方式。"""
        if count > 0:
            self.status_lbl.setText(f"当前生效方式：人工框选 · {count} 个焊点")
            self.status_lbl.setStyleSheet("color: #FF8000; font-weight: 600;")
            if count == 1:
                self.hint_lbl.setText(
                    "单焊点：检盘内四种缺陷。再框 ≥1 个改检连锡；Delete 删选中")
            else:
                self.hint_lbl.setText(
                    f"多焊点（{count}）：仅检连锡。可继续追加；Delete 删选中")
            self.btn_clear_roi.setVisible(True)
            self.radio_ellipse.setEnabled(False)
            self.radio_contour.setEnabled(False)
        else:
            text = _MODE_TEXT.get(auto_mode, auto_mode)
            self.status_lbl.setText(f"当前生效方式：{text}")
            self.status_lbl.setStyleSheet(f"color: {theme.ACCENT_Hi}; font-weight: 600;")
            self.hint_lbl.setText(
                "自动锡面：检盘内四种。画 1 框仍盘内；≥2 框仅检连锡。")
            self.btn_clear_roi.setVisible(False)
            self.radio_ellipse.setEnabled(True)
            self.radio_contour.setEnabled(True)

    def _on_radio_toggled(self, checked: bool):
        if self._building or not checked:
            return
        self.shape_choice_changed.emit(self.shape_choice())
