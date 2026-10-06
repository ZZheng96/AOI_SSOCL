"""通用 UI 控件：卡片容器、参数滑杆、彩色圆点、状态徽标。"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from . import theme
from ..metadata import Param


def _tooltip_text(param: Param) -> str:
    if param.hint:
        return param.hint
    return f"参数键：{param.key}"


def _apply_tooltip(widget: QWidget, text: str):
    widget.setToolTip(text)
    widget.setToolTipDuration(8000)
    widget.setStatusTip(text)


class Card(QFrame):
    """带标题的圆角面板容器。"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("class", "Panel")
        self.setObjectName("Card")
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(14, 12, 14, 14)
        self._v.setSpacing(10)
        if title:
            lbl = QLabel(title)
            lbl.setProperty("class", "PanelTitle")
            self._v.addWidget(lbl)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        self._v.addLayout(self.body)

    def add(self, w):
        self.body.addWidget(w)

    def add_layout(self, lay):
        self.body.addLayout(lay)


class ColorDot(QWidget):
    def __init__(self, color: str, size: int = 12, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self.setFixedSize(size + 2, size + 2)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self._size, self._size)


class StatusPill(QLabel):
    """OK / NG / ERROR 状态徽标。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(22)
        self.setContentsMargins(10, 0, 10, 0)
        self.set_status("")

    def set_status(self, status: str):
        colors = {"OK": theme.OK, "NG": theme.NG, "ERROR": theme.WARN}
        c = colors.get(status, theme.TEXT_MUTE)
        txt = status or "—"
        self.setText(txt)
        self.setStyleSheet(
            f"background: {c}22; color: {c}; border: 1px solid {c}; "
            f"border-radius: 11px; padding: 0 10px; font-weight: 600;"
        )


class ParamSlider(QWidget):
    """一行参数控件：标签 + 滑杆 + 数值框，支持 float/int/choice。"""

    changed = Signal()

    def __init__(self, param: Param, parent=None):
        super().__init__(parent)
        self.param = param
        tip = _tooltip_text(param)
        _apply_tooltip(self, tip)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        name = QLabel(param.label)
        name.setMinimumWidth(96)
        _apply_tooltip(name, tip)
        name.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(name)

        if tip:
            info = QLabel("ⓘ")
            info.setProperty("class", "ParamHintIcon")
            info.setFixedWidth(16)
            info.setAlignment(Qt.AlignCenter)
            _apply_tooltip(info, tip)
            lay.addWidget(info)

        self._kind = param.kind
        if param.kind == "bool":
            self.check = QCheckBox()
            self.check.setChecked(bool(param.default))
            _apply_tooltip(self.check, tip)
            self.check.toggled.connect(lambda *_: self.changed.emit())
            lay.addWidget(self.check)
            lay.addStretch(1)
            self.slider = None
            self.spin = None
            self.combo = None
            return
        if param.kind == "choice":
            self.combo = QComboBox()
            self.combo.addItems(param.choices or [])
            if param.default in (param.choices or []):
                self.combo.setCurrentText(str(param.default))
            _apply_tooltip(self.combo, tip)
            self.combo.currentTextChanged.connect(lambda *_: self.changed.emit())
            lay.addWidget(self.combo, 1)
            self.slider = None
            self.spin = None
            self.check = None
            return

        self.slider = QSlider(Qt.Horizontal)
        self._scale = 1 if param.kind == "int" else int(round(1.0 / max(param.step, 1e-6)))
        self.slider.setMinimum(int(round(param.minimum * self._scale)))
        self.slider.setMaximum(int(round(param.maximum * self._scale)))
        self.slider.setSingleStep(max(1, int(round(param.step * self._scale))))
        _apply_tooltip(self.slider, tip)
        lay.addWidget(self.slider, 1)

        if param.kind == "int":
            self.spin = QSpinBox()
            self.spin.setRange(int(param.minimum), int(param.maximum))
            self.spin.setSingleStep(max(1, int(param.step)))
            self.spin.setValue(int(param.default))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(param.minimum, param.maximum)
            self.spin.setSingleStep(param.step)
            self.spin.setDecimals(self._decimals(param.step))
            self.spin.setValue(float(param.default))
        self.spin.setFixedWidth(78)
        _apply_tooltip(self.spin, tip)
        lay.addWidget(self.spin)

        self.slider.setValue(int(round(float(param.default) * self._scale)))
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    @staticmethod
    def _decimals(step: float) -> int:
        s = f"{step:.6f}".rstrip("0")
        return max(1, len(s.split(".")[1])) if "." in s else 0

    def _on_slider(self, v: int):
        val = v / self._scale
        self.spin.blockSignals(True)
        if self._kind == "int":
            self.spin.setValue(int(round(val)))
        else:
            self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.changed.emit()

    def _on_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(float(v) * self._scale)))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> Any:
        if self._kind == "bool":
            return bool(self.check.isChecked())
        if self._kind == "choice":
            return self.combo.currentText()
        if self._kind == "int":
            return int(self.spin.value())
        return float(self.spin.value())

    def set_value(self, v: Any):
        if self._kind == "bool":
            self.check.setChecked(bool(v))
        elif self._kind == "choice":
            self.combo.setCurrentText(str(v))
        else:
            self.spin.setValue(v)
