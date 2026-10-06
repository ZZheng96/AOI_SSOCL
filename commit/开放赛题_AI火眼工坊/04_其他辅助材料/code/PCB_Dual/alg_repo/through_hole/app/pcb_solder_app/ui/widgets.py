"""通用 UI 控件：卡片容器、参数滑杆、彩色圆点、状态徽标"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QSizePolicy,
    QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from . import theme
from ..metadata import Param


def apply_tooltip(widget: QWidget, text: str) -> None:
    if not text:
        return
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
    """OK / NG / REVIEW / ERROR 状态徽标。"""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(26)
        self.setContentsMargins(10, 0, 10, 0)
        self.set_status("")

    def set_status(self, status: str):
        colors = {"OK": theme.OK, "NG": theme.NG, "REVIEW": theme.REVIEW, "ERROR": theme.WARN}
        c = colors.get(status, theme.TEXT_MUTE)
        self.setText(status or "—")
        self.setStyleSheet(
            f"background: {c}22; color: {c}; border: 1px solid {c}; "
            f"border-radius: 13px; padding: 0 12px; font-weight: 700; font-size: 14px;"
        )


class ParamSlider(QWidget):
    """参数控件：上方参数名；下方左侧滑杆、右侧数值框（含上下步进按钮）。

    ``unit="%"`` 时滑杆/数值框按百分数显示，``value()`` / ``set_value()``
    仍使用算法侧的 0~1 比例。
    """

    changed = Signal()

    def __init__(self, param: Param, parent=None):
        super().__init__(parent)
        self.param = param
        tip = param.hint or f"参数键：{param.key}"
        apply_tooltip(self, tip)

        # 百分比：界面 ×100 显示，与算法 0~1 互转；其它单位仅作后缀
        self._is_percent = param.unit == "%"
        self._ui_scale = 100.0 if self._is_percent else 1.0
        ui_min = param.minimum * self._ui_scale
        ui_max = param.maximum * self._ui_scale
        ui_step = param.step * self._ui_scale
        ui_default = param.default * self._ui_scale

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 6)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        name = QLabel(param.label)
        name.setWordWrap(True)
        apply_tooltip(name, tip)
        name.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
        head.addWidget(name, 1)

        info = QLabel("ⓘ")
        info.setProperty("class", "ParamHintIcon")
        info.setFixedWidth(16)
        info.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        apply_tooltip(info, tip)
        head.addWidget(info, 0, Qt.AlignTop)
        root.addLayout(head)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        self._kind = param.kind
        self.slider = QSlider(Qt.Horizontal)
        # 滑杆刻度按「界面显示值」步进，再换算回算法值
        self._scale = 1 if param.kind == "int" else int(round(1.0 / max(ui_step, 1e-6)))
        step_ticks = max(1, int(round(ui_step * self._scale)))
        self.slider.setMinimum(int(round(ui_min * self._scale)))
        self.slider.setMaximum(int(round(ui_max * self._scale)))
        self.slider.setSingleStep(step_ticks)
        self.slider.setPageStep(step_ticks * 5)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.setMinimumWidth(120)
        self.slider.setFixedHeight(24)
        apply_tooltip(self.slider, tip)
        row.addWidget(self.slider, 1)

        if param.kind == "int":
            self.spin = QSpinBox()
            self.spin.setRange(int(round(ui_min)), int(round(ui_max)))
            self.spin.setSingleStep(max(1, int(round(ui_step))))
            self.spin.setValue(int(round(ui_default)))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(ui_min, ui_max)
            self.spin.setSingleStep(ui_step)
            self.spin.setDecimals(self._decimals(ui_step))
            self.spin.setValue(float(ui_default))
        if param.unit:
            # 百分数用紧凑 "%"；其它单位前加空格更易读（如 " ×"）
            suffix = param.unit if self._is_percent else f" {param.unit}"
            self.spin.setSuffix(suffix)
        # 隐藏原生按钮，右侧自绘上下小箭头；整体包在同一边框里与数值等高对齐
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setKeyboardTracking(False)
        self.spin.setAccelerated(True)
        self.spin.setObjectName("ParamValueSpin")
        self.spin.setFixedWidth(92 if param.unit else 70)
        self.spin.setFixedHeight(26)
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin.setFrame(False)
        apply_tooltip(self.spin, tip)

        step_box = QFrame()
        step_box.setObjectName("ParamStepBox")
        step_box.setFixedWidth(20)
        step_lay = QVBoxLayout(step_box)
        step_lay.setContentsMargins(0, 0, 0, 0)
        step_lay.setSpacing(0)

        self.btn_up = QToolButton()
        self.btn_up.setObjectName("ParamStepUp")
        self.btn_up.setText("▴")
        self.btn_up.setAutoRepeat(True)
        self.btn_up.setAutoRepeatDelay(300)
        self.btn_up.setAutoRepeatInterval(50)
        self.btn_up.setCursor(Qt.PointingHandCursor)
        self.btn_up.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        apply_tooltip(self.btn_up, "增大")
        self.btn_up.clicked.connect(self.spin.stepUp)

        self.btn_down = QToolButton()
        self.btn_down.setObjectName("ParamStepDown")
        self.btn_down.setText("▾")
        self.btn_down.setAutoRepeat(True)
        self.btn_down.setAutoRepeatDelay(300)
        self.btn_down.setAutoRepeatInterval(50)
        self.btn_down.setCursor(Qt.PointingHandCursor)
        self.btn_down.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        apply_tooltip(self.btn_down, "减小")
        self.btn_down.clicked.connect(self.spin.stepDown)

        step_lay.addWidget(self.btn_up, 1)
        step_lay.addWidget(self.btn_down, 1)

        value_box = QFrame()
        value_box.setObjectName("ParamValueBox")
        value_box.setFixedHeight(28)
        value_lay = QHBoxLayout(value_box)
        value_lay.setContentsMargins(0, 0, 0, 0)
        value_lay.setSpacing(0)
        value_lay.addWidget(self.spin, 1)
        value_lay.addWidget(step_box)
        row.addWidget(value_box, 0, Qt.AlignVCenter)
        root.addLayout(row)

        self.slider.setValue(int(round(float(ui_default) * self._scale)))
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    @staticmethod
    def _decimals(step: float) -> int:
        s = f"{step:.6f}".rstrip("0").rstrip(".")
        if "." not in s:
            return 0
        return max(1, len(s.split(".")[1]))

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
        """返回算法侧数值（百分比参数已从界面百分数换算回 0~1）。"""
        ui_val = self.spin.value()
        engine = float(ui_val) / self._ui_scale
        if self._kind == "int":
            return int(round(engine))
        return engine

    def set_value(self, v: Any):
        """接受算法侧数值，换算为界面显示值后写入控件。"""
        self.spin.setValue(float(v) * self._ui_scale)
