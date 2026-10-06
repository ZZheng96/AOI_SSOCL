"""通用 UI 控件：图片标签、卡片容器、状态徽标、参数滑杆。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import theme


def apply_tooltip(widget: QWidget, text: str) -> None:
    if not text:
        return
    widget.setToolTip(text)
    widget.setToolTipDuration(8000)
    widget.setStatusTip(text)


class _NoWheelMixin:
    """忽略滚轮，避免在参数区误触改值；事件交给上层滚动区域。"""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelSlider(_NoWheelMixin, QSlider):
    pass


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    pass


class ImageLabel(QLabel):
    """QLabel that keeps aspect ratio when showing pixmaps.

    sizeHint 固定为最小尺寸，避免 setPixmap 后 QLabel 按原图像素尺寸撑破布局。
    """

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = title or "无图像"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background:{theme.BG_1}; color:{theme.TEXT_MUTE}; "
            f"border:1px solid {theme.BORDER}; border-radius:12px;"
        )
        self.setText(self._placeholder)

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        return QSize(self.minimumWidth(), self.minimumHeight())

    def minimumSizeHint(self):  # noqa: N802
        return self.sizeHint()

    def setText(self, text: str) -> None:  # noqa: N802 - keep placeholder in sync when set directly
        self._placeholder = text
        super().setText(text)

    def set_image(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.setPixmap(QPixmap())
            super().setText(self._placeholder)
            return
        super().setText("")
        self._update_scaled()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class Card(QFrame):
    """带标题的圆角面板；可选左侧强调条提升层级。"""

    def __init__(self, title: str = "", parent=None, *, elevated: bool = False, accent: bool = True):
        super().__init__(parent)
        self.setProperty("class", "Panel")
        self.setObjectName("CardElevated" if elevated else "Card")
        self._v = QVBoxLayout(self)
        self._v.setContentsMargins(16, 14, 16, 14)
        self._v.setSpacing(10)
        self._title_label: QLabel | None = None
        if title:
            head = QHBoxLayout()
            head.setSpacing(8)
            if accent:
                bar = QFrame()
                bar.setFixedWidth(3)
                bar.setFixedHeight(14)
                bar.setStyleSheet(
                    f"background:{theme.ACCENT}; border:none; border-radius:2px;"
                )
                head.addWidget(bar, 0, Qt.AlignmentFlag.AlignVCenter)
            self._title_label = QLabel(title)
            self._title_label.setProperty("class", "PanelTitle")
            self._title_label.setWordWrap(True)
            head.addWidget(self._title_label, 1)
            self._v.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        self._v.addLayout(self.body, 1)

    def set_title(self, title: str) -> None:
        if self._title_label is not None:
            self._title_label.setText(title)

    def add(self, w):
        self.body.addWidget(w)

    def add_layout(self, lay):
        self.body.addLayout(lay)


class ColorDot(QWidget):
    def __init__(self, color: str, size: int = 12, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # 外圈淡环
        ring = QColor(self._color)
        ring.setAlpha(70)
        p.setBrush(QBrush(ring))
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, self._size + 4, self._size + 4)
        p.setBrush(QBrush(self._color))
        p.drawEllipse(2, 2, self._size, self._size)


class StatusPill(QLabel):
    """OK / NG / REVIEW / SKIP / ERROR 状态徽标。"""

    def __init__(self, text: str = "", parent=None, *, hero: bool = False):
        super().__init__(text, parent)
        self._hero = hero
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(36 if hero else 26)
        self.setMinimumWidth(72 if hero else 0)
        self.setContentsMargins(12, 0, 12, 0)
        self.set_status("")

    def set_status(self, status: str):
        colors = {
            "OK": theme.OK,
            "NG": theme.NG,
            "REVIEW": theme.REVIEW,
            "ERROR": theme.ERROR,
            "SKIP": theme.SKIP,
            "RUNNING": theme.RUNNING,
            "PUB": theme.OK,
            "DRAFT": theme.REVIEW,
        }
        c = colors.get(status, theme.TEXT_MUTE)
        label = status or "—"
        self.setText(label)
        radius = 10 if self._hero else 8
        size = 18 if self._hero else 13
        weight = 800 if self._hero else 700
        self.setStyleSheet(
            f"background: {c}18; color: {c}; border: 1px solid {c}66; "
            f"border-radius: {radius}px; padding: 0 14px; "
            f"font-weight: {weight}; font-size: {size}px; letter-spacing: 0.5px;"
        )


class ToolbarStrip(QFrame):
    """浅抬升工具条容器，统一页顶操作区视觉。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolbarStrip")
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(10, 8, 10, 8)
        self.layout_.setSpacing(8)


class ParamSlider(QWidget):
    """一行参数控件：标签 + 滑杆拖拽 + 右侧上下按钮微调。

    调节说明（hint）只挂在左侧 ⓘ 上；标签 / 滑杆 / 数值框 / 下拉 均不弹提示，
    避免各缺陷大类调参时误触弹出。适用于本体 / 焊锡 / 插件 / 金手指 / 表面等全部算法。

    刻意禁用鼠标滚轮改值，避免在参数列表里滚动时误触。
    ``spec`` 需具备 ``key/label/kind/default/minimum/maximum/step/hint/choices``
    字段（对应 :class:`app.detect.adapters.base.ParamSpec`）。kind 支持
    "int" / "float" / "enum"。
    """

    changed = Signal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        tip = (spec.hint or "").strip() or f"参数键：{spec.key}"
        # 整行与各控件默认不挂提示
        self.setToolTip("")
        self.setStatusTip("")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        name = QLabel(spec.label)
        name.setMinimumWidth(56)
        name.setMaximumWidth(120)
        name.setWordWrap(True)
        name.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        name.setStyleSheet(f"color: {theme.TEXT_DIM};")
        name.setToolTip("")
        name.setStatusTip("")
        lay.addWidget(name, 0)

        # 调节说明只挂在 ⓘ（所有算法大类统一）
        info = QLabel("ⓘ")
        info.setProperty("class", "ParamHintIcon")
        info.setFixedWidth(16)
        info.setAlignment(Qt.AlignCenter)
        apply_tooltip(info, tip)
        lay.addWidget(info)

        self._kind = spec.kind
        self.slider: NoWheelSlider | None = None
        self.combo: NoWheelComboBox | None = None

        if self._kind == "enum":
            self.combo = NoWheelComboBox()
            self.combo.addItems(list(spec.choices))
            if str(spec.default) in spec.choices:
                self.combo.setCurrentText(str(spec.default))
            self.combo.setToolTip("")
            self.combo.setStatusTip("")
            lay.addWidget(self.combo, 1)
            self.combo.currentTextChanged.connect(lambda _t: self.changed.emit())
            self.spin = None
            return

        # 1) 滑杆拖拽粗调（不挂提示）
        self.slider = NoWheelSlider(Qt.Horizontal)
        self._scale = 1 if spec.kind == "int" else int(round(1.0 / max(spec.step, 1e-6)))
        self.slider.setMinimum(int(round(spec.minimum * self._scale)))
        self.slider.setMaximum(int(round(spec.maximum * self._scale)))
        self.slider.setSingleStep(max(1, int(round(spec.step * self._scale))))
        self.slider.setPageStep(max(1, int(round(spec.step * self._scale)) * 5))
        self.slider.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.slider.setToolTip("")
        self.slider.setStatusTip("")
        lay.addWidget(self.slider, 1)

        # 2) 右侧数值框 + 独立上下按钮（不挂提示）
        if spec.kind == "int":
            self.spin = NoWheelSpinBox()
            self.spin.setRange(int(spec.minimum), int(spec.maximum))
            self.spin.setSingleStep(max(1, int(spec.step)))
            self.spin.setValue(int(spec.default))
        else:
            self.spin = NoWheelDoubleSpinBox()
            self.spin.setRange(spec.minimum, spec.maximum)
            self.spin.setSingleStep(spec.step)
            self.spin.setDecimals(self._decimals(spec.step))
            self.spin.setValue(float(spec.default))
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.spin.setToolTip("")
        self.spin.setStatusTip("")
        # 内部输入框也清掉，避免悬停数字区仍弹出
        line = self.spin.lineEdit()
        if line is not None:
            line.setToolTip("")
            line.setStatusTip("")
        lay.addWidget(self.spin)

        step_btns = QVBoxLayout()
        step_btns.setContentsMargins(8, 0, 0, 0)  # 与数值框拉开，减少误点
        step_btns.setSpacing(2)
        self.btn_up = QToolButton()
        self.btn_down = QToolButton()
        # 纯箭头，不要文案/提示（样式表下 setArrowType 常不显示，改用字符）
        self.btn_up.setText("▲")
        self.btn_down.setText("▼")
        self.btn_up.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_down.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for b in (self.btn_up, self.btn_down):
            b.setObjectName("ParamStepBtn")
            b.setFixedSize(26, 18)
            b.setAutoRepeat(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip("")
            b.setStatusTip("")
        step_btns.addWidget(self.btn_up)
        step_btns.addWidget(self.btn_down)
        lay.addLayout(step_btns)

        self.slider.setValue(int(round(float(spec.default) * self._scale)))
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.btn_up.clicked.connect(self.spin.stepUp)
        self.btn_down.clicked.connect(self.spin.stepDown)

    def wheelEvent(self, event) -> None:  # noqa: N802
        # 整行也忽略滚轮，交给外层 QScrollArea 滚动列表
        event.ignore()

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
        if self._kind == "enum":
            return self.combo.currentText()
        if self._kind == "int":
            return int(self.spin.value())
        return float(self.spin.value())

    def set_value(self, v: Any):
        if self._kind == "enum":
            if v is not None and str(v) in self.spec.choices:
                self.combo.setCurrentText(str(v))
            return
        self.spin.setValue(v)

    def set_readonly(self, readonly: bool) -> None:
        if self._kind == "enum":
            if self.combo is not None:
                self.combo.setEnabled(not readonly)
            return
        if self.slider is not None:
            self.slider.setEnabled(not readonly)
        if self.spin is not None:
            self.spin.setEnabled(not readonly)
        for btn in (getattr(self, "btn_up", None), getattr(self, "btn_down", None)):
            if btn is not None:
                btn.setEnabled(not readonly)
                btn.setVisible(not readonly)
