"""自定义无边框标题栏：拖动、最小化/最大化/关闭"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from . import theme
from .widgets import ColorDot


class TitleBar(QWidget):
    def __init__(self, window, title="插件焊点缺陷检测", subtitle="Through-Hole Solder Inspection"):
        super().__init__(window)
        self._win = window
        self._drag_pos = None
        self.setObjectName("TitleBar")
        self.setFixedHeight(46)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 8, 0)
        lay.setSpacing(10)

        lay.addWidget(ColorDot(theme.ACCENT, 14))
        title_lbl = QLabel(title)
        title_lbl.setObjectName("AppTitle")
        lay.addWidget(title_lbl)
        sub = QLabel("·  " + subtitle)
        sub.setObjectName("AppSub")
        lay.addWidget(sub)
        lay.addStretch(1)

        self.btn_min = QPushButton("\u2013")
        self.btn_max = QPushButton("\u25A1")
        self.btn_close = QPushButton("\u2715")
        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setProperty("class", "WinBtn")
        self.btn_close.setObjectName("CloseBtn")
        self.btn_min.clicked.connect(self._win.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close.clicked.connect(self._win.close)
        lay.addWidget(self.btn_min)
        lay.addWidget(self.btn_max)
        lay.addWidget(self.btn_close)

    def _toggle_max(self):
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()

    # ---- 拖动窗口 ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self._win.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.LeftButton:
            if self._win.isMaximized():
                self._win.showNormal()
            self._win.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, e):
        self._toggle_max()
