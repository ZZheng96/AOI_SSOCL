"""无边框窗口边缘拖拽缩放：在窗口四周放置透明感应带。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRect, QEvent, QObject
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget


class _EdgeBand(QWidget):
    """窗口边缘/角落的透明拖拽带。"""

    _CURSORS = {
        "left": Qt.SizeHorCursor,
        "right": Qt.SizeHorCursor,
        "top": Qt.SizeVerCursor,
        "bottom": Qt.SizeVerCursor,
        "tl": Qt.SizeFDiagCursor,
        "tr": Qt.SizeBDiagCursor,
        "bl": Qt.SizeBDiagCursor,
        "br": Qt.SizeFDiagCursor,
    }

    def __init__(self, window: QWidget, edge: str, thickness: int = 6):
        super().__init__(window)
        self._win = window
        self._edge = edge
        self._thickness = thickness
        self._active = False
        self._origin = None
        self._geom = None
        self.setCursor(QCursor(self._CURSORS[edge]))
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: transparent;")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._active = True
            self._origin = e.globalPosition().toPoint()
            self._geom = self._win.geometry()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if not self._active or self._origin is None or self._geom is None:
            super().mouseMoveEvent(e)
            return
        delta = e.globalPosition().toPoint() - self._origin
        g = QRect(self._geom)
        min_w = self._win.minimumWidth()
        min_h = self._win.minimumHeight()

        if "left" in self._edge:
            g.setLeft(g.left() + delta.x())
            if g.width() < min_w:
                g.setLeft(g.right() - min_w)
        if "right" in self._edge:
            g.setRight(g.right() + delta.x())
            if g.width() < min_w:
                g.setRight(g.left() + min_w)
        if "top" in self._edge:
            g.setTop(g.top() + delta.y())
            if g.height() < min_h:
                g.setTop(g.bottom() - min_h)
        if "bottom" in self._edge:
            g.setBottom(g.bottom() + delta.y())
            if g.height() < min_h:
                g.setBottom(g.top() + min_h)

        self._win.setGeometry(g)
        e.accept()

    def mouseReleaseEvent(self, e):
        self._active = False
        self._origin = None
        self._geom = None
        super().mouseReleaseEvent(e)


class FramelessResizeGrips(QObject):
    """管理窗口八向边缘缩放带。"""

    def __init__(self, window: QWidget, thickness: int = 6):
        super().__init__(window)
        self._win = window
        self._t = thickness
        self._bands: list[_EdgeBand] = []
        for edge in ("left", "right", "top", "bottom", "tl", "tr", "bl", "br"):
            band = _EdgeBand(window, edge, thickness)
            band.raise_()
            self._bands.append(band)
        window.installEventFilter(self)
        self.relayout()

    def relayout(self):
        w, h = self._win.width(), self._win.height()
        t, c = self._t, max(self._t + 2, 10)
        for band in self._bands:
            if band._edge == "left":
                band.setGeometry(0, c, t, h - 2 * c)
            elif band._edge == "right":
                band.setGeometry(w - t, c, t, h - 2 * c)
            elif band._edge == "top":
                band.setGeometry(c, 0, w - 2 * c, t)
            elif band._edge == "bottom":
                band.setGeometry(c, h - t, w - 2 * c, t)
            elif band._edge == "tl":
                band.setGeometry(0, 0, c, c)
            elif band._edge == "tr":
                band.setGeometry(w - c, 0, c, c)
            elif band._edge == "bl":
                band.setGeometry(0, h - c, c, c)
            elif band._edge == "br":
                band.setGeometry(w - c, h - c, c, c)
            band.raise_()

    def eventFilter(self, obj, event):
        if obj is self._win and event.type() == QEvent.Resize:
            self.relayout()
        return False
