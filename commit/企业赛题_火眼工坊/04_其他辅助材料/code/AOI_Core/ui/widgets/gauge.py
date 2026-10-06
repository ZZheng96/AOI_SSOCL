"""ScoreGauge：圆环仪表盘，显示异常分数（0-1）。

超过阈值时圆环与文字变红，下方显示"正常 / 异常"判定。
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.theme import DANGER, PRIMARY, SUCCESS, TEXT_SUB


class ScoreGauge(QWidget):
    """异常分数圆环仪表盘。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._score = 0.0        # 0~1
        self._threshold = 0.5    # 判定阈值
        self.setMinimumSize(150, 170)

    def set_score(self, score: float | None, threshold: float | None = None) -> None:
        """设置分数（自动截断到 0~1）与阈值。"""
        try:
            self._score = max(0.0, min(1.0, float(score or 0.0)))
        except (TypeError, ValueError):
            self._score = 0.0
        if threshold is not None:
            self._threshold = max(0.01, min(1.0, float(threshold)))
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self._threshold = max(0.01, min(1.0, float(threshold)))
        self.update()

    @property
    def is_anomaly(self) -> bool:
        return self._score >= self._threshold

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h - 30) - 16
        rect = QRectF((w - side) / 2, 8, side, side)

        anomaly = self.is_anomaly
        arc_color = QColor(DANGER if anomaly else PRIMARY)

        # 底环（浅灰）
        pen = QPen(QColor("#E5E7EB"), 10)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        # 分数弧（从顶部顺时针）
        pen.setColor(arc_color)
        painter.setPen(pen)
        span = int(-self._score * 360 * 16)
        painter.drawArc(rect, 90 * 16, span)

        # 阈值刻度线
        pen = QPen(QColor(TEXT_SUB), 2)
        painter.setPen(pen)
        import math
        ang = math.radians(90 - self._threshold * 360)
        cx, cy = rect.center().x(), rect.center().y()
        r1, r2 = side / 2 - 12, side / 2 + 2
        painter.drawLine(
            int(cx + r1 * math.cos(ang)), int(cy - r1 * math.sin(ang)),
            int(cx + r2 * math.cos(ang)), int(cy - r2 * math.sin(ang)))

        # 中央分数
        painter.setPen(arc_color)
        font = QFont(self.font())
        font.setPointSizeF(side / 9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, f"{self._score:.3f}")

        # 副标题"异常分数"
        painter.setPen(QColor(TEXT_SUB))
        font.setPointSizeF(8)
        font.setBold(False)
        painter.setFont(font)
        sub_rect = QRectF(rect.x(), rect.center().y() + side / 8,
                          rect.width(), 20)
        painter.drawText(sub_rect, Qt.AlignHCenter | Qt.AlignTop, "异常分数")

        # 下方判定文字
        font.setPointSizeF(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(DANGER if anomaly else SUCCESS))
        text_rect = QRectF(0, h - 26, w, 24)
        painter.drawText(text_rect, Qt.AlignCenter,
                         "异 常" if anomaly else "正 常")
        painter.end()
