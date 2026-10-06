"""pyqtgraph 图表封装。

- LineChart：多序列折线图（图例 + 网格 + 日期/类目横轴）
- BarChart：分组柱状图（可用于评估对比）
- LatencyChart：延迟柱状图 + 200ms 预算参考线
- DonutChart：环形占比图（QPainter 自绘，不依赖 QtCharts）
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.theme import DANGER, PRIMARY, SUCCESS, TEXT, TEXT_SUB, WARNING

# 序列配色
SERIES_COLORS = ["#2F6FED", "#22C55E", "#F59E0B", "#EF4444",
                 "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", TEXT_SUB)


def _style_axis(plot: pg.PlotWidget, x_labels: list[str] | None = None) -> None:
    """统一坐标轴样式：浅灰网格、无右边框。"""
    plot.showGrid(x=True, y=True, alpha=0.25)
    for name in ("left", "bottom"):
        ax = plot.getAxis(name)
        ax.setPen(pg.mkPen(color=TEXT_SUB))
        ax.setTextPen(pg.mkPen(color=TEXT_SUB))
    plot.getPlotItem().hideAxis("top")
    plot.getPlotItem().hideAxis("right")
    if x_labels is not None:
        # 刻度抽稀：标签过多（如近30天日期）时按间隔抽样，
        # 否则 30 个刻度全部画上会重叠看不清（前端反馈 v5-4）。
        n = len(x_labels)
        max_ticks = 8
        step = max(1, (n + max_ticks - 1) // max_ticks)
        ticks = [[(i, x_labels[i]) for i in range(0, n, step)]]
        plot.getAxis("bottom").setTicks(ticks)


def _empty_hint(plot: pg.PlotWidget, text: str = "暂无数据") -> None:
    """空数据态：居中灰字提示，避免空白图表让人误以为没加载出来。"""
    plot.setXRange(0, 1, padding=0)
    plot.setYRange(0, 1, padding=0)
    t = pg.TextItem(text, color=TEXT_SUB, anchor=(0.5, 0.5))
    t.setPos(0.5, 0.5)
    plot.addItem(t)


def _reset_view(plot: pg.PlotWidget) -> None:
    """重新开启自动缩放：空数据态把范围钉在 0-1（手动），
    后续恢复真实数据前需先复位，否则新数据可能落在固定范围外看不见。"""
    plot.getPlotItem().getViewBox().enableAutoRange()


class LineChart(QWidget):
    """多序列折线图：add_series(name, x, y, color?)，自动图例 + 网格。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(8, 8), pen=pg.mkPen(color="#E5E7EB"),
                            brush=pg.mkBrush(255, 255, 255, 220),
                            labelTextColor=TEXT)
        lay.addWidget(self.plot)
        self.setMinimumHeight(200)

    def clear(self) -> None:
        self.plot.clear()

    def add_series(self, name: str, x: list, y: list,
                   color: str | None = None, width: int = 2,
                   symbol: bool = True) -> None:
        idx = len(self.plot.listDataItems())
        c = color or SERIES_COLORS[idx % len(SERIES_COLORS)]
        pen = pg.mkPen(color=c, width=width)
        kwargs = {}
        if symbol:
            kwargs = dict(symbol="o", symbolSize=6,
                          symbolBrush=pg.mkBrush(c), symbolPen=pg.mkPen(c))
        self.plot.plot(x, y, pen=pen, name=name, **kwargs)

    def set_labels(self, x_labels: list[str] | None = None,
                   y_title: str = "") -> None:
        _reset_view(self.plot)
        _style_axis(self.plot, x_labels)
        if y_title:
            self.plot.setLabel("left", y_title, color=TEXT_SUB)
        if not self.plot.listDataItems() and not x_labels:
            _empty_hint(self.plot)


class BarChart(QWidget):
    """分组柱状图：set_data(categories, {序列名: [值...]}, colors?)。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(8, 8), pen=pg.mkPen(color="#E5E7EB"),
                            brush=pg.mkBrush(255, 255, 255, 220),
                            labelTextColor=TEXT)
        lay.addWidget(self.plot)
        self.setMinimumHeight(200)

    def set_data(self, categories: list[str], series: dict,
                 colors: dict | None = None) -> None:
        """categories: 横轴类目；series: {序列名: [各类目值]}。"""
        self.plot.clear()
        _reset_view(self.plot)
        names = list(series.keys())
        n_cat = len(categories)
        if n_cat == 0:
            # 空数据态（如帕累托无复核确认缺陷）：提示而非空白
            _style_axis(self.plot, None)
            _empty_hint(self.plot)
            return
        n_ser = max(1, len(names))
        width = 0.8 / n_ser
        for si, name in enumerate(names):
            vals = series[name]
            xs = [i - 0.4 + width * (si + 0.5) for i in range(n_cat)]
            color = (colors or {}).get(
                name, SERIES_COLORS[si % len(SERIES_COLORS)])
            bg = pg.BarGraphItem(x=xs, height=vals, width=width * 0.9,
                                 brush=pg.mkBrush(color),
                                 pen=pg.mkPen(None), name=name)
            self.plot.addItem(bg)
        _style_axis(self.plot, categories)
        self.plot.setXRange(-0.6, n_cat - 0.4)


class LatencyChart(QWidget):
    """延迟柱状图 + 预算参考线（默认 200ms 红色虚线）。"""

    def __init__(self, budget_ms: float = 200.0,
                 parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        lay.addWidget(self.plot)
        self._budget = budget_ms
        self._ref_line = pg.InfiniteLine(
            pos=budget_ms, angle=0, movable=False,
            pen=pg.mkPen(DANGER, width=2, style=Qt.DashLine),
            label=f"预算 {budget_ms:.0f}ms",
            labelOpts={"color": DANGER, "position": 0.95})
        self.plot.addItem(self._ref_line)
        self.setMinimumHeight(180)

    def set_budget(self, budget_ms: float) -> None:
        self._budget = budget_ms
        self._ref_line.setPos(budget_ms)
        self._ref_line.label.setText(f"预算 {budget_ms:.0f}ms")

    def set_values(self, values: list[float],
                   labels: list[str] | None = None) -> None:
        """柱状展示各样本延迟，超预算的柱子标红。"""
        self.plot.clear()
        _reset_view(self.plot)
        self.plot.addItem(self._ref_line)
        if not values:
            _style_axis(self.plot, labels)
            _empty_hint(self.plot, "暂无延迟数据")
            return
        brushes = [pg.mkBrush(DANGER if v > self._budget else PRIMARY)
                   for v in values]
        bg = pg.BarGraphItem(x=list(range(len(values))), height=values,
                             width=0.7, brushes=brushes, pen=pg.mkPen(None))
        self.plot.addItem(bg)
        _style_axis(self.plot, labels)
        self.plot.setLabel("left", "延迟 (ms)", color=TEXT_SUB)

    def set_series(self, values: list[float],
                   labels: list[str] | None = None) -> None:
        """折线展示延迟趋势（用于统计页近 14 天平均延迟）。"""
        self.plot.clear()
        _reset_view(self.plot)
        self.plot.addItem(self._ref_line)
        if values:
            pen = pg.mkPen(color=PRIMARY, width=2)
            self.plot.plot(list(range(len(values))), values, pen=pen,
                           symbol="o", symbolSize=6,
                           symbolBrush=pg.mkBrush(PRIMARY))
        else:
            _empty_hint(self.plot, "暂无延迟数据")
        _style_axis(self.plot, labels)
        self.plot.setLabel("left", "延迟 (ms)", color=TEXT_SUB)


class DonutChart(QWidget):
    """环形占比图：set_data({标签: 值}, colors?)，QPainter 自绘。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._data: dict[str, float] = {}
        self._colors: dict[str, str] = {}
        self.setMinimumSize(190, 190)

    def set_data(self, data: dict, colors: dict | None = None) -> None:
        self._data = {k: max(0.0, float(v)) for k, v in (data or {}).items()}
        keys = list(self._data.keys())
        self._colors = dict(colors or {})
        for i, k in enumerate(keys):
            self._colors.setdefault(k, SERIES_COLORS[i % len(SERIES_COLORS)])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        legend_w = 92
        side = min(w - legend_w - 12, h - 12)
        rect = QRectF(6, (h - side) / 2, side, side)

        total = sum(self._data.values())
        if total <= 0:
            # 空数据态：灰环 + 提示
            pen = QPen(QColor("#E5E7EB"), 14)
            painter.setPen(pen)
            painter.drawArc(rect.adjusted(7, 7, -7, -7), 0, 360 * 16)
            painter.setPen(QColor(TEXT_SUB))
            painter.drawText(rect, Qt.AlignCenter, "暂无数据")
            painter.end()
            return

        start = 90 * 16
        inner = rect.adjusted(7, 7, -7, -7)
        for name, val in self._data.items():
            if val <= 0:
                continue
            span = int(-val / total * 360 * 16)
            pen = QPen(QColor(self._colors[name]), 14)
            pen.setCapStyle(Qt.FlatCap)
            painter.setPen(pen)
            painter.drawArc(inner, start, span)
            start += span

        # 中心总数
        painter.setPen(QColor(TEXT))
        font = QFont(self.font())
        font.setPointSizeF(13)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, str(int(total)))

        # 图例
        font.setPointSizeF(8.5)
        font.setBold(False)
        painter.setFont(font)
        ly = rect.top() + 4
        for name, val in self._data.items():
            y = int(ly)
            painter.setPen(QPen(QColor(self._colors[name]), 10))
            painter.drawPoint(int(rect.right() + 16), int(y + 6))
            painter.setPen(QColor(TEXT))
            pct = val / total * 100
            painter.drawText(int(rect.right() + 26), int(y + 11),
                             f"{name} {pct:.0f}%")
            ly += 22
        painter.end()
