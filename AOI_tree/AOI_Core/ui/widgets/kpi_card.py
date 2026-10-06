"""KpiCard：KPI 指标卡片（标题 + 大数值 + 副文本），白色卡片样式。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from ui.theme import TEXT


class KpiCard(QFrame):
    """单指标卡片：标题（灰）/ 大数值（加粗）/ 副文本（灰小字）。"""

    def __init__(self, title: str = "", value: str = "-",
                 sub_text: str = "", accent: str = TEXT,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setMinimumHeight(96)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self._title = QLabel(title)
        self._title.setProperty("subtext", True)
        lay.addWidget(self._title)

        self._value = QLabel(value)
        font = self._value.font()
        font.setPointSizeF(17)
        font.setBold(True)
        self._value.setFont(font)
        self._accent = accent
        self._value.setStyleSheet(f"color: {accent};")
        lay.addWidget(self._value)

        self._sub = QLabel(sub_text)
        self._sub.setProperty("subtext", True)
        self._sub.setWordWrap(True)
        lay.addWidget(self._sub)
        lay.addStretch(1)

    def set_value(self, value: str, sub_text: str | None = None) -> None:
        """更新大数值，可同步更新副文本。"""
        self._value.setText(value)
        if sub_text is not None:
            self._sub.setText(sub_text)

    def set_accent(self, color: str) -> None:
        """更新数值颜色（如正常绿/异常红）。"""
        self._accent = color
        self._value.setStyleSheet(f"color: {color};")

    def set_title(self, title: str) -> None:
        self._title.setText(title)
