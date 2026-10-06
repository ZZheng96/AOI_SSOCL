"""L2 缺陷表：类型、位置、数量；点行高亮并放大对应框。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout

from app.detect.scheduler import merge_defect_overlays
from app.detect.types import DetectSummary
from app.ui import theme


@dataclass
class OverlayDefect:
    x: int
    y: int
    w: int
    h: int
    label: str
    color: str = ""
    index: int = 0
    item_id: str = ""
    score: float | None = None
    roi_id: str | None = None


class DefectTable(QWidget):
    defect_selected = Signal(int)  # overlay index; -1 = 清除

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.overlays: list[OverlayDefect] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "类型", "位置", "尺寸"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table)
        self.table.currentCellChanged.connect(self._on_row)

    def clear(self) -> None:
        self.overlays = []
        self.table.setRowCount(0)

    def bind_summary(self, summary: DetectSummary | None, *, error: str = "") -> list[OverlayDefect]:
        self.table.blockSignals(True)
        self.clear()
        if error:
            self.table.setRowCount(1)
            item = QTableWidgetItem(error)
            item.setForeground(QColor(theme.ERROR))
            self.table.setItem(0, 1, item)
            self.table.blockSignals(False)
            return []
        if summary is None:
            self.table.blockSignals(False)
            return []
        merged = merge_defect_overlays(summary.results)
        self.overlays = [
            OverlayDefect(
                x=int(m["x"]),
                y=int(m["y"]),
                w=int(m["w"]),
                h=int(m["h"]),
                label=str(m.get("label") or ""),
                color=str(m.get("color") or theme.NG),
                index=int(m.get("index") or i + 1),
                item_id=str(m.get("item_id") or ""),
                score=m.get("score"),
                roi_id=m.get("roi_id"),
            )
            for i, m in enumerate(merged)
        ]
        if not self.overlays:
            rows = []
            for r in summary.results:
                status = r.status or ("SKIP" if r.skipped else ("OK" if r.ok else "NG"))
                if status == "OK":
                    continue
                rows.append((status, r.display_name or r.algorithm, r.message or r.error_message or r.skip_reason or ""))
            self.table.setRowCount(len(rows))
            for i, (status, name, msg) in enumerate(rows):
                self.table.setItem(i, 0, QTableWidgetItem(status))
                self.table.setItem(i, 1, QTableWidgetItem(name))
                self.table.setItem(i, 2, QTableWidgetItem(msg))
            self.table.blockSignals(False)
            return []

        self.table.setRowCount(len(self.overlays))
        for i, d in enumerate(self.overlays):
            num = QTableWidgetItem(str(d.index))
            typ = QTableWidgetItem(d.label)
            pos = QTableWidgetItem(f"{d.x},{d.y}")
            size = QTableWidgetItem(f"{d.w}×{d.h}")
            color = QColor(d.color or theme.NG)
            num.setForeground(color)
            typ.setForeground(color)
            for col, item in enumerate((num, typ, pos, size)):
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.table.setItem(i, col, item)
        self.table.blockSignals(False)
        return list(self.overlays)

    def select_index(self, index: int) -> None:
        if 0 <= index < self.table.rowCount():
            self.table.blockSignals(True)
            self.table.selectRow(index)
            self.table.blockSignals(False)

    def _on_row(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0 or row >= len(self.overlays):
            self.defect_selected.emit(-1)
            return
        self.defect_selected.emit(row)
