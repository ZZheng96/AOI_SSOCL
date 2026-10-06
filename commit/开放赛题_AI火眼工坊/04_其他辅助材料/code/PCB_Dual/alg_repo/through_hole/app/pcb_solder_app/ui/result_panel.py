"""中下：检测结果（状态 + 耗时 + 缺陷明细表 + 图层显隐）"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, StatusPill
from ..core.models import DefectBox, DetectionResult
from ..metadata import DEFECT_BY_ID

_COLS = ["缺陷类型", "坐标(x,y)", "尺寸(w×h)", "置信度", "判定依据"]

# 判定依据人话模板；原始 reason 仅作单元格 tooltip。
_REASON_TEMPLATES = {
    12: "锡面内存在明显凹陷或暗色区域，面积、形状及深浅综合评分超过判定阈值，判定为孔洞。",
    13: {
        "薄锡": "锡层明显偏薄、覆盖不足，判定为少锡（薄锡型）。",
        "露铜": "存在明显裸铜/发红区域，锡层未完全覆盖焊盘，判定为少锡（露铜型）。",
        "": "锡层覆盖不足或存在裸铜现象，判定为少锡。",
    },
    14: {
        "包锡": "引脚结构仍存在，但锡面凸起、反光明显增多，判定为多锡（包锡型）。",
        "": "锡面面积相比标准明显增大，伴随高亮反光或形态异常，判定为多锡（外扩型）。",
    },
    15: {
        "颜色桥连": "相邻焊点之间出现锡色连通区域，判定为连锡。",
        "缝隙收窄": "相邻焊点之间缝隙相对标准明显收窄或闭合，判定为连锡。",
        "": "相邻焊点被锡桥连，判定为连锡。",
    },
    16: "焊盘中心结构、颜色或亮度与标准图存在显著差异（引脚特征缺失/退化），判定为不出脚。",
}


def _human_reason(defect_id: int, raw_reason: str) -> str:
    tpl = _REASON_TEMPLATES.get(defect_id)
    if tpl is None:
        return "综合多项特征判定为异常。"
    if isinstance(tpl, str):
        return tpl
    for kw, text in tpl.items():
        if kw and kw in raw_reason:
            return text
    return tpl.get("", "综合多项特征判定为异常。")


def _human_review_reason(raw: str) -> str:
    if not raw:
        return "数据不可靠，请人工复检。"
    if "配准超出范围" in raw:
        return "图像位置偏差过大，无法准确比对，请人工复检。"
    if "覆盖率不足" in raw:
        return "有效检测区域覆盖不足，图像可能存在遮挡或严重偏移，请人工复检。"
    return "数据不可靠，请人工复检。"


class ResultPanel(Card):
    locate = Signal(object)              # DefectBox
    visibility_changed = Signal(object)  # set[int] 或 None

    def __init__(self, parent=None):
        super().__init__("检测结果", parent)

        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)

        self.pill = StatusPill()
        tl.addWidget(self.pill)

        self.info_lbl = QLabel("等待检测…")
        self.info_lbl.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.info_lbl.setWordWrap(True)
        tl.addWidget(self.info_lbl, 1)

        self.time_lbl = QLabel("耗时 —")
        self.time_lbl.setStyleSheet(f"color: {theme.TEXT_MUTE};")
        tl.addWidget(self.time_lbl)
        self.add(top)

        self._layer_box = QWidget()
        self._layer_l = QHBoxLayout(self._layer_box)
        self._layer_l.setContentsMargins(0, 0, 0, 0)
        self._layer_l.setSpacing(10)
        lb = QLabel("图层:")
        lb.setStyleSheet(f"color: {theme.TEXT_MUTE};")
        self._layer_l.addWidget(lb)
        self._layer_l.addStretch(1)
        self.add(self._layer_box)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(len(_COLS) - 1, QHeaderView.Stretch)
        for i in range(len(_COLS) - 1):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(120)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.add(self.table)

        self._defects: List[DefectBox] = []
        self._layer_checks: Dict[int, QCheckBox] = {}

    # ---- API ----
    def clear(self):
        self.pill.set_status("")
        self.info_lbl.setText("等待检测…")
        self.time_lbl.setText("耗时 —")
        self.table.setRowCount(0)
        self._clear_layers()
        self._defects = []

    def show_error(self, message: str):
        self.pill.set_status("ERROR")
        self.info_lbl.setText(message)
        self.time_lbl.setText("耗时 —")
        self.table.setRowCount(0)
        self._clear_layers()
        self._defects = []

    def show_result(self, result: DetectionResult):
        self.pill.set_status(result.status)
        if result.status == "REVIEW":
            self.info_lbl.setText(_human_review_reason(result.review_reason))
        elif result.status == "OK":
            self.info_lbl.setText("未发现缺陷")
        elif result.status == "NG":
            self.info_lbl.setText(f"发现 {len(result.defects)} 处缺陷，详见下表")
        else:
            self.info_lbl.setText("—")
        self.time_lbl.setText(f"耗时 {result.cost_ms:.0f} ms")

        self._defects = list(result.defects)
        self.table.setRowCount(0)
        for d in self._defects:
            self._append_row(d)
        self._rebuild_layers()

    def _append_row(self, d: DefectBox):
        meta = DEFECT_BY_ID.get(d.defect_id)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = meta.name if meta else d.label
        human_reason = _human_reason(d.defect_id, d.reason)
        vals = [name, f"{d.x}, {d.y}", f"{d.width}×{d.height}",
                f"{d.confidence:.2f}", human_reason]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            self.table.setItem(row, c, item)
        reason_item = self.table.item(row, len(_COLS) - 1)
        if reason_item is not None and d.reason:
            reason_item.setToolTip(d.reason or "")
        self.table.item(row, 0).setData(Qt.UserRole, d)

    def _on_select(self):
        items = self.table.selectedItems()
        if not items:
            return
        d = self.table.item(items[0].row(), 0).data(Qt.UserRole)
        if isinstance(d, DefectBox):
            self.locate.emit(d)

    def _clear_layers(self):
        for chk in self._layer_checks.values():
            chk.setParent(None)
        self._layer_checks.clear()

    def _rebuild_layers(self):
        self._clear_layers()
        ids: List[int] = []
        for d in self._defects:
            if d.defect_id not in ids:
                ids.append(d.defect_id)
        for did in ids:
            meta = DEFECT_BY_ID.get(did)
            chk = QCheckBox(meta.name if meta else str(did))
            chk.setChecked(True)
            chk.toggled.connect(self._emit_visibility)
            self._layer_l.insertWidget(self._layer_l.count() - 1, chk)
            self._layer_checks[did] = chk

    def _emit_visibility(self):
        if not self._layer_checks:
            self.visibility_changed.emit(None)
            return
        vis = {did for did, chk in self._layer_checks.items() if chk.isChecked()}
        self.visibility_changed.emit(vis)
