"""模板选择组件：列表 / 下拉，显示名称、版本、类别、状态。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.detect.catalog import get_catalog
from app.template.model import InspectionTemplate
from app.template.regions import count_regions, region_sets_from_calibration
from app.template.store import TemplateStore
from app.ui import theme


class TemplatePicker(QWidget):
    """模板下拉选择。默认仅 published；算法调试可设 published_only=False。"""

    template_changed = Signal(object)  # InspectionTemplate | None

    def __init__(
        self,
        store: TemplateStore | None = None,
        parent=None,
        *,
        published_only: bool = True,
        label: str = "检测模板",
        empty_text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store or TemplateStore()
        self.published_only = published_only
        self._empty_text = empty_text or (
            "（请选择已发布模板）" if published_only else "（请选择模板）"
        )
        self._templates: list[InspectionTemplate] = []

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        lay.addWidget(self.combo, 1)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setProperty("class", "Hint")
        lay.addWidget(self.lbl_meta)

        self.combo.currentIndexChanged.connect(self._on_changed)

    def refresh(self, *, keep_id: str | None = None) -> None:
        current = keep_id or (self.current_template().id if self.current_template() else None)
        self._templates = self.store.list_templates(
            status="published" if self.published_only else None
        )
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(self._empty_text, None)
        for tpl in self._templates:
            mark = "已发布" if tpl.status == "published" else "草稿"
            text = f"{tpl.display_name}  ·  v{tpl.version}"
            if not self.published_only:
                text += f"  ·  {mark}"
            if tpl.category:
                text += f"  ·  {tpl.category}"
            self.combo.addItem(text, tpl.id)
        self.combo.blockSignals(False)
        if current:
            idx = self.combo.findData(current)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
                self._on_changed(idx)
                return
        self.combo.setCurrentIndex(0)
        self._on_changed(0)

    def current_template(self) -> InspectionTemplate | None:
        tid = self.combo.currentData()
        if not tid:
            return None
        return self.store.load(str(tid))

    def _on_changed(self, _index: int) -> None:
        tpl = self.current_template()
        if tpl is None:
            self.lbl_meta.setText("")
            self.template_changed.emit(None)
            return
        algs = [a for a in tpl.enabled_algorithm_ids() if get_catalog().is_job_item(a)]
        std_name = tpl.standard_image.display_name
        sets = tpl.region_sets or region_sets_from_calibration(tpl.calibration)
        self.lbl_meta.setText(f"检测项 {len(algs)} · 区域 {count_regions(sets)} · {std_name}")
        self.template_changed.emit(tpl)


class TemplateListPanel(QWidget):
    """建模页左侧模板列表。"""

    selection_changed = Signal(object)  # InspectionTemplate | None

    def __init__(self, store: TemplateStore | None = None, parent=None) -> None:
        super().__init__(parent)
        self.store = store or TemplateStore()
        self._templates: list[InspectionTemplate] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        title = QLabel("模板库")
        title.setProperty("class", "PanelTitle")
        root.addWidget(title)
        self.list = QListWidget()
        root.addWidget(self.list, 1)
        self.list.currentRowChanged.connect(self._on_row)

    def refresh(self, *, keep_id: str | None = None, reload_selection: bool = True) -> None:
        """刷新列表。

        reload_selection=False：只更新列表外观与当前行高亮，不发射 selection_changed
        （用于从其它页签切回时，避免重复重建参数面板/触发预览）。
        """
        current = keep_id
        if current is None and 0 <= self.list.currentRow() < len(self._templates):
            current = self._templates[self.list.currentRow()].id
        prev_id = current
        self._templates = self.store.list_templates()
        self.list.blockSignals(True)
        self.list.clear()
        select_row = 0
        for i, tpl in enumerate(self._templates):
            mark = "●" if tpl.status == "published" else "○"
            color_hint = "已发布" if tpl.status == "published" else "草稿"
            item = QListWidgetItem(f"{mark} {tpl.display_name}  [{color_hint}]  v{tpl.version}")
            std = tpl.standard_image.display_name
            n_items = len([a for a in tpl.enabled_algorithm_ids() if get_catalog().is_job_item(a)])
            sets = tpl.region_sets or region_sets_from_calibration(tpl.calibration)
            tip = (
                f"ID: {tpl.id}\n标准图: {std}\n"
                f"检测项: {n_items} · 区域: {count_regions(sets)}\n更新: {tpl.updated_at}"
            )
            item.setToolTip(tip)
            if tpl.status == "published":
                item.setForeground(QColor(theme.OK))
            self.list.addItem(item)
            if current and tpl.id == current:
                select_row = i
        if self._templates:
            self.list.setCurrentRow(select_row)
        self.list.blockSignals(False)
        if not self._templates:
            if reload_selection:
                self.selection_changed.emit(None)
            return
        new_id = self._templates[select_row].id
        if reload_selection:
            # 只通过信号通知一次（勿再手动 + setCurrentRow 各调一次）
            self._on_row(select_row)
        elif new_id != prev_id:
            # 原选中项已不存在，必须重绑
            self._on_row(select_row)

    def current_template(self) -> InspectionTemplate | None:
        row = self.list.currentRow()
        if row < 0 or row >= len(self._templates):
            return None
        return self.store.load(self._templates[row].id)

    def _on_row(self, row: int) -> None:
        if row < 0 or row >= len(self._templates):
            self.selection_changed.emit(None)
            return
        self.selection_changed.emit(self.store.load(self._templates[row].id))


def Path_name(path: str) -> str:
    from pathlib import Path

    return Path(path).name if path else "（未关联）"
