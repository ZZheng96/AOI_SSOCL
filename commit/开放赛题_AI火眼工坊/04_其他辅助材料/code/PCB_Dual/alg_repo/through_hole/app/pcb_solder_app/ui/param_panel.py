"""右侧：按选中缺陷类型展示"判定挡位 + 精选参数滑杆 + 恢复默认"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, ColorDot, ParamSlider, apply_tooltip
from ..metadata import DEFECT_BY_ID, PRESET_CHOICES, PRESET_HINT

# 孔洞/少锡挡位表；挡位切换时刷新未手动覆盖的滑杆显示值。延迟导入 detect。
_VOID_PRESETS: Optional[Dict[str, Dict[str, Any]]] = None
_INSUF_PRESETS: Optional[Dict[str, Dict[str, Any]]] = None


def _preset_tables():
    global _VOID_PRESETS, _INSUF_PRESETS
    if _VOID_PRESETS is None or _INSUF_PRESETS is None:
        from .. import config as app_config
        app_config.ensure_detect_on_path()
        from algorithms.pcb_through_hole import tuning as T  # noqa: WPS433
        from algorithms.pcb_through_hole import config as INS_C  # noqa: WPS433
        _VOID_PRESETS = T.VOID_PRESETS
        _INSUF_PRESETS = INS_C.INSUF_PRESETS
    return _VOID_PRESETS, _INSUF_PRESETS


class ParamPanel(Card):
    config_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("参数", parent)
        self._current_id: Optional[int] = None
        self._advanced: Dict[str, Any] = {}
        self._void_preset = "MED"
        self._insuf_preset = "MED"
        self._building = False

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._host = QWidget()
        self._host.setObjectName("ScrollHost")
        self._hl = QVBoxLayout(self._host)
        self._hl.setContentsMargins(0, 0, 6, 0)
        self._hl.setSpacing(12)
        self._scroll.setWidget(self._host)
        self.add(self._scroll)

        self._placeholder = QLabel("在左侧选择一个缺陷类型以查看/调整其参数")
        self._placeholder.setProperty("class", "Hint")
        self._placeholder.setWordWrap(True)
        self._hl.addWidget(self._placeholder)

        self._param_section: Optional[QWidget] = None
        self._hl.addStretch(1)

    @staticmethod
    def _sep_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("class", "SectionLabel")
        return lbl

    # ---- API ----
    def show_defect(self, defect_id: int):
        self._current_id = defect_id
        self._rebuild_params()

    def set_presets(self, void_preset: str, insuf_preset: str):
        self._void_preset = void_preset or "MED"
        self._insuf_preset = insuf_preset or "MED"
        self._rebuild_params()

    def void_preset(self) -> str:
        return self._void_preset

    def insuf_preset(self) -> str:
        return self._insuf_preset

    def set_advanced(self, advanced: Dict[str, Any]):
        self._advanced = dict(advanced or {})
        self._rebuild_params()

    def advanced(self) -> Dict[str, Any]:
        return dict(self._advanced)

    # ---- internal ----
    def _clear_params(self):
        if self._param_section is not None:
            self._hl.removeWidget(self._param_section)
            self._param_section.setParent(None)
            self._param_section = None

    def _effective_value(self, defect_id: int, key: str, default: Any) -> Any:
        """没有手动覆盖时，滑杆应显示的值：优先取当前挡位表里的值，
        该键不受挡位管理时回退到 ``Param.default``。"""
        if defect_id not in (12, 13):
            return default
        try:
            void_presets, insuf_presets = _preset_tables()
        except Exception:
            return default
        level = self._void_preset if defect_id == 12 else self._insuf_preset
        table = void_presets if defect_id == 12 else insuf_presets
        preset = table.get(level) or {}
        return preset[key] if key in preset else default

    def _rebuild_params(self):
        self._clear_params()
        meta = DEFECT_BY_ID.get(self._current_id) if self._current_id is not None else None
        self._placeholder.setVisible(meta is None)
        if meta is None:
            return

        self._building = True
        section = QFrame()
        sl = QVBoxLayout(section)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(10)

        head = QFrame()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        hl.addWidget(ColorDot(meta.color, 14))
        title = QLabel(meta.name)
        title.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {theme.TEXT};")
        hl.addWidget(title)
        hl.addStretch(1)
        sl.addWidget(head)

        self.preset_combo: Optional[QComboBox] = None
        if meta.preset_key:
            row = QFrame()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(8)
            lbl = QLabel("判定挡位")
            lbl.setStyleSheet(f"color: {theme.TEXT_DIM};")
            rl.addWidget(lbl)
            self.preset_combo = QComboBox()
            self.preset_combo.addItems(PRESET_CHOICES)
            current = self._void_preset if meta.defect_id == 12 else self._insuf_preset
            self.preset_combo.setCurrentText(current or "MED")
            apply_tooltip(self.preset_combo, PRESET_HINT)
            self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
            rl.addWidget(self.preset_combo, 1)
            sl.addWidget(row)

        if meta.params:
            sl.addWidget(self._sep_label("参数"))
            self._sliders = []
            for p in meta.params:
                w = ParamSlider(p)
                if p.key in self._advanced:
                    w.set_value(self._advanced[p.key])
                else:
                    w.set_value(self._effective_value(meta.defect_id, p.key, p.default))
                w.changed.connect(self._make_param_cb(p.key, w))
                sl.addWidget(w)
                self._sliders.append(w)

        btn_reset = QPushButton("恢复默认参数")
        apply_tooltip(btn_reset, "恢复挡位 MED，并清除当前缺陷的手动参数覆盖")
        btn_reset.clicked.connect(self._on_reset_defaults)
        sl.addWidget(btn_reset)

        self._hl.insertWidget(1, section)
        self._param_section = section
        self._building = False

    def _make_param_cb(self, key: str, widget: ParamSlider):
        def cb():
            if self._building:
                return
            self._advanced[key] = widget.value()
            self.config_changed.emit()
        return cb

    def _on_preset_changed(self, value: str):
        if self._building or self._current_id is None:
            return
        if self._current_id == 12:
            self._void_preset = value
        elif self._current_id == 13:
            self._insuf_preset = value
        # 挡位变化后，没有手动覆盖的滑杆需要刷新为新挡位下的实际值。
        self._rebuild_params()
        self.config_changed.emit()

    def _on_reset_defaults(self):
        meta = DEFECT_BY_ID.get(self._current_id) if self._current_id is not None else None
        if meta is None:
            return
        if meta.defect_id == 12:
            self._void_preset = "MED"
        elif meta.defect_id == 13:
            self._insuf_preset = "MED"
        for p in meta.params:
            self._advanced.pop(p.key, None)
        self._rebuild_params()
        self.config_changed.emit()
