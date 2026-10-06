"""右侧：数据驱动的参数面板（判据开关 + 阈值滑杆）。

按当前选中的瑕疵类型，依据 ``metadata`` 自动生成"判据多选 + 参数"控件；
每个瑕疵的配置独立保存在 ``self._state[code]``，随时可取用于检测。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from . import theme
from .widgets import Card, ColorDot, _apply_tooltip
from .params_state import make_default_state
from ..metadata import DEFECT_BY_CODE, DefectMeta, Param

_TOLERANCE_LEVELS = [("low", "低"), ("mid", "中"), ("high", "高")]
_TOLERANCE_CUSTOM = "自定义"


class ParamPanel(Card):
    config_changed = Signal(str)   # code

    def __init__(self, parent=None):
        super().__init__("检测参数", parent)
        self._state: Dict[str, Dict[str, Any]] = make_default_state()
        self._current: Optional[str] = None
        self._building = False
        self._advanced_mode = False
        self._tolerance_combo: Optional[QComboBox] = None
        self._tolerance_widgets: Dict[str, "ParamSlider"] = {}

        adv_row = QWidget()
        adv_l = QHBoxLayout(adv_row)
        adv_l.setContentsMargins(0, 0, 0, 0)
        adv_l.setSpacing(6)
        self._advanced_chk = QCheckBox("高级模式（显示全部参数）")
        self._advanced_chk.setToolTip(
            "开启后显示更多底层技术参数，供工程师精细调试；日常使用无需打开")
        self._advanced_chk.toggled.connect(self._on_advanced_toggled)
        adv_l.addWidget(self._advanced_chk)
        adv_l.addStretch(1)
        self.add(adv_row)

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

        self._placeholder = QLabel("选择左侧某个瑕疵类型以配置其判据与参数")
        self._placeholder.setProperty("class", "Hint")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._hl.addWidget(self._placeholder)
        self._hl.addStretch(1)

    # ------------------------------------------------------------------ API --
    def show_defect(self, code: str):
        self._current = code
        self._rebuild()

    def get_config(self, code: str) -> Dict[str, Any]:
        return dict(self._state.get(code, {}))

    def all_state(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self._state.items()}

    def reset_defaults(self, code: Optional[str] = None) -> bool:
        """将指定瑕疵（默认当前查看项）的判据与参数恢复为元数据默认值。"""
        target = code or self._current
        if not target:
            return False
        meta = DEFECT_BY_CODE.get(target)
        if meta is None:
            return False
        self._state[target] = meta.default_config()
        if target == self._current:
            self._rebuild()
        self.config_changed.emit(target)
        return True

    # -------------------------------------------------------------- internal --
    def _clear(self):
        while self._hl.count():
            item = self._hl.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _rebuild(self):
        self._clear()
        self._tolerance_combo = None
        self._tolerance_widgets = {}
        meta = DEFECT_BY_CODE.get(self._current or "")
        if meta is None:
            self._hl.addWidget(self._placeholder)
            self._placeholder.setVisible(True)
            self._hl.addStretch(1)
            return

        self._building = True
        cfg = self._state.setdefault(meta.code, meta.default_config())

        # 标题
        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        hl.addWidget(ColorDot(meta.color, 14))
        t = QLabel(meta.name)
        t.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {theme.TEXT};")
        hl.addWidget(t)
        hl.addStretch(1)
        btn_reset = QPushButton("恢复默认")
        btn_reset.setToolTip("将当前瑕疵类型的判据与参数一键恢复为默认值")
        btn_reset.clicked.connect(self._on_reset_defaults)
        hl.addWidget(btn_reset)
        self._hl.addWidget(head)
        desc = QLabel(meta.summary)
        desc.setProperty("class", "Hint")
        desc.setWordWrap(True)
        self._hl.addWidget(desc)

        # 判据（多选）：show_judges=False 时固定生效，界面不展示开关
        if meta.judges and meta.show_judges:
            gb = QGroupBox("判据（多选，投票）")
            gl = QVBoxLayout(gb)
            gl.setSpacing(6)
            for j in meta.judges:
                chk = QCheckBox(j.label)
                tip = j.hint or j.key
                _apply_tooltip(chk, tip)
                chk.setChecked(bool(cfg.get(j.key, j.default)))
                chk.toggled.connect(self._make_judge_cb(meta.code, j.key))
                gl.addWidget(chk)
            self._hl.addWidget(gb)

        # 检出容忍度：低/中/高 一键联动所有带 tolerance 的参数
        tol_params = [p for p in meta.params if p.tolerance]
        if tol_params:
            tol_row = QWidget()
            tl = QHBoxLayout(tol_row)
            tl.setContentsMargins(0, 0, 0, 0)
            tl.setSpacing(8)
            lbl = QLabel("检出容忍度")
            lbl.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
            _apply_tooltip(lbl, "低=更严格，容易报异常；高=更宽松，容易判合格；中=默认")
            tl.addWidget(lbl)
            combo = QComboBox()
            combo.addItems([label for _, label in _TOLERANCE_LEVELS] + [_TOLERANCE_CUSTOM])
            combo.setCurrentText(self._tolerance_level_label(tol_params, cfg))
            combo.currentTextChanged.connect(self._make_tolerance_cb(meta.code, tol_params))
            tl.addWidget(combo)
            tl.addStretch(1)
            self._hl.addWidget(tol_row)
            self._tolerance_combo = combo

        # 参数（滑杆）：hidden=True 的彻底不显示；advanced=True 的仅在高级模式下显示。
        # 两者都不影响其 default 值参与检测（相当于固定值）。
        visible_params = [
            p for p in meta.params
            if not p.hidden and (self._advanced_mode or not p.advanced)
        ]
        if visible_params:
            gb2 = QGroupBox("阈值参数")
            gl2 = QVBoxLayout(gb2)
            gl2.setSpacing(8)
            from .widgets import ParamSlider
            for prm in visible_params:
                w = ParamSlider(prm)
                if prm.key in cfg:
                    w.set_value(cfg[prm.key])
                w.changed.connect(self._make_param_cb(meta.code, prm.key, w))
                gl2.addWidget(w)
                if prm.tolerance:
                    self._tolerance_widgets[prm.key] = w
            self._hl.addWidget(gb2)

        self._hl.addStretch(1)
        self._building = False

    @staticmethod
    def _tolerance_level_label(tol_params: List[Param], cfg: Dict[str, Any]) -> str:
        for level_key, level_label in _TOLERANCE_LEVELS:
            matched = True
            for p in tol_params:
                target = p.tolerance.get(level_key)
                value = cfg.get(p.key, p.default)
                eps = max(float(p.step), 1e-6) / 2 + 1e-9
                if abs(float(value) - float(target)) > eps:
                    matched = False
                    break
            if matched:
                return level_label
        return _TOLERANCE_CUSTOM

    def _make_judge_cb(self, code: str, key: str):
        def cb(checked: bool):
            if self._building:
                return
            self._state.setdefault(code, {})[key] = bool(checked)
            self.config_changed.emit(code)
        return cb

    def _make_param_cb(self, code: str, key: str, widget):
        def cb():
            if self._building:
                return
            self._state.setdefault(code, {})[key] = widget.value()
            if self._tolerance_combo is not None and key in self._tolerance_widgets:
                meta = DEFECT_BY_CODE.get(code)
                tol_params = [p for p in meta.params if p.tolerance] if meta else []
                cfg = self._state.get(code, {})
                self._tolerance_combo.blockSignals(True)
                self._tolerance_combo.setCurrentText(
                    self._tolerance_level_label(tol_params, cfg))
                self._tolerance_combo.blockSignals(False)
            self.config_changed.emit(code)
        return cb

    def _make_tolerance_cb(self, code: str, tol_params: List[Param]):
        def cb(label: str):
            if self._building or label == _TOLERANCE_CUSTOM:
                return
            level_key = next((k for k, lbl in _TOLERANCE_LEVELS if lbl == label), None)
            if level_key is None:
                return
            cfg = self._state.setdefault(code, {})
            for p in tol_params:
                cfg[p.key] = p.tolerance[level_key]
                w = self._tolerance_widgets.get(p.key)
                if w is not None:
                    w.set_value(cfg[p.key])
            self.config_changed.emit(code)
        return cb

    def _on_advanced_toggled(self, checked: bool):
        self._advanced_mode = checked
        if self._current:
            self._rebuild()

    def _on_reset_defaults(self):
        if self._current:
            self.reset_defaults(self._current)
