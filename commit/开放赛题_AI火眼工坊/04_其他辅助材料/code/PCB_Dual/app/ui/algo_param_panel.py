"""内嵌式算法参数面板：挂在检测页每个缺陷勾选行下方，点开即可直接调参，
不再需要单独弹一个参数窗口。拖动滑杆时会防抖（350ms）自动用当前标准图/
测试图跑一次该算法，实时把 OK/NG 判定 + 缺陷框画出来，不用点“运行检测”
就能看到调参效果。

若参数带有 ``tolerance``（如元件本体检测的低/中/高挡位），面板顶部会显示
「检出容忍度」下拉，一键联动所有挡位参数（与 ``元件本体检测`` 验证台一致）。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.detect.registry import get_adapter_for
from app.detect.scheduler import DetectScheduler
from app.detect.types import DetectRequest
from app.recipe.param_store import ParamStore
from app.ui.widgets import ImageLabel, NoWheelComboBox, ParamSlider, StatusPill, apply_tooltip
from app.utils.image_qt import numpy_to_qpixmap

_PREVIEW_DEBOUNCE_MS = 350

# 与元件本体检测验证台一致：低=更严格，高=更宽松
_TOLERANCE_LEVELS = [("low", "低"), ("mid", "中"), ("high", "高")]
_TOLERANCE_CUSTOM = "自定义"


class AlgorithmParamPanel(QWidget):
    """单个 algorithm_id 的参数编辑区。``template_name_getter`` 延迟获取当前
    模板名，保证用户先改模板名、后展开面板时读到的还是最新模板。
    ``get_std_bgr``/``get_test_bgr`` 用来实时拿当前标准图/测试图做预览检测，
    不传时该面板仅退化为静态调参（不做实时反馈）。"""

    params_changed = Signal()

    def __init__(
        self,
        algorithm_id: str,
        template_name_getter: Callable[[], str | None],
        param_store: ParamStore,
        get_std_bgr: Callable[[], np.ndarray | None] | None = None,
        get_test_bgr: Callable[[], np.ndarray | None] | None = None,
        base_params_getter: Callable[[], dict[str, Any]] | None = None,
        *,
        readonly: bool = False,
        values_provider: Callable[[], tuple[dict[str, Any], str]] | None = None,
        show_preview: bool = True,
        compact: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.algorithm_id = algorithm_id
        self.template_name_getter = template_name_getter
        self.param_store = param_store
        self.get_std_bgr = get_std_bgr
        self.get_test_bgr = get_test_bgr
        self.base_params_getter = base_params_getter
        self.readonly = readonly
        self.values_provider = values_provider
        self.show_preview = show_preview
        self.compact = compact
        self.adapter = get_adapter_for(algorithm_id)
        self.specs = self.adapter.param_specs(algorithm_id) if self.adapter is not None else []
        self.sliders: dict[str, ParamSlider] = {}
        self._tolerance_keys: list[str] = [
            s.key for s in self.specs if getattr(s, "tolerance", None)
        ]
        self._tolerance_combo: NoWheelComboBox | None = None
        self._applying_tolerance = False
        self._loaded = False
        self._save_row_widget: QWidget | None = None
        self.preview_image: ImageLabel | None = None
        self._preview_timer: QTimer | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8 if compact else 26, 2, 4, 8)
        root.setSpacing(6)
        # 避免在窄侧栏里被竖向 Expanding 控件撑破裁切
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        if self.adapter is None:
            lbl = QLabel("该算法尚未接入真实适配器，无可调参数（对照检测将返回 stub OK）。")
            lbl.setProperty("class", "Hint")
            root.addWidget(lbl)
            return
        if not self.specs:
            lbl = QLabel("无可调参数")
            lbl.setProperty("class", "Hint")
            root.addWidget(lbl)
            return

        self.lbl_hit = QLabel("")
        self.lbl_hit.setProperty("class", "Hint")
        root.addWidget(self.lbl_hit)

        if self._tolerance_keys:
            tol_row = QHBoxLayout()
            tol_lbl = QLabel("检出容忍度")
            tol_lbl.setStyleSheet("font-weight: 600;")
            apply_tooltip(
                tol_lbl,
                "与元件本体检测验证台一致：\n"
                "低=更严格，容易报异常；中=默认；高=更宽松，容易判合格。\n"
                "选择挡位会联动下方所有带预设的参数；手动改滑杆后变为「自定义」。",
            )
            tol_row.addWidget(tol_lbl)
            self._tolerance_combo = NoWheelComboBox()
            self._tolerance_combo.addItems([lbl for _, lbl in _TOLERANCE_LEVELS] + [_TOLERANCE_CUSTOM])
            self._tolerance_combo.setCurrentText("中")
            apply_tooltip(
                self._tolerance_combo,
                "低 / 中 / 高 为预设挡位；自定义表示当前参数与任一挡位不完全一致。",
            )
            self._tolerance_combo.currentTextChanged.connect(self._on_tolerance_changed)
            tol_row.addWidget(self._tolerance_combo, 1)
            root.addLayout(tol_row)

        for spec in self.specs:
            slider = ParamSlider(spec)
            slider.changed.connect(self._on_param_changed)
            self.sliders[spec.key] = slider
            root.addWidget(slider)

        save_host = QWidget()
        save_lay = QVBoxLayout(save_host)
        save_lay.setContentsMargins(0, 0, 0, 0)
        save_lay.setSpacing(4)
        scope_row = QHBoxLayout()
        self.combo_scope = QComboBox()
        # 默认优先保存到当前模板专用（改参通常针对当前料号）
        self.combo_scope.addItem("当前模板专用", "template_algorithm")
        self.combo_scope.addItem("算法默认（全局，所有模板生效）", "algorithm_default")
        self.combo_scope.setCurrentIndex(0)
        scope_row.addWidget(QLabel("保存到"))
        scope_row.addWidget(self.combo_scope, 1)
        save_lay.addLayout(scope_row)
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("保存")
        self.btn_reset = QPushButton("恢复默认")
        self.btn_clear = QPushButton("清除层级")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        save_lay.addLayout(btn_row)
        root.addWidget(save_host)
        self._save_row_widget = save_host

        self.preview_pill = StatusPill("")
        self.lbl_preview_msg = QLabel("")
        if show_preview:
            preview_row = QHBoxLayout()
            self.lbl_preview_msg = QLabel(
                "对照预览（参数只读）" if readonly else "调整参数后这里会实时预览检测效果"
            )
            self.lbl_preview_msg.setWordWrap(True)
            self.lbl_preview_msg.setProperty("class", "Hint")
            preview_row.addWidget(self.preview_pill)
            preview_row.addWidget(self.lbl_preview_msg, 1)
            root.addLayout(preview_row)

            self.preview_image = ImageLabel("")
            # 固定预览区高度，避免大图 sizeHint 把侧栏/分割条不断撑大
            preview_h = 140 if compact else 170
            self.preview_image.setMinimumHeight(preview_h)
            self.preview_image.setMaximumHeight(preview_h)
            self.preview_image.setMinimumWidth(0)
            self.preview_image.setText(
                "选择模板并打开测试图后，可在此对照预览"
                if readonly
                else "打开标准图和测试图后，调参即可在此实时预览"
            )
            root.addWidget(self.preview_image)

            self._preview_timer = QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
            self._preview_timer.timeout.connect(self._run_preview)

        self.btn_save.clicked.connect(self._save)
        self.btn_reset.clicked.connect(self._reset_defaults)
        self.btn_clear.clicked.connect(self._clear_scope)
        if readonly:
            self.set_readonly(True)

    def has_params(self) -> bool:
        return self.adapter is not None and bool(self.specs)

    def current_values(self) -> dict[str, Any]:
        return {key: slider.value() for key, slider in self.sliders.items()}

    def set_readonly(self, readonly: bool) -> None:
        self.readonly = readonly
        if self._save_row_widget is not None:
            self._save_row_widget.setVisible(not readonly)
        if self._tolerance_combo is not None:
            self._tolerance_combo.setEnabled(not readonly)
        for slider in self.sliders.values():
            slider.set_readonly(readonly)

    def ensure_loaded(self, *, preview: bool = True) -> None:
        """首次展开时才去解析当前生效值，避免面板刚创建时模板名还没确定。"""
        if not self._loaded:
            self.refresh(preview=preview)
            self._loaded = True
        elif preview:
            self._schedule_preview()

    def refresh(self, *, preview: bool = True) -> None:
        if self.adapter is None or not self.sliders:
            return
        defaults = self.adapter.default_params(self.algorithm_id)
        if self.values_provider is not None:
            provided, hit = self.values_provider()
            resolved = {**defaults, **(provided or {})}
        else:
            template_name = self.template_name_getter()
            resolved, hit = self.param_store.resolve(template_name, None, self.algorithm_id, defaults)
        self.lbl_hit.setText(f"当前生效：{hit}")
        self._applying_tolerance = True
        try:
            for key, slider in self.sliders.items():
                slider.set_value(resolved.get(key, slider.spec.default))
        finally:
            self._applying_tolerance = False
        self._sync_tolerance_combo()
        if preview:
            self._schedule_preview()

    def _reset_defaults(self) -> None:
        if self.readonly or self.adapter is None:
            return
        defaults = self.adapter.default_params(self.algorithm_id)
        self._applying_tolerance = True
        try:
            for key, slider in self.sliders.items():
                slider.set_value(defaults.get(key, slider.spec.default))
        finally:
            self._applying_tolerance = False
        self._sync_tolerance_combo()
        self._schedule_preview()
        self.params_changed.emit()

    def _clear_scope(self) -> None:
        if self.readonly:
            return
        scope = self.combo_scope.currentData()
        template_name = self.template_name_getter() if scope == "template_algorithm" else None
        self.param_store.clear(scope, self.algorithm_id, template_name=template_name)
        self.refresh()

    def _save(self) -> None:
        if self.readonly or self.adapter is None or not self.sliders:
            return
        scope = self.combo_scope.currentData()
        template_name = self.template_name_getter() if scope == "template_algorithm" else None
        if scope == "template_algorithm" and not template_name:
            QMessageBox.information(self, "提示", "请先设置模板名（打开/命名标准图）再保存到“当前模板专用”层级")
            return
        values = {key: slider.value() for key, slider in self.sliders.items()}
        self.param_store.save(scope, self.algorithm_id, values, template_name=template_name)
        self.refresh()

    # ---- 检出容忍度挡位（同步自元件本体检测） ----
    def _tolerance_level_label(self) -> str:
        if not self._tolerance_keys:
            return _TOLERANCE_CUSTOM
        for level_key, level_label in _TOLERANCE_LEVELS:
            matched = True
            for key in self._tolerance_keys:
                slider = self.sliders.get(key)
                if slider is None or not slider.spec.tolerance:
                    matched = False
                    break
                target = slider.spec.tolerance.get(level_key)
                if target is None:
                    matched = False
                    break
                value = slider.value()
                eps = max(float(slider.spec.step), 1e-6) / 2 + 1e-9
                try:
                    if abs(float(value) - float(target)) > eps:
                        matched = False
                        break
                except (TypeError, ValueError):
                    matched = False
                    break
            if matched:
                return level_label
        return _TOLERANCE_CUSTOM

    def _sync_tolerance_combo(self) -> None:
        if self._tolerance_combo is None:
            return
        label = self._tolerance_level_label()
        self._tolerance_combo.blockSignals(True)
        self._tolerance_combo.setCurrentText(label)
        self._tolerance_combo.blockSignals(False)

    def _on_tolerance_changed(self, label: str) -> None:
        if self.readonly or self._applying_tolerance or label == _TOLERANCE_CUSTOM:
            return
        level_key = next((k for k, lbl in _TOLERANCE_LEVELS if lbl == label), None)
        if level_key is None:
            return
        self._applying_tolerance = True
        try:
            for key in self._tolerance_keys:
                slider = self.sliders.get(key)
                if slider is None or not slider.spec.tolerance:
                    continue
                if level_key not in slider.spec.tolerance:
                    continue
                slider.set_value(slider.spec.tolerance[level_key])
        finally:
            self._applying_tolerance = False
        self._schedule_preview()
        self.params_changed.emit()

    def _on_param_changed(self) -> None:
        # 批量 set_value（refresh）时不要触发预览，避免切页/重建面板时连跑算法
        if self._applying_tolerance:
            return
        self._sync_tolerance_combo()
        self._schedule_preview()
        self.params_changed.emit()

    # ---- 实时预览 ----
    def _schedule_preview(self) -> None:
        if not self.show_preview or self._preview_timer is None:
            return
        if not self.has_params() or not self.isVisible():
            return
        self._preview_timer.start()

    def _run_preview(self) -> None:
        if not self.show_preview or self.preview_image is None:
            return
        if not self.has_params() or not self.isVisible():
            return
        std = self.get_std_bgr() if self.get_std_bgr else None
        test = self.get_test_bgr() if self.get_test_bgr else None
        need_std = getattr(self.adapter, "requires_standard", True)
        if test is None or (need_std and std is None):
            self.preview_pill.set_status("")
            tip = "打开测试图后，这里会实时预览调参效果" if not need_std else "打开标准图和测试图后，这里会实时预览调参效果"
            self.lbl_preview_msg.setText(tip)
            self.preview_image.set_image(None)
            return

        values: dict[str, Any] = dict(self.base_params_getter()) if self.base_params_getter else {}
        values.update({key: slider.value() for key, slider in self.sliders.items()})
        req = DetectRequest(
            image_test=test,
            image_std=std if std is not None else test,
            image_std_raw=std if std is not None else test,
            image_test_raw=test,
            template_name=self.template_name_getter(),
            algorithm=self.algorithm_id,
        )
        try:
            result = self.adapter.run(self.algorithm_id, req, values)
        except Exception as exc:  # noqa: BLE001 - 预览异常直接展示，不让面板崩掉
            self.preview_pill.set_status("ERROR")
            self.lbl_preview_msg.setText(f"预览出错：{exc}")
            self.preview_image.set_image(None)
            return

        status = "SKIP" if result.skipped else ("OK" if result.ok else "NG")
        self.preview_pill.set_status(status)
        self.lbl_preview_msg.setText(result.message)
        if result.diff_image is not None and not need_std:
            vis = result.diff_image
        else:
            vis = DetectScheduler.draw_result(test, [result], result.ok or result.skipped)
        self.preview_image.set_image(numpy_to_qpixmap(vis))
