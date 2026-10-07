"""模板建模页：新建/编辑/发布检测配方。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.calibration.body_roi import get_body_rois, set_body_rois
from app.calibration.gold_seed import get_gold_seed, set_gold_seed
from app.calibration.shift_base import get_shift_bases, set_shift_bases
from app.calibration.smt_pads import (
    get_smt_pads,
    get_smt_rim,
    get_smt_toe,
    set_smt_pads,
    set_smt_rim,
    set_smt_toe,
)
from app.calibration.th_roi import get_th_roi, set_th_pads
from app.detect.catalog import get_catalog
from app.detect.registry import get_adapter_for
from app.inspect.service import InspectService
from app.recipe.param_store import ParamStore
from app.result.store import ResultStore
from app.template.model import InspectionTemplate, TemplateAlgorithm
from app.template.store import TemplateStore
from app.ui import theme
from app.ui.algo_param_panel import AlgorithmParamPanel
from app.ui.image_canvas import ImageCanvas, LayerConfig
from app.ui.template_picker import TemplateListPanel
from app.ui.widgets import Card, StatusPill, ToolbarStrip
from app.utils.cv_io import imread_unicode


def _all_algo_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    groups = []
    for _gid, name, items in get_catalog().groups(job_only=True):
        groups.append((name, [(m.id, m.display_name) for m in items]))
    return groups


_CALIB_NONE = "（先勾选检测项）"


def _calib_kind_for(algorithm_id: str) -> str | None:
    kind = get_catalog().region_kind(algorithm_id)
    return None if not kind or kind == "none" else kind


def _major_group_of(alg_id: str) -> str:
    return get_catalog().group_name(alg_id)


def _rects_to_shapes(rects: list[tuple[int, int, int, int]]) -> list[dict]:
    return [{"shape": "rect", "x": x, "y": y, "w": w, "h": h} for x, y, w, h in rects]


def _shapes_to_rects(shapes: list[dict]) -> list[tuple[float, float, float, float]]:
    out = []
    for s in shapes:
        if s.get("shape") == "rect":
            out.append((float(s["x"]), float(s["y"]), float(s["w"]), float(s["h"])))
    return out


class TemplateStudioPage(QWidget):
    """工程师：以标准图为基本单位建模（标准图→模板→检测项→检测区域）。"""

    request_open_inspect = Signal(str)
    request_preprocess = Signal()

    def __init__(
        self,
        template_store: TemplateStore | None = None,
        param_store: ParamStore | None = None,
        inspect_service: InspectService | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.template_store = template_store or TemplateStore()
        self.param_store = param_store or ParamStore()
        self.inspect_service = inspect_service or InspectService(
            result_store=ResultStore(),
            template_store=self.template_store,
        )
        self.current: InspectionTemplate | None = None
        self.std_bgr = None
        self.preview_test_bgr = None
        self.preview_test_path = ""
        self.algo_checks: dict[str, QCheckBox] = {}
        self._algo_groups: dict[str, list[tuple[str, str]]] = {}
        self._algo_label_by_id: dict[str, str] = {}
        self._tool_buttons: dict[str, QToolButton] = {}
        self._suppress_shape_save = False
        self._param_panels: dict[str, AlgorithmParamPanel] = {}
        self._suppress_algo_toggle = False
        self._algo_hint = ""
        self._group_hosts: dict[str, QWidget] = {}
        self._build()
        self.list_panel.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)

        toolbar = ToolbarStrip()
        self.btn_std = QPushButton("选择标准图新建")
        self.btn_std.setObjectName("Primary")
        self.btn_std.setToolTip("每张标准图建立独立模板；不可在同一模板内更换标准图")
        self.btn_save = QPushButton("保存模板")
        self.btn_publish = QPushButton("发布")
        self.btn_publish.setObjectName("Primary")
        self.btn_delete = QPushButton("删除")
        self.btn_delete.setObjectName("Danger")
        toolbar.layout_.addWidget(self.btn_std)
        self.btn_trial = QPushButton("试跑一张")
        self.btn_trial.setObjectName("Ghost")
        toolbar.layout_.addWidget(self.btn_trial)
        self.btn_tuner = QPushButton("调参辅助")
        self.btn_tuner.setObjectName("Ghost")
        self.btn_tuner.setToolTip("HSV 抽色 / 色板取色 / 预处理预览")
        toolbar.layout_.addWidget(self.btn_tuner)
        toolbar.layout_.addStretch()
        toolbar.layout_.addWidget(self.btn_delete)
        toolbar.layout_.addWidget(self.btn_save)
        toolbar.layout_.addWidget(self.btn_publish)
        root.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.list_panel = TemplateListPanel(self.template_store)
        self.list_panel.setMinimumWidth(220)
        splitter.addWidget(self.list_panel)

        center = QWidget()
        c_lay = QVBoxLayout(center)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.addLayout(self._build_canvas_toolbar())
        card = Card("标准图 · 检测区域", elevated=True)
        self.canvas = ImageCanvas("")
        self.canvas.setMinimumSize(280, 220)
        self.canvas.set_placeholder_text("选择标准图以新建模板，再勾选检测项并框选区域")
        self.canvas.set_interaction_mode("view")
        card.body.addWidget(self.canvas, 1)
        c_lay.addWidget(card, 1)
        self.lbl_assoc = QLabel("")
        self.lbl_assoc.setWordWrap(True)
        self.lbl_assoc.setProperty("class", "Hint")
        c_lay.addWidget(self.lbl_assoc)
        self.lbl_checklist = QLabel("")
        self.lbl_checklist.setWordWrap(True)
        self.lbl_checklist.setProperty("class", "Hint")
        c_lay.addWidget(self.lbl_checklist)
        splitter.addWidget(center)

        right = self._build_right()
        right.setMinimumWidth(380)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 720, 460])
        root.addWidget(splitter, 1)

        self.list_panel.selection_changed.connect(self._on_select)
        self.btn_std.clicked.connect(self._choose_standard_image)
        self.btn_save.clicked.connect(self._save_draft)
        self.btn_publish.clicked.connect(self._publish)
        self.btn_tuner.clicked.connect(self._open_tuner)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_trial.clicked.connect(self._trial_run)
        self.combo_calib_target.currentIndexChanged.connect(self._on_calib_target_changed)
        self.rb_rect.toggled.connect(self._on_shape_kind)
        self.rb_circle.toggled.connect(self._on_shape_kind)
        self.canvas.shapes_changed.connect(self._on_shapes_changed)
        self._set_tool("view")

    def _build_canvas_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("区域"))
        self.combo_calib_target = QComboBox()
        self.combo_calib_target.setMinimumWidth(130)
        row.addWidget(self.combo_calib_target)

        self.rb_rect = QRadioButton("矩形")
        self.rb_circle = QRadioButton("圆形")
        self.rb_rect.setChecked(True)
        row.addWidget(self.rb_rect)
        row.addWidget(self.rb_circle)

        row.addSpacing(8)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        for mode, text in (
            ("view", "浏览"),
            ("draw", "框选"),
            ("move", "移动"),
            ("scale", "缩放"),
        ):
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, m=mode: self._set_tool(m))
            self._tool_group.addButton(btn)
            self._tool_buttons[mode] = btn
            row.addWidget(btn)

        self.btn_delete_shape = QPushButton("删除")
        self.btn_clear_shapes = QPushButton("清空")
        self.btn_fit = QPushButton("自适应")
        self.btn_fit.setObjectName("Ghost")
        self.btn_fit.setToolTip("图片自适应显示区域")
        self.btn_delete_shape.clicked.connect(self._delete_selected_shape)
        self.btn_clear_shapes.clicked.connect(self._clear_current_shapes)
        self.btn_fit.clicked.connect(self._fit_canvas)
        row.addWidget(self.btn_delete_shape)
        row.addWidget(self.btn_clear_shapes)
        row.addWidget(self.btn_fit)
        row.addStretch()
        return row

    def _build_right(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        form_box = QGroupBox("模板（绑定标准图）")
        form = QFormLayout(form_box)
        self.edit_name = QLineEdit()
        self.pill_status = StatusPill("")
        self.lbl_std_path = QLabel("—")
        self.lbl_std_path.setWordWrap(True)
        self.lbl_std_path.setProperty("class", "Hint")
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("传统差分", "traditional")
        self.combo_engine.addItem("特征模型", "feature")
        self.combo_engine.addItem("双引擎并行", "dual")
        self.combo_engine.setToolTip(
            "传统差分=模板比对；特征模型=品类特征引擎；双引擎=两者并行融合判定"
        )
        self.combo_model_category = QComboBox()
        self.combo_model_category.setEditable(True)
        self.combo_model_category.setToolTip(
            "特征/双引擎模式必须绑定品类模型（先在「品类模型」页准备）；传统模式可留空"
        )
        form.addRow("名称", self.edit_name)
        form.addRow("状态", self.pill_status)
        form.addRow("标准图", self.lbl_std_path)
        form.addRow("引擎模式", self.combo_engine)
        form.addRow("品类模型", self.combo_model_category)
        self.combo_engine.currentIndexChanged.connect(self._on_engine_changed)
        self.combo_model_category.currentTextChanged.connect(lambda _t: self._refresh_checklist())
        lay.addWidget(form_box)

        algo_box = QGroupBox("检测项（缺陷类型）")
        algo_lay = QVBoxLayout(algo_box)
        algo_lay.setContentsMargins(8, 8, 8, 8)
        algo_lay.setSpacing(8)

        self.algo_checks.clear()
        self._algo_groups.clear()
        self._algo_label_by_id.clear()
        self._group_hosts.clear()

        major_row = QHBoxLayout()
        major_row.addWidget(QLabel("检测类别"))
        self.combo_major = QComboBox()
        self.combo_major.setMinimumWidth(160)
        for group_name, items in _all_algo_groups():
            self.combo_major.addItem(group_name, group_name)
            self._algo_groups[group_name] = list(items)
        major_row.addWidget(self.combo_major, 1)
        algo_lay.addLayout(major_row)

        actions = QHBoxLayout()
        self.lbl_item_hint = QLabel("勾选当前类别下的检测项（来自 Catalog）")
        self.lbl_item_hint.setProperty("class", "Hint")
        self.lbl_item_hint.setWordWrap(True)
        self.btn_items_all = QPushButton("全选")
        self.btn_items_none = QPushButton("全不选")
        self.btn_items_all.setFixedWidth(60)
        self.btn_items_none.setFixedWidth(70)
        actions.addWidget(self.lbl_item_hint, 1)
        actions.addWidget(self.btn_items_all)
        actions.addWidget(self.btn_items_none)
        algo_lay.addLayout(actions)

        item_scroll = QScrollArea()
        item_scroll.setWidgetResizable(True)
        item_scroll.setFrameShape(QFrame.Shape.NoFrame)
        item_scroll.setMinimumHeight(120)
        item_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        item_host = QWidget()
        item_host.setObjectName("ScrollHost")
        item_host_lay = QVBoxLayout(item_host)
        item_host_lay.setContentsMargins(0, 0, 0, 0)
        item_host_lay.setSpacing(4)
        for group_name, items in _all_algo_groups():
            group_w = QWidget()
            group_lay = QVBoxLayout(group_w)
            group_lay.setContentsMargins(0, 0, 0, 0)
            group_lay.setSpacing(2)
            for alg_id, label in items:
                cb = QCheckBox(label)
                cb.toggled.connect(self._on_algo_toggled)
                self.algo_checks[alg_id] = cb
                self._algo_label_by_id[alg_id] = label
                group_lay.addWidget(cb)
            item_host_lay.addWidget(group_w)
            self._group_hosts[group_name] = group_w
        item_host_lay.addStretch(1)
        item_scroll.setWidget(item_host)
        algo_lay.addWidget(item_scroll, 1)

        self.combo_major.currentIndexChanged.connect(self._on_major_changed)
        self.btn_items_all.clicked.connect(lambda: self._set_current_group_checked(True))
        self.btn_items_none.clicked.connect(lambda: self._set_current_group_checked(False))
        lay.addWidget(algo_box)

        param_box = QGroupBox("检测参数")
        param_lay = QVBoxLayout(param_box)
        param_lay.setContentsMargins(8, 8, 8, 8)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.param_scroll.setMinimumHeight(160)
        self.param_host = QWidget()
        self.param_host.setObjectName("ScrollHost")
        self.param_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.param_host_lay = QVBoxLayout(self.param_host)
        self.param_host_lay.setContentsMargins(4, 4, 8, 8)
        self.param_host_lay.setSpacing(10)
        self.param_host_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lbl_param_empty = QLabel("选择检测项目后在此调参")
        self.lbl_param_empty.setProperty("class", "Hint")
        self.param_host_lay.addWidget(self.lbl_param_empty)
        self.param_scroll.setWidget(self.param_host)
        param_lay.addWidget(self.param_scroll, 1)
        lay.addWidget(param_box, 1)

        self._refresh_item_host()
        return host

    def _current_major_group(self) -> str:
        return str(self.combo_major.currentData() or self.combo_major.currentText() or "")

    def _on_major_changed(self, _index: int = 0) -> None:
        """切换检测类别时清空其它类别勾选，保证一模板一类别。"""
        group = self._current_major_group()
        allowed = {alg_id for alg_id, _ in self._algo_groups.get(group, [])}
        had_other = any(cb.isChecked() and alg_id not in allowed for alg_id, cb in self.algo_checks.items())
        self._suppress_algo_toggle = True
        try:
            for alg_id, cb in self.algo_checks.items():
                if alg_id not in allowed and cb.isChecked():
                    cb.setChecked(False)
        finally:
            self._suppress_algo_toggle = False
        if had_other:
            self._set_algo_hint(f"已切换到「{group}」（同一模板仅绑定一个检测类别）")
        self._refresh_item_host()
        self._after_algo_selection_changed()

    def _set_current_group_checked(self, checked: bool) -> None:
        group = self._current_major_group()
        allowed = {alg_id for alg_id, _ in self._algo_groups.get(group, [])}
        self._suppress_algo_toggle = True
        try:
            for alg_id, cb in self.algo_checks.items():
                if alg_id in allowed:
                    cb.setChecked(checked)
        finally:
            self._suppress_algo_toggle = False
        self._after_algo_selection_changed()

    def _refresh_item_host(self) -> None:
        """只展示当前类别的 Catalog 勾选，避免弹层黑盒。"""
        group = self._current_major_group()
        for name, widget in self._group_hosts.items():
            widget.setVisible(name == group)
        selected = [
            self._algo_label_by_id.get(alg_id, alg_id)
            for alg_id, cb in self.algo_checks.items()
            if cb.isChecked()
        ]
        if selected:
            self.lbl_item_hint.setText(f"已选 {len(selected)} 项：{'、'.join(selected)}")
        else:
            self.lbl_item_hint.setText("勾选当前类别下的检测项（来自 Catalog）")

    # ---- 工具 / 标定图层 ----
    def _set_tool(self, mode: str) -> None:
        btn = self._tool_buttons.get(mode)
        if btn is not None:
            btn.setChecked(True)
        self.canvas.set_interaction_mode(mode)

    def _fit_canvas(self) -> None:
        if self.canvas.has_image():
            self.canvas.fit_to_view()
            self.canvas.update()

    def _current_calib_key(self) -> str | None:
        if self.current is None:
            return None
        return self.current.id

    def _refresh_association_label(self) -> None:
        if self.current is None:
            self.lbl_assoc.setText("")
            return
        s = self.current.association_summary()
        kinds = "、".join(s["region_kinds"]) if s["region_kinds"] else "—"
        self.lbl_assoc.setText(
            f"关联：标准图 → 模板「{self.current.display_name}」→ "
            f"检测项 {s['detection_item_count']} · 区域种类 {kinds} · 框/点 {s['region_count']}"
        )

    def _available_calib_targets(self) -> list[tuple[str, str]]:
        selected = self._selected_ids()
        kinds = {k for k in (_calib_kind_for(a) for a in selected) if k}
        out: list[tuple[str, str]] = []
        if "body" in kinds:
            out.append(("body", "本体重点区域"))
        if "th" in kinds:
            out.append(("th", "插件锡面/焊点"))
        if "smt" in kinds:
            out.extend(
                [
                    ("smt_pads", "贴片焊盘"),
                    ("smt_toe", "贴片 toe"),
                    ("smt_rim", "贴片 rim"),
                ]
            )
        if "shift" in kinds:
            out.append(("shift", "元件底座框"))
        if "gold" in kinds:
            out.append(("gold", "金手指取色点"))
        return out

    def _refresh_calib_targets(self, *, prefer: str | None = None) -> None:
        targets = self._available_calib_targets()
        cur = prefer or self.combo_calib_target.currentData()
        self.combo_calib_target.blockSignals(True)
        self.combo_calib_target.clear()
        if not targets:
            self.combo_calib_target.addItem(_CALIB_NONE, None)
        else:
            for key, label in targets:
                self.combo_calib_target.addItem(label, key)
        self.combo_calib_target.blockSignals(False)
        if cur:
            idx = self.combo_calib_target.findData(cur)
            if idx >= 0:
                self.combo_calib_target.setCurrentIndex(idx)
        self._reload_calib_layer()
        point_mode = self.combo_calib_target.currentData() == "gold"
        self.rb_rect.setEnabled(not point_mode and bool(targets))
        self.rb_circle.setEnabled(not point_mode and bool(targets))

    def _on_calib_target_changed(self, _index: int = 0) -> None:
        self._reload_calib_layer()
        point_mode = self.combo_calib_target.currentData() == "gold"
        self.rb_rect.setEnabled(not point_mode)
        self.rb_circle.setEnabled(not point_mode)

    def _on_shape_kind(self, checked: bool = False) -> None:
        if not checked:
            return
        self.canvas.set_draw_shape_kind("circle" if self.rb_circle.isChecked() else "rect")

    def _layer_config_for(self, target: str) -> LayerConfig:
        if target == "gold":
            return LayerConfig(color="#2bb0ed", shape="point", multi=True, label="seed")
        if target == "body":
            return LayerConfig(color=theme.ACCENT, shape="rect_or_circle", multi=True, label="ROI")
        if target == "th":
            return LayerConfig(color=theme.OK, shape="rect_or_circle", multi=True, label="TH")
        if target == "smt_toe":
            return LayerConfig(color=theme.REVIEW, shape="rect", multi=True, label="toe")
        if target == "smt_rim":
            return LayerConfig(color="#e06c75", shape="rect", multi=True, label="rim")
        if target == "shift":
            return LayerConfig(color="#0891b2", shape="rect", multi=True, label="base")
        return LayerConfig(color=theme.REVIEW, shape="rect", multi=True, label="pad")

    def _reload_calib_layer(self) -> None:
        target = self.combo_calib_target.currentData()
        self._suppress_shape_save = True
        try:
            self.canvas.clear_all_layers()
            if self.current is None:
                return
            all_targets = [key for key, _label in self._available_calib_targets()]
            for key in all_targets:
                cfg = self._layer_config_for(key)
                cfg.faded = key != target
                cfg.editable = key == target
                self.canvas.set_layer(key, cfg, self._load_shapes_for_target(key))
            if target:
                self.canvas.set_active_layer(str(target))
                self.canvas.set_draw_shape_kind("circle" if self.rb_circle.isChecked() else "rect")
        finally:
            self._suppress_shape_save = False

    def _load_shapes_for_target(self, target: str) -> list[dict]:
        if self.current is None:
            return []
        owned = self.current.get_region_shapes(target)
        if owned:
            return owned
        key = self.current.id
        if target == "body":
            return [dict(p) for p in get_body_rois(key)]
        if target == "th":
            return [dict(p) for p in (get_th_roi(key).get("pads") or [])]
        if target == "smt_pads":
            return _rects_to_shapes(get_smt_pads(key))
        if target == "smt_toe":
            return _rects_to_shapes(get_smt_toe(key))
        if target == "smt_rim":
            return _rects_to_shapes(get_smt_rim(key))
        if target == "shift":
            return _rects_to_shapes(get_shift_bases(key))
        if target == "gold":
            seeds = get_gold_seed(key).get("seeds") or []
            return [{"shape": "point", "x": float(x), "y": float(y)} for x, y in seeds]
        return []

    def _on_shapes_changed(self, _layer: str = "") -> None:
        if self._suppress_shape_save:
            return
        self._persist_current_shapes()
        self._refresh_association_label()
        self._refresh_checklist()

    def _persist_current_shapes(self) -> None:
        target = self.combo_calib_target.currentData()
        if not target or self.current is None:
            return
        shapes = self.canvas.get_layer_shapes(str(target))
        normalized: list[dict] = []
        if target in ("body", "th"):
            for s in shapes:
                if s.get("shape") == "circle":
                    normalized.append({"shape": "circle", "cx": s["cx"], "cy": s["cy"], "r": s["r"]})
                elif s.get("shape") == "rect":
                    normalized.append({"shape": "rect", "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})
        elif str(target).startswith("smt_") or target == "shift":
            for s in shapes:
                if s.get("shape") == "rect":
                    normalized.append({"shape": "rect", "x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]})
        elif target == "gold":
            for s in shapes:
                if s.get("shape") == "point":
                    normalized.append({"shape": "point", "x": float(s["x"]), "y": float(s["y"])})
        else:
            normalized = [dict(s) for s in shapes]

        self.current.set_region_shapes(str(target), normalized)
        key = self.current.id
        if target == "body":
            set_body_rois(key, normalized)
        elif target == "th":
            set_th_pads(key, normalized)
        elif target == "smt_pads":
            set_smt_pads(key, _shapes_to_rects(normalized))
        elif target == "smt_toe":
            set_smt_toe(key, _shapes_to_rects(normalized))
        elif target == "smt_rim":
            set_smt_rim(key, _shapes_to_rects(normalized))
        elif target == "shift":
            set_shift_bases(key, _shapes_to_rects(normalized))
        elif target == "gold":
            seeds = [(int(round(s["x"])), int(round(s["y"]))) for s in normalized]
            prev = get_gold_seed(key)
            set_gold_seed(key, seeds, float(prev.get("tol", 1.0)), prev.get("ranges") or [])
        self.template_store.sync_calibration_to_legacy(self.current)

    def _delete_selected_shape(self) -> None:
        if not self.canvas.delete_selected():
            QMessageBox.information(self, "提示", "请先选中一个框，或切换到「移动/缩放」再点选")
            return
        self._persist_current_shapes()
        self._refresh_checklist()

    def _clear_current_shapes(self) -> None:
        target = self.combo_calib_target.currentData()
        if not target:
            return
        self.canvas.clear_layer(str(target))
        self._persist_current_shapes()
        self._refresh_checklist()

    # ---- 数据绑定 ----
    def _on_select(self, tpl: InspectionTemplate | None) -> None:
        self.current = tpl
        if tpl is None:
            self.edit_name.clear()
            self.pill_status.set_status("")
            self.lbl_std_path.setText("—")
            self.combo_engine.setCurrentIndex(2)  # 与模板默认 engine_mode=dual 对齐
            self._refresh_model_categories(keep="")
            self._on_engine_changed()
            self.std_bgr = None
            self.canvas.set_image(None)
            for cb in self.algo_checks.values():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            self._sync_major_combo_from_selection()
            self._refresh_item_host()
            self._refresh_calib_targets()
            self._refresh_param_panels()
            self.lbl_checklist.setText("")
            self._refresh_association_label()
            return

        self.edit_name.setText(tpl.display_name)
        self.pill_status.set_status("PUB" if tpl.status == "published" else "DRAFT")
        eng_idx = self.combo_engine.findData(tpl.engine_mode or "dual")
        self.combo_engine.setCurrentIndex(eng_idx if eng_idx >= 0 else 2)
        self._refresh_model_categories(keep=(tpl.model_category or tpl.category or "").strip())
        self._on_engine_changed()
        std_path = tpl.standard_image.path or "—"
        wh = ""
        if tpl.standard_image.width and tpl.standard_image.height:
            wh = f"  ({tpl.standard_image.width}×{tpl.standard_image.height})"
        self.lbl_std_path.setText(f"{tpl.standard_image.display_name}{wh}" if std_path != "—" else "—")
        self.lbl_std_path.setToolTip(std_path)

        enabled = set(tpl.enabled_algorithm_ids())
        self._suppress_algo_toggle = True
        try:
            for alg_id, cb in self.algo_checks.items():
                cb.setChecked(alg_id in enabled)
        finally:
            self._suppress_algo_toggle = False
        self._sync_major_combo_from_selection()
        self._refresh_item_host()

        self._load_standard_image(tpl)
        self._refresh_calib_targets()
        self._refresh_param_panels()
        self._refresh_association_label()
        self._refresh_checklist()

    def _load_standard_image(self, tpl: InspectionTemplate) -> None:
        path = tpl.standard_image.path
        if path and Path(path).exists():
            self.std_bgr = imread_unicode(path)
            self.canvas.set_image(self.std_bgr)
        else:
            self.std_bgr = None
            self.canvas.set_image(None)
            self.canvas.set_placeholder_text("标准图缺失：请用「选择标准图新建」重建模板")

    def _collect_into_current(self) -> InspectionTemplate | None:
        if self.current is None:
            return None
        self.current.display_name = self.edit_name.text().strip() or self.current.id
        self.current.engine_mode = str(self.combo_engine.currentData() or "dual")
        self.current.model_category = self.combo_model_category.currentText().strip()
        selected = [alg_id for alg_id, cb in self.algo_checks.items() if cb.isChecked()]
        self.current.set_algorithm_ids(selected, keep_params=True)
        return self.current

    def _selected_ids(self) -> list[str]:
        return [alg_id for alg_id, cb in self.algo_checks.items() if cb.isChecked()]

    def _set_algo_hint(self, text: str) -> None:
        """非模态提示，避免勾选时连弹 QMessageBox。"""
        self._algo_hint = text or ""

    def _sync_major_combo_from_selection(self) -> None:
        selected = self._selected_ids()
        group = _major_group_of(selected[0]) if selected else self._current_major_group()
        idx = self.combo_major.findData(group)
        if idx < 0:
            idx = self.combo_major.findText(group)
        if idx >= 0 and self.combo_major.currentIndex() != idx:
            self.combo_major.blockSignals(True)
            self.combo_major.setCurrentIndex(idx)
            self.combo_major.blockSignals(False)

    def _refresh_model_categories(self, keep: str = "") -> None:
        """刷新品类模型下拉候选（来自树干 AOI_Core 品类列表），保留当前绑定值。"""
        current = keep if keep else self.combo_model_category.currentText().strip()
        cats: list[str] = []
        try:
            from app.engines.feature import get_engine

            cats = sorted(str(c["category"]) for c in get_engine().list_categories())
        except Exception:  # noqa: BLE001 Core 不可达时仅提供手输
            cats = []
        self.combo_model_category.blockSignals(True)
        self.combo_model_category.clear()
        self.combo_model_category.addItem("")
        for c in cats:
            self.combo_model_category.addItem(c)
        if current:
            self.combo_model_category.setCurrentText(current)
        self.combo_model_category.blockSignals(False)

    def _on_engine_changed(self, _idx: int = 0) -> None:
        need_model = str(self.combo_engine.currentData() or "dual") in ("feature", "dual")
        self.combo_model_category.setEnabled(need_model)
        self._refresh_checklist()

    def _on_algo_toggled(self, _checked: bool = False) -> None:
        if getattr(self, "_suppress_algo_toggle", False):
            return
        selected = self._selected_ids()
        groups: list[str] = []
        for alg_id in selected:
            g = _major_group_of(alg_id)
            if g not in groups:
                groups.append(g)
        if len(groups) > 1:
            sender = self.sender()
            if isinstance(sender, QCheckBox) and sender.isChecked():
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
                kept = "、".join(groups[:-1] or groups[:1])
                self._set_algo_hint(f"同一模板只能绑定一个检测类别；当前已保留：{kept}")
        else:
            self._set_algo_hint("")
            if selected:
                self._sync_major_combo_from_selection()
        self._after_algo_selection_changed()

    def _after_algo_selection_changed(self) -> None:
        self._refresh_item_host()
        self._refresh_calib_targets()
        self._refresh_param_panels()
        self._refresh_association_label()
        self._refresh_checklist()

    def _refresh_param_panels(self) -> None:
        selected = self._selected_ids()
        # 清掉旧面板
        while self.param_host_lay.count():
            item = self.param_host_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._param_panels.clear()
        if not selected:
            self.lbl_param_empty = QLabel("选择检测项目后在此调参")
            self.lbl_param_empty.setProperty("class", "Hint")
            self.lbl_param_empty.setWordWrap(True)
            self.param_host_lay.addWidget(self.lbl_param_empty)
            return
        catalog = get_catalog()
        shown_shared: list[str] = []
        for alg_id in selected:
            shared = catalog.shared_param_id(alg_id)
            if shared and shared not in shown_shared:
                shown_shared.append(shared)
                sm = catalog.get(shared)
                shared_title = QLabel((sm.display_name if sm else "共用参数") + "（对本组检测项统一生效）")
                shared_title.setProperty("class", "SectionLabel")
                shared_title.setWordWrap(True)
                self.param_host_lay.addWidget(shared_title)
                shared_panel = AlgorithmParamPanel(
                    shared,
                    template_name_getter=lambda: self._current_calib_key(),
                    param_store=self.param_store,
                    get_std_bgr=lambda: self.std_bgr,
                    get_test_bgr=lambda: self.std_bgr,
                    show_preview=False,
                    compact=True,
                )
                shared_panel.ensure_loaded(preview=False)
                self._param_panels[shared] = shared_panel
                self.param_host_lay.addWidget(shared_panel)
            title = QLabel(self._algo_label_by_id.get(alg_id, alg_id))
            title.setProperty("class", "SectionLabel")
            title.setWordWrap(True)
            self.param_host_lay.addWidget(title)
            panel = AlgorithmParamPanel(
                alg_id,
                template_name_getter=lambda: self._current_calib_key(),
                param_store=self.param_store,
                get_std_bgr=lambda: self.std_bgr,
                get_test_bgr=lambda: self.std_bgr,
                show_preview=False,
                compact=True,
            )
            panel.ensure_loaded(preview=False)
            self._param_panels[alg_id] = panel
            self.param_host_lay.addWidget(panel)

    def _refresh_checklist(self) -> None:
        tpl = self._collect_into_current()
        if tpl is None:
            self.lbl_checklist.setText(self._algo_hint)
            return
        # 以模板内 region_sets 为准同步 legacy，再做发布检查
        try:
            self.template_store.sync_calibration_to_legacy(tpl)
            issues = self.template_store.publish_checklist(tpl)
        except Exception as exc:  # noqa: BLE001
            self.lbl_checklist.setText(f"发布检查失败：{exc}")
            self.lbl_checklist.setStyleSheet(f"color: {theme.NG};")
            return
        blockers = [i for i in issues if i.startswith("[阻断]")]
        warns = [i for i in issues if i.startswith("[警告]")]
        lines: list[str] = []
        if self._algo_hint:
            lines.append(self._algo_hint)
        if blockers:
            lines.append("发布检查 · 阻断")
            lines.extend(blockers)
            color = theme.NG
        elif warns:
            lines.append("发布检查 · 警告")
            lines.extend(warns)
            color = theme.REVIEW
        else:
            lines.append("发布检查 · 可发布")
            color = theme.OK
        self.lbl_checklist.setText("\n".join(lines))
        self.lbl_checklist.setStyleSheet(f"color: {color};")

    # ---- 操作 ----
    def _choose_standard_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标准图（新建独立模板）",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not path:
            return
        existing = self.template_store.find_by_standard_path(path)
        if existing is not None:
            ret = QMessageBox.question(
                self,
                "标准图已绑定模板",
                f"该标准图已绑定模板「{existing.display_name}」（{existing.id}）。\n"
                "一图一模板，不可在其它模板中复用。\n是否打开已有模板？",
            )
            if ret == QMessageBox.StandardButton.Yes:
                self.list_panel.refresh(keep_id=existing.id)
            return
        try:
            tpl = self.template_store.create_from_standard_image(path)
        except (ValueError, FileNotFoundError, OSError) as exc:
            QMessageBox.warning(self, "无法创建模板", str(exc))
            return
        self.list_panel.refresh(keep_id=tpl.id)

    def _freeze_params(self, tpl: InspectionTemplate) -> None:
        # 先把面板当前值写入 ParamStore（模板专用）
        key = tpl.id
        for alg_id, panel in self._param_panels.items():
            if panel.has_params():
                values = {k: s.value() for k, s in panel.sliders.items()}
                self.param_store.save("template_algorithm", alg_id, values, template_name=key)

        ids = list(tpl.enabled_algorithm_ids())
        freeze_ids = list(ids)
        catalog = get_catalog()
        for item_id in ids:
            shared = catalog.shared_param_id(item_id)
            if shared and shared not in freeze_ids:
                freeze_ids.append(shared)
        defaults_by_alg = {}
        for alg_id in freeze_ids:
            adapter = get_adapter_for(alg_id)
            defaults_by_alg[alg_id] = adapter.default_params(alg_id) if adapter else {}
        exported = self.param_store.export_for_template(
            key,
            freeze_ids,
            category=tpl.category or None,
            defaults_by_alg=defaults_by_alg,
        )
        for alg_id, params in exported.items():
            if not catalog.is_job_item(alg_id):
                found = False
                for a in tpl.algorithms:
                    if a.algorithm_id == alg_id:
                        a.params = params
                        a.enabled = False
                        found = True
                        break
                if not found:
                    tpl.algorithms.append(
                        TemplateAlgorithm(algorithm_id=alg_id, enabled=False, params=params)
                    )
            else:
                tpl.upsert_params(alg_id, params)

    def _open_tuner(self) -> None:
        """调参辅助：按 zz pcba_tuner 成熟方案，与模板 ROI/算法参数打通。

        流程：模板已导入 ROI（region_sets 的 pad/toe/rim）→ 已分配算法（勾选
        检测项）→ 调出本控件辅助调参。对话框显示模板 ROI 框、自动注册勾选
        算法的 HSV 抽色对象，抽色结果写回共用参数层。
        """
        tpl = self.current
        if tpl is None or not getattr(tpl.standard_image, "path", ""):
            QMessageBox.information(self, "提示", "请先选择标准图新建/加载模板")
            return
        from app.ui.tuner_dialog import TunerDialog
        # 模板 ROI 框（pad/toe/rim），叠加显示在调参画布上
        roi_boxes: list[tuple[str, int, int, int, int]] = []
        for key in ("smt_pads", "smt_toe", "smt_rim"):
            for s in tpl.get_region_shapes(key):
                if str(s.get("shape") or "rect") == "rect":
                    roi_boxes.append((key, int(s["x"]), int(s["y"]),
                                      int(s["w"]), int(s["h"])))
        # 模板当前算法参数合并（供各抽色对象初始化）
        merged_params: dict = {}
        for a in tpl.detection_items:
            if a.params:
                merged_params.update(a.params)
        dlg = TunerDialog(
            image_path=tpl.standard_image.path,
            apply_callback=lambda spec, hsv: self._apply_hsv_params(tpl, spec, hsv),
            roi_boxes=roi_boxes,
            initial_params=merged_params,
            parent=self,
        )
        dlg.exec()

    def _apply_hsv_params(self, tpl: InspectionTemplate, spec, hsv: dict) -> None:
        """把抽色对象的 HSV 六参写回所有勾选算法的"共用参数"层。

        写入口与右侧 AlgorithmParamPanel 一致（param_store 的
        template_algorithm 层），保存模板时 ``_freeze_params`` 再把该层固化为
        模板快照，保证发布后检测读到同一组参数。
        """
        import traceback
        try:
            self._apply_hsv_params_impl(tpl, spec, hsv)
        except Exception as exc:  # noqa: BLE001 - 槽函数异常会 abort 进程，必须兜底
            traceback.print_exc()
            QMessageBox.warning(self, "应用失败", f"{exc}")

    def _apply_hsv_params_impl(self, tpl: InspectionTemplate, spec, hsv: dict) -> None:
        from app.tools.tuner import hsv_to_params
        params = hsv_to_params(spec, hsv)
        if not params:
            QMessageBox.information(self, "提示", "当前对象没有可映射的算法参数键")
            return
        selected = self._selected_ids()
        if not selected:
            QMessageBox.information(self, "提示", "请先勾选至少一个检测项")
            return
        catalog = get_catalog()
        written: list[str] = []
        seen_shared: set[str] = set()
        for alg_id in selected:
            shared = catalog.shared_param_id(alg_id)
            adapter = get_adapter_for(alg_id)
            if not shared or shared in seen_shared or adapter is None:
                continue
            seen_shared.add(shared)
            try:
                shared_specs = list(adapter.param_specs(shared) or [])
            except Exception:  # noqa: BLE001
                continue
            spec_keys = {p.key for p in shared_specs}
            sub = {k: v for k, v in params.items() if k in spec_keys}
            if not sub:
                continue
            defaults = {p.key: p.default for p in shared_specs}
            resolved, _hit = self.param_store.resolve(tpl.id, None, shared, defaults)
            merged = {**resolved, **sub}
            self.param_store.save("template_algorithm", shared, merged,
                                  template_name=tpl.id)
            written.append(f"{shared}({len(sub)}项)")
        if not written:
            QMessageBox.information(
                self, "提示",
                "勾选算法的参数中不含该对象的颜色键，未写入。\n"
                "（可改用「复制参数」手工填入算法调试页）")
            return
        self._refresh_param_panels()
        QMessageBox.information(self, "已应用", "已写入参数：" + "；".join(written))

    def _save_draft(self) -> None:
        tpl = self._collect_into_current()
        if tpl is None:
            QMessageBox.information(self, "提示", "请先选择标准图新建模板")
            return
        self._persist_current_shapes()
        self._freeze_params(tpl)
        if tpl.status == "published":
            ret = QMessageBox.question(
                self,
                "已发布模板将升版本",
                f"「{tpl.display_name}」已发布为 v{tpl.version}。\n"
                "改参数后需升版本并重新发布才进产线。\n是否升版本并保存为草稿？",
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
            tpl.version = int(tpl.version or 1) + 1
            tpl.status = "draft"
        else:
            tpl.status = "draft"
        self.template_store.save(tpl)
        self.param_store.import_from_template(
            tpl.id,
            {a.algorithm_id: a.params for a in tpl.detection_items if a.params},
        )
        self.list_panel.refresh(keep_id=tpl.id)
        self._refresh_association_label()
        QMessageBox.information(self, "已保存", f"模板「{tpl.display_name}」已保存（标准图独立绑定）")

    def _publish(self) -> None:
        tpl = self._collect_into_current()
        if tpl is None:
            QMessageBox.information(self, "提示", "请先选择标准图新建模板")
            return
        try:
            self._persist_current_shapes()
            self._freeze_params(tpl)
            self.template_store.publish(tpl)
        except ValueError as exc:
            self._refresh_checklist()
            QMessageBox.warning(self, "无法发布", str(exc) or "发布检查未通过，请查看页面底部清单。")
            return
        except Exception as exc:  # noqa: BLE001
            self._refresh_checklist()
            QMessageBox.warning(self, "无法发布", f"发布时出错：{exc}")
            return
        self.param_store.import_from_template(
            tpl.id,
            {a.algorithm_id: a.params for a in tpl.detection_items if a.params},
        )
        self.list_panel.refresh(keep_id=tpl.id)
        self.pill_status.set_status("PUB")
        ret = QMessageBox.question(
            self,
            "发布成功",
            f"模板「{tpl.display_name}」已发布为 v{tpl.version}。\n是否前往自动检测？",
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.request_open_inspect.emit(tpl.id)

    def _delete(self) -> None:
        if self.current is None:
            return
        ret = QMessageBox.question(self, "确认删除", f"确定删除模板「{self.current.display_name}」？")
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.template_store.delete(self.current.id)
        self.list_panel.refresh()

    def _trial_run(self) -> None:
        tpl = self._collect_into_current()
        if tpl is None:
            QMessageBox.information(self, "提示", "请先选择模板")
            return
        self._persist_current_shapes()
        self._freeze_params(tpl)
        path, _ = QFileDialog.getOpenFileName(
            self, "试跑一张（不归档）", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not path:
            return
        from app.ui.defect_table import DefectTable
        from app.ui.inspect_canvas import InspectCanvas
        from app.ui.result_hud import ResultHud
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        run = self.inspect_service.run_with_template(
            tpl, path, archive=False, allow_stub=True, source="trial"
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("试跑结果（不写入产线历史）")
        dlg.resize(980, 640)
        lay = QVBoxLayout(dlg)
        hud = ResultHud()
        hud.bind_summary(
            run.summary,
            error=run.error,
            test_path=path,
            template_name=tpl.display_name,
            template_version=tpl.version,
        )
        lay.addWidget(hud)
        canvas = InspectCanvas("试跑图")
        table = DefectTable()
        overlays = table.bind_summary(run.summary, error=run.error)
        img = imread_unicode(path)
        canvas.bind_job(img, run.summary, overlays, filename=path, region_sets=tpl.region_sets)
        table.defect_selected.connect(canvas.highlight)
        canvas.defect_clicked.connect(table.select_index)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(canvas)
        split.addWidget(table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # 切回本页只刷新列表文案/状态，不重复触发选中重载（否则会重建参数面板并连闪）
        keep = self.current.id if self.current else None
        self.list_panel.refresh(keep_id=keep, reload_selection=False)
        # 品类模型页可能刚准备了新模型，刷新候选但保留当前绑定
        self._refresh_model_categories()
