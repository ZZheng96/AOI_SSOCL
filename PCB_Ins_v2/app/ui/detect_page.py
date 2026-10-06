from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.config import ROOT_DIR
from app.detect.catalog import get_catalog
from app.detect.registry import get_adapter_for
from app.detect.roi import rois_from_region_sets
from app.detect.scheduler import DetectScheduler
from app.result.store import ResultStore
from app.template.model import InspectionTemplate
from app.template.store import TemplateStore
from app.ui import theme
from app.ui.algo_param_panel import AlgorithmParamPanel
from app.ui.defect_table import DefectTable
from app.ui.image_canvas import ImageCanvas
from app.ui.inspect_canvas import InspectCanvas
from app.ui.result_hud import ResultHud
from app.ui.template_picker import TemplatePicker
from app.ui.widgets import Card, ColorDot
from app.utils.cv_io import imread_unicode


# 各大类状态灯颜色（收起时也能从外侧辨认勾选了哪一类）
def _calib_kind_for(algorithm_id: str) -> str | None:
    kind = get_catalog().region_kind(algorithm_id)
    return None if not kind or kind == "none" else kind


def _calib_hint_for_kind(kind: str) -> str:
    return {
        "body": "元件本体 → 重点区域框选（可选，不画则整图）",
        "smt": "元件焊锡 → 焊盘/toe/rim 标定（必需）",
        "th": "插件焊点 → 锡面区域标定（可选）",
        "gold": "板面金手指 → 板面取色种子点（可选）",
    }.get(kind, kind)


class DetectPage(QWidget):
    """算法调试：引用模板，调参并对照标准图/测试图验证效果。"""

    request_preprocess = Signal()
    request_goto_studio = Signal(str)

    def __init__(
        self,
        scheduler: DetectScheduler,
        result_store: ResultStore,
        template_store: TemplateStore | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scheduler = scheduler
        self.result_store = result_store
        self.template_store = template_store or TemplateStore()
        self.bound_template: InspectionTemplate | None = None

        self.std_bgr = None
        self.test_bgr = None
        self.std_path = ""
        self.test_path = ""
        self.last_summary = None
        self.last_result_folder: Path | None = None
        self.algo_checks: dict[str, QCheckBox] = {}
        self.board_checks: dict[str, QCheckBox] = {}
        self.surface_checks: dict[str, QCheckBox] = {}
        self._param_panels: list[AlgorithmParamPanel] = []
        self._panel_by_alg: dict[str, AlgorithmParamPanel] = {}
        self._open_param_toggle: QToolButton | None = None
        self._open_group_toggle: QToolButton | None = None
        self._drawer_refreshers: list = []
        self._drawer_frames: list[tuple[QWidget, set[str]]] = []
        self._live_focus_alg: str | None = None
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(350)
        self._live_timer.timeout.connect(self._run_live_detect)
        self._splitter_save_timer = QTimer(self)
        self._splitter_save_timer.setSingleShot(True)
        self._splitter_save_timer.setInterval(250)
        self._splitter_save_timer.timeout.connect(self._persist_splitter_sizes)
        self.memory_file = ROOT_DIR / "ui_state.json"
        self._build()
        for cb in self._all_checkboxes():
            cb.setEnabled(False)
        self._refresh_all_drawer_headers()
        self._update_calib_button()
        self._splitters_restored = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        keep = self.bound_template.id if self.bound_template else None
        self.template_picker.refresh(keep_id=keep)
        # 窗口真正有尺寸后再应用一次，避免启动时分割条比例被挤扁
        if not self._splitters_restored:
            self._splitters_restored = True
            self._restore_splitter_sizes()

    # ---- UI 构建 ----
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        self.template_picker = TemplatePicker(
            self.template_store,
            published_only=False,
            label="引用模板",
            empty_text="（请选择模板以对照跑图）",
        )
        toolbar.addWidget(self.template_picker, 1)
        self.btn_open_test = QPushButton("打开测试图")
        self.btn_calib = QPushButton("去建模改区域")
        self.btn_calib.setObjectName("Ghost")
        self.btn_calib.setToolTip("检测区域标定请在「模板建模」中编辑")
        self.btn_run = QPushButton("对照跑图")
        self.btn_run.setObjectName("Primary")
        self.cb_archive = QCheckBox("归档本次")
        self.cb_archive.setToolTip("默认不写入产线历史；勾选后才归档")
        self.btn_to_preprocess = QPushButton("去预处理调参")
        self.btn_to_preprocess.setObjectName("Ghost")
        toolbar.addWidget(self.btn_open_test)
        toolbar.addWidget(self.btn_calib)
        toolbar.addWidget(self.btn_to_preprocess)
        toolbar.addWidget(self.cb_archive)
        toolbar.addWidget(self.btn_run)
        root.addLayout(toolbar)

        hint = QLabel(
            "算法调试：选择模板并打开测试图，在右侧调整参数，结果实时反馈到左侧。"
            "未在模板中的检测项已隐藏；默认不归档，需要时勾选「归档本次」。"
        )
        hint.setProperty("class", "Hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = self._build_left()
        right = self._build_right()
        left.setMinimumWidth(360)
        right.setMinimumWidth(360)
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Expanding：避免抽屉标题变长时 Preferred sizeHint 把左右分割条来回挤
        right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        # 默认给右侧更多宽度，参数抽屉标题/勾选摘要不易被挤掉
        splitter.setSizes([820, 680])
        self.main_splitter = splitter
        root.addWidget(splitter, 1)
        splitter.splitterMoved.connect(self._on_splitter_moved)

        self.template_picker.template_changed.connect(self._on_template_changed)
        self.btn_open_test.clicked.connect(self.open_test)
        self.btn_calib.clicked.connect(self._goto_studio_for_edit)
        self.btn_run.clicked.connect(self.run_detect)
        self.btn_to_preprocess.clicked.connect(self.request_preprocess.emit)

    def _build_image_card(
        self,
        title: str,
        placeholder: str,
        *,
        accept_drops: bool = False,
        min_size: tuple[int, int] = (160, 140),
    ) -> tuple[Card, ImageCanvas]:
        """统一结构：一行标题 + 一个图片显示框。"""
        card = Card(title)
        canvas = ImageCanvas("")
        canvas.setMinimumSize(*min_size)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.set_placeholder_text(placeholder)
        canvas.setAcceptDrops(accept_drops)
        card.body.addWidget(canvas, 1)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return card, canvas

    def _build_left(self) -> QWidget:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        images = QHBoxLayout()
        images.setSpacing(10)
        self.card_std, self.canvas_std = self._build_image_card(
            "标准图（来自模板）",
            "请先在上方选择模板\n标准图将自动从模板加载",
            accept_drops=False,
            min_size=(160, 160),
        )
        self.canvas_test = InspectCanvas("测试图 / 结果")
        self.canvas_test.set_placeholder("拖拽图片到此处，或点击上方「打开测试图」")
        self.canvas_test.canvas.setAcceptDrops(True)
        self.canvas_test.canvas.setMinimumSize(160, 160)
        images.addWidget(self.card_std, 1)
        images.addWidget(self.canvas_test, 1)
        layout.addLayout(images, 3)

        card_result = Card("检测结果")
        self.result_hud = ResultHud()
        self.defect_table = DefectTable()
        card_result.add(self.result_hud)
        card_result.add(self.defect_table)
        layout.addWidget(card_result, 1)

        self.canvas_test.image_dropped.connect(self.load_test)
        self.defect_table.defect_selected.connect(self._on_defect_row)
        self.canvas_test.defect_clicked.connect(self.defect_table.select_index)
        return host

    def _build_right(self) -> QWidget:
        host = QWidget()
        host.setMinimumWidth(360)
        host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # 右侧仅参数/算法列表，与模板建模侧栏一致；预览反馈在左侧测试图
        layout.addWidget(self._build_algo_card(), 1)
        return host

    def _build_algo_card(self) -> QWidget:
        card = Card("检测项与参数（与模板建模一致）")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # 禁止横向滚动条：标题变长时 H/V 滚动条互抢会导致整栏抖动
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        host = QWidget()
        host.setObjectName("ScrollHost")
        host.setMinimumWidth(0)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(4, 4, 4, 4)
        host_layout.setSpacing(8)
        catalog = get_catalog()
        self._drawer_frames.clear()
        for _gid, group_name, manifests in catalog.groups(job_only=True):
            items = [(m.id, m.display_name) for m in manifests]
            extra_rows = []
            seen_shared: list[str] = []
            ids = {m.id for m in manifests}
            for m in manifests:
                if m.shared_param_id and m.shared_param_id not in seen_shared:
                    seen_shared.append(m.shared_param_id)
                    ids.add(m.shared_param_id)
                    sm = catalog.get(m.shared_param_id)
                    extra_rows.append(
                        self._build_shared_param_row(
                            m.shared_param_id,
                            (sm.display_name if sm else "共用参数") + "（对本组检测项统一生效）",
                        )
                    )
            drawer = self._build_algo_drawer(group_name, items, self.algo_checks, extra_rows or None)
            drawer.setVisible(False)
            self._drawer_frames.append((drawer, ids))
            host_layout.addWidget(drawer)
        host_layout.addStretch()
        scroll.setWidget(host)
        card.body.addWidget(scroll, 1)
        return card

    def _build_algo_drawer(
        self,
        group_name: str,
        items: list[tuple[str, str]],
        checks_dict: dict,
        extra_rows: list[QWidget] | None = None,
    ) -> QWidget:
        """大类抽屉：点标题展开/收起；外侧用颜色灯 + 勾选摘要辨认状态。"""
        frame = QFrame()
        frame.setProperty("class", "Panel")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        active_color = get_catalog().color(items[0][0]) if items else theme.ACCENT
        idle_color = theme.TEXT_MUTE

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        indicator = ColorDot(idle_color, size=12)
        indicator.setToolTip("灰色=未勾选；亮色=该大类已勾选缺陷")
        header_row.addWidget(indicator)

        header = QToolButton()
        header.setObjectName("DrawerHeader")
        header.setCheckable(True)
        header.setChecked(False)
        header.setMinimumWidth(0)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        header.setText(f"▸  {group_name}")
        header_row.addWidget(header, 1)
        outer.addLayout(header_row)

        body = QWidget()
        body.setVisible(False)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(4)

        actions = QHBoxLayout()
        actions.addStretch()
        btn_all = QPushButton("全选")
        btn_none = QPushButton("全不选")
        btn_all.setFixedWidth(60)
        btn_none.setFixedWidth(70)
        actions.addWidget(btn_all)
        actions.addWidget(btn_none)
        lay.addLayout(actions)

        for row in extra_rows or []:
            lay.addWidget(row)

        row_checks: list[QCheckBox] = []
        id_to_label = {alg_id: label for alg_id, label in items}
        for alg_id, label in items:
            row = self._build_algo_row(alg_id, label, checks_dict)
            lay.addWidget(row)
            row_checks.append(checks_dict[alg_id])

        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in row_checks])
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in row_checks])
        outer.addWidget(body)

        def _refresh_header() -> None:
            selected_labels = [
                id_to_label[alg_id]
                for alg_id, cb in checks_dict.items()
                if alg_id in id_to_label and cb.isChecked()
            ]
            arrow = "▾" if header.isChecked() else "▸"
            # 标题只显示短摘要，避免 sizeHint 变宽触发滚动条闪烁/分割条抖动
            if selected_labels:
                n = len(selected_labels)
                brief = selected_labels[0] if n == 1 else f"{selected_labels[0]}等{n}项"
                header.setText(f"{arrow}  {group_name}  ·  {brief}")
                header.setToolTip("已勾选：" + "、".join(selected_labels))
                indicator.set_color(active_color)
            else:
                header.setText(f"{arrow}  {group_name}")
                header.setToolTip("")
                indicator.set_color(idle_color)

        def _on_drawer(checked: bool) -> None:
            body.setVisible(checked)
            _refresh_header()
            if checked:
                if self._open_group_toggle is not None and self._open_group_toggle is not header:
                    self._open_group_toggle.setChecked(False)
                self._open_group_toggle = header
            elif self._open_group_toggle is header:
                self._open_group_toggle = None

        header.toggled.connect(_on_drawer)
        for cb in row_checks:
            cb.toggled.connect(lambda _c=False: (_refresh_header(), self._update_calib_button()))
        self._drawer_refreshers.append(_refresh_header)
        return frame

    def _shared_base_params(self, shared_id: str) -> dict:
        panel = self._panel_by_alg.get(shared_id)
        if panel is not None and getattr(panel, "_loaded", False) and panel.has_params():
            return panel.current_values()
        frozen = self._frozen_params_for(shared_id)
        if frozen:
            return frozen
        adapter = get_adapter_for(shared_id)
        if adapter is None:
            return {}
        defaults = adapter.default_params(shared_id)
        resolved, _ = self.scheduler.param_store.resolve(self.current_template_name(), None, shared_id, defaults)
        return resolved

    def _frozen_params_for(self, alg_id: str) -> dict:
        if self.bound_template is None:
            return {}
        return dict(self.bound_template.params_map().get(alg_id) or {})

    def _build_algo_row(self, alg_id: str, label: str, checks_dict: dict) -> QWidget:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QHBoxLayout()
        cb = QCheckBox(label)
        cb.setProperty("algorithm_id", alg_id)
        checks_dict[alg_id] = cb
        cb.toggled.connect(lambda *_: self._schedule_live_detect())
        header.addWidget(cb, 1)

        shared_id = get_catalog().shared_param_id(alg_id)
        base_getter = (lambda sid=shared_id: self._shared_base_params(sid)) if shared_id else None
        panel = AlgorithmParamPanel(
            alg_id,
            self.current_template_name,
            self.scheduler.param_store,
            get_std_bgr=lambda: self.std_bgr,
            get_test_bgr=lambda: self.test_bgr,
            base_params_getter=base_getter,
            show_preview=False,
            compact=True,
        )
        panel.params_changed.connect(lambda aid=alg_id: self._schedule_live_detect(aid))
        self._panel_by_alg[alg_id] = panel
        self._append_param_toggle(v, header, panel)
        return container

    def _build_shared_param_row(self, alg_id: str, label: str) -> QWidget:
        """SMT 共用参数（交互与模板建模一致）。"""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QHBoxLayout()
        title = QLabel(label)
        title.setProperty("class", "SectionLabel")
        title.setWordWrap(True)
        header.addWidget(title, 1)

        panel = AlgorithmParamPanel(
            alg_id,
            self.current_template_name,
            self.scheduler.param_store,
            get_std_bgr=lambda: self.std_bgr,
            get_test_bgr=lambda: self.test_bgr,
            show_preview=False,
            compact=True,
        )
        panel.params_changed.connect(lambda aid=alg_id: self._schedule_live_detect(aid))
        self._panel_by_alg[alg_id] = panel
        self._append_param_toggle(v, header, panel)
        return container

    def _append_param_toggle(self, v: QVBoxLayout, header: QHBoxLayout, panel: AlgorithmParamPanel) -> QToolButton:
        panel.setVisible(False)
        self._param_panels.append(panel)

        toggle = QToolButton()
        toggle.setCheckable(True)
        if panel.has_params():
            toggle.setText("参数 ▸")
        else:
            toggle.setText("无参数")
            toggle.setEnabled(False)
        header.addWidget(toggle)
        v.addLayout(header)
        v.addWidget(panel)

        def _on_toggle(checked: bool) -> None:
            panel.setVisible(checked)
            toggle.setText("参数 ▾" if checked else "参数 ▸")
            if checked:
                panel.ensure_loaded(preview=False)
                if self._open_param_toggle is not None and self._open_param_toggle is not toggle:
                    self._open_param_toggle.setChecked(False)  # 手风琴：同一时刻只展开一个
                self._open_param_toggle = toggle
            elif self._open_param_toggle is toggle:
                self._open_param_toggle = None

        toggle.toggled.connect(_on_toggle)
        return toggle

    def _schedule_live_detect(self, alg_id: str | None = None) -> None:
        self._live_focus_alg = alg_id
        self._live_timer.start()

    def _collect_param_overrides(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.bound_template is not None:
            out.update({k: dict(v) for k, v in self.bound_template.params_map().items()})
        for alg_id, panel in self._panel_by_alg.items():
            if getattr(panel, "_loaded", False) and panel.has_params():
                out[alg_id] = panel.current_values()
        return out

    def _apply_summary_display(self, summary, *, persist: bool = False) -> None:
        """把检测结果刷到公共判定条、缺陷表与测试图画布。"""
        self.last_summary = summary
        regions = self.bound_template.region_sets if self.bound_template else {}
        extra = summary.gate_note or ""
        error = summary.gate_message if summary.gate_blocked else ""
        self.result_hud.bind_summary(
            summary,
            error=error,
            test_path=self.test_path,
            template_name=self.bound_template.display_name if self.bound_template else "",
            template_version=self.bound_template.version if self.bound_template else None,
            extra=extra,
        )
        overlays = self.defect_table.bind_summary(summary, error=error)
        if self.test_bgr is not None:
            self.canvas_test.bind_job(
                self.test_bgr,
                summary,
                overlays,
                filename=self.test_path,
                region_sets=regions,
            )
        if persist:
            self.last_result_folder = self.result_store.save(
                self.current_template_name() or None,
                self.std_path or "",
                self.test_path or "memory_test.png",
                None,
                self.selected_algorithms(),
                summary,
            )

    def _on_defect_row(self, index: int) -> None:
        self.canvas_test.highlight(None if index < 0 else index)

    def _run_live_detect(self) -> None:
        """调参防抖后静默跑检测，结果实时刷到测试图/结果区（不落库）。"""
        if self.bound_template is None or self.test_bgr is None:
            return
        selected = self.selected_algorithms()
        focus = self._live_focus_alg
        if not selected:
            if not focus:
                return
            selected = [focus]
        elif focus and focus not in selected:
            selected = [focus]

        need_std = self._selection_requires_standard(selected)
        if need_std and self.std_bgr is None:
            return

        groups = self._selected_major_groups(selected)
        if len(groups) > 1:
            return

        self.template_store.sync_calibration_to_legacy(self.bound_template)
        summary = self.scheduler.run(
            self.std_bgr,
            self.test_bgr,
            self.bound_template.category or None,
            self.current_template_name() or None,
            selected,
            self.algorithm_labels(),
            param_overrides=self._collect_param_overrides(),
            prefer_frozen_params=True,
            allow_stub=True,
            region_sets=self.bound_template.region_sets or {},
            template_id=self.bound_template.id,
            template_version=self.bound_template.version,
            trigger="live",
            source="debug",
        )
        self._apply_summary_display(summary, persist=False)

    def _on_template_changed(self, tpl: InspectionTemplate | None) -> None:
        self.bound_template = tpl
        if tpl is None:
            self.std_bgr = None
            self.std_path = ""
            self.canvas_std.set_image(None)
            self.card_std.set_title("标准图（来自模板）")
            for cb in self._all_checkboxes():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.setEnabled(False)
                cb.blockSignals(False)
            for frame, _ids in self._drawer_frames:
                frame.setVisible(False)
            self._refresh_all_drawer_headers()
            self._update_calib_button()
            self._refresh_open_param_panels()
            return

        # 同步标定与冻结参数，保证适配器就绪且侧栏与建模页同源
        self.template_store.sync_calibration_to_legacy(tpl)
        self.scheduler.param_store.import_from_template(tpl.id, tpl.params_map())
        path = tpl.standard_image.path
        if path and Path(path).exists():
            self.load_standard(path)
            self.card_std.set_title(f"标准图 — {tpl.display_name}")
        else:
            self.std_bgr = None
            self.std_path = path or ""
            self.canvas_std.set_image(None)
            self.canvas_std.set_placeholder_text("模板标准图缺失，请到建模页检查")

        enabled = set(tpl.enabled_algorithm_ids())
        # 模板内允许取消勾选做子集对照；未在模板中的模块隐藏
        for alg_id, cb in self._all_checkboxes_map().items():
            cb.blockSignals(True)
            in_tpl = alg_id in enabled
            cb.setEnabled(in_tpl)
            cb.setChecked(in_tpl)
            cb.blockSignals(False)
        for frame, ids in self._drawer_frames:
            frame.setVisible(bool(ids & enabled))
        self._refresh_all_drawer_headers()
        self._update_calib_button()
        self._refresh_open_param_panels()

    def _all_checkboxes(self) -> list[QCheckBox]:
        return list(self._all_checkboxes_map().values())

    def _all_checkboxes_map(self) -> dict[str, QCheckBox]:
        out: dict[str, QCheckBox] = {}
        out.update(self.algo_checks)
        out.update(self.board_checks)
        out.update(self.surface_checks)
        return out

    def _goto_studio_for_edit(self) -> None:
        tid = self.bound_template.id if self.bound_template else ""
        self.request_goto_studio.emit(tid)

    def open_test(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择测试图", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.load_test(path)

    def load_standard(self, path: str) -> None:
        image = imread_unicode(path)
        if image is None:
            QMessageBox.warning(self, "打开失败", f"无法读取标准图：{path}")
            return
        self.std_bgr = image
        self.std_path = path
        self.card_std.set_title(f"标准图 — {Path(path).stem}")
        self.canvas_std.set_image(image)

    def load_test(self, path: str) -> None:
        image = imread_unicode(path)
        if image is None:
            QMessageBox.warning(self, "打开失败", f"无法读取测试图：{path}")
            return
        self.test_bgr = image
        self.test_path = path
        regions = self.bound_template.region_sets if self.bound_template else {}
        self.canvas_test.set_image(image, filename=path)
        self.canvas_test.set_rois(regions)
        self.canvas_test.set_defects([])
        self.last_summary = None
        self.last_result_folder = None
        self.result_hud.clear()
        self.defect_table.clear()
        self._schedule_live_detect()

    def _refresh_open_param_panels(self) -> None:
        for panel in self._param_panels:
            if getattr(panel, "_loaded", False) or panel.isVisible():
                panel.refresh(preview=False)

    # ---- 记忆（不再按类别区分，扁平记住上次勾选） ----
    def _memory(self) -> dict:
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_memory(self, selected: list[str]) -> None:
        data = self._memory()
        data["last_selected"] = selected
        self.memory_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _on_splitter_moved(self, *_args) -> None:
        # 拖动中只排队保存，避免每像素写盘；程序化 setSizes 不走这里也没关系
        self._splitter_save_timer.start()

    def _persist_splitter_sizes(self) -> None:
        data = self._memory()
        if hasattr(self, "main_splitter"):
            data["main_splitter"] = self.main_splitter.sizes()
        try:
            self.memory_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _restore_splitter_sizes(self) -> None:
        data = self._memory()
        main = data.get("main_splitter")
        if isinstance(main, list) and len(main) == 2 and all(isinstance(v, int) and v > 40 for v in main):
            self.main_splitter.setSizes(main)

    def _refresh_all_drawer_headers(self) -> None:
        for fn in self._drawer_refreshers:
            fn()

    def _update_calib_button(self) -> None:
        self.btn_calib.setText("去建模改区域")
        if self.bound_template is None:
            self.btn_calib.setToolTip("先选择模板；检测区域请在「模板建模」中编辑")
            return
        selected = self.selected_algorithms()
        kinds = {k for k in (_calib_kind_for(a) for a in selected) if k}
        tip = f"打开模板建模编辑「{self.bound_template.display_name}」的检测区域"
        if kinds:
            tip += "\n当前检测项涉及：\n" + "\n".join(
                _calib_hint_for_kind(k) for k in ("body", "smt", "th", "gold") if k in kinds
            )
        self.btn_calib.setToolTip(tip)

    def _load_memory(self) -> None:
        selected = self._memory().get("last_selected") or get_catalog().default_selected_ids()
        for alg_id, cb in self.algo_checks.items():
            cb.blockSignals(True)
            cb.setChecked(alg_id in selected)
            cb.blockSignals(False)
        for alg_id, cb in self.board_checks.items():
            cb.blockSignals(True)
            cb.setChecked(alg_id in selected)
            cb.blockSignals(False)
        for alg_id, cb in self.surface_checks.items():
            cb.blockSignals(True)
            cb.setChecked(alg_id in selected)
            cb.blockSignals(False)

    def selected_algorithms(self) -> list[str]:
        ids = [alg_id for alg_id, cb in self.algo_checks.items() if cb.isChecked()]
        ids.extend(alg_id for alg_id, cb in self.board_checks.items() if cb.isChecked())
        ids.extend(alg_id for alg_id, cb in self.surface_checks.items() if cb.isChecked())
        return ids

    def algorithm_labels(self) -> dict[str, str]:
        return get_catalog().labels()

    @staticmethod
    def _major_group_of(alg_id: str) -> str:
        return get_catalog().group_name(alg_id)

    def _selected_major_groups(self, selected: list[str]) -> list[str]:
        groups: list[str] = []
        for alg_id in selected:
            g = self._major_group_of(alg_id)
            if g not in groups:
                groups.append(g)
        return groups

    def _selection_requires_standard(self, selected: list[str]) -> bool:
        return get_catalog().requires_standard(selected)

    def current_template_name(self) -> str:
        """调试页以绑定模板 id 为标定/参数键（与模板建模一致）。"""
        if self.bound_template is not None:
            return self.bound_template.id
        return ""

    # ---- 检测 ----
    def _confirm_ready(self, selected: list[str], template_name: str | None) -> bool:
        """运行前先检查一遍标定/前置条件，未就绪的算法提前弹提示，而不是等
        跑完才在结果列表里看到 SKIP。返回 True 表示可以继续运行。"""
        not_ready: list[str] = []
        labels = self.algorithm_labels()
        for alg_id in selected:
            adapter = get_adapter_for(alg_id)
            if adapter is None:
                continue
            kind = get_catalog().region_kind(alg_id)
            rois = rois_from_region_sets(
                self.bound_template.region_sets if self.bound_template else {},
                kind,
            )
            ready, reason = adapter.is_ready(alg_id, template_name, rois=rois)
            if not ready:
                not_ready.append(f"- {labels.get(alg_id, alg_id)}：{reason}")
        if not not_ready:
            return True
        detail = "\n".join(not_ready)
        ret = QMessageBox.question(
            self,
            "存在未完成标定的算法",
            f"以下算法尚未完成标定，运行后会被跳过（不影响其余算法）：\n{detail}\n\n"
            f"是否仍要继续运行？（选“否”可先到「模板建模」补齐区域）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ret == QMessageBox.StandardButton.Yes

    def run_detect(self) -> None:
        if self.bound_template is None:
            QMessageBox.information(self, "提示", "请先选择要引用的模板")
            return
        selected = self.selected_algorithms()
        if not selected:
            QMessageBox.information(self, "提示", "请至少勾选一个算法（模板内检测项）")
            return

        groups = self._selected_major_groups(selected)
        if len(groups) > 1:
            QMessageBox.information(
                self,
                "提示",
                "请只勾选一个大类后再运行。\n当前同时勾选了：\n- " + "\n- ".join(groups),
            )
            return

        need_std = self._selection_requires_standard(selected)
        if need_std and self.std_bgr is None:
            QMessageBox.information(self, "提示", "模板标准图无效，请到「模板建模」检查")
            return
        if self.test_bgr is None:
            QMessageBox.information(self, "提示", "请先打开测试图")
            return

        template_name = self.current_template_name()
        # 确保最新区域写入 legacy
        self.template_store.sync_calibration_to_legacy(self.bound_template)
        if not self._confirm_ready(selected, template_name or None):
            return

        self._save_memory(selected)

        # 确保已展开面板的滑杆值参与跑图
        for panel in self._param_panels:
            if panel.isVisible():
                panel.ensure_loaded(preview=False)

        summary = self.scheduler.run(
            self.std_bgr,
            self.test_bgr,
            self.bound_template.category or None,
            template_name or None,
            selected,
            self.algorithm_labels(),
            param_overrides=self._collect_param_overrides(),
            prefer_frozen_params=True,
            allow_stub=True,
            region_sets=self.bound_template.region_sets or {},
            template_id=self.bound_template.id,
            template_version=self.bound_template.version,
            trigger="manual",
            source="debug",
        )
        if summary.gate_blocked:
            self._apply_summary_display(summary, persist=False)
            self.last_result_folder = None
            QMessageBox.warning(self, "质量门禁拦截", summary.gate_message)
            return
        self._apply_summary_display(summary, persist=self.cb_archive.isChecked())
