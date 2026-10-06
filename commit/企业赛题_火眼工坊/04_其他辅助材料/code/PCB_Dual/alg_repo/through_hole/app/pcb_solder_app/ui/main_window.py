"""主窗口：装配标题栏、工具条、双画布、左右面板，串联"标准图->锡面区域->待检图->检测"全流程"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSizeGrip, QSplitter, QToolButton, QVBoxLayout,
    QWidget,
)

from . import theme
from .defect_panel import DefectPanel
from .image_canvas import ImageCanvas
from .param_panel import ParamPanel
from .resize_grips import FramelessResizeGrips
from .result_panel import ResultPanel
from .solder_area_panel import SolderAreaPanel
from .title_bar import TitleBar
from .workers import AlignPreviewWorker, DetectWorker, PrebuildWorker, ServiceInitWorker
from ..core.models import (
    DefectBox, ManualRoi, TemplateConfig, bridge_joints_from_rois,
)
from ..core.template_store import TemplateStore, template_id_for_path

# ROI / 参数变化后的防抖间隔（毫秒）
_REACTIVE_DEBOUNCE_MS = 400


def _imread_unicode(path: str) -> Optional[np.ndarray]:
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("RootWindow")
        self.setWindowTitle("插件焊点缺陷检测")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setMinimumSize(1100, 700)
        self.resize(1560, 960)

        self.template_store = TemplateStore()
        self.tpl_img: Optional[np.ndarray] = None
        self.tst_img: Optional[np.ndarray] = None
        self.tcfg: Optional[TemplateConfig] = None
        self.service = None
        self._service_ready = False
        self._init_thread: Optional[QThread] = None
        self._pre_thread: Optional[QThread] = None
        self._det_thread: Optional[QThread] = None
        self._align_thread: Optional[QThread] = None
        self._pending_prebuild = False
        # 异步请求代数：丢弃过期的 prebuild / 配准预览 / 检测回调
        self._pre_gen = 0
        self._align_gen = 0
        self._det_gen = 0
        # 仍在运行的后台线程（关窗前须等其退出）
        self._active_threads: list[QThread] = []

        self._reactive_timer = QTimer(self)
        self._reactive_timer.setSingleShot(True)
        self._reactive_timer.setInterval(_REACTIVE_DEBOUNCE_MS)
        self._reactive_timer.timeout.connect(self._on_reactive_refresh)

        # 调参防抖后自动重跑检测
        self._live_detect_timer = QTimer(self)
        self._live_detect_timer.setSingleShot(True)
        self._live_detect_timer.setInterval(_REACTIVE_DEBOUNCE_MS)
        self._live_detect_timer.timeout.connect(self._on_live_detect)

        self._build_ui()
        self._resize_grips = FramelessResizeGrips(self)
        self._start_service_init()

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.title_bar = TitleBar(self)
        root.addWidget(self.title_bar)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(10, 10, 10, 6)
        bl.setSpacing(8)
        root.addWidget(body, 1)

        self.main_split = QSplitter(Qt.Horizontal)
        bl.addWidget(self.main_split, 1)

        # 左列：锡面区域 + 缺陷类型
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        self.solder_area_panel = SolderAreaPanel()
        self.defect_panel = DefectPanel()
        ll.addWidget(self.solder_area_panel)
        ll.addWidget(self.defect_panel)
        ll.addStretch(1)
        left.setMinimumWidth(280)
        self.main_split.addWidget(left)

        # 中列：工具条 + 双画布 + 结果
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        cl.addWidget(self._build_toolbar())

        self.center_split = QSplitter(Qt.Vertical)
        cl.addWidget(self.center_split, 1)

        canvas_wrap = QSplitter(Qt.Horizontal)
        self.tpl_canvas = ImageCanvas("标准图", editable=True)
        self.tst_canvas = ImageCanvas("待检图", editable=False)
        canvas_wrap.addWidget(self._wrap_panel(self.tpl_canvas))
        canvas_wrap.addWidget(self._wrap_panel(self.tst_canvas))
        canvas_wrap.setSizes([700, 700])
        self.center_split.addWidget(canvas_wrap)

        self.result_panel = ResultPanel()
        self.center_split.addWidget(self.result_panel)
        self.center_split.setSizes([620, 260])
        self.main_split.addWidget(center)

        # 右列：检测参数
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumWidth(300)
        self.main_split.addWidget(self.param_panel)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setStretchFactor(2, 0)
        self.main_split.setSizes([320, 900, 340])

        status = QWidget()
        status.setObjectName("StatusBar")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(6, 0, 4, 0)
        self.status_lbl = QLabel("正在加载算法引擎…")
        self.status_lbl.setStyleSheet(f"color: {theme.TEXT_MUTE}; font-size: 12px;")
        sl.addWidget(self.status_lbl)
        sl.addStretch(1)
        grip = QSizeGrip(self)
        grip.setToolTip("拖拽右下角缩放窗口")
        sl.addWidget(grip, 0, Qt.AlignRight | Qt.AlignBottom)
        bl.addWidget(status)

        self._wire_signals()

    def _wrap_panel(self, canvas: ImageCanvas) -> QWidget:
        f = QFrame()
        f.setProperty("class", "Panel")
        v = QVBoxLayout(f)
        v.setContentsMargins(6, 6, 6, 6)
        v.addWidget(canvas)
        return f

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setProperty("class", "Panel")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 8, 10, 8)
        h.setSpacing(8)

        self.btn_open_tpl = QPushButton("打开标准图")
        self.btn_open_tst = QPushButton("打开待检图")
        self.btn_open_tpl.clicked.connect(self._open_template)
        self.btn_open_tst.clicked.connect(self._open_test)
        h.addWidget(self.btn_open_tpl)
        h.addWidget(self.btn_open_tst)
        h.addWidget(self._sep())

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.btn_select = self._tool_btn("选择", "select", True)
        self.btn_rect = self._tool_btn("矩形框", "rect")
        self.btn_circle = self._tool_btn("圆形框", "circle")
        for b in (self.btn_select, self.btn_rect, self.btn_circle):
            h.addWidget(b)
        h.addWidget(self._sep())

        self.btn_fit = QPushButton("适应窗口")
        self.btn_fit.clicked.connect(self._fit_all)
        h.addWidget(self.btn_fit)

        h.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        h.addWidget(self.progress)

        self.btn_run = QPushButton("运行检测")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_detection)
        h.addWidget(self.btn_run)
        return bar

    def _tool_btn(self, text: str, tool: str, checked: bool = False) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setCheckable(True)
        b.setChecked(checked)
        b.setMinimumWidth(64)
        b.clicked.connect(lambda: self._set_tool(tool))
        self.tool_group.addButton(b)
        return b

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet(f"color: {theme.BORDER};")
        return f

    # ---- signals ----
    def _wire_signals(self):
        self.solder_area_panel.shape_choice_changed.connect(self._on_shape_choice_changed)
        self.solder_area_panel.clear_manual_roi_requested.connect(self._on_clear_manual_pads)

        self.defect_panel.changed.connect(self._on_defect_panel_changed)
        self.defect_panel.defect_selected.connect(self.param_panel.show_defect)

        self.param_panel.config_changed.connect(self._on_param_panel_changed)

        # 右侧参数面板默认展示左侧当前选中的缺陷类型（构造时二者尚未连线）。
        self.param_panel.show_defect(self.defect_panel.selected_defect_id())

        # 标准图始终多框追加：1 个框=其它缺陷锡面；≥2 个框=另可检连锡
        self.tpl_canvas.set_annotate_bridge(True)
        self.tpl_canvas.bridge_rois_changed.connect(self._on_manual_pads_changed)
        self.tpl_canvas.image_dropped.connect(lambda p: self._load_template(p))
        self.tst_canvas.image_dropped.connect(lambda p: self._load_test(p))

        self.result_panel.locate.connect(self._locate_defect)
        self.result_panel.visibility_changed.connect(self._on_visibility)

    # ---- service init ----
    def _start_service_init(self):
        self._init_thread = QThread(self)
        self._init_worker = ServiceInitWorker()
        self._init_worker.moveToThread(self._init_thread)
        self._init_thread.started.connect(self._init_worker.run)
        self._init_worker.ready.connect(self._on_service_ready)
        self._init_worker.failed.connect(self._on_service_failed)
        self._init_thread.finished.connect(self._init_thread.deleteLater)
        self._init_thread.finished.connect(self._init_worker.deleteLater)
        self._track_thread(self._init_thread)
        self._init_thread.start()

    def _on_service_ready(self, svc):
        self.service = svc
        self._service_ready = True
        self.status_lbl.setText("算法引擎就绪")
        self._init_thread.quit()
        self._update_run_enabled()
        if self._pending_prebuild:
            self._pending_prebuild = False
            self._prebuild_template()

    def _on_service_failed(self, msg: str):
        self.status_lbl.setText("算法引擎加载失败")
        QMessageBox.critical(self, "算法引擎加载失败", msg)
        self._init_thread.quit()

    # ---- image load ----
    def _open_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择标准图", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._load_template(path)

    def _open_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待检图", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._load_test(path)

    def _load_template(self, path: str):
        img = _imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片：\n{path}")
            return
        self.tpl_img = img
        template_id = template_id_for_path(path)
        cfg = self.template_store.load(template_id)
        if cfg is None:
            cfg = TemplateConfig(template_id=template_id, template_path=path)
            note = "新标准图，已使用默认配置（圆形自动拟合）"
        else:
            cfg.template_path = path
            note = "已载入该标准图上次保存的配置"
        self.tcfg = cfg

        self.tpl_canvas.set_image(img)
        self.tpl_canvas.set_roi(None)  # 锡面单框不再单独使用，统一为焊点列表
        pads = cfg.manual_pad_rois()
        self.tpl_canvas.set_bridge_rois(pads)
        self.tpl_canvas.set_annotate_bridge(True)
        self._sync_panels_from_tcfg()
        self._refresh_pad_status()
        self._reset_detection()
        self.status_lbl.setText(f"{note} · template_id={template_id}")
        self._prebuild_template()
        self._update_run_enabled()

    def _load_test(self, path: str):
        img = _imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片：\n{path}")
            return
        self.tst_img = img
        self.tst_canvas.set_image(img)
        self._reset_detection()
        self._update_run_enabled()
        self._refresh_roi_preview()

    def _reset_detection(self):
        self.result_panel.clear()
        self.tst_canvas.set_defects([])
        self.tst_canvas.set_visible_defect_ids(None)
        self.tst_canvas.update()

    def _sync_panels_from_tcfg(self):
        cfg = self.tcfg
        if cfg is None:
            return
        self.solder_area_panel.set_shape_choice(cfg.solder_mode)
        self.solder_area_panel.set_effective_mode(cfg.effective_solder_mode())
        n_pads = len(cfg.manual_pad_rois())
        self.defect_panel.apply_pad_count_mode(n_pads, emit=False)
        if n_pads < 2:
            self.defect_panel.set_enabled_defects(cfg.effective_enabled_defects())
        cfg.enabled_defects = self.defect_panel.enabled_defects()
        self.param_panel.set_presets(cfg.void_preset, cfg.insuf_preset)
        self.param_panel.set_advanced(cfg.advanced)

    # ---- tools ----
    def _set_tool(self, tool: str):
        self.tpl_canvas.set_tool(tool)

    def _fit_all(self):
        self.tpl_canvas.fit_to_view()
        self.tpl_canvas.update()
        self.tst_canvas.fit_to_view()
        self.tst_canvas.update()

    # ---- 锡面 / 人工焊点框 ----
    def _on_shape_choice_changed(self, mode: str):
        if self.tcfg is None:
            return
        # 仅在无人工框时切换自动方式才会生效
        self.tcfg.solder_mode = mode
        if not self.tpl_canvas.bridge_rois:
            self._schedule_reactive_refresh()

    def _apply_manual_pads_to_tcfg(self):
        """画布焊点列表 → 配置：有框则人工模式（关闭自动锡面）。"""
        if self.tcfg is None:
            return
        pads = list(self.tpl_canvas.bridge_rois)
        self.tcfg.bridge_joints = bridge_joints_from_rois(pads) if pads else None
        self.tcfg.solder_roi = pads[0] if pads else None
        self._refresh_pad_status()

    def _refresh_pad_status(self):
        n = len(self.tpl_canvas.bridge_rois)
        auto = self.solder_area_panel.shape_choice()
        if self.tcfg is not None and n == 0:
            auto = self.tcfg.solder_mode if self.tcfg.solder_mode in ("ellipse", "contour") else auto
        self.solder_area_panel.set_pad_status(n, auto_mode=auto)

    def _on_manual_pads_changed(self):
        if self.tcfg is None:
            if self.tpl_canvas.bridge_rois:
                QMessageBox.information(self, "请先打开标准图", "画框前请先打开一张标准图。")
                self.tpl_canvas.clear_bridge_rois()
            return
        self._apply_manual_pads_to_tcfg()
        n = len(self.tpl_canvas.bridge_rois)
        self.defect_panel.apply_pad_count_mode(n, emit=False)
        self.tcfg.enabled_defects = self.defect_panel.enabled_defects()
        self._reset_detection()
        self._schedule_reactive_refresh()
        if n == 0:
            self.status_lbl.setText("已清除人工框，改回自动锡面（盘内四种缺陷）")
        elif n == 1:
            self.status_lbl.setText("单焊点框：检测孔洞/少锡/多锡/不出脚；连锡请再框 ≥1 个")
        else:
            self.status_lbl.setText(f"{n} 个焊点框：仅检测连锡（焊盘之间）")

    def _on_clear_manual_pads(self):
        self.tpl_canvas.clear_bridge_rois()
        self._on_manual_pads_changed()

    # ---- 缺陷/参数 ----
    def _on_defect_panel_changed(self):
        if self.tcfg is None:
            return
        self.tcfg.enabled_defects = self.defect_panel.enabled_defects()
        self._schedule_live_detect()

    def _on_param_panel_changed(self):
        if self.tcfg is None:
            return
        self.tcfg.void_preset = self.param_panel.void_preset()
        self.tcfg.insuf_preset = self.param_panel.insuf_preset()
        self.tcfg.advanced = self.param_panel.advanced()
        self._schedule_live_detect()

    def _sync_tcfg_from_panels(self):
        if self.tcfg is None:
            return
        self.tcfg.solder_mode = self.solder_area_panel.shape_choice()
        self.tcfg.enabled_defects = self.defect_panel.enabled_defects()
        self._apply_manual_pads_to_tcfg()
        self.tcfg.void_preset = self.param_panel.void_preset()
        self.tcfg.insuf_preset = self.param_panel.insuf_preset()
        self.tcfg.enable_review = True
        self.tcfg.advanced = self.param_panel.advanced()

    # ---- 自动保存 + 配准预览(防抖) ----
    def _schedule_reactive_refresh(self):
        """ROI/锡面方式变化后防抖：重建锡面、保存配置、刷新配准预览。"""
        self._reactive_timer.start()

    def _on_reactive_refresh(self):
        self._prebuild_template()

    def _schedule_live_detect(self):
        """参数变化后防抖：保存配置并自动重跑检测。"""
        self._live_detect_timer.start()

    def _on_live_detect(self):
        if self.tcfg is None:
            return
        self._sync_tcfg_from_panels()
        self._auto_save_template_config()
        if (self._service_ready and self.service is not None
                and self.tpl_img is not None and self.tst_img is not None):
            self._run_detection(quiet=True)
        else:
            self.status_lbl.setText("参数已保存 · 打开标准图与待检图后，调参将实时刷新检测结果")

    def _auto_save_template_config(self):
        if self.tcfg is None:
            return
        self.template_store.save(self.tcfg)

    # ---- 预生成 ----
    def _prebuild_template(self):
        if self.tpl_img is None or self.tcfg is None:
            return
        if not self._service_ready or self.service is None:
            self._pending_prebuild = True
            return
        self._sync_tcfg_from_panels()
        self.status_lbl.setText("正在生成锡面区域…")
        self._pre_gen += 1
        gen = self._pre_gen
        thread = QThread(self)
        worker = PrebuildWorker(self.service, self.tpl_img, self.tcfg)
        worker.gen = gen
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # 须连到 QObject 槽（勿用 lambda），否则跨线程回调可能在子线程执行
        worker.finished.connect(self._on_prebuild_finished)
        worker.failed.connect(self._on_prebuild_failed)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        self._track_thread(thread)
        self._pre_thread = thread
        self._pre_worker = worker
        thread.start()

    def _on_prebuild_finished(self, result: dict):
        worker = self.sender()
        gen = worker.gen
        worker.thread().quit()
        if gen != self._pre_gen:
            return  # 已被更晚一次的请求取代，丢弃这个过期结果
        if result.get("ok"):
            self._refresh_pad_status()
            if self.tpl_canvas.bridge_rois:
                self.tpl_canvas.set_solder_mask(None)
            else:
                self.tpl_canvas.set_solder_mask(result.get("solder_mask"))
            self._auto_save_template_config()
            self.status_lbl.setText(
                f"锡面区域已就绪 · template_id={result.get('template_id')} · 已自动保存配置")
            self._refresh_roi_preview()
        else:
            self.tpl_canvas.set_solder_mask(None)
            self.status_lbl.setText("锡面区域生成失败")
            QMessageBox.warning(self, "锡面区域生成失败", str(result.get("error")))
        self._update_run_enabled()

    def _on_prebuild_failed(self, msg: str):
        worker = self.sender()
        gen = worker.gen
        worker.thread().quit()
        if gen != self._pre_gen:
            return
        self.status_lbl.setText("锡面区域生成失败")
        QMessageBox.critical(self, "锡面区域生成失败", msg)

    # ---- 测试图配准预览 ----
    def _refresh_roi_preview(self):
        """在测试图上叠加显示"标准图 ROI 配准映射后的对应检测区域"预览框。"""
        if self.tpl_img is None or self.tst_img is None or self.tcfg is None:
            self.tst_canvas.set_preview_roi(None)
            return
        if not self._service_ready or self.service is None:
            return
        self._align_gen += 1
        gen = self._align_gen
        thread = QThread(self)
        worker = AlignPreviewWorker(self.service, self.tpl_img, self.tst_img, self.tcfg)
        worker.gen = gen
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_align_preview_finished)
        worker.failed.connect(self._on_align_preview_failed)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        self._track_thread(thread)
        self._align_thread = thread
        self._align_worker = worker
        thread.start()

    def _on_align_preview_finished(self, result: dict):
        worker = self.sender()
        gen = worker.gen
        worker.thread().quit()
        if gen != self._align_gen:
            return  # 已被更晚一次的预览请求取代，丢弃这个过期结果
        if not result.get("ok"):
            self.tst_canvas.set_preview_rois([])
            return
        rois_raw = result.get("rois")
        if isinstance(rois_raw, list) and rois_raw:
            rois = [ManualRoi.from_dict(r) for r in rois_raw]
            self.tst_canvas.set_preview_rois([r for r in rois if r is not None])
        else:
            roi = ManualRoi.from_dict(result.get("roi"))
            self.tst_canvas.set_preview_rois([roi] if roi is not None else [])

    def _on_align_preview_failed(self, _msg: str):
        worker = self.sender()
        gen = worker.gen
        worker.thread().quit()
        if gen != self._align_gen:
            return
        self.tst_canvas.set_preview_rois([])

    # ---- 检测 ----
    def _update_run_enabled(self):
        ok = (self._service_ready and self.service is not None
              and self.tpl_img is not None and self.tst_img is not None
              and self.tcfg is not None)
        self.btn_run.setEnabled(bool(ok))

    def _run_detection(self, quiet: bool = False):
        if self.tpl_img is None or self.tst_img is None or self.tcfg is None or self.service is None:
            return
        self._sync_tcfg_from_panels()
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.status_lbl.setText("检测中…" if not quiet else "参数已更新，正在刷新结果…")

        self._det_gen += 1
        gen = self._det_gen
        thread = QThread(self)
        worker = DetectWorker(self.service, self.tpl_img, self.tst_img, self.tcfg)
        worker.gen = gen
        worker.quiet = quiet
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detect_finished)
        worker.failed.connect(self._on_detect_failed)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        self._track_thread(thread)
        self._det_thread = thread
        self._det_worker = worker
        thread.start()

    def _on_detect_finished(self, result):
        worker = self.sender()
        gen = worker.gen
        quiet = bool(getattr(worker, "quiet", False))
        worker.thread().quit()
        if gen != self._det_gen:
            return  # 已被更晚一次的检测请求取代，丢弃这个过期结果
        self.progress.setVisible(False)
        if not result.ok:
            self.result_panel.show_error(result.error or "检测失败")
            self.status_lbl.setText("检测失败")
            if not quiet:
                QMessageBox.warning(self, "检测失败", result.error or "检测失败")
        else:
            self.result_panel.show_result(result)
            self.tst_canvas.set_defects(result.defects)
            self.tst_canvas.update()
            self.status_lbl.setText(
                f"检测完成：{result.status} · 缺陷数 {len(result.defects)} · {result.cost_ms:.0f}ms")
        self._update_run_enabled()

    def _on_detect_failed(self, msg: str):
        worker = self.sender()
        gen = worker.gen
        quiet = bool(getattr(worker, "quiet", False))
        worker.thread().quit()
        if gen != self._det_gen:
            return
        self.progress.setVisible(False)
        self.status_lbl.setText("检测失败")
        if not quiet:
            QMessageBox.critical(self, "检测失败", msg)
        self._update_run_enabled()

    # ---- 结果交互 ----
    def _locate_defect(self, d: DefectBox):
        cv = self.tst_canvas
        if not cv.has_image():
            return
        cx = d.x + d.width / 2.0
        cy = d.y + d.height / 2.0
        cv.ox = cv.width() / 2.0 - cx * cv.scale
        cv.oy = cv.height() / 2.0 - cy * cv.scale
        cv.update()

    def _on_visibility(self, ids):
        self.tst_canvas.set_visible_defect_ids(ids)

    # ---- 窗口 ----
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            # 画布内部会 emit bridge_rois_changed / roi_cleared，无需再同步
            self.tpl_canvas.delete_selected_roi()
        super().keyPressEvent(e)

    def _track_thread(self, thread: QThread):
        """登记后台线程，结束后自动摘除。"""
        self._active_threads.append(thread)
        thread.finished.connect(self._untrack_thread)

    def _untrack_thread(self):
        thread = self.sender()
        if thread in self._active_threads:
            self._active_threads.remove(thread)

    def closeEvent(self, event):
        """关窗前等待后台线程退出，避免 QThread 仍在运行时被销毁。"""
        self._reactive_timer.stop()
        self._live_detect_timer.stop()
        for thread in list(self._active_threads):
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        super().closeEvent(event)
