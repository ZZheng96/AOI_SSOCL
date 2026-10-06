"""主窗口：装配标题栏、工具条、双图画布、四大面板，串联检测流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, QPointF
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSizeGrip, QSplitter, QToolButton, QVBoxLayout,
    QWidget,
)

from . import theme
from .title_bar import TitleBar
from .resize_grips import FramelessResizeGrips
from .image_canvas import ImageCanvas
from .defect_panel import DefectPanel
from .roi_panel import ROIPanel
from .param_panel import ParamPanel
from .result_panel import ResultPanel
from .workers import ServiceInitWorker, DetectWorker, RoiMatchWorker
from ..core.models import ROI, DefectBox, RunSummary
from .. import config as app_config


def _imread_unicode(path: str) -> Optional[np.ndarray]:
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("RootWindow")
        self.setWindowTitle("PCB 缺陷检测算法验证台")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setMinimumSize(1024, 680)
        self.resize(1500, 940)

        self.rois: List[ROI] = []
        self.tpl_img: Optional[np.ndarray] = None
        self.tst_img: Optional[np.ndarray] = None
        self.service = None
        self._init_thread = None
        self._det_thread = None
        self._last_summary: Optional[RunSummary] = None
        # ROI 画完/挪完后的即时结构匹配预览：按 rid 记录最新请求代号，
        # 用于丢弃过期结果；_roi_match_jobs 防止线程对象被提前回收。
        self._roi_match_gen: Dict[int, int] = {}
        self._roi_match_jobs: list = []

        self._build_ui()
        self._resize_grips = FramelessResizeGrips(self)
        self._start_service_init()

    # ------------------------------------------------------------------ UI --
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

        # 主横向分割：左 | 中 | 右
        self.main_split = QSplitter(Qt.Horizontal)
        bl.addWidget(self.main_split, 1)

        # 左列
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)
        self.defect_panel = DefectPanel()
        self.roi_panel = ROIPanel()
        ll.addWidget(self.defect_panel, 3)
        ll.addWidget(self.roi_panel, 2)
        left.setMinimumWidth(260)
        self.main_split.addWidget(left)

        # 中列
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        cl.addWidget(self._build_toolbar())

        self.center_split = QSplitter(Qt.Vertical)
        cl.addWidget(self.center_split, 1)

        canvas_wrap = QSplitter(Qt.Horizontal)
        self.tpl_canvas = ImageCanvas("模板图（金板）", editable=True)
        self.tst_canvas = ImageCanvas("待检图（来料）", editable=False)
        self.tpl_canvas.set_rois(self.rois)
        self.tst_canvas.set_rois(self.rois)
        canvas_wrap.addWidget(self._wrap_panel(self.tpl_canvas))
        canvas_wrap.addWidget(self._wrap_panel(self.tst_canvas))
        canvas_wrap.setSizes([700, 700])
        self.center_split.addWidget(canvas_wrap)

        self.result_panel = ResultPanel()
        self.center_split.addWidget(self.result_panel)
        self.center_split.setSizes([620, 260])
        self.main_split.addWidget(center)

        # 右列
        self.param_panel = ParamPanel()
        self.param_panel.setMinimumWidth(280)
        self.main_split.addWidget(self.param_panel)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setStretchFactor(2, 0)
        self.main_split.setSizes([320, 900, 340])

        # 底部状态条 + resize grip
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

        self.btn_open_tpl = QPushButton("打开模板图")
        self.btn_open_tst = QPushButton("打开待检图")
        self.btn_open_tpl.clicked.connect(lambda: self._open_image("tpl"))
        self.btn_open_tst.clicked.connect(lambda: self._open_image("tst"))
        h.addWidget(self.btn_open_tpl)
        h.addWidget(self.btn_open_tst)
        h.addWidget(self._sep())

        # 画布工具
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
        self.btn_clear = QPushButton("清空ROI")
        self.btn_clear.clicked.connect(self._clear_rois)
        h.addWidget(self.btn_clear)

        h.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
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

    # -------------------------------------------------------------- signals --
    def _wire_signals(self):
        self.defect_panel.selection_changed.connect(self._on_defect_selection)
        self.defect_panel.current_changed.connect(self.param_panel.show_defect)

        for cv in (self.tpl_canvas, self.tst_canvas):
            cv.roi_selected.connect(self._on_roi_selected)
        self.tpl_canvas.roi_created.connect(self._on_roi_created)
        self.tpl_canvas.roi_changed.connect(self._refresh_rois)
        self.tpl_canvas.roi_geometry_ready.connect(self._on_roi_geometry_ready)
        self.tpl_canvas.image_dropped.connect(lambda p: self._load_image("tpl", p))
        self.tst_canvas.image_dropped.connect(lambda p: self._load_image("tst", p))

        self.roi_panel.roi_selected.connect(self._on_roi_selected)
        self.roi_panel.roi_removed.connect(self._remove_roi)
        self.roi_panel.roi_rebound.connect(self._refresh_rois)

        self.result_panel.locate.connect(self._locate_defect)
        self.result_panel.visibility_changed.connect(self._on_visibility)

    # ------------------------------------------------------ service init ----
    def _start_service_init(self):
        self._init_thread = QThread(self)
        self._init_worker = ServiceInitWorker()
        self._init_worker.moveToThread(self._init_thread)
        self._init_thread.started.connect(self._init_worker.run)
        self._init_worker.ready.connect(self._on_service_ready)
        self._init_worker.failed.connect(self._on_service_failed)
        self._init_thread.start()

    def _on_service_ready(self, svc):
        self.service = svc
        avail = svc.available()
        roi_note = "" if svc.detector_supports_roi else " · 算法包较旧，ROI 由应用侧裁剪"
        self.status_lbl.setText(f"算法引擎就绪 · 已加载 {len(avail)} 个算法{roi_note}")
        self._init_thread.quit()
        self._update_run_enabled()

    def _on_service_failed(self, msg: str):
        self.status_lbl.setText("算法引擎加载失败")
        QMessageBox.critical(self, "算法引擎加载失败", msg)
        self._init_thread.quit()

    # ------------------------------------------------------- image load ----
    def _open_image(self, which: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._load_image(which, path)

    def _load_image(self, which: str, path: str):
        img = _imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片：\n{path}")
            return
        # 换图即清场：清除上次检测结果叠加，并清空之前画的 ROI（新样本从零开始）
        had_state = bool(self.rois) or self._last_summary is not None
        self._reset_detection()
        if self.rois:
            if self.service is not None:
                for r in self.rois:
                    self.service.forget_roi_match(r.rid)
            self._roi_match_gen.clear()
            self.rois.clear()
            self._refresh_rois()

        if which == "tpl":
            self.tpl_img = img
            self.tpl_canvas.set_image(img)
        else:
            self.tst_img = img
            self.tst_canvas.set_image(img)
        self._sync_view_hint()
        if had_state:
            self.status_lbl.setText("已载入新图，已自动清除上次的 ROI 与检测结果")
        self._update_run_enabled()

    def _reset_detection(self):
        """清除上次检测的缺陷叠加与结果表。"""
        self._last_summary = None
        self.tst_canvas.set_defects([])
        self.tst_canvas.set_matched_rois([])
        self.tst_canvas.set_visible_codes(None)
        self.tst_canvas.update()
        self.result_panel.clear()

    def _sync_view_hint(self):
        if self.tpl_img is not None and self.tst_img is not None:
            th, tw = self.tpl_img.shape[:2]
            sh, sw = self.tst_img.shape[:2]
            extra = "" if (th, tw) == (sh, sw) else "  （模板与待检尺寸不同，算法会自动对齐）"
            self.status_lbl.setText(f"模板 {tw}×{th} · 待检 {sw}×{sh}{extra}")

    # ---------------------------------------------------------- tools ----
    def _set_tool(self, tool: str):
        self.tpl_canvas.set_tool(tool)
        self.tst_canvas.set_tool("select" if tool in ("rect", "circle") else tool)

    def _fit_all(self):
        self.tpl_canvas.fit_to_view()
        self.tst_canvas.fit_to_view()

    def _clear_rois(self):
        if self.service is not None:
            for r in self.rois:
                self.service.forget_roi_match(r.rid)
        self._roi_match_gen.clear()
        self.rois.clear()
        self.tpl_canvas.selected_rid = None
        self.tst_canvas.selected_rid = None
        self._refresh_rois()
        # ROI 变了，之前基于 ROI 的检测结果已失效，一并清除叠加
        self._reset_detection()
        self._update_run_enabled()

    # ---------------------------------------------------------- ROI ----
    def _on_defect_selection(self, codes: set):
        # 新建 ROI 默认绑定"当前所有已勾选"的瑕疵类型（可多种）
        self.tpl_canvas.pending_defect_codes = set(codes)
        self._update_run_enabled()

    def _on_roi_created(self, roi: ROI):
        # 若未带任何默认绑定，兜底绑定"当前查看项"
        if not roi.defect_codes and self.param_panel._current:
            roi.defect_codes = [self.param_panel._current]
        self._refresh_rois()
        self._on_roi_selected(roi.rid)
        self._update_run_enabled()

    def _on_roi_selected(self, rid):
        self.tpl_canvas.selected_rid = rid
        self.tst_canvas.selected_rid = rid
        self.tpl_canvas.update()
        self.tst_canvas.update()
        self.roi_panel.set_selected(rid)
        # 若该 ROI 有绑定瑕疵，右侧联动显示其中一种（默认第一种）的参数
        for roi in self.rois:
            if roi.rid == rid and roi.defect_codes:
                self.param_panel.show_defect(roi.defect_codes[0])
                break

    def _remove_roi(self, rid: int):
        self.rois[:] = [r for r in self.rois if r.rid != rid]
        self._roi_match_gen.pop(rid, None)
        self.tst_canvas.set_matched_rois(
            [m for m in self.tst_canvas.matched_rois if m.rid != rid])
        if self.service is not None:
            self.service.forget_roi_match(rid)
        self._refresh_rois()
        self._update_run_enabled()

    def _on_roi_geometry_ready(self, roi: ROI):
        """ROI 刚画完或挪完：立即在整图范围内做一次结构匹配预览，不必等点击"运行检测"。"""
        if self.service is None or self.tpl_img is None or self.tst_img is None:
            return
        rid = roi.rid
        gen = self._roi_match_gen.get(rid, 0) + 1
        self._roi_match_gen[rid] = gen

        thread = QThread(self)
        worker = RoiMatchWorker(self.service, self.tpl_img, self.tst_img, roi)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda matched: self._on_roi_match_ready(rid, gen, matched))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        job = (thread, worker)
        self._roi_match_jobs.append(job)
        thread.finished.connect(lambda j=job: self._roi_match_jobs.remove(j)
                                 if j in self._roi_match_jobs else None)
        thread.start()

    def _on_roi_match_ready(self, rid: int, gen: int, matched):
        if self._roi_match_gen.get(rid) != gen:
            return  # 该 ROI 又有了更新的匹配请求，丢弃这个过期结果
        merged = [m for m in self.tst_canvas.matched_rois if m.rid != rid]
        merged.append(matched)
        self.tst_canvas.set_matched_rois(merged)

    def _refresh_rois(self):
        self.tpl_canvas.set_rois(self.rois)
        self.tst_canvas.set_rois(self.rois)
        self.roi_panel.rebuild(self.rois, self.tpl_canvas.selected_rid)
        self.tpl_canvas.update()
        self.tst_canvas.update()
        self._update_run_enabled()

    # ---------------------------------------------------------- 检测 ----
    def _build_tasks(self) -> List[dict]:
        active = set(self.defect_panel.active_codes())
        bound = {c for r in self.rois for c in r.defect_codes}
        codes = active | bound
        tasks: List[dict] = []
        for code in codes:
            cfg = self.param_panel.get_config(code)
            # 同一个 ROI 若绑定了多种瑕疵类型，会分别为每种类型生成一条任务，
            # 因此同一区域可以被多种已启用的检测算法各自检测一遍。
            rois_for = [r for r in self.rois if code in r.defect_codes]
            if rois_for:
                for r in rois_for:
                    tasks.append({"defect_code": code, "config": cfg,
                                  "roi": r, "roi_name": r.name})
            else:
                tasks.append({"defect_code": code, "config": cfg,
                              "roi": None, "roi_name": "整图"})
        return tasks

    def _warn_no_task(self):
        """无检测任务时给出针对性的说明。"""
        unbound = [r for r in self.rois if not r.defect_codes]
        active = self.defect_panel.active_codes()
        if self.rois and unbound and not active:
            msg = (f"当前有 {len(unbound)} 个 ROI 尚未指定要检测的瑕疵类型，"
                   f"因此没有可执行的检测任务。\n\n"
                   f"请在左下「ROI 重点区域」列表里，为每个框在右侧勾选一种或多种瑕疵"
                   f"（如：极反 / 错件 / 破损…可多选），或在左上「瑕疵类型」中勾选后再运行。")
            title = "ROI 未绑定瑕疵类型"
        elif self.rois and unbound:
            msg = (f"有 {len(unbound)} 个 ROI 未绑定瑕疵类型，将被忽略。\n"
                   f"请为它们在「ROI 重点区域」中勾选一种或多种瑕疵类型，或先补全再运行。")
            title = "部分 ROI 未绑定"
        else:
            msg = ("还没有可执行的检测任务。\n\n"
                   "请先在左上「瑕疵类型」勾选至少一种要检测的类型；"
                   "如需聚焦局部，可用工具条的「矩形框/圆形框」在模板图上画 ROI 并绑定瑕疵类型。")
            title = "请先选择检测内容"
        QMessageBox.information(self, title, msg)

    def _update_run_enabled(self):
        ok = (self.service is not None and self.tpl_img is not None
              and self.tst_img is not None and len(self._build_tasks()) > 0)
        self.btn_run.setEnabled(ok)

    def _run_detection(self):
        tasks = self._build_tasks()
        if not tasks:
            self._warn_no_task()
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(tasks))
        self.progress.setValue(0)
        self.status_lbl.setText("检测中…")

        self._det_thread = QThread(self)
        self._det_worker = DetectWorker(
            self.service, self.tpl_img, self.tst_img, tasks,
            sessions_dir=app_config.DEFAULT_SESSIONS_DIR)
        self._det_worker.moveToThread(self._det_thread)
        self._det_thread.started.connect(self._det_worker.run)
        self._det_worker.progress.connect(self._on_progress)
        self._det_worker.finished.connect(self._on_detect_finished)
        self._det_worker.failed.connect(self._on_detect_failed)
        self._det_thread.start()

    def _on_progress(self, done: int, total: int, text: str):
        self.progress.setValue(done)
        self.status_lbl.setText(f"检测中… {done}/{total}  {text}")

    def _on_detect_finished(self, summary: RunSummary):
        self._last_summary = summary
        self.progress.setVisible(False)
        self.tst_canvas.set_defects(summary.all_defects)
        self.tst_canvas.set_matched_rois(summary.matched_rois)
        self.tst_canvas.update()
        self.result_panel.show_summary(summary)
        verdict = "NG" if summary.ng_count else ("异常" if summary.err_count else "OK")
        self.status_lbl.setText(
            f"检测完成：{verdict} · NG {summary.ng_count} / OK {summary.ok_count} / "
            f"异常 {summary.err_count} · {summary.total_ms:.0f}ms · 已自动保存")
        self._det_thread.quit()
        self._update_run_enabled()

    def _on_detect_failed(self, msg: str):
        self.progress.setVisible(False)
        self.status_lbl.setText("检测失败")
        QMessageBox.critical(self, "检测失败", msg)
        self._det_thread.quit()
        self._update_run_enabled()

    # ---------------------------------------------------------- 结果交互 ----
    def _locate_defect(self, d: DefectBox):
        cv = self.tst_canvas
        if not cv.has_image():
            return
        cx = d.x + d.width / 2.0
        cy = d.y + d.height / 2.0
        cv.ox = cv.width() / 2.0 - cx * cv.scale
        cv.oy = cv.height() / 2.0 - cy * cv.scale
        cv.selected_rid = None
        cv.update()

    def _on_visibility(self, codes):
        self.tst_canvas.set_visible_codes(codes)

    # ---------------------------------------------------------- 窗口 ----
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete and self.tpl_canvas.selected_rid is not None:
            self._remove_roi(self.tpl_canvas.selected_rid)
        super().keyPressEvent(e)
