"""自动检测（操作台）：选模板、上料、过板、复判。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.detect.catalog import get_catalog
from app.inspect.service import InspectRunResult, InspectService
from app.inspect.worker import InspectWorker
from app.result.store import ResultStore
from app.template.model import InspectionTemplate
from app.template.store import TemplateStore
from app.ui import theme
from app.ui.defect_table import DefectTable
from app.ui.inspect_canvas import InspectCanvas
from app.ui.result_hud import ResultHud
from app.ui.template_picker import TemplatePicker
from app.ui.widgets import Card, ToolbarStrip
from app.utils.cv_io import imread_unicode


class InspectPage(QWidget):
    request_goto_studio = Signal(str)
    session_stats_changed = Signal()

    def __init__(
        self,
        inspect_service: InspectService | None = None,
        template_store: TemplateStore | None = None,
        result_store: ResultStore | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.template_store = template_store or TemplateStore()
        self.result_store = result_store or ResultStore()
        self.service = inspect_service or InspectService(
            result_store=self.result_store,
            template_store=self.template_store,
        )
        self.current_template: InspectionTemplate | None = None
        self.queue: list[Path] = []
        self.current_index = -1
        self.last_run: InspectRunResult | None = None
        self.last_folder: Path | None = None
        self._runs_by_path: dict[str, InspectRunResult] = {}
        self._worker: InspectWorker | None = None
        self._single_index = -1  # 单张检测时记录队列索引（worker 内子队列索引恒为 0）
        self._queue_filter = "all"  # all | ng | error
        self._build()
        self._bind_shortcuts()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        top = ToolbarStrip()
        self.picker = TemplatePicker(self.template_store)
        top.layout_.addWidget(self.picker, 1)
        self.lbl_queue_meta = QLabel("0/0")
        self.lbl_queue_meta.setProperty("class", "SectionLabel")
        self.lbl_yield = QLabel("直通率 —")
        self.lbl_yield.setProperty("class", "Hint")
        top.layout_.addWidget(self.lbl_queue_meta)
        top.layout_.addWidget(self.lbl_yield)
        self.btn_studio = QPushButton("建模")
        self.btn_studio.setObjectName("Ghost")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setVisible(False)
        self.btn_run = QPushButton("开始检测")
        self.btn_run.setObjectName("Primary")
        self.btn_run.setMinimumWidth(140)
        top.layout_.addWidget(self.btn_studio)
        top.layout_.addWidget(self.btn_stop)
        top.layout_.addWidget(self.btn_run)
        root.addWidget(top)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)
        queue_btns = QHBoxLayout()
        self.btn_add_files = QPushButton("导入图片")
        self.btn_add_folder = QPushButton("导入文件夹")
        self.btn_clear_queue = QPushButton("清空")
        self.btn_clear_queue.setObjectName("Ghost")
        queue_btns.addWidget(self.btn_add_files)
        queue_btns.addWidget(self.btn_add_folder)
        queue_btns.addWidget(self.btn_clear_queue)
        left_lay.addLayout(queue_btns)
        filt = QHBoxLayout()
        self.btn_filt_all = QToolButton()
        self.btn_filt_all.setText("全部")
        self.btn_filt_all.setCheckable(True)
        self.btn_filt_all.setChecked(True)
        self.btn_filt_ng = QToolButton()
        self.btn_filt_ng.setText("NG")
        self.btn_filt_ng.setCheckable(True)
        self.btn_filt_err = QToolButton()
        self.btn_filt_err.setText("ERROR")
        self.btn_filt_err.setCheckable(True)
        self.btn_next_ng = QPushButton("下一张 NG")
        self.btn_next_ng.setObjectName("Ghost")
        filt.addWidget(self.btn_filt_all)
        filt.addWidget(self.btn_filt_ng)
        filt.addWidget(self.btn_filt_err)
        filt.addWidget(self.btn_next_ng)
        left_lay.addLayout(filt)
        self.list_queue = QListWidget()
        self.list_queue.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left_lay.addWidget(self.list_queue, 1)
        splitter.addWidget(left)

        self.inspect_canvas = InspectCanvas("检测图")
        self.inspect_canvas.set_placeholder("导入测试图 · Enter 测当前 · Ctrl+Enter 整批")
        splitter.addWidget(self.inspect_canvas)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(10)
        hud_card = Card("判定", elevated=True)
        self.result_hud = ResultHud()
        hud_card.body.addWidget(self.result_hud)
        right_lay.addWidget(hud_card)

        table_card = Card("缺陷")
        self.defect_table = DefectTable()
        table_card.body.addWidget(self.defect_table, 1)
        right_lay.addWidget(table_card, 1)

        rejudge = QHBoxLayout()
        self.btn_fp = QPushButton("复判合格")
        self.btn_fp.setObjectName("Ghost")
        self.btn_confirm_ng = QPushButton("复判不合格")
        self.btn_confirm_ng.setObjectName("Danger")
        rejudge.addWidget(self.btn_fp)
        rejudge.addWidget(self.btn_confirm_ng)
        right_lay.addLayout(rejudge)

        self.box_rules = Card("模板摘要")
        self.lbl_rules = QLabel("选择已发布模板")
        self.lbl_rules.setWordWrap(True)
        self.lbl_rules.setProperty("class", "Hint")
        self.box_rules.body.addWidget(self.lbl_rules)
        right_lay.addWidget(self.box_rules)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 720, 380])
        root.addWidget(splitter, 1)

        self.lbl_status = QLabel("Enter 当前张 · Ctrl+Enter 整批 · N 下一 NG · 1/2 复判 · F 自适应 · Esc 停批")
        self.lbl_status.setProperty("class", "Hint")
        root.addWidget(self.lbl_status)

        self.picker.template_changed.connect(self._on_template)
        self.btn_studio.clicked.connect(self._goto_studio)
        self.btn_run.clicked.connect(self._start_detect)
        self.btn_stop.clicked.connect(self._cancel_batch)
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_clear_queue.clicked.connect(self._clear_queue)
        self.list_queue.currentRowChanged.connect(self._on_queue_row)
        self.btn_fp.clicked.connect(lambda: self._manual_verdict("false_positive"))
        self.btn_confirm_ng.clicked.connect(lambda: self._manual_verdict("confirmed_ng"))
        self.defect_table.defect_selected.connect(self._on_defect_row)
        self.inspect_canvas.defect_clicked.connect(self.defect_table.select_index)
        self.inspect_canvas.image_dropped.connect(lambda p: self._append_paths([Path(p)]))
        self.btn_filt_all.clicked.connect(lambda: self._set_queue_filter("all"))
        self.btn_filt_ng.clicked.connect(lambda: self._set_queue_filter("ng"))
        self.btn_filt_err.clicked.connect(lambda: self._set_queue_filter("error"))
        self.btn_next_ng.clicked.connect(self._next_ng)
        self._update_run_enabled()

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Return), self, self._run_current)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self._run_current)
        QShortcut(QKeySequence("Ctrl+Return"), self, self._run_batch)
        QShortcut(QKeySequence("N"), self, self._next_ng)
        QShortcut(QKeySequence("1"), self, lambda: self._manual_verdict("false_positive"))
        QShortcut(QKeySequence("2"), self, lambda: self._manual_verdict("confirmed_ng"))
        QShortcut(QKeySequence("F"), self, self.inspect_canvas.fit)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._cancel_batch)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.picker.refresh(keep_id=self.current_template.id if self.current_template else None)

    def select_template(self, template_id: str) -> None:
        self.picker.refresh(keep_id=template_id)

    def session_counts(self) -> tuple[int, int, int]:
        ok = ng = err = 0
        for run in self._runs_by_path.values():
            tag = self._run_tag(run)
            if tag == "OK":
                ok += 1
            elif tag in ("NG", "GRAY"):
                ng += 1  # 灰区待复判按不通过口径统计，不计入直通
            else:
                err += 1
        return ok, ng, err

    def header_snapshot(self) -> dict:
        tpl = self.current_template
        ok, ng, err = self.session_counts()
        done = ok + ng + err
        yield_txt = f"{100 * ok / (ok + ng):.0f}%" if (ok + ng) else "—"
        return {
            "template": tpl.display_name if tpl else "未选模板",
            "version": tpl.version if tpl else None,
            "index": max(self.current_index + 1, 0) if self.queue else 0,
            "total": len(self.queue),
            "yield": yield_txt,
            "running": self._worker is not None and self._worker.isRunning(),
        }

    def _on_template(self, tpl: InspectionTemplate | None) -> None:
        self.current_template = tpl
        self._fill_rules(tpl)
        self._update_run_enabled()
        self._refresh_current_view()
        self.session_stats_changed.emit()

    def _fill_rules(self, tpl: InspectionTemplate | None) -> None:
        if tpl is None:
            self.lbl_rules.setText("选择已发布模板")
            return
        catalog = get_catalog()
        labels = catalog.labels()
        std = tpl.standard_image.display_name
        items = [labels.get(i.algorithm_id, i.algorithm_id) for i in tpl.enabled_detection_items() if catalog.is_job_item(i.algorithm_id)]
        self.lbl_rules.setText(
            f"{tpl.display_name}  v{tpl.version}\n标准图 {std}\n检测项：{'、'.join(items) or '—'}"
        )

    def _update_run_enabled(self) -> None:
        running = self._worker is not None and self._worker.isRunning()
        enabled = bool(self.queue) and self.current_template is not None and not running
        if enabled and self.current_template is not None:
            ok, _msg = self.service.can_run(self.current_template)
            enabled = ok
        self.btn_run.setEnabled(enabled)
        self.btn_stop.setVisible(running)

    def _goto_studio(self) -> None:
        self.request_goto_studio.emit(self.current_template.id if self.current_template else "")

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择测试图", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if paths:
            self._append_paths([Path(p) for p in paths])

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择测试图文件夹")
        if not folder:
            return
        files = self.service.collect_images_in_folder(folder)
        if not files:
            QMessageBox.information(self, "提示", "文件夹内没有图片")
            return
        self._append_paths(files)

    def _append_paths(self, paths: list[Path]) -> None:
        existing = {str(p) for p in self.queue}
        for p in paths:
            if str(p) not in existing:
                self.queue.append(p)
                existing.add(str(p))
        self._reload_queue_list()
        if self.queue and self.list_queue.currentRow() < 0:
            self.list_queue.setCurrentRow(0)
        self._update_run_enabled()
        self.session_stats_changed.emit()

    def _clear_queue(self) -> None:
        self.queue.clear()
        self.current_index = -1
        self._runs_by_path.clear()
        self.last_run = None
        self.last_folder = None
        self._reload_queue_list()
        self.inspect_canvas.set_image(None)
        self.inspect_canvas.set_defects([])
        self.result_hud.clear()
        self.defect_table.clear()
        self._update_run_enabled()
        self.session_stats_changed.emit()

    def _set_queue_filter(self, filt: str) -> None:
        self._queue_filter = filt
        self.btn_filt_all.setChecked(filt == "all")
        self.btn_filt_ng.setChecked(filt == "ng")
        self.btn_filt_err.setChecked(filt == "error")
        self._reload_queue_list()

    def _visible_indices(self) -> list[int]:
        out = []
        for i, path in enumerate(self.queue):
            run = self._runs_by_path.get(str(path))
            tag = self._run_tag(run) if run else ""
            if self._queue_filter == "ng" and tag not in ("NG", "GRAY"):
                # 口径与 session_counts/_next_ng 对齐：灰区待复判按不通过计入 NG 过滤
                continue
            if self._queue_filter == "error" and tag != "ERR":
                continue
            out.append(i)
        return out

    def _reload_queue_list(self) -> None:
        visible = self._visible_indices()
        keep = self.queue[self.current_index] if 0 <= self.current_index < len(self.queue) else None
        self.list_queue.blockSignals(True)
        self.list_queue.clear()
        select = 0
        for row, qi in enumerate(visible):
            p = self.queue[qi]
            run = self._runs_by_path.get(str(p))
            tag = self._run_tag(run) if run else ""
            item = QListWidgetItem(f"[{tag}] {p.name}" if tag else p.name)
            item.setData(Qt.ItemDataRole.UserRole, qi)
            if tag == "OK":
                item.setForeground(QColor(theme.OK))
            elif tag == "NG":
                item.setForeground(QColor(theme.NG))
            elif tag == "GRAY":
                item.setForeground(QColor(theme.REVIEW))
            elif tag == "ERR":
                item.setForeground(QColor(theme.ERROR))
            self.list_queue.addItem(item)
            if keep is not None and p == keep:
                select = row
        self.list_queue.blockSignals(False)
        if visible:
            self.list_queue.setCurrentRow(select)
        self._refresh_queue_meta()

    def _refresh_queue_meta(self) -> None:
        ok, ng, err = self.session_counts()
        n = len(self.queue)
        idx = self.current_index + 1 if 0 <= self.current_index < n else 0
        self.lbl_queue_meta.setText(f"{idx}/{n}")
        denom = ok + ng
        self.lbl_yield.setText(f"直通率 {100 * ok / denom:.0f}%" if denom else "直通率 —")

    def _run_tag(self, run: InspectRunResult) -> str:
        if run.error or run.summary.gate_blocked or run.summary.overall == "ERROR":
            return "ERR"
        if run.summary.overall == "GRAY":
            return "GRAY"  # 灰区待复判：不计 OK，也不混进 NG
        return "OK" if run.summary.overall_ok else "NG"

    def _on_queue_row(self, row: int) -> None:
        item = self.list_queue.item(row)
        if item is None:
            return
        qi = int(item.data(Qt.ItemDataRole.UserRole))
        self.current_index = qi
        self._refresh_current_view()
        self._refresh_queue_meta()
        self.session_stats_changed.emit()

    def _on_defect_row(self, index: int) -> None:
        self.inspect_canvas.highlight(index if index >= 0 else None)

    def _current_path(self) -> Path | None:
        if 0 <= self.current_index < len(self.queue):
            return self.queue[self.current_index]
        return None

    def _std_wh(self) -> tuple[int, int] | None:
        tpl = self.current_template
        if tpl is None:
            return None
        ref = tpl.standard_image
        if ref.width > 0 and ref.height > 0:
            return ref.width, ref.height
        if ref.path:
            std = imread_unicode(ref.path)
            if std is not None:
                return int(std.shape[1]), int(std.shape[0])
        return None

    def _load_display_image(self, path: Path) -> np.ndarray | None:
        img = imread_unicode(path)
        if img is None:
            return None
        wh = self._std_wh()
        if wh is None:
            return img
        tw, th = wh
        if img.shape[1] == tw and img.shape[0] == th:
            return img
        interp = cv2.INTER_AREA if (img.shape[1] > tw or img.shape[0] > th) else cv2.INTER_LINEAR
        return cv2.resize(img, (tw, th), interpolation=interp)

    def _refresh_current_view(self) -> None:
        path = self._current_path()
        if path is None:
            self.last_run = None
            self.last_folder = None
            self.inspect_canvas.set_image(None)
            self.inspect_canvas.set_defects([])
            self.result_hud.clear()
            self.defect_table.clear()
            return
        img = self._load_display_image(path)
        run = self._runs_by_path.get(str(path))
        rois = self.current_template.region_sets if self.current_template else None
        if run is None:
            # 未检测图：必须清空上次检测上下文，否则复判会错写到上一张图的归档
            self.last_run = None
            self.last_folder = None
            self.inspect_canvas.set_image(img, filename=str(path))
            self.inspect_canvas.set_rois(rois)
            self.inspect_canvas.set_defects([])
            self.result_hud.clear()
            self.defect_table.clear()
            return
        self.last_run = run
        self.last_folder = run.folder
        overlays = self.defect_table.bind_summary(run.summary, error=run.error or run.summary.gate_message)
        self.result_hud.bind_summary(
            run.summary,
            error=run.error,
            test_path=str(path),
            template_name=self.current_template.display_name if self.current_template else "",
            template_version=self.current_template.version if self.current_template else None,
        )
        self.inspect_canvas.bind_job(img, run.summary, overlays, filename=str(path), region_sets=rois)

    def _start_detect(self) -> None:
        if not self.queue or self.current_template is None:
            return
        ok, msg = self.service.can_run(self.current_template)
        if not ok:
            QMessageBox.warning(self, "无法检测", msg)
            return
        if len(self.queue) == 1:
            self._run_one(0)
            return
        self._run_batch()

    def _run_current(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self.current_template is None:
            return
        if not (0 <= self.current_index < len(self.queue)):
            return
        ok, msg = self.service.can_run(self.current_template)
        if not ok:
            QMessageBox.warning(self, "无法检测", msg)
            return
        self._run_one(self.current_index)

    def _run_one(self, index: int) -> None:
        if self.current_template is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        path = self.queue[index]
        self.current_index = index
        self._single_index = index
        self.result_hud.set_running("检测中…")
        # 单张同样走后台线程：同步检测会阻塞 UI 事件循环造成假死
        self._worker = InspectWorker(self.service, self.current_template, [path], allow_stub=False, parent=self)
        self._worker.item_done.connect(self._on_single_item)
        self._worker.batch_finished.connect(self._on_single_finished)
        self._worker.failed.connect(lambda m: QMessageBox.warning(self, "检测失败", m))
        self._update_run_enabled()
        self._worker.start()

    def _on_single_item(self, _idx: int, run: object) -> None:
        if not isinstance(run, InspectRunResult):
            return
        if 0 <= self._single_index < len(self.queue):
            self._apply_run(self._single_index, run)

    def _on_single_finished(self, _batch: object) -> None:
        self._worker = None
        self._reload_queue_list()
        self._update_run_enabled()

    def _apply_run(self, index: int, run: InspectRunResult) -> None:
        path = self.queue[index]
        self.last_run = run
        self.last_folder = run.folder
        self._runs_by_path[str(path)] = run
        self.current_index = index
        self._refresh_current_view()
        self.session_stats_changed.emit()

    def _run_batch(self) -> None:
        if self.current_template is None or not self.queue:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        ok, msg = self.service.can_run(self.current_template)
        if not ok:
            QMessageBox.warning(self, "无法检测", msg)
            return
        self._worker = InspectWorker(self.service, self.current_template, self.queue, allow_stub=False, parent=self)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.item_done.connect(self._on_batch_item)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.failed.connect(lambda m: QMessageBox.warning(self, "批量失败", m))
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.queue))
        self.progress.setValue(0)
        self.result_hud.set_running("批量检测中…")
        self._update_run_enabled()
        self._worker.start()

    def _on_batch_progress(self, i: int, total: int, path: str) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(i)
        self.lbl_status.setText(f"检测中 {i}/{total}  {Path(path).name}")

    def _on_batch_item(self, index: int, run: object) -> None:
        if not isinstance(run, InspectRunResult):
            return
        self._runs_by_path[str(self.queue[index])] = run
        self.current_index = index
        self._refresh_current_view()
        self.session_stats_changed.emit()

    def _on_batch_finished(self, batch: object) -> None:
        self.progress.setVisible(False)
        self._worker = None
        self._reload_queue_list()
        self._update_run_enabled()
        first_ng = None
        for i, p in enumerate(self.queue):
            run = self._runs_by_path.get(str(p))
            if run is not None and self._run_tag(run) in ("NG", "GRAY"):
                first_ng = i
                break
        if first_ng is not None:
            self.current_index = first_ng
            self._reload_queue_list()
        cancelled = getattr(batch, "cancelled", False)
        if cancelled:
            self.lbl_status.setText("已取消")
        elif first_ng is not None:
            self.lbl_status.setText("批量完成 · 已停在首张 NG")
        else:
            self.lbl_status.setText("批量完成")
        self.session_stats_changed.emit()

    def _cancel_batch(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.lbl_status.setText("正在停止…")

    def _next_ng(self) -> None:
        start = self.current_index + 1
        for i in range(start, len(self.queue)):
            run = self._runs_by_path.get(str(self.queue[i]))
            if run is not None and self._run_tag(run) in ("NG", "GRAY"):
                self.current_index = i
                self._reload_queue_list()
                return
        QMessageBox.information(self, "提示", "没有后续 NG")

    def _manual_verdict(self, verdict: str) -> None:
        try:
            self._manual_verdict_impl(verdict)
        except Exception as exc:  # noqa: BLE001 槽函数异常会穿过 Qt 事件循环 abort 进程，必须兜底
            QMessageBox.warning(self, "复判失败", str(exc))

    def _manual_verdict_impl(self, verdict: str) -> None:
        path = self._current_path()
        # 校验复判对象确为当前图片的本次检测，防止误写到其它图的归档
        if (
            self.last_folder is None
            or self.last_run is None
            or path is None
            or str(path) != self.last_run.test_path
        ):
            QMessageBox.information(self, "提示", "请先对当前图片完成检测")
            return
        if not self.result_store.set_manual_verdict(self.last_folder, None, verdict):
            QMessageBox.warning(self, "复判失败", "归档写入失败，请检查输出目录权限")
            return
        extra = "复判合格" if verdict == "false_positive" else "复判不合格"
        learn_note = self._submit_verdict_feedback(verdict)
        if learn_note:
            extra = f"{extra} · {learn_note}"
        self.result_hud.bind_summary(
            self.last_run.summary,
            error=self.last_run.error,
            test_path=self.last_run.test_path,
            template_name=self.current_template.display_name if self.current_template else "",
            template_version=self.current_template.version if self.current_template else None,
            extra=extra,
        )

    def _submit_verdict_feedback(self, verdict: str) -> str:
        """复判即学：把人工复判结论反馈给模板绑定的品类模型（学习闭环）。

        返回给 HUD 的备注；未绑定品类或机器无有效判定时返回空串。
        语义映射（label：0=正常 1=缺陷；与 routes_feedback 的 /feedback 口径一致）：
        - 复判合格：机器 NG/GRAY → wrong+0（误报放行）；机器 OK → correct+0
        - 复判不合格：机器 NG/GRAY → correct+1（确认缺陷）；机器 OK → wrong+1（漏检）
        """
        tpl = self.current_template
        run = self.last_run
        if tpl is None or run is None or not run.test_path:
            return ""
        category = (tpl.model_category or "").strip() or (tpl.category or "").strip()
        if not category:
            return ""
        summary = run.summary
        overall = summary.overall or (
            "ERROR" if summary.gate_blocked else ("OK" if summary.overall_ok else "NG")
        )
        if overall == "ERROR" or run.error:
            return ""  # 机器未给出有效判定，不作为学习样本
        # 学习样本以 AOI 引擎自身判定为准（双检时融合结论可能来自传统 CV，
        # 不能把传统 CV 的 NG 记成 AOI 的误报）；无 AOI 行时回退融合结论
        aoi_meta = next((r.metadata for r in summary.results
                         if r.item_id == "aoi_feature"), None) or {}
        aoi_decision = aoi_meta.get("decision")
        if aoi_decision in ("anomaly", "gray", "normal"):
            machine_ng = aoi_decision in ("anomaly", "gray")
        else:
            machine_ng = overall in ("NG", "GRAY")
        human_ng = verdict == "confirmed_ng"
        if human_ng:
            fb_verdict, label = ("correct", 1) if machine_ng else ("wrong", 1)
        else:
            fb_verdict, label = ("wrong", 0) if machine_ng else ("correct", 0)
        try:
            from app.engines.feature import get_engine

            info = get_engine().submit_feedback(category, run.test_path, fb_verdict, label=label,
                                                detection_id=aoi_meta.get("detection_id"))
        except Exception as exc:  # noqa: BLE001 学习失败不影响已落盘的复判结论
            return f"学习反馈未生效（{str(exc)[:60]}）"
        if isinstance(info, dict) and info.get("duplicate"):
            return "该次检测已反馈过（未重复学习）"
        return "已反馈学习（AOI_Core）" if aoi_meta.get("backend") == "aoi_core" else "已反馈学习"
