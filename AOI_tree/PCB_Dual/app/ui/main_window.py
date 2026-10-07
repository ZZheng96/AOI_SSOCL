from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import ensure_dirs, get_outputs_dir, set_outputs_dir
from app.detect.registry import load_errors
from app.detect.scheduler import DetectScheduler
from app.inspect.service import InspectService
from app.recipe.param_store import ParamStore
from app.recipe.store import RecipeStore
from app.result.store import ResultStore
from app.template.migrate import migrate_legacy_to_templates
from app.template.store import TemplateStore
from app.ui import theme
from app.ui.detect_page import DetectPage
from app.ui.history_page import HistoryPage
from app.ui.inspect_page import InspectPage
from app.ui.preprocess_page import PreprocessPage
from app.ui.settings_dialog import SettingsDialog
from app.ui.template_studio_page import TemplateStudioPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("PCB 缺陷检测系统")
        self.resize(1540, 920)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(theme.QSS)

        self.recipe_store = RecipeStore()
        self.param_store = ParamStore()
        self.scheduler = DetectScheduler(self.recipe_store, self.param_store)
        self.result_store = ResultStore()
        self.template_store = TemplateStore()
        self.inspect_service = InspectService(
            scheduler=self.scheduler,
            result_store=self.result_store,
            template_store=self.template_store,
        )

        shell = QWidget()
        shell_lay = QVBoxLayout(shell)
        shell_lay.setContentsMargins(0, 0, 0, 0)
        shell_lay.setSpacing(0)
        shell_lay.addWidget(self._build_header())

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(12, 10, 12, 10)
        body_lay.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setDocumentMode(True)

        self.inspect_page = InspectPage(
            inspect_service=self.inspect_service,
            template_store=self.template_store,
            result_store=self.result_store,
        )
        self.history_page = HistoryPage(self.result_store)
        self.studio_page = TemplateStudioPage(
            template_store=self.template_store,
            param_store=self.param_store,
            inspect_service=self.inspect_service,
        )
        self.preprocess_page = PreprocessPage(self.recipe_store)
        self.detect_page = DetectPage(
            self.scheduler,
            self.result_store,
            template_store=self.template_store,
        )
        # 树枝/树干：特征学习侧（数据/模型/评估/复核/学习/统计）直接用树干 AOI_Core 的页面
        from app.ui.core_pages import CoreHubPage
        self.model_page = CoreHubPage()
        self._model_tab_name = "特征学习（AOI_Core）"

        self._tab_inspect = self.tabs.addTab(self.inspect_page, "自动检测")
        self._tab_history = self.tabs.addTab(self.history_page, "历史")
        self._tab_studio = self.tabs.addTab(self.studio_page, "模板建模")
        self._tab_models = self.tabs.addTab(self.model_page, self._model_tab_name)
        self._tab_detect = self.tabs.addTab(self.detect_page, "算法调试")
        self._tab_preprocess = self.tabs.addTab(self.preprocess_page, "预处理")
        body_lay.addWidget(self.tabs, 1)
        shell_lay.addWidget(body, 1)
        self.setCentralWidget(shell)

        self.studio_page.request_open_inspect.connect(self._goto_inspect_with_template)
        self.studio_page.request_preprocess.connect(self._goto_preprocess_from_studio)
        self.inspect_page.request_goto_studio.connect(self._goto_studio)
        self.inspect_page.session_stats_changed.connect(self._refresh_header_runtime)
        self.detect_page.request_goto_studio.connect(self._goto_studio)
        self.detect_page.request_preprocess.connect(self._goto_preprocess_with_current)
        self.preprocess_page.btn_use_detect.clicked.connect(self._goto_preprocess_with_current)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._build_menu()
        self.statusBar().showMessage("就绪")
        self._check_adapter_errors()
        self._on_tab_changed(self.tabs.currentIndex())

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("AppHeader")
        header.setFixedHeight(52)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 8, 18, 8)
        lay.setSpacing(12)

        mark = QLabel("◈")
        mark.setObjectName("BrandMark")
        title = QLabel("PCB 缺陷检测系统")
        title.setObjectName("BrandTitle")
        lay.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(title)
        lay.addStretch(1)

        self.lbl_header_runtime = QLabel("")
        self.lbl_header_runtime.setObjectName("HeaderMeta")
        self.lbl_header_tab = QLabel("")
        self.lbl_header_tab.setObjectName("HeaderMeta")
        lay.addWidget(self.lbl_header_runtime, 0, Qt.AlignmentFlag.AlignVCenter)
        sep = QLabel("·")
        sep.setObjectName("HeaderMeta")
        lay.addWidget(sep)
        lay.addWidget(self.lbl_header_tab, 0, Qt.AlignmentFlag.AlignVCenter)
        return header

    def _refresh_header_runtime(self) -> None:
        snap = self.inspect_page.header_snapshot()
        running = "检测中" if snap["running"] else "就绪"
        ver = f" v{snap['version']}" if snap.get("version") is not None else ""
        self.lbl_header_runtime.setText(
            f"{snap['template']}{ver}  ·  {snap['index']}/{snap['total']}  ·  直通率 {snap['yield']}  ·  {running}"
        )

    def _on_tab_changed(self, index: int) -> None:
        names = {
            self.tabs.indexOf(self.inspect_page): "自动检测",
            self.tabs.indexOf(self.history_page): "历史",
            self.tabs.indexOf(self.studio_page): "模板建模",
            self.tabs.indexOf(self.model_page): self._model_tab_name,
            self.tabs.indexOf(self.detect_page): "算法调试",
            self.tabs.indexOf(self.preprocess_page): "预处理",
        }
        if index == self.tabs.indexOf(self.model_page) and hasattr(self.model_page, "ensure_built"):
            try:
                self.model_page.ensure_built()
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"AOI_Core 页面加载失败：{exc}", 8000)
        self.lbl_header_tab.setText(names.get(index, ""))
        self._refresh_header_runtime()
        self.statusBar().showMessage(names.get(index, "就绪"))

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("设置")
        act_settings = QAction("系统设置（尺寸门禁 / 输出目录）…", self)
        act_settings.triggered.connect(self._open_settings)
        menu.addAction(act_settings)
        act_out = QAction("设置输出目录…", self)
        act_out.triggered.connect(self._choose_outputs)
        menu.addAction(act_out)
        act_show = QAction("显示当前输出目录", self)
        act_show.triggered.connect(lambda: QMessageBox.information(self, "输出目录", str(get_outputs_dir())))
        menu.addAction(act_show)
        menu.addSeparator()
        act_errs = QAction("查看算法适配器加载状态…", self)
        act_errs.triggered.connect(self._show_adapter_errors)
        menu.addAction(act_errs)

        tpl_menu = self.menuBar().addMenu("模板")
        act_migrate = QAction("导入旧标定 → 生成模板草稿…", self)
        act_migrate.triggered.connect(self._migrate_legacy)
        tpl_menu.addAction(act_migrate)
        act_open_tpl = QAction("打开模板库目录", self)
        act_open_tpl.triggered.connect(self._open_templates_dir)
        tpl_menu.addAction(act_open_tpl)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.result_store.root = get_outputs_dir()

    def _choose_outputs(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出根目录", str(get_outputs_dir()))
        if path:
            set_outputs_dir(path)
            self.result_store.root = get_outputs_dir()
            QMessageBox.information(self, "已更新", f"输出目录已设为：\n{path}")

    def _check_adapter_errors(self) -> None:
        errors = load_errors()
        if errors:
            # v2：不弹模态窗（启动流畅）；状态栏提示 + 菜单可查
            names = list(errors.keys())
            self.statusBar().showMessage(
                f"警告：{len(errors)} 个算法仓库未加载（{', '.join(names[:3])}…）"
                f"，产线模式对应算法将记 ERROR。详见 设置→算法适配器加载状态。",
                15000,
            )

    def _show_adapter_errors(self) -> None:
        errors = load_errors()
        if not errors:
            QMessageBox.information(self, "算法适配器状态", "外部算法仓库均已成功加载。")
            return
        detail = "\n".join(f"- {name}: {msg}" for name, msg in errors.items())
        QMessageBox.warning(self, "算法适配器状态", detail)

    def _migrate_legacy(self) -> None:
        ret = QMessageBox.question(
            self,
            "导入旧标定",
            "将扫描 template_calib / template_map / recipes/params，\n"
            "为每个 stem 生成草稿模板（已存在的默认跳过）。\n\n是否继续？",
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        result = migrate_legacy_to_templates(store=self.template_store, param_store=self.param_store)
        msg = (
            f"新建：{len(result['created'])}\n"
            f"跳过：{len(result['skipped'])}\n"
            f"更新：{len(result['updated'])}\n"
            f"目录：{result['root']}"
        )
        warns = result.get("warnings") or []
        if warns:
            msg += "\n\n警告（前 8 条）：\n" + "\n".join(warns[:8])
        QMessageBox.information(self, "迁移完成", msg)
        self.studio_page.list_panel.refresh()
        self.tabs.setCurrentWidget(self.studio_page)

    def _open_templates_dir(self) -> None:
        from app.config import TEMPLATES_DIR
        import os
        import sys

        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(TEMPLATES_DIR))  # type: ignore[attr-defined]
        else:
            QMessageBox.information(self, "模板库", str(TEMPLATES_DIR))

    def _goto_inspect_with_template(self, template_id: str) -> None:
        self.tabs.setCurrentWidget(self.inspect_page)
        if template_id:
            self.inspect_page.select_template(template_id)

    def _goto_studio(self, template_id: str = "") -> None:
        self.tabs.setCurrentWidget(self.studio_page)
        if template_id:
            self.studio_page.list_panel.refresh(keep_id=template_id)

    def _goto_preprocess_with_current(self) -> None:
        bgr = None
        path = ""
        if getattr(self.inspect_page, "queue", None) and self.inspect_page.current_index >= 0:
            from app.utils.cv_io import imread_unicode

            p = self.inspect_page.queue[self.inspect_page.current_index]
            bgr = imread_unicode(p)
            path = str(p)
        elif self.detect_page.test_bgr is not None:
            bgr = self.detect_page.test_bgr
            path = self.detect_page.test_path
        if bgr is not None:
            self.preprocess_page.set_image_from_detect(bgr, path)
        self.tabs.setCurrentWidget(self.preprocess_page)

    def _goto_preprocess_from_studio(self) -> None:
        self.tabs.setCurrentWidget(self.preprocess_page)
