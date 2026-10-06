"""品类模型管理页：AOI 特征学习能力入口。

- 品类模型列表（版本 / 激活态）
- 准备新模型（正常图目录 + 缺陷图目录 → fit）
- 版本激活 / 回滚
- 学习状态透出（双库 / 孵育头 / 权重门控）

用户无需知道 AOI_sys 存在——这里就是「准备/管理检测模型」。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class PrepareWorker(QThread):
    done = Signal(int, int, int)     # version, n_normal, n_defect
    failed = Signal(str)

    def __init__(self, category: str, normal_dir: str, defect_dir: str,
                 scenario: str, profile: str, parent=None) -> None:
        super().__init__(parent)
        self.category = category
        self.normal_dir = normal_dir
        self.defect_dir = defect_dir
        self.scenario = scenario
        self.profile = profile

    def run(self) -> None:  # noqa: N802
        try:
            from app.engines.feature import get_engine
            ver = get_engine().prepare(
                self.category,
                {"init_normal": self.normal_dir, "init_defect": self.defect_dir,
                 "val": [], "test": []},
                scenario=self.scenario, profile=self.profile,
                force=True,
            )
            self.done.emit(ver, len(self.normal_dir), len(self.defect_dir))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ModelPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: PrepareWorker | None = None
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # 顶部：准备新模型
        top = QWidget()
        form = QFormLayout(top)
        self.edit_category = QLineEdit()
        self.edit_category.setPlaceholderText("品类名（如 smt_solder / 插件器件）")
        self.edit_normal = QLineEdit()
        self.edit_normal.setPlaceholderText("正常图目录（good/normal 子目录 → normal，其余 → anomaly）")
        self.btn_normal = QPushButton("选择…")
        self.edit_defect = QLineEdit()
        self.edit_defect.setPlaceholderText("缺陷图目录（可选；空则只用正常图 L0）")
        self.btn_defect = QPushButton("选择…")
        self.cmb_scenario = QComboBox()
        self.cmb_scenario.addItems(["L0", "L1a", "L1b", "L2", "L3"])
        self.cmb_scenario.setCurrentText("L1a")
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["fast", "accuracy", "cpu"])
        self.btn_prepare = QPushButton("准备品类模型（fit + 快照）")
        self.btn_prepare.setObjectName("PrimaryButton")

        form.addRow("品类名", self.edit_category)
        form.addRow("正常图目录", self._row(self.edit_normal, self.btn_normal))
        form.addRow("缺陷图目录", self._row(self.edit_defect, self.btn_defect))
        form.addRow("场景层", self.cmb_scenario)
        form.addRow("工作点", self.cmb_profile)
        form.addRow("", self.btn_prepare)
        lay.addWidget(top)

        # 中部：模型列表 + 学习状态
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["品类", "当前版本", "版本数", "操作"])
        self.table.horizontalHeader().setStretchLastSection(True)
        split.addWidget(self.table)

        self.lbl_learning = QLabel("学习状态：选择品类后查看")
        self.lbl_learning.setWordWrap(True)
        self.lbl_learning.setAlignment(Qt.AlignmentFlag.AlignTop)
        split.addWidget(self.lbl_learning)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        # 底部：按钮
        btns = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_activate = QPushButton("激活所选版本")
        self.btn_rollback = QPushButton("回滚")
        btns.addWidget(self.btn_refresh)
        btns.addStretch(1)
        btns.addWidget(self.btn_activate)
        btns.addWidget(self.btn_rollback)
        lay.addLayout(btns)

        self.btn_normal.clicked.connect(lambda: self._pick_dir(self.edit_normal))
        self.btn_defect.clicked.connect(lambda: self._pick_dir(self.edit_defect))
        self.btn_prepare.clicked.connect(self._prepare)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_activate.clicked.connect(self._activate)
        self.btn_rollback.clicked.connect(self._rollback)
        self.table.cellClicked.connect(self._on_row_clicked)
        self.refresh()

    def _row(self, edit, btn) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        lay.addWidget(btn)
        return w

    def _pick_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录", str(Path.home()))
        if path:
            edit.setText(path)

    def _collect_images(self, folder: str) -> list[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        root = Path(folder)
        if not root.is_dir():
            return []
        return [str(p) for p in sorted(root.rglob("*"))
                if p.is_file() and p.suffix.lower() in exts]

    def _prepare(self) -> None:
        category = self.edit_category.text().strip()
        normal_dir = self.edit_normal.text().strip()
        if not category or not normal_dir:
            QMessageBox.warning(self, "缺参数", "品类名与正常图目录必填")
            return
        if not Path(normal_dir).is_dir():
            QMessageBox.warning(self, "目录无效", f"正常图目录不存在：{normal_dir}")
            return
        # 整理到 train 域（红线要求），再准备
        try:
            from app.config import get_settings
            import shutil
            settings = get_settings()
            train_cat = settings.storage("train") / category
            n_dir = train_cat / "normal"
            d_dir = train_cat / "defect"
            for d in (n_dir, d_dir):
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
            normals = self._collect_images(normal_dir)
            defect_dir = self.edit_defect.text().strip()
            defects = self._collect_images(defect_dir) if defect_dir else []
            for i, p in enumerate(normals):
                shutil.copy2(p, n_dir / f"n{i:04d}{Path(p).suffix.lower()}")
            for i, p in enumerate(defects):
                shutil.copy2(p, d_dir / f"d{i:04d}{Path(p).suffix.lower()}")
            train_normal = [str(p) for p in sorted(n_dir.iterdir())]
            train_defect = [str(p) for p in sorted(d_dir.iterdir())]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "准备失败", f"整理数据失败：{exc}")
            return

        self.btn_prepare.setEnabled(False)
        self.btn_prepare.setText("拟合中…（含 DINO 特征提取，稍候）")
        self._worker = PrepareWorker(category, train_normal, train_defect,
                                     self.cmb_scenario.currentText(),
                                     self.cmb_profile.currentText())
        self._worker.done.connect(self._on_prepared)
        self._worker.failed.connect(self._on_prepare_failed)
        self._worker.start()

    def _on_prepared(self, version: int, n_normal: int, n_defect: int) -> None:
        self.btn_prepare.setEnabled(True)
        self.btn_prepare.setText("准备品类模型（fit + 快照）")
        QMessageBox.information(self, "品类已准备",
                                f"品类模型 v{version} 已激活\n正常 {n_normal} / 缺陷 {n_defect}")
        self.refresh()

    def _on_prepare_failed(self, msg: str) -> None:
        self.btn_prepare.setEnabled(True)
        self.btn_prepare.setText("准备品类模型（fit + 快照）")
        QMessageBox.critical(self, "准备失败", msg)

    def refresh(self) -> None:
        from app.engines.feature import get_engine
        try:
            engine = get_engine()
        except Exception:
            return
        from app.config import get_settings
        snap_root = get_settings().snapshots_dir
        self.table.setRowCount(0)
        if not snap_root.is_dir():
            return
        for cat_dir in sorted(snap_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            versions = engine.list_snapshots(cat_dir.name)
            cur = engine.current_version(cat_dir.name)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(cat_dir.name))
            self.table.setItem(row, 1, QTableWidgetItem(f"v{cur}" if cur else "-"))
            self.table.setItem(row, 2, QTableWidgetItem(str(len(versions))))
            self.table.setItem(row, 3, QTableWidgetItem("双击版本激活"))
        self.table.resizeColumnsToContents()

    def _on_row_clicked(self, row: int, _col: int) -> None:
        category = self.table.item(row, 0).text()
        self._show_learning(category)

    def _show_learning(self, category: str) -> None:
        from app.engines.feature import get_engine
        try:
            engine = get_engine()
            pipe = engine.get_pipeline(category)
        except Exception as exc:  # noqa: BLE001
            self.lbl_learning.setText(f"品类 {category} 未准备：{exc}")
            return
        h = getattr(pipe, "handler", None)
        if h is None:
            self.lbl_learning.setText(f"品类 {category}：SSOCL 未启用（无学习状态）")
            return
        nb = h.normal_bank
        db = h.defect_bank
        hm = getattr(pipe, "head_mgr", None)
        st = getattr(h, "stats", {}) or {}
        text = (
            f"品类：{category}  (v{engine.current_version(category)})\n"
            f"训练 AUROC：{getattr(pipe, 'train_auroc', '?')}\n"
            f"正常库：core {int(nb.core.shape[0]) if getattr(nb, 'core', None) is not None else 0} patches"
            f" / 扩展 {len(getattr(nb, 'ext', []) or [])} 条"
            f"（熔断：{'是' if getattr(nb, 'fuse_blocked', False) else '否'}）\n"
            f"缺陷库：{len(getattr(db, 'samples', []) or [])} 条\n"
            f"孵育头：{'已训练' if (hm and hm.head) else '未训练'}（"
            f"{int(hm.head.n_pos) if (hm and hm.head) else 0} 正样本）\n"
            f"主动选样队列：{len(getattr(pipe, 'selector', None).queue or []) if getattr(pipe, 'selector', None) else 0}\n"
            f"权重学习：应用 {st.get('weight_apply', 0)} / 拒绝 {st.get('weight_reject', 0)}"
        )
        self.lbl_learning.setText(text)

    def _activate(self) -> None:
        item = self.table.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "先选中一个品类")
            return
        category = self.table.item(item.row(), 0).text()
        from app.engines.feature import get_engine
        engine = get_engine()
        versions = engine.list_snapshots(category)
        cur = engine.current_version(category)
        # 简单循环：当前版本 → 下一个（版本选择用表格双击未来增强）
        if not versions:
            return
        vs = [v["version"] for v in versions]
        idx = vs.index(cur) if cur in vs else -1
        target = vs[(idx + 1) % len(vs)]
        engine.activate(category, target)
        self.refresh()
        QMessageBox.information(self, "已激活", f"{category} → v{target}")

    def _rollback(self) -> None:
        item = self.table.currentItem()
        if item is None:
            return
        category = self.table.item(item.row(), 0).text()
        from app.engines.feature import get_engine
        try:
            target = get_engine().rollback(category)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "回滚失败", str(exc))
            return
        self.refresh()
        QMessageBox.information(self, "已回滚", f"{category} → v{target}")
