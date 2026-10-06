"""AOI_feature GUI：特征分析与管理界面（PySide6）

三页签：
  ① 特征集管理 —— 启停勾选 / 参数编辑 / 自定义增删 / reset / parity
  ② 单图分析   —— 选图提取特征向量（按组着色）+ 可选参考集组级 z 对比
  ③ 贡献评估   —— 后台线程跑评估，结果表格 + 三层证据图表

启动：python app.py
线程规范：评估走 EvalWorker（QThread），主线程不阻塞；关窗时置停止标志并 join。
"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from AOI_feature.feature_lib import FEATURE_FUNCS                     # noqa: E402
from AOI_feature.feature_set import FeatureSet, load_custom_features  # noqa: E402
from AOI_feature.run_eval import load_image_unicode, run_evaluation   # noqa: E402
from AOI_feature.scoring import extract_features                      # noqa: E402

plt_rc = matplotlib.rcParams
plt_rc["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt_rc["axes.unicode_minus"] = False

GROUP_COLORS = matplotlib.colormaps["tab20"].colors


def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(arr)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ── 评估后台线程 ──────────────────────────────────────────────────────

class EvalWorker(QThread):
    progressed = Signal(str)
    done = Signal(list, list)
    failed = Signal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            fs = FeatureSet.load()
            results, meta = run_evaluation(
                fs, self.cfg["dataset"], self.cfg["root"] or None,
                self.cfg["category"] or None,
                n_init_normal=self.cfg["n_init_normal"],
                n_init_defect=self.cfg["n_init_defect"],
                max_test=self.cfg["max_test"], k=self.cfg["k"],
                n_perm=self.cfg["n_perm"], seed=self.cfg["seed"],
                out=self.cfg["out"],
                progress=lambda m: self.progressed.emit(m))
            self.done.emit(results, meta)
        except Exception as e:
            self.failed.emit(str(e))


# ── 页签 1：特征集管理 ────────────────────────────────────────────────

class ManageTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fs = FeatureSet.load()
        self._build()
        self.reload()

    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)

        # 左：特征组表
        left = QWidget()
        ll = QVBoxLayout(left)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["启用", "key", "组名", "维", "参数改动"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 46)
        self.table.setColumnWidth(1, 72)
        self.table.setColumnWidth(3, 40)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.currentCellChanged.connect(self._on_select)
        ll.addWidget(self.table)
        btns = QHBoxLayout()
        for text, fn in [("停用选中", self._disable), ("启用选中", self._enable),
                         ("删除选中", self._remove), ("恢复主干预设", self._reset),
                         ("parity 校验", self._parity)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        ll.addLayout(btns)
        addrow = QHBoxLayout()
        self.custom_combo = QComboBox()
        b = QPushButton("加入特征集")
        b.clicked.connect(self._add_custom)
        addrow.addWidget(QLabel("自定义特征:"))
        addrow.addWidget(self.custom_combo)
        addrow.addWidget(b)
        ll.addLayout(addrow)
        split.addWidget(left)

        # 右：详情 + 参数编辑
        right = QWidget()
        rl = QVBoxLayout(right)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(180)
        rl.addWidget(self.info)
        rl.addWidget(QLabel("参数（直接编辑值，YAML 语法；应用后写回配置）:"))
        self.params = QTableWidget(0, 2)
        self.params.setHorizontalHeaderLabels(["参数", "值"])
        self.params.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        rl.addWidget(self.params)
        b = QPushButton("应用参数修改")
        b.clicked.connect(self._apply_params)
        rl.addWidget(b)
        split.addWidget(right)
        split.setSizes([560, 420])

    # ── 数据装载 ──

    def reload(self):
        self.fs = FeatureSet.load()
        load_custom_features()
        meta = {m["key"]: m for m in self.fs.probe()}
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for g in self.fs.groups:
            key = g["key"]
            reg = FEATURE_FUNCS[key]
            row = self.table.rowCount()
            self.table.insertRow(row)
            cb = QCheckBox()
            cb.setChecked(g["enabled"])
            cb.stateChanged.connect(
                lambda _s, k=key, c=cb: self._toggle(k, c.isChecked()))
            self.table.setCellWidget(row, 0, cb)
            self.table.setItem(row, 1, QTableWidgetItem(key))
            self.table.setItem(row, 2, QTableWidgetItem(reg["name"]))
            dim = str(meta[key]["dim"]) if g["enabled"] and key in meta else "-"
            self.table.setItem(row, 3, QTableWidgetItem(dim))
            diff = {k: v for k, v in g["params"].items()
                    if v != reg["default_params"].get(k)}
            it = QTableWidgetItem(str(diff) if diff else "")
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 4, it)
        for r in range(self.table.rowCount()):
            for c in (1, 2, 3):
                it = self.table.item(r, c)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.table.blockSignals(False)
        # 自定义特征下拉：已注册但未入列
        inset = {g["key"] for g in self.fs.groups}
        self.custom_combo.clear()
        for k in sorted(set(FEATURE_FUNCS) - inset):
            self.custom_combo.addItem(f"{k}（{FEATURE_FUNCS[k]['name']}）", k)
        if self.table.rowCount():
            self.table.setCurrentCell(0, 0)   # 默认选中首行，右侧面板即有内容

    def _selected_key(self):
        r = self.table.currentRow()
        return self.table.item(r, 1).text() if r >= 0 else None

    def _on_select(self, row, *_):
        if row < 0:
            return
        key = self.table.item(row, 1).text()
        g = self.fs._find(key)
        reg = FEATURE_FUNCS[key]
        self.info.setPlainText(
            f"{reg['name']}（{'启用' if g['enabled'] else '停用'}）\n\n"
            f"选择理由: {reg['rationale']}\n目标缺陷: {reg['targets']}")
        params = {**reg["default_params"], **g["params"]}
        self.params.setRowCount(0)
        for k, v in params.items():
            r = self.params.rowCount()
            self.params.insertRow(r)
            it = QTableWidgetItem(k)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.params.setItem(r, 0, it)
            self.params.setItem(r, 1, QTableWidgetItem(yaml.safe_dump(
                v, default_flow_style=True).strip()))

    # ── 操作 ──

    def _mutate(self, fn, msg):
        key = self._selected_key()
        if not key:
            return
        try:
            fn(key)
            self.fs.save()
            self.reload()
            QMessageBox.information(self, "完成", msg.format(key))
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _toggle(self, key, on):
        self.fs.set_enabled(key, on)
        self.fs.save()
        self.reload()

    def _disable(self):
        self._mutate(lambda k: self.fs.set_enabled(k, False), "{} 已停用")

    def _enable(self):
        self._mutate(lambda k: self.fs.set_enabled(k, True), "{} 已启用")

    def _remove(self):
        self._mutate(self.fs.remove, "{} 已删除")

    def _add_custom(self):
        key = self.custom_combo.currentData()
        if not key:
            QMessageBox.information(self, "提示", "没有可加入的自定义特征\n"
                                    "（在 custom_features.py 用 @feature 定义）")
            return
        try:
            self.fs.add(key)
            self.fs.save()
            self.reload()
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _apply_params(self):
        key = self._selected_key()
        if not key:
            return
        try:
            for r in range(self.params.rowCount()):
                k = self.params.item(r, 0).text()
                v = yaml.safe_load(self.params.item(r, 1).text())
                self.fs.set_param(key, k, v)
            self.fs.save()
            self.reload()
            QMessageBox.information(self, "完成", f"{key} 参数已更新")
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _reset(self):
        from AOI_feature.presets import trunk_preset
        if QMessageBox.question(self, "确认", "恢复主干预设？当前修改将丢失") \
                != QMessageBox.Yes:
            return
        FeatureSet(trunk_preset()).save()
        self.reload()

    def _parity(self):
        import numpy as np
        from AOI_feature import ensure_core_importable
        ensure_core_importable()
        from algo.vendor.traditional import TraditionalFeatureExtractor
        trunk = TraditionalFeatureExtractor()
        rng = np.random.default_rng(0)
        max_diff = 0.0
        for _ in range(5):
            img = rng.integers(0, 256, (128, 96, 3), dtype=np.uint8)
            state = np.random.get_state()
            np.random.seed(0)
            try:
                v_t = trunk.extract(img)
            finally:
                np.random.set_state(state)
            state = np.random.get_state()
            np.random.seed(0)
            try:
                v_m = self.fs.extract(img)
            finally:
                np.random.set_state(state)
            if len(v_t) != len(v_m):
                QMessageBox.warning(self, "parity",
                                    f"维度不一致: 主干 {len(v_t)} vs 配置 {len(v_m)}"
                                    "（有组停用/改维度属预期）")
                return
            max_diff = max(max_diff, float(np.abs(v_t - v_m).max()))
        ok = max_diff < 1e-4
        QMessageBox.information(self, "parity",
                                f"5 图最大逐维差 = {max_diff:.3e}\n"
                                + ("与主干一致 ✓" if ok else "与主干有差异 ✗"))


# ── 页签 2：单图分析 ──────────────────────────────────────────────────

class AnalyzeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ref_vecs = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        row = QHBoxLayout()
        b = QPushButton("选择图片…")
        b.clicked.connect(self._pick_image)
        row.addWidget(b)
        b = QPushButton("选择参考集（正常图文件夹，可选）…")
        b.clicked.connect(self._pick_ref)
        row.addWidget(b)
        self.ref_label = QLabel("未加载参考集")
        row.addWidget(self.ref_label)
        row.addStretch()
        ll.addLayout(row)
        self.img_label = QLabel("未加载图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumHeight(280)
        self.img_label.setStyleSheet("background:#222;color:#888")
        ll.addWidget(self.img_label)
        self.info = QLabel("")
        ll.addWidget(self.info)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        rl.addWidget(self.canvas)
        self.fig2 = Figure(figsize=(6, 3))
        self.canvas2 = FigureCanvasQTAgg(self.fig2)
        rl.addWidget(self.canvas2)
        split.addWidget(right)
        split.setSizes([420, 620])

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.analyze(path)

    def _pick_ref(self):
        d = QFileDialog.getExistingDirectory(self, "选择参考集文件夹")
        if not d:
            return
        paths = sorted(str(p) for p in Path(d).rglob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if not paths:
            QMessageBox.warning(self, "空目录", "该文件夹下没有图像文件")
            return
        fs = FeatureSet.load()
        vecs, _ = extract_features(fs, paths, load_image_unicode)
        self.ref_vecs = vecs
        self.ref_label.setText(f"参考集: {len(vecs)} 张")

    def analyze(self, path: str):
        img = load_image_unicode(path)
        if img is None:
            QMessageBox.warning(self, "失败", f"无法解码: {path}")
            return
        fs = FeatureSet.load()
        meta = fs.probe()
        vec = fs.extract(img).astype(np.float64)

        pm = QPixmap.fromImage(ndarray_to_qimage(img))
        self.img_label.setPixmap(pm.scaled(
            self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.info.setText(f"{os.path.basename(path)}  |  {len(vec)} 维 / "
                          f"{len(meta)} 组启用")

        # 上图：特征向量按组着色
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for i, m in enumerate(meta):
            sl = m["slice"]
            xs = np.arange(sl.start, sl.stop)
            ax.bar(xs, vec[sl], width=1.0, color=GROUP_COLORS[i % 20],
                   label=m["key"])
        ax.legend(fontsize=7, ncol=4)
        ax.set_title("特征向量（按组着色）")
        self.canvas.draw()

        # 下图：有参考集时画组级 mean|z|；否则画组级均值
        self.fig2.clear()
        ax2 = self.fig2.add_subplot(111)
        keys = [m["key"] for m in meta]
        if self.ref_vecs is not None and len(self.ref_vecs):
            mean = self.ref_vecs.mean(axis=0)
            std = self.ref_vecs.std(axis=0) + 1e-6
            z = np.clip(np.abs((vec - mean) / std), 0, 10)
            vals = [float(z[m["slice"]].mean()) for m in meta]
            ax2.set_title("组级 mean|z|（相对参考集，>3 即显著偏离）")
            ax2.axhline(3, color="r", ls="--", lw=1)
        else:
            vals = [float(np.abs(vec[m["slice"]]).mean()) for m in meta]
            ax2.set_title("组级 mean|特征值|（未加载参考集）")
        ax2.bar(keys, vals, color=[GROUP_COLORS[i % 20] for i in range(len(keys))])
        ax2.tick_params(axis="x", rotation=45)
        self.fig2.tight_layout()
        self.canvas2.draw()


# ── 页签 3：贡献评估 ──────────────────────────────────────────────────

class EvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.dataset = QComboBox()
        self.dataset.addItems(["synthetic", "mvtec", "mpdd", "btad", "datalocal"])
        form.addRow("数据集", self.dataset)
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        self.root_edit = QLineEdit()
        b = QPushButton("浏览…")
        b.clicked.connect(self._browse)
        hl.addWidget(self.root_edit)
        hl.addWidget(b)
        form.addRow("数据根目录", row)
        self.category = QLineEdit()
        self.category.setPlaceholderText("缺省=synthetic；all=全品类；或单品类名")
        form.addRow("品类", self.category)
        self.spins = {}
        for label, key, val in [("参考图数", "n_init_normal", 60),
                                ("test 上限", "max_test", 150),
                                ("kNN k", "k", 3), ("排列次数", "n_perm", 5),
                                ("随机种子", "seed", 42)]:
            s = QSpinBox()
            s.setRange(0, 100000)
            s.setValue(val)
            self.spins[key] = s
            form.addRow(label, s)
        root.addLayout(form)
        self.run_btn = QPushButton("运行评估（后台）")
        self.run_btn.clicked.connect(self._run)
        root.addWidget(self.run_btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        root.addWidget(self.log)

        split = QSplitter()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["组", "维", "独立AUROC", "留一损失", "排列损失"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        split.addWidget(self.table)
        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        split.addWidget(self.canvas)
        split.setSizes([460, 560])
        root.addWidget(split)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "数据根目录")
        if d:
            self.root_edit.setText(d)

    def _run(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "评估进行中…")
            return
        out = str(Path(__file__).resolve().parent / "outputs")
        cfg = {"dataset": self.dataset.currentText(),
               "root": self.root_edit.text().strip(),
               "category": self.category.text().strip(),
               "out": out,
               **{k: s.value() for k, s in self.spins.items()},
               "n_init_defect": 30}
        self.log.clear()
        self.run_btn.setEnabled(False)
        self.worker = EvalWorker(cfg, self)
        self.worker.progressed.connect(lambda m: self.log.append(m))
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_fail(self, msg):
        self.run_btn.setEnabled(True)
        QMessageBox.warning(self, "评估失败", msg)

    def _on_done(self, results, groups_meta):
        self.run_btn.setEnabled(True)
        if not results:
            self.log.append("[warn] 无结果（数据为空？）")
            return
        res = results[0]
        g = res["groups"]
        keys = list(g)
        self.table.setRowCount(0)
        for k in keys:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, txt in enumerate([
                    g[k]["name"], str(g[k]["dim"]),
                    f"{g[k]['standalone_auroc']:.4f}",
                    f"{g[k]['loo_drop']:+.4f}",
                    f"{g[k]['perm_drop_mean']:+.4f}±{g[k]['perm_drop_std']:.4f}"]):
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, c, it)
        # 三联图
        self.fig.clear()
        titles = ["独立 AUROC", "留一损失", "排列损失"]
        series = [[g[k]["standalone_auroc"] for k in keys],
                  [g[k]["loo_drop"] for k in keys],
                  [g[k]["perm_drop_mean"] for k in keys]]
        for i, (t, vals) in enumerate(zip(titles, series)):
            ax = self.fig.add_subplot(3, 1, i + 1)
            colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
            ax.bar(keys, vals, color=colors)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{res['category']} {t}", fontsize=10)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()
        self.log.append(f"[done] 报告已写 outputs/（summary.md 含全量）")


# ── 主窗口 ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI 特征分析与管理")
        self.resize(1180, 760)
        tabs = QTabWidget()
        self.manage_tab = ManageTab()
        self.analyze_tab = AnalyzeTab()
        self.eval_tab = EvalTab()
        tabs.addTab(self.manage_tab, "特征集管理")
        tabs.addTab(self.analyze_tab, "单图分析")
        tabs.addTab(self.eval_tab, "贡献评估")
        self.setCentralWidget(tabs)

    def closeEvent(self, e):
        w = self.eval_tab.worker
        if w and w.isRunning():
            w.requestInterruption()
            w.wait(3000)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 9))   # CJK 字体显式指定（离屏/真实显示一致）
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
