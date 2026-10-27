"""评估看板：三个独立评估入口、任务进度、指标记录与内嵌鲁棒性曲线。"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (attach_level_badge, make_banner, make_card,
                             run_async, warn)
from ui.pages.stats_page import biz_budget_value
from ui.theme import DANGER, PRIMARY, SUCCESS, WARNING
from ui.widgets.charts import BarChart, LatencyChart, LineChart, num_or_nan
from ui.widgets.task_progress import TaskMonitor, TaskProgressBar

RUN_TYPE_NAMES = {"accuracy": "精度评估", "benchmark": "延迟基准",
                  "robustness": "噪声鲁棒性",
                  "ab_compare": "A/B 版本对比", "learning_curve": "学习曲线回放",
                  "contribution": "贡献档案"}

# 噪声扰动中文名（后端 noise_kinds 契约）
NOISE_KIND_CN = {"brightness": "亮度", "gauss": "高斯噪声", "blur": "模糊"}


def _fmt(v) -> str:
    """指标容错格式化：None / 非数值显示 '-'。"""
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


def _rate_text(metrics: dict, numerator: str, other: str) -> str:
    a, b = metrics.get(numerator), metrics.get(other)
    if not all(isinstance(v, (int, float)) for v in (a, b)) or a + b == 0:
        return "未知"
    return f"{a / (a + b) * 100:.1f}%"


class RobustnessPanel(QWidget):
    """噪声鲁棒性详情（2026-08-29 用户要求补充）：基线 AUROC + 各扰动
    类型 A_rob（分数-噪声曲线面积）+ 分档 AUC 衰减曲线。

    数据来自 EvalRun.metrics.curves（后端 run_robustness_eval 已算好，
    此前 UI 只在记录表格显示 baseline_auroc 单值，无详情入口）。
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)

    def set_run(self, run: dict) -> None:
        lay = self._layout
        while lay.count():
            widget = lay.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        name = run.get("name") or f"运行 #{run.get('id')}"
        head = QLabel(f"{name}　（{RUN_TYPE_NAMES.get(run.get('run_type'), '')}）")
        head.setProperty("heading", True)
        lay.addWidget(head)

        metrics = run.get("metrics") or {}
        curves = metrics.get("curves") if isinstance(metrics.get("curves"), dict) else {}
        base = metrics.get("baseline_auroc")

        kpi = QLabel()
        parts = [f"基线 AUROC（扰动前）＝ {_fmt(base)}"]
        for kind, c in curves.items():
            arob = c.get("a_rob") if isinstance(c, dict) else None
            parts.append(f"{NOISE_KIND_CN.get(kind, kind)} A_rob＝{_fmt(arob)}")
        kpi.setText("　".join(parts))
        kpi.setProperty("subtext", True)
        lay.addWidget(kpi)

        chart = LineChart()
        levels = metrics.get("levels")
        n = max((len(c.get("auc_levels") or []) for c in curves.values()
                 if isinstance(c, dict)), default=0)
        real_levels = isinstance(levels, list) and len(levels) >= n and n > 0
        labels = [str(v) for v in levels[:n]] if real_levels else [str(i + 1) for i in range(n)]
        for kind, c in curves.items():
            aucs = c.get("auc_levels") or [] if isinstance(c, dict) else []
            if aucs:
                color = SUCCESS if kind == "brightness" else PRIMARY if kind == "gauss" else WARNING
                chart.add_series(NOISE_KIND_CN.get(kind, kind), list(range(len(aucs))),
                                 [num_or_nan(v) for v in aucs], color=color)
        chart.set_labels(labels, "AUROC")
        lay.addWidget(chart, 1)

        x_hint = "横轴按位置展示真实扰动强度标签" if real_levels else "历史记录未保存完整真实扰动档位，横轴为序号"
        hint = QLabel("A_rob＝未归一化的 AUROC-扰动强度梯形面积；默认强度区间 0→0.8，"
                      "理想上限为 0.8，不是 1.0。" + x_hint + "；AUC 随强度下降为预期衰减")
        hint.setProperty("subtext", True)
        hint.setWordWrap(True)
        lay.addWidget(hint)


class CompareDialog(QDialog):
    """两次评估对比对话框：双系列分组柱状图。"""

    def __init__(self, run_a: dict, run_b: dict,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("评估结果对比")
        self.resize(640, 420)
        lay = QVBoxLayout(self)

        name_a = run_a.get("name") or f"运行 #{run_a.get('id')}"
        name_b = run_b.get("name") or f"运行 #{run_b.get('id')}"
        head = QLabel(f"{name_a}　VS　{name_b}")
        head.setProperty("heading", True)
        lay.addWidget(head)

        categories: list[str] = []
        va: list[float] = []
        vb: list[float] = []
        for key, label in (("auroc", "AUROC"), ("ap", "AP"), ("f1", "F1")):
            ma = (run_a.get("metrics") or {}).get(key)
            mb = (run_b.get("metrics") or {}).get(key)
            # 学习曲线等记录的指标是时序列表，回退到 final_*
            if not isinstance(ma, (int, float)):
                ma = (run_a.get("metrics") or {}).get(f"final_{key}")
            if not isinstance(mb, (int, float)):
                mb = (run_b.get("metrics") or {}).get(f"final_{key}")
            if ma is None and mb is None:
                continue
            categories.append(label)
            va.append(num_or_nan(ma))  # 缺失留空，不画成 0
            vb.append(num_or_nan(mb))
        la = (run_a.get("latency") or {}).get("mean")
        lb_ = (run_b.get("latency") or {}).get("mean")
        if la is not None or lb_ is not None:
            categories.append("延迟mean(ms)")
            va.append(num_or_nan(la))
            vb.append(num_or_nan(lb_))

        chart = BarChart()
        chart.set_data(categories, {str(name_a): va, str(name_b): vb})
        lay.addWidget(chart, 1)

        btn = QPushButton("关闭")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn, 0, Qt.AlignRight)


class EvalPage(QWidget):
    """评估看板页。"""

    def __init__(self, client: ApiClient, get_category, get_budget=None,
                 task_monitor: TaskMonitor | None = None,
                 parent: QWidget | None = None, get_biz_budget=None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._get_budget = get_budget or (lambda: 200.0)
        self._get_biz_budget = get_biz_budget
        self._monitor = task_monitor
        self._runs: list[dict] = []
        self._benchmark_default = 0
        self._task_id = None
        self._submitting = False
        self._polling = False
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._poll_task)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner("评估看板：精度指标 + 延迟基准 + 噪声鲁棒性"))
        attach_level_badge(self, top)   # U-workorder：数据条件徽标
        self.lbl_tip = QLabel("")
        self.lbl_tip.setProperty("subtext", True)
        top.addWidget(self.lbl_tip)
        top.addStretch(1)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setSizes([300, 760])
        root.addWidget(splitter, 1)

        self.progress = TaskProgressBar("评估任务")
        root.addWidget(self.progress)
        if self._monitor is not None:
            self._monitor.task_updated.connect(self._on_task)
        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    # ══════════════════ 左：新建评估表单 ══════════════════
    def _build_form_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head = QLabel("新建评估")
        head.setProperty("heading", True)
        lay.addWidget(head)

        form = QFormLayout()
        self.combo_category = QComboBox()
        self.combo_category.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_category.setMinimumContentsLength(8)
        self.combo_category.currentIndexChanged.connect(
            lambda _i: self._on_category_changed())
        form.addRow("评估品类", self.combo_category)
        self.run_buttons = {}
        self.sample_limits = {}
        self.combo_split = QComboBox()
        self.combo_split.addItems(["test", "val", "train"])
        form.addRow("数据划分", self.combo_split)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("可选，便于对比记录")
        form.addRow("评估名称", self.edit_name)
        lay.addLayout(form)

        for text, rtype, default in (("运行精度评估", "accuracy", 20),
                                      ("运行延迟基准", "benchmark", 0),
                                      ("运行噪声鲁棒性", "robustness", 12)):
            row = QHBoxLayout()
            btn = QPushButton(text)
            btn.clicked.connect(lambda _checked=False, t=rtype: self._on_run(t))
            spin = QSpinBox()
            spin.setRange(0 if rtype == "benchmark" else 8 if rtype == "robustness" else 1, 10000)
            if rtype == "benchmark":
                spin.setSpecialValueText("使用配置")
            spin.setValue(default)
            spin.setSuffix(" 张")
            spin.setToolTip("本入口样本上限；延迟基准预热次数由配置决定，且可能循环已有图片")
            self.run_buttons[rtype] = btn
            self.sample_limits[rtype] = spin
            row.addWidget(btn)
            row.addWidget(spin)
            lay.addLayout(row)

        self.note = QLabel("")
        self.note.setProperty("subtext", True)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        lay.addStretch(1)
        return card

    def _update_run_buttons(self) -> None:
        enabled = self.combo_category.count() > 0 and not self._submitting and self._task_id is None
        for btn in self.run_buttons.values():
            btn.setEnabled(enabled)

    def _on_run(self, run_type: str) -> None:
        if self._submitting or self._task_id is not None:
            return
        cat = (self.combo_category.currentData()
               or (self._get_category() if self._get_category else "")
               or "")
        if not cat:
            warn(self, "请先在左侧选择评估品类")
            return
        name = self.edit_name.text().strip()
        limit = self.sample_limits[run_type].value()
        split = self.combo_split.currentText()
        if run_type == "robustness" and QMessageBox.question(
                self, "确认鲁棒性前向成本",
                f"最多 {limit} 张 × 16 次前向（基线 + 3 类扰动 × 5 档）＝{limit * 16} 次。\n"
                "不是单次检测耗时；确认单独提交？") != QMessageBox.Yes:
            return
        self._submitting = True
        self._update_run_buttons()

        if run_type == "accuracy":
            fn = lambda: self._client.eval_accuracy(  # noqa: E731
                cat, split=split, name=name, limit=limit)
        elif run_type == "robustness":
            n = limit if limit > 0 else 12
            fn = lambda: self._client.eval_robustness(  # noqa: E731
                cat, split=split, limit=n)
        else:
            n = limit if limit > 0 else None
            fn = lambda: self._client.eval_benchmark(cat, n_images=n)  # noqa: E731

        def _failed(message="提交失败，未取得任务 ID") -> None:
            self._submitting = False
            self._update_run_buttons()
            warn(self, str(message))

        def _started(res) -> None:
            self._submitting = False
            if not res or not res.get("task_id"):
                _failed()
                return
            self._task_id = str(res["task_id"])
            self._update_run_buttons()
            self.progress.update_task({"id": self._task_id,
                "status": "pending", "progress": 0,
                "message": "评估任务已提交，正在执行…"})
            self._poll_timer.start()
            self._poll_task()
        run_async(self, fn, _started, _failed)

    def _poll_task(self) -> None:
        if self._task_id is None or self._polling:
            return
        task_id = self._task_id
        self._polling = True

        def _received(task) -> None:
            self._polling = False
            if task:
                self._on_task(task)

        def _failed(_message) -> None:
            self._polling = False
        run_async(self, lambda: self._client.get_task(task_id), _received, _failed)

    # ══════════════════ 右：记录表格 + 延迟图 ══════════════════
    def _build_result_panel(self) -> QWidget:
        card = make_card()
        lay = QVBoxLayout(card)
        head_row = QHBoxLayout()
        head = QLabel("评估记录")
        head.setProperty("heading", True)
        head_row.addWidget(head)
        head_row.addStretch(1)
        # 前端反馈 2026-09-13 #6：评估记录全选切换
        btn_all = QPushButton("全选")
        btn_all.setToolTip("勾选/取消勾选全部评估记录")
        btn_all.clicked.connect(self._on_toggle_all)
        btn_compare = QPushButton("对比选中（两行）")
        btn_compare.clicked.connect(self._on_compare)
        btn_delete = QPushButton("删除所选")
        btn_delete.setToolTip("删除勾选的评估记录（admin 权限；仅删记录，"
                              "不影响模型/图片数据）")
        btn_delete.clicked.connect(self._on_delete_selected)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        head_row.addWidget(btn_all)
        head_row.addWidget(btn_compare)
        head_row.addWidget(btn_delete)
        head_row.addWidget(btn_refresh)
        lay.addLayout(head_row)

        self.table = QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(
            ["对比", "类型", "品类", "AUROC", "AP", "F1",
             "延迟mean(ms)", "通过 / 时间", "版本", "样本数", "检出率", "误报率(FPR)"])
        # U-opsflow：指标人话解释（鼠标悬停表头可见）
        for col, tip in (
                (3, "AUROC：把异常排在正常前面的能力，1.0=完美，0.5=瞎猜"),
                (4, "AP：精确率-召回率的综合，越高越好"),
                (5, "F1：精确率与召回率的调和平均，越高越好"),
                (7, "是否达标（与基线对比）及记录时间"),
                (10, "检出率=TP/(TP+FN)：真实缺陷被系统检出的比例；缺标签或零分母为未知"),
                (11, "误报率(FPR)=FP/(FP+TN)：正常样本被系统判为缺陷的比例；缺标签或零分母为未知")): 
            h = self.table.horizontalHeaderItem(col)
            if h is not None:
                h.setToolTip(tip)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)

        # ── 下区：指标说明 + 延迟明细（2026-10-14 布局调整）──
        # 前端反馈：原实现把「说明文字 + 延迟柱状图」和记录表平铺在一个卡片里，
        # 图最小高 180px、说明文字又占 4~5 行，导致**记录表空间严重不足**。
        # 改为上下 QSplitter：记录表占大头、下区可拖拽，延迟图最小高降到 110px。
        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(2)
        # 混淆矩阵派生指标明细（2026-08-30）：选中行时填充，见 _on_row_selected
        self.lbl_metrics_detail = QLabel(
            "选中记录查看精度、延迟或内嵌鲁棒性曲线")
        self.lbl_metrics_detail.setProperty("subtext", True)
        self.lbl_metrics_detail.setWordWrap(True)
        dl.addWidget(self.lbl_metrics_detail)
        # 指标含义说明区（固定展示）
        meaning = QLabel(
            "指标含义（最优 F1 阈值口径）：准确率=全部样本判对比例；"
            "精确率=判为缺陷中真缺陷占比；检出率(缺陷召回)=缺陷被判出的比例；"
            "误报率(FPR)=正常被判缺陷的比例；特异度=正常判正常比例。"
            "最佳 F1 阈值来自本次评估集，不等于生产阈值；缺失或零分母为未知。")
        meaning.setProperty("subtext", True)
        meaning.setWordWrap(True)
        dl.addWidget(meaning)

        # M16c 流程引导：验收后的下一步动作提示（review R7）
        hint = QLabel("流程引导：✓ 通过 → 去「模型管理」激活版本投产；"
                      "✗ 未达标 → 回「数据管理」补缺陷数据或用「数据增强」"
                      "合成后重新准备模型")
        hint.setProperty("subtext", True)
        dl.addWidget(hint)

        self.lat_head = QLabel("")
        self.lat_head.setProperty("heading", True)
        dl.addWidget(self.lat_head)
        self.latency_chart = LatencyChart(budget_ms=float(self._get_budget()))
        self.latency_chart.setMinimumHeight(110)   # 让位给上方记录表
        self.latency_chart.set_values([], [])
        dl.addWidget(self.latency_chart, 1)
        self.lbl_latency_hint = QLabel("选中一条评估记录查看延迟分布柱状图")
        self.lbl_latency_hint.setProperty("subtext", True)
        dl.addWidget(self.lbl_latency_hint)
        self.robustness_panel = RobustnessPanel()
        self.robustness_panel.hide()
        dl.addWidget(self.robustness_panel, 1)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.addWidget(self.table)
        vsplit.addWidget(detail)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([430, 240])
        lay.addWidget(vsplit, 1)
        return card

    def _refresh_budget_texts(self) -> None:
        """预算相关文案/参考线随设置热更新（设置页保存后主窗口重拉 system_info，
        get_budget 回调即返回新值；此前多处硬编码 <200ms，改预算不生效）。"""
        budget = float(self._get_budget() or 200.0)
        self.lbl_tip.setText(
            f"验证 2060 GPU / 2500×2500 大图分块 / 总延迟 <{budget:.0f}ms 的实时性指标")
        self.note.setText(
            "精度与鲁棒性：任务执行时固定当前快照，适用于初次及后续模型；版本见记录。\n"
            "精度评估：AUROC / AP / F1 + 延迟分布；检出率与误报率基于评估集最佳 F1 阈值。\n"
            "延迟基准使用在线服务，不固定版本，不用于固定版本精度比较。\n"
            f"延迟基准：预热+大图优先的严格计时，mean/p50/p95/max 对比 {budget:.0f}ms 预算\n"
            "噪声鲁棒性：亮度/高斯/模糊扰动下 AUC 衰减（≈样本数×16 次前向，建议小样本）")
        self.lat_head.setText(
            f"延迟明细（本机 GPU · 2500×2500 大图分块 · 预算 <{budget:.0f}ms）")
        self.latency_chart.set_budget(budget)
        # 预算变化后重绘当前选中行（hint/柱体红绿随新预算口径）
        if getattr(self, "table", None) is not None and self.table.currentRow() >= 0:
            self._on_row_selected()

    def _fill_benchmark_config(self, data) -> None:
        if not isinstance(data, dict):
            return
        values = {item["key"]: item.get("value") for item in data.get("items", [])
                  if item.get("section") == "evaluation"}
        n = values.get("benchmark_n_images")
        spin = self.sample_limits["benchmark"]
        if isinstance(n, int) and n > 0:
            if spin.value() == self._benchmark_default:
                spin.setValue(n)
            self._benchmark_default = n
        warmup = values.get("benchmark_warmup")
        if isinstance(warmup, int):
            spin.setToolTip(f"延迟基准另含 {warmup} 次预热（来自配置），可能循环已有图片")

    def reload(self) -> None:
        run_async(self, self._client.get_config, self._fill_benchmark_config)
        self._refresh_categories()
        self._refresh_budget_texts()
        run_async(self, lambda: self._client.list_eval_runs(all_pages=True), self._fill_runs)

    def _on_category_changed(self) -> None:
        """品类变化时同步"数据划分"默认值（test 域不可用则回退 val）。"""
        # 保持简单：不做联动，split 由用户自选。

    def _refresh_categories(self) -> None:
        """填充品类下拉：多品类工单/无工单时用户可手动选。

        （前端反馈 v3 移除顶栏品类下拉后，评估不再能靠"当前品类"兜底；
        品类来源：当前工单 categories → 单品类推导 → 空（按钮禁用）。）
        """
        combo = getattr(self, "combo_category", None)
        if combo is None:
            return
        wo = getattr(self, "_get_workorder", lambda: None)()
        cats = []
        if wo:
            cats = [str(c) for c in (wo.get("categories") or [])]
        if not cats:
            cur = self._get_category() if self._get_category else ""
            cats = [cur] if cur else []
        combo.blockSignals(True)
        combo.clear()
        for c in cats:
            combo.addItem(c, c)
        combo.blockSignals(False)
        cur = self._get_category() if self._get_category else ""
        idx = combo.findData(cur)
        combo.setCurrentIndex(idx if idx >= 0 and idx < combo.count() else 0)
        self._update_run_buttons()

    def _fill_runs(self, runs) -> None:
        runs = runs or []
        self._runs = runs
        self.table.clearSpans()
        if not runs:
            self.table.setRowCount(1)
            self.table.setSpan(0, 0, 1, self.table.columnCount())
            self.robustness_panel.hide()
            self.lbl_metrics_detail.setText("暂无评估记录")
            self.table.setItem(0, 0, QTableWidgetItem(
                "暂无评估记录，可在左侧新建评估"))
            return
        budget = float(self._get_budget() or 200.0)
        self.table.setRowCount(len(runs))
        for r, run in enumerate(runs):
            chk = QCheckBox()
            chk.setProperty("row", r)
            self.table.setCellWidget(r, 0, chk)
            rtype = str(run.get("run_type", ""))
            self.table.setItem(r, 1, QTableWidgetItem(
                RUN_TYPE_NAMES.get(rtype, rtype)))
            self.table.setItem(r, 2, QTableWidgetItem(str(run.get("category", ""))))
            metrics = run.get("metrics") or {}
            latency = run.get("latency") or {}
            for c, key in ((3, "auroc"), (4, "ap"), (5, "f1")):
                v = metrics.get(key)
                if not isinstance(v, (int, float)):
                    # 学习曲线等记录的 auroc 是时序列表，回退到 final_auroc
                    v = metrics.get(f"final_{key}")
                if not isinstance(v, (int, float)) and key == "auroc":
                    # 噪声鲁棒性记录：显示基线 AUROC（扰动前）
                    v = metrics.get("baseline_auroc")
                self.table.setItem(r, c, QTableWidgetItem(
                    f"{float(v):.3f}" if isinstance(v, (int, float)) else "-"))
            mean = latency.get("mean")
            self.table.setItem(r, 6, QTableWidgetItem(
                f"{float(mean):.1f}" if mean is not None else "-"))
            # 通过判定：与后端口径一致——benchmark 用后端 metrics.pass
            # （p95<预算，2026-08-30 起统一；此前前端按 mean 自判会漏判
            # p95 超标）；accuracy 看 AUROC≥0.9。旧记录无 pass 字段时
            # 回退本地 p95 判定
            mark_override = None
            if rtype == "benchmark":
                if isinstance(metrics.get("pass"), bool):
                    passed = bool(metrics["pass"])
                elif latency.get("p95") is not None:
                    passed = float(latency["p95"]) < budget
                else:
                    passed = None
            elif rtype == "accuracy" and metrics.get("auroc") is not None:
                # B4 修复（2026-08-30 首用质检）：accuracy 验收须同时满足
                # 精度（AUROC≥0.9）与延迟（latency_pass，p95<预算）——此前
                # 只看 AUROC，latency_pass=false 的记录也显示"✓ 通过"
                acc_ok = float(metrics["auroc"]) >= 0.9
                lat_ok = metrics.get("latency_pass")
                passed = False if not acc_ok or lat_ok is False else True if lat_ok is True else None
                if acc_ok and lat_ok is None:
                    mark_override = "未知（缺延迟判定）"
                if acc_ok and lat_ok is False:
                    mark_override = "✗ 延迟未达标"
                elif not acc_ok:
                    mark_override = "✗ 精度未达标"
            else:
                passed = None
            mark = mark_override or {True: "✓ 通过", False: "✗ 未达标",
                                     None: "-"}[passed]
            item = QTableWidgetItem(f"{mark}　{run.get('created_at', '')}")
            if passed is True:
                item.setForeground(QColor(SUCCESS))
            elif passed is False:
                item.setForeground(QColor(DANGER))
            self.table.setItem(r, 7, item)
            version = metrics.get("model_version")
            version_text = "在线服务（不固定）" if rtype == "benchmark" else (f"v{version}" if version is not None else "未知")
            self.table.setItem(r, 8, QTableWidgetItem(version_text))
            self.table.setItem(r, 9, QTableWidgetItem(str(run.get("n_images") if run.get("n_images") is not None else "未知")))
            self.table.setItem(r, 10, QTableWidgetItem(_rate_text(metrics, "tp", "fn")))
            self.table.setItem(r, 11, QTableWidgetItem(_rate_text(metrics, "fp", "tn")))
        # 走查 2026-08-29 修复：刷新后自动选中最先记录并画延迟图——此前
        # 评估完成 reload 后无选中行，延迟明细一直空（"跑完看不到内容"）
        self.table.selectRow(0)

    def _on_toggle_all(self) -> None:
        """全选/全不选切换（前端反馈 2026-09-13 #6）：有未勾选的则全勾，否则全清。"""
        boxes = [self.table.cellWidget(r, 0)
                 for r in range(self.table.rowCount())]
        boxes = [b for b in boxes if isinstance(b, QCheckBox)]
        if not boxes:
            return
        target = not all(b.isChecked() for b in boxes)
        for b in boxes:
            b.setChecked(target)

    def _checked_runs(self) -> list[dict]:
        rows = []
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 0)
            if isinstance(w, QCheckBox) and w.isChecked():
                row = int(w.property("row"))
                if 0 <= row < len(self._runs):
                    rows.append(self._runs[row])
        return rows

    def _on_compare(self) -> None:
        runs = self._checked_runs()
        if len(runs) != 2:
            warn(self, "请勾选恰好两行评估记录进行对比")
            return
        dlg = CompareDialog(runs[0], runs[1], self)
        dlg.exec()

    def _on_delete_selected(self) -> None:
        """删除勾选的评估记录（admin；2026-08-29 用户要求补充）。

        只删 EvalRun 记录行，不触碰快照/图片/模型；成功后刷新列表。
        """
        runs = self._checked_runs()
        if not runs:
            warn(self, "请先勾选要删除的评估记录")
            return
        names = "、".join(
            r.get("name") or f"运行 #{r.get('id')}" for r in runs[:5])
        if len(runs) > 5:
            names += f" 等 {len(runs)} 条"
        if QMessageBox.question(
                self, "删除评估记录",
                f"确定删除以下 {len(runs)} 条评估记录？\n{names}\n\n"
                "仅删除记录行，不影响模型与图片数据。") != QMessageBox.Yes:
            return
        ids = [r["id"] for r in runs]

        def _del() -> dict | None:
            last = None
            for rid in ids:
                last = self._client.delete_eval_run(rid)
            return last

        run_async(self, _del, lambda _r: self.reload())

    def _budget_verdict(self, mets: dict, tp, tn, fp, fn_) -> str:
        """AC-072：FPR / 缺陷召回对照业务预算达标线（设置页可配）。"""
        lim_fpr = biz_budget_value(self._get_biz_budget, "g2_max_fpr_pct")
        lim_rec = biz_budget_value(self._get_biz_budget, "g2_min_recall_pct")

        def _v(key, a, b):
            v = (a / b) if b else None
            return None if v is None else float(v) * 100

        fpr, rec = _v("fpr", fp, fp + tn), _v("recall_defect", tp, tp + fn_)
        s_fpr = ("无定义" if fpr is None else
                 ("超限" if fpr > lim_fpr else "达标"))
        s_rec = ("无定义" if rec is None else
                 ("低于下限" if rec < lim_rec else "达标"))
        return (f"业务预算达标线：误报率(FPR) ≤{lim_fpr:g}%（{s_fpr}）　"
                f"缺陷召回 ≥{lim_rec:g}%（{s_rec}）")

    @staticmethod
    def _working_point_text(mets: dict) -> str:
        """FPR≤预算时的最大 recall 参考工作点（不修改生产阈值）。"""
        wp = mets.get("budget_working_point")
        if not isinstance(wp, dict):
            return ""
        return (f"参考工作点（FPR≤{float(wp.get('max_fpr', 0)) * 100:g}%）："
                f"最大缺陷召回 {float(wp['recall']) * 100:.1f}%　"
                f"阈值 {float(wp['threshold']):.4g}　FPR {float(wp['fpr']) * 100:.1f}%"
                "（仅供参考，不修改生产阈值）")

    @staticmethod
    def _coverage_text(mets: dict) -> str:
        """批次覆盖告警：未覆盖批次 + 域外比例 + 纳入 train/good 建议。"""
        cov = mets.get("batch_coverage")
        if not isinstance(cov, dict):
            return ""
        ood = cov.get("out_of_domain_ratio")
        s_ood = "-" if ood is None else f"{float(ood) * 100:.1f}%"
        unc = cov.get("uncovered_batches") or []
        if not cov.get("train_batches"):
            return f"批次覆盖：train/good 来源未知，仅统计域外比例 {s_ood}"
        if not unc and not (ood or 0):
            return "批次覆盖：评估正常样本批次均已被 train/good 覆盖"
        appr = set(cov.get("approved_extra_good_batches") or [])
        parts = [b + ("（已登记）" if b in appr else "") for b in unc]
        txt = (f"⚠ 批次覆盖告警：未覆盖批次 {('、'.join(parts) or '无')}　"
               f"域外比例 {s_ood}")
        sugg = cov.get("suggestions") or []
        if sugg:
            txt += "\n建议：" + "；".join(sugg)
        return txt

    def _on_row_selected(self) -> None:
        row = self.table.currentRow()
        if not (0 <= row < len(self._runs)):
            return
        run = self._runs[row]
        robust = run.get("run_type") == "robustness"
        self.robustness_panel.setVisible(robust)
        for widget in (self.lat_head, self.latency_chart, self.lbl_latency_hint):
            widget.setVisible(not robust)
        if robust:
            self.robustness_panel.set_run(run)
        # 混淆矩阵派生指标明细：优先用后端已算好的键；旧记录（无派生键）
        # 由 tn/fp/fn/tp 本地等价推导；分母为 0（除零）显示 "-"
        mets = run.get("metrics") or {}
        tp, tn = mets.get("tp"), mets.get("tn")
        fp, fn_ = mets.get("fp"), mets.get("fn")
        if all(isinstance(v, (int, float)) for v in (tp, tn, fp, fn_)):
            def _cell(key, a, b):
                v = (a / b) if b else None
                return (f"{float(v) * 100:.1f}%" if v is not None
                        else "未知（零分母）")
            n_all = tp + tn + fp + fn_
            self.lbl_metrics_detail.setText(
                f"混淆矩阵：tn={int(tn)} fp={int(fp)} fn={int(fn_)} "
                f"tp={int(tp)}　|　准确率 {_cell('accuracy', tp + tn, n_all)}　"
                f"精确率 {_cell('precision', tp, tp + fp)}　"
                f"检出率(缺陷召回) {_cell('recall_defect', tp, tp + fn_)}　"
                f"误报率(FPR) {_cell('fpr', fp, fp + tn)}　"
                f"特异度 {_cell('specificity', tn, tn + fp)}"
                f"\n{self._budget_verdict(mets, tp, tn, fp, fn_)}"
                + "".join(f"\n{t}" for t in (self._working_point_text(mets),
                                             self._coverage_text(mets)) if t))
        else:
            self.lbl_metrics_detail.setText(
                "该记录无混淆矩阵数据（仅精度评估记录含 tn/fp/fn/tp）")
        latency = run.get("latency") or {}
        # 走查 2026-08-29 修复：不限定 benchmark 才画图——accuracy 记录也有
        # 延迟（mean/p95），有键就画；benchmark 才有完整 p50/max/min 统计量
        if not isinstance(latency, dict) or not latency:
            self.latency_chart.set_values([], [])
            self.lbl_latency_hint.setText(
                "该记录无延迟数据（延迟基准才会统计完整延迟分布）")
            return
        keys = [("mean", "mean"), ("p50", "p50"), ("p95", "p95"),
                ("max", "max"), ("min", "min")]
        values, labels = [], []
        for key, label in keys:
            v = latency.get(key)
            if v is not None:
                values.append(float(v))
                labels.append(label)
        self.latency_chart.set_budget(float(self._get_budget() or 200.0))
        self.latency_chart.set_values(values, labels)
        name = run.get("name") or f"运行 #{run.get('id')}"
        rtype = str(run.get("run_type", ""))
        self.lbl_latency_hint.setText(
            f"{name}（{RUN_TYPE_NAMES.get(rtype, rtype)}）："
            f"超预算柱体为红色，红色虚线为 {self._get_budget():.0f}ms 预算线")

    def _on_task(self, task: dict) -> None:
        if self._task_id is None or str(task.get("id", task.get("task_id"))) != self._task_id:
            return
        self.progress.update_task(task)
        if str(task.get("status")) in ("done", "success", "completed", "failed", "error", "cancelled", "canceled"):
            self._task_id = None
            self._poll_timer.stop()
            self._update_run_buttons()
            self.reload()
            # 前端反馈 2026-09-13 #6：鲁棒性等任务"完成后记录没立刻出现"
            # （WS 终态推送可能先于 DB 提交落库可读），延迟 1.5s 兜底二次刷新
            QTimer.singleShot(1500, self.reload)
