"""统计报表页（W-report 2026-08-29 重构，docs/重构设计_学习与统计页.md）。

工单质检报表——复核后口径（用户明确："不要今日检测和今日不良，用今日来统计
没有意义，要用工单来统计"）：
- KPI：检测数 / 复核覆盖率 / 真实不良率（复核口径）/ 误报率 / 漏检率 / 平均延迟
- 误判率趋势（按日 FP 率/FN 率，学习价值的运营侧证据）
- 缺陷类型帕累托（复核确认缺陷按 defect_type 分布）
- 批次对比表（按导入批次聚合检测/不良/误报/漏检）
- 产线健康信号（灰区率 / open 未解释信号 / 对位预警，algo 每图产出落库聚合）

操作日志已移至「系统设置」页（审计功能不占报表版面）。
全部数据为纯 DB 聚合，零模型前向。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (attach_level_badge, make_banner, make_card,
                             notify, run_async, warn)
from ui.theme import DANGER, PRIMARY, SUCCESS, WARNING
from ui.widgets.charts import BarChart, LatencyChart, LineChart
from ui.widgets.kpi_card import KpiCard

# 时间范围选项：(名称, range 参数)
RANGE_OPTIONS = [("全部", "all"), ("近 7 天", "7d"), ("近 30 天", "30d")]


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


class StatsPage(QWidget):
    """工单质检报表页（复核后口径）。"""

    def __init__(self, client: ApiClient, get_category, get_budget=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category
        self._get_budget = get_budget or (lambda: 200.0)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner(
            "工单质检报表：复核后口径（误报率/漏检率以操作员复核真值为准）"))
        attach_level_badge(self, top)
        top.addStretch(1)
        self.lbl_scope = QLabel("统计口径：当前工单")
        self.lbl_scope.setProperty("subtext", True)
        top.addWidget(self.lbl_scope)
        top.addWidget(QLabel("范围"))
        self.combo_range = QComboBox()
        for name, val in RANGE_OPTIONS:
            self.combo_range.addItem(name, val)
        self.combo_range.setCurrentIndex(2)   # 默认近 30 天
        self.combo_range.currentIndexChanged.connect(lambda _i: self.reload())
        top.addWidget(self.combo_range)
        top.addWidget(QLabel("品类"))
        self.combo_category = QComboBox()
        self.combo_category.addItem("全部品类", "")
        self.combo_category.currentIndexChanged.connect(lambda _i: self.reload())
        top.addWidget(self.combo_category)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        btn_export = QPushButton("导出报表 CSV")
        btn_export.setToolTip("导出当前口径的误判率趋势 + 批次对比（BOM 头，"
                              "Excel 直接打开）")
        btn_export.clicked.connect(self._export_report)
        top.addWidget(btn_export)
        root.addLayout(top)

        # ── KPI 卡片行（复核后口径）──
        kpi_card = make_card()
        kpi_row = QHBoxLayout(kpi_card)
        self.kpi_det = KpiCard("检测数", "-")
        self.kpi_coverage = KpiCard("复核覆盖率", "-", accent=PRIMARY)
        self.kpi_coverage.setToolTip("有操作员复核真值的检测占比")
        self.kpi_true_rate = KpiCard("真实不良率", "-", accent=WARNING)
        self.kpi_true_rate.setToolTip("复核确认缺陷 / 复核数（区别于系统判定口径）")
        self.kpi_fp = KpiCard("误报率", "-", accent=DANGER)
        self.kpi_fp.setToolTip("过杀：系统判异常而实为正常 / (误报+确认正常)")
        self.kpi_fn = KpiCard("漏检率", "-", accent=DANGER)
        self.kpi_fn.setToolTip("逃逸：系统判正常而实为缺陷 / (漏检+确认缺陷)")
        self.kpi_latency = KpiCard("平均延迟", "-", accent=PRIMARY)
        for k in (self.kpi_det, self.kpi_coverage, self.kpi_true_rate,
                  self.kpi_fp, self.kpi_fn, self.kpi_latency):
            kpi_row.addWidget(k)
        root.addWidget(kpi_card)

        # ── 第一行图：误判率趋势 | 缺陷类型帕累托 ──
        row1 = QHBoxLayout()
        trend_card = make_card()
        t_lay = QVBoxLayout(trend_card)
        t_head = QLabel("误判率趋势（按日；学习生效应持续下降）")
        t_head.setProperty("heading", True)
        t_lay.addWidget(t_head)
        self.trend_chart = LineChart()
        t_lay.addWidget(self.trend_chart, 1)
        row1.addWidget(trend_card, 3)

        pareto_card = make_card()
        p_lay = QVBoxLayout(pareto_card)
        p_head = QLabel("缺陷类型帕累托（复核确认缺陷分布）")
        p_head.setProperty("heading", True)
        p_lay.addWidget(p_head)
        self.pareto_chart = BarChart()
        p_lay.addWidget(self.pareto_chart, 1)
        row1.addWidget(pareto_card, 2)
        root.addLayout(row1, 1)

        # ── 第二行：批次对比表 | 产线健康 + 延迟趋势 ──
        splitter = QSplitter(Qt.Horizontal)
        batch_card = make_card()
        b_lay = QVBoxLayout(batch_card)
        b_head = QLabel("批次对比（哪个批次不良/误判集中）")
        b_head.setProperty("heading", True)
        b_lay.addWidget(b_head)
        self.table_batch = QTableWidget(0, 7)
        self.table_batch.setHorizontalHeaderLabels(
            ["批次", "检测数", "系统不良", "复核数", "误报率", "漏检率",
             "平均延迟"])
        self.table_batch.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_batch.setAlternatingRowColors(True)
        self.table_batch.horizontalHeader().setStretchLastSection(True)
        b_lay.addWidget(self.table_batch, 1)
        splitter.addWidget(batch_card)

        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(8)
        health_card = make_card()
        h_lay = QVBoxLayout(health_card)
        h_head = QLabel("产线健康信号（近 7 天检测聚合）")
        h_head.setProperty("heading", True)
        h_lay.addWidget(h_head)
        h_row = QHBoxLayout()
        self.kpi_gray = KpiCard("灰区率", "-", accent=WARNING)
        self.kpi_gray.setToolTip("模型不确定判定占比→复判工作量信号")
        self.kpi_open = KpiCard("未解释信号", "-", accent=DANGER)
        self.kpi_open.setToolTip("open 哨兵：体系外缺陷预警（现有特征组无法解释）")
        self.kpi_align = KpiCard("对位预警", "-", accent=WARNING)
        self.kpi_align.setToolTip("对位偏移超阈值次数（治具/传送带漂移信号）")
        for k in (self.kpi_gray, self.kpi_open, self.kpi_align):
            h_row.addWidget(k)
        h_lay.addLayout(h_row)
        self.lbl_health_hint = QLabel("")
        self.lbl_health_hint.setProperty("subtext", True)
        h_lay.addWidget(self.lbl_health_hint)
        r_lay.addWidget(health_card)

        lat_card = make_card()
        l_lay = QVBoxLayout(lat_card)
        l_head = QLabel("平均延迟趋势")
        l_head.setProperty("heading", True)
        l_lay.addWidget(l_head)
        self.latency_chart = LatencyChart(budget_ms=float(self._get_budget()))
        l_lay.addWidget(self.latency_chart, 1)
        r_lay.addWidget(lat_card, 1)
        splitter.addWidget(right)
        splitter.setSizes([620, 420])
        root.addWidget(splitter, 1)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        run_async(self, self._client.list_categories, self._fill_categories)
        self.reload()

    # ══════════════════ 口径 ══════════════════
    def _workorder_id(self) -> int | None:
        fn = getattr(self, "_get_workorder_id", None)
        return fn() if callable(fn) else None

    def _range(self) -> str:
        return str(self.combo_range.currentData() or "30d")

    def _category(self) -> str:
        return str(self.combo_category.currentData() or "")

    def _fill_categories(self, cats) -> None:
        cats = [c for c in (cats or []) if c]
        cur = self._category()
        self.combo_category.blockSignals(True)
        self.combo_category.clear()
        self.combo_category.addItem("全部品类", "")
        for c in cats:
            self.combo_category.addItem(str(c), str(c))
        idx = self.combo_category.findData(cur)
        if idx >= 0:
            self.combo_category.setCurrentIndex(idx)
        self.combo_category.blockSignals(False)

    # ══════════════════ 数据加载 ══════════════════
    def reload(self) -> None:
        wid = self._workorder_id()
        cat = self._category()
        rng = self._range()
        self.lbl_scope.setText(
            "统计口径：当前工单" if wid is not None
            else "统计口径：全部工单（未选工单）")
        run_async(self, lambda: self._client.stats_quality(
            workorder_id=wid, category=cat, range_name=rng),
            self._fill_quality)
        run_async(self, lambda: self._client.stats_mistake_trend(
            workorder_id=wid, category=cat, range_name=rng),
            self._fill_trend)
        run_async(self, lambda: self._client.stats_defect_types(
            workorder_id=wid, category=cat, range_name=rng),
            self._fill_pareto)
        run_async(self, lambda: self._client.stats_batch_summary(
            workorder_id=wid, category=cat, range_name=rng),
            self._fill_batch)
        run_async(self, lambda: self._client.stats_health(
            workorder_id=wid, category=cat, range_name="7d"),
            self._fill_health)
        # 延迟趋势沿用旧端点（按日聚合）
        days = {"all": 30, "7d": 7, "30d": 30, "today": 1}.get(rng, 30)
        run_async(self, lambda: self._client.stats_timeseries(
            days=days, category=cat), self._fill_latency)

    def _fill_quality(self, data) -> None:
        if not isinstance(data, dict):
            return
        self.kpi_det.set_value(str(data.get("n_detections", 0)))
        n_rev = int(data.get("n_reviewed", 0) or 0)
        self.kpi_coverage.set_value(_pct(data.get("coverage")),
                                    f"复核 {n_rev} 条")
        if n_rev > 0:
            self.kpi_true_rate.set_value(_pct(data.get("true_defect_rate")))
            self.kpi_fp.set_value(
                _pct(data.get("fp_rate")), f"误报 {data.get('fp', 0)} 条")
            self.kpi_fn.set_value(
                _pct(data.get("fn_rate")), f"漏检 {data.get('fn', 0)} 条")
        else:
            self.kpi_true_rate.set_value("-", "暂无复核")
            self.kpi_fp.set_value("-", "暂无复核")
            self.kpi_fn.set_value("-", "暂无复核")
        lat = data.get("avg_latency_ms")
        budget = float(self._get_budget() or 200.0)
        self.kpi_latency.set_value(
            f"{float(lat):.0f} ms" if lat is not None else "-",
            f"预算 <{budget:.0f} ms")
        self.kpi_latency.set_accent(
            SUCCESS if (lat is not None and float(lat) < budget) else DANGER)

    def _fill_trend(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        self.trend_chart.clear()
        if not items:
            self.trend_chart.set_labels([], "比率")
            return
        xs = list(range(len(items)))
        self.trend_chart.add_series(
            "误报率", xs, [float(it.get("fp_rate", 0) or 0) for it in items],
            color=WARNING)
        self.trend_chart.add_series(
            "漏检率", xs, [float(it.get("fn_rate", 0) or 0) for it in items],
            color=DANGER)
        labels = [str(it.get("date", ""))[5:] for it in items]
        self.trend_chart.set_labels(labels, "比率")

    def _fill_pareto(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        if not items:
            self.pareto_chart.set_data([], {})
            return
        top = items[:8]
        self.pareto_chart.set_data(
            [str(it.get("type", "?")) for it in top],
            {"缺陷数": [int(it.get("count", 0)) for it in top]})

    def _fill_batch(self, data) -> None:
        items = (data or {}).get("items", []) if isinstance(data, dict) else []
        self.table_batch.setRowCount(len(items))
        for r, it in enumerate(items):
            vals = [str(it.get("name", "-")),
                    str(it.get("n_detections", 0)),
                    str(it.get("n_anomaly", 0)),
                    str(it.get("n_reviewed", 0)),
                    _pct(it.get("fp_rate")) if it.get("n_reviewed") else "-",
                    _pct(it.get("fn_rate")) if it.get("n_reviewed") else "-",
                    f"{float(it.get('avg_latency_ms') or 0):.0f} ms"]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c > 0:
                    cell.setTextAlignment(Qt.AlignCenter)
                self.table_batch.setItem(r, c, cell)
        if not items:
            self.table_batch.setRowCount(0)

    def _fill_health(self, data) -> None:
        if not isinstance(data, dict):
            return
        n = int(data.get("scanned", 0) or 0)
        self.kpi_gray.set_value(_pct(data.get("gray_rate")),
                                f"{data.get('gray_count', 0)} 帧")
        self.kpi_open.set_value(str(data.get("open_alerts", 0)),
                                _pct(data.get("open_rate")))
        self.kpi_align.set_value(str(data.get("align_warns", 0)),
                                 _pct(data.get("align_rate")))
        self.lbl_health_hint.setText(
            f"扫描最近 {n} 条检测记录" if n else "暂无检测记录")

    def _fill_latency(self, data) -> None:
        items = (data or {}).get("items", [])
        labels = [str(i.get("date", ""))[5:] for i in items]
        latencies = [float(i.get("avg_latency_ms", 0) or 0) for i in items]
        self.latency_chart.set_budget(float(self._get_budget() or 200.0))
        self.latency_chart.set_series(latencies, labels)

    # ══════════════════ 导出 ══════════════════
    def _export_report(self) -> None:
        """导出当前口径报表：误判率趋势 + 批次对比合成一个 CSV。"""
        import csv
        import io

        from PySide6.QtWidgets import QFileDialog
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出报表", "质检报表.csv", "CSV 文件 (*.csv)")
        if not save_path:
            return
        wid = self._workorder_id()
        cat = self._category()
        rng = self._range()
        trend = self._client.stats_mistake_trend(
            workorder_id=wid, category=cat, range_name=rng)
        batch = self._client.stats_batch_summary(
            workorder_id=wid, category=cat, range_name=rng)
        quality = self._client.stats_quality(
            workorder_id=wid, category=cat, range_name=rng)
        buf = io.StringIO()
        w = csv.writer(buf)
        q = quality or {}
        w.writerow(["# 质检报表（复核后口径）",
                    f"范围={rng}", f"品类={cat or '全部'}"])
        w.writerow(["检测数", q.get("n_detections", 0),
                    "复核数", q.get("n_reviewed", 0),
                    "真实不良率", round(float(q.get("true_defect_rate") or 0), 4),
                    "误报率", round(float(q.get("fp_rate") or 0), 4),
                    "漏检率", round(float(q.get("fn_rate") or 0), 4)])
        w.writerow([])
        w.writerow(["## 误判率趋势"])
        w.writerow(["date", "reviewed", "fp_rate", "fn_rate", "mistake_rate"])
        for it in (trend or {}).get("items", []):
            w.writerow([it.get("date"), it.get("reviewed"),
                        round(float(it.get("fp_rate") or 0), 4),
                        round(float(it.get("fn_rate") or 0), 4),
                        round(float(it.get("mistake_rate") or 0), 4)])
        w.writerow([])
        w.writerow(["## 批次对比"])
        w.writerow(["batch", "n_detections", "n_anomaly", "n_reviewed",
                    "fp_rate", "fn_rate", "avg_latency_ms"])
        for it in (batch or {}).get("items", []):
            w.writerow([it.get("name"), it.get("n_detections"),
                        it.get("n_anomaly"), it.get("n_reviewed"),
                        round(float(it.get("fp_rate") or 0), 4),
                        round(float(it.get("fn_rate") or 0), 4),
                        round(float(it.get("avg_latency_ms") or 0), 1)])
        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(buf.getvalue())
        notify(self, f"已保存：{save_path}")
