"""存档统计页（2026-10-09 重构，原「统计报表」）。

以工单为基础的历史管理 + 统计：
- 左侧工单档案列表：进行中 / 已存档 两组；选中工单驱动右侧全部统计
- 存档管理：存档（数据保留转历史）/ 还原 / 删除（级联清检测与反馈）
- 右侧统计（复核后口径）：KPI（检测数/复核覆盖率/真实不良率/误报率/
  漏检率/平均延迟）+ 误判率趋势 + 缺陷帕累托 + 批次对比 + 产线健康 +
  延迟趋势；固定「全部」时间范围（工单本身就是时间边界，不再用 7d/30d）
- 未选工单时右侧为空态提示

操作日志在「系统设置」页。全部数据为纯 DB 聚合，零模型前向。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (attach_level_badge, make_banner, make_card,
                             notify, run_async, warn)
from ui.theme import DANGER, PRIMARY, SUCCESS, WARNING
from ui.widgets.charts import BarChart, LatencyChart, LineChart
from ui.widgets.kpi_card import KpiCard


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


class StatsPage(QWidget):
    """存档统计页：工单档案管理 + 按工单统计（复核后口径）。"""

    def __init__(self, client: ApiClient, get_category, get_budget=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._get_category = get_category   # 兼容旧注入（统计已按工单）
        self._get_budget = get_budget or (lambda: 200.0)
        self._wo_items: list[dict] = []     # 进行中 + 已存档工单合并缓存

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(make_banner(
            "存档统计：以工单为单位的历史管理与统计"
            "（工单即时间边界，统计固定全量口径）"))
        attach_level_badge(self, top)
        top.addStretch(1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.setProperty("flat", True)
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        # ══════════ 左：工单档案（进行中 / 已存档）══════════
        left = make_card()
        l_lay = QVBoxLayout(left)
        l_head = QLabel("工单档案（单击查看统计）")
        l_head.setProperty("heading", True)
        l_lay.addWidget(l_head)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["工单", "检测", "不良率"])
        self.tree.setColumnWidth(0, 180)
        self.tree.itemSelectionChanged.connect(self._on_select)
        l_lay.addWidget(self.tree, 1)
        op = QHBoxLayout()
        self.btn_archive = QPushButton("存档")
        self.btn_archive.setToolTip("进行中工单转入历史管理（数据保留）")
        self.btn_archive.clicked.connect(self._on_archive)
        op.addWidget(self.btn_archive)
        self.btn_unarchive = QPushButton("还原")
        self.btn_unarchive.setToolTip("已存档工单还原回进行中列表")
        self.btn_unarchive.clicked.connect(self._on_unarchive)
        op.addWidget(self.btn_unarchive)
        self.btn_delete = QPushButton("删除")
        self.btn_delete.setProperty("danger", True)
        self.btn_delete.setToolTip("级联删除该工单的全部检测/反馈记录，不可恢复")
        self.btn_delete.clicked.connect(self._on_delete)
        op.addWidget(self.btn_delete)
        self.btn_export = QPushButton("导出 CSV")
        self.btn_export.setToolTip("导出所选工单报表（误判率趋势 + 批次对比，"
                                   "BOM 头，Excel 直接打开）")
        self.btn_export.clicked.connect(self._export_report)
        op.addWidget(self.btn_export)
        l_lay.addLayout(op)
        splitter.addWidget(left)

        # ══════════ 右：所选工单统计（复核后口径）══════════
        right = QWidget()
        r_root = QVBoxLayout(right)
        r_root.setContentsMargins(0, 0, 0, 0)
        r_root.setSpacing(8)

        self.lbl_scope = QLabel("未选工单：请在左侧选择要查看的工单")
        self.lbl_scope.setProperty("subtext", True)
        r_root.addWidget(self.lbl_scope)

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
        r_root.addWidget(kpi_card)

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
        r_root.addLayout(row1, 1)

        split2 = QSplitter(Qt.Horizontal)
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
        split2.addWidget(batch_card)

        right2 = QWidget()
        r2_lay = QVBoxLayout(right2)
        r2_lay.setContentsMargins(0, 0, 0, 0)
        r2_lay.setSpacing(8)
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
        r2_lay.addWidget(health_card)

        lat_card = make_card()
        l2_lay = QVBoxLayout(lat_card)
        l_head = QLabel("平均延迟趋势")
        l_head.setProperty("heading", True)
        l2_lay.addWidget(l_head)
        self.latency_chart = LatencyChart(budget_ms=float(self._get_budget()))
        l2_lay.addWidget(self.latency_chart, 1)
        r2_lay.addWidget(lat_card, 1)
        split2.addWidget(right2)
        split2.setSizes([620, 420])
        r_root.addWidget(split2, 1)

        splitter.addWidget(right)
        splitter.setSizes([300, 900])
        root.addWidget(splitter, 1)

        self._client.error_occurred.connect(lambda m: warn(self, m))
        self.reload()

    # ══════════════════ 工单档案列表 ══════════════════
    def reload(self) -> None:
        """刷新工单档案列表（进行中 + 已存档）。"""
        run_async(self,
                  lambda: (self._client.list_workorders("all"),
                           self._client.list_workorders("all", archived=True)),
                  self._fill_tree)

    def _fill_tree(self, data) -> None:
        active, archived = data if isinstance(data, tuple) else ([], [])
        active = [w for w in (active or []) if isinstance(w, dict)]
        archived = [w for w in (archived or []) if isinstance(w, dict)]
        self._wo_items = active + archived
        cur_wid = self._selected_wo_id()
        self.tree.blockSignals(True)
        self.tree.clear()
        for title, items in (("进行中", active), ("已存档", archived)):
            grp = QTreeWidgetItem([title, "", ""])
            grp.setFlags(grp.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(grp)
            for wo in items:
                stats = wo.get("stats") or {}
                n_ins = int(stats.get("inspected", 0) or 0)
                rate = float(stats.get("rate", 0.0) or 0.0)
                it = QTreeWidgetItem([
                    str(wo.get("name") or "-"), str(n_ins),
                    f"{rate * 100:.1f}%" if n_ins else "-"])
                it.setData(0, Qt.UserRole, wo.get("id"))
                it.setToolTip(0, f"工单 #{wo.get('id')}｜"
                                 f"数据源：{'、'.join(str(d.get('name')) for d in (wo.get('datasources') or [])) or '-'}")
                grp.addChild(it)
                if wo.get("id") == cur_wid:
                    self.tree.setCurrentItem(it)
        grp0 = self.tree.topLevelItem(0)
        if grp0 is not None:
            grp0.setExpanded(True)
        grp1 = self.tree.topLevelItem(1)
        if grp1 is not None:
            grp1.setExpanded(bool(archived))
        self.tree.blockSignals(False)
        if not active and not archived:
            self.tree.blockSignals(True)
            grp0 = self.tree.topLevelItem(0)
            if grp0 is not None:
                grp0.addChild(QTreeWidgetItem(["（暂无工单）", "", ""]))
            self.tree.blockSignals(False)
        self._refresh_op_buttons()

    def _selected_wo_id(self) -> int | None:
        it = self.tree.currentItem()
        if it is None:
            return None
        wid = it.data(0, Qt.UserRole)
        return int(wid) if wid is not None else None

    def _selected_wo(self) -> dict | None:
        wid = self._selected_wo_id()
        if wid is None:
            return None
        return next((w for w in self._wo_items if w.get("id") == wid), None)

    def _on_select(self) -> None:
        self._refresh_op_buttons()
        wo = self._selected_wo()
        if wo is None:
            self.lbl_scope.setText("未选工单：请在左侧选择要查看的工单")
            self._clear_stats()
            return
        self.lbl_scope.setText(
            f"统计口径：工单「{wo.get('name')}」"
            f"（{'已存档' if wo.get('archived') else '进行中'}，全量）")
        self._load_stats(int(wo.get("id")), wo)

    def _refresh_op_buttons(self) -> None:
        wo = self._selected_wo()
        self.btn_archive.setEnabled(bool(wo) and not wo.get("archived"))
        self.btn_unarchive.setEnabled(bool(wo) and bool(wo.get("archived")))
        self.btn_delete.setEnabled(bool(wo))
        self.btn_export.setEnabled(bool(wo))

    # ══════════════════ 存档管理操作 ══════════════════
    def _on_archive(self) -> None:
        wo = self._selected_wo()
        if wo is None:
            return
        if QMessageBox.question(
                self, "存档工单",
                f"确定存档工单「{wo.get('name')}」？\n"
                "数据全部保留并转入历史管理，不再参与全局统计；"
                "可随时在此还原。") != QMessageBox.Yes:
            return

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("archived"):
                notify(self, f"工单已存档：{wo.get('name')}")
                self.reload()
            else:
                warn(self, "存档失败，请重试")

        run_async(self, lambda: self._client.archive_workorder(
            int(wo.get("id"))), _done)

    def _on_unarchive(self) -> None:
        wo = self._selected_wo()
        if wo is None:
            return

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("archived") is False:
                notify(self, f"工单已还原：{wo.get('name')}")
                self.reload()
            else:
                warn(self, "还原失败，请重试")

        run_async(self, lambda: self._client.unarchive_workorder(
            int(wo.get("id"))), _done)

    def _on_delete(self) -> None:
        wo = self._selected_wo()
        if wo is None:
            return
        if QMessageBox.question(
                self, "删除工单",
                f"确定删除工单「{wo.get('name')}」？\n"
                "将级联删除该工单名下的全部检测记录与反馈记录，"
                "不可恢复！\n数据源、图片与模型保留。") != QMessageBox.Yes:
            return

        def _done(res) -> None:
            if isinstance(res, dict) and res.get("deleted"):
                notify(self, f"工单已删除：{wo.get('name')}")
                self.reload()
            else:
                warn(self, "删除失败，请重试")

        run_async(self, lambda: self._client.delete_workorder(
            int(wo.get("id"))), _done)

    # ══════════════════ 数据加载（按工单，固定全量）══════════════════
    def _clear_stats(self) -> None:
        for k in (self.kpi_det, self.kpi_coverage, self.kpi_true_rate,
                  self.kpi_fp, self.kpi_fn, self.kpi_latency,
                  self.kpi_gray, self.kpi_open, self.kpi_align):
            k.set_value("-")
        self.trend_chart.clear()
        self.trend_chart.set_labels([], "比率")
        self.pareto_chart.set_data([], {})
        self.table_batch.setRowCount(0)
        self.latency_chart.set_series([], [])
        self.lbl_health_hint.setText("")

    def _load_stats(self, wid: int, wo: dict) -> None:
        run_async(self, lambda: self._client.stats_quality(
            workorder_id=wid, range_name="all"), self._fill_quality)
        run_async(self, lambda: self._client.stats_mistake_trend(
            workorder_id=wid, range_name="all"), self._fill_trend)
        run_async(self, lambda: self._client.stats_defect_types(
            workorder_id=wid, range_name="all"), self._fill_pareto)
        run_async(self, lambda: self._client.stats_batch_summary(
            workorder_id=wid, range_name="all"), self._fill_batch)
        run_async(self, lambda: self._client.stats_health(
            workorder_id=wid, range_name="7d"), self._fill_health)
        # 延迟趋势端点基于 StatsDaily 聚合表、无法按工单拆解——
        # 用工单首个品类近似（无品类则全局近 30 天）
        cats = [str(c) for c in (wo.get("categories") or []) if c]
        cat = cats[0] if len(cats) == 1 else ""
        run_async(self, lambda: self._client.stats_timeseries(
            days=30, category=cat), self._fill_latency)

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
        """导出所选工单报表：误判率趋势 + 批次对比合成一个 CSV。"""
        import csv
        import io

        from PySide6.QtWidgets import QFileDialog
        wo = self._selected_wo()
        if wo is None:
            warn(self, "请先在左侧选择要导出的工单")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出报表", f"工单报表_{wo.get('name')}.csv",
            "CSV 文件 (*.csv)")
        if not save_path:
            return
        wid = int(wo.get("id"))
        trend = self._client.stats_mistake_trend(
            workorder_id=wid, range_name="all")
        batch = self._client.stats_batch_summary(
            workorder_id=wid, range_name="all")
        quality = self._client.stats_quality(
            workorder_id=wid, range_name="all")
        buf = io.StringIO()
        w = csv.writer(buf)
        q = quality or {}
        w.writerow(["# 工单质检报表（复核后口径）",
                    f"工单={wo.get('name')}",
                    f"状态={'已存档' if wo.get('archived') else '进行中'}"])
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
