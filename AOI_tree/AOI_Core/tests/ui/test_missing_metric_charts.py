"""学习页缺失指标不画成 0 + contribution latest 透出 trad_groups（AC-061~070 UI 缺口）。"""
from __future__ import annotations

import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from ui.pages.learning_page import _num
from ui.widgets.charts import BarChart, LatencyChart, LineChart, num_or_nan


def _bars(chart: BarChart):
    return [it for it in chart.plot.getPlotItem().items
            if isinstance(it, pg.BarGraphItem)]


class MissingMetricChartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_num_missing_is_nan(self):
        self.assertTrue(math.isnan(_num(None)))
        self.assertTrue(math.isnan(_num("x")))
        self.assertEqual(_num(0), 0.0)
        self.assertEqual(_num("0.5"), 0.5)

    def test_bar_skips_missing(self):
        ch = BarChart()
        ch.set_data(["a", "b", "c"], {"s1": [0.9, None, float("nan")],
                                      "s2": [None, None, None]})
        bars = _bars(ch)
        self.assertEqual(len(bars), 1)  # 全缺失序列不画
        heights = list(bars[0].opts["height"])
        self.assertEqual(heights, [0.9])  # 缺失不补 0
        ch.close()

    def test_line_breaks_on_nan(self):
        ch = LineChart()
        ch.add_series("r", [0, 1, 2], [0.1, float("nan"), 0.3])
        items = ch.plot.listDataItems()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].opts.get("connect"), "finite")
        ch.close()


    def test_latency_series_breaks_on_nan(self):
        ch = LatencyChart()
        ch.set_series([120.0, float("nan"), 90.0], ["a", "b", "c"])
        items = ch.plot.getPlotItem().listDataItems()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].opts.get("connect"), "finite")
        ch.close()

    def test_stats_trend_and_latency_skip_empty_days(self):
        from ui.pages.stats_page import StatsPage
        trend, lat = Mock(), Mock()
        fake = SimpleNamespace(trend_chart=trend, latency_chart=lat,
                               _get_budget=lambda: 200)
        StatsPage._fill_trend(fake, {"items": [
            {"date": "2026-10-01", "reviewed": 0, "fp_rate": 0.0, "recall_defect": None},
            {"date": "2026-10-02", "reviewed": 5, "fp_rate": 0.2, "recall_defect": 1.0}]})
        fp = trend.add_series.call_args_list[0].args[2]
        self.assertTrue(math.isnan(fp[0]))
        self.assertEqual(fp[1], 0.2)
        recall = trend.add_series.call_args_list[1].args[2]
        self.assertEqual(recall[1], 1.0)  # 有复核的真实检出率 100% 保留
        StatsPage._fill_latency(fake, {"items": [
            {"date": "2026-10-01", "n_inspected": 0, "avg_latency_ms": 0.0},
            {"date": "2026-10-02", "n_inspected": 3, "avg_latency_ms": 80.0}]})
        vals = lat.set_series.call_args.args[0]
        self.assertTrue(math.isnan(vals[0]))
        self.assertEqual(vals[1], 80.0)

    def test_num_or_nan(self):
        self.assertTrue(math.isnan(num_or_nan(None)))
        self.assertEqual(num_or_nan(0), 0.0)


class ContributionLatestTradGroupsTest(unittest.TestCase):
    def test_latest_returns_trad_groups(self):
        from backend.api import routes_learning
        tg = {"status": "available", "groups": {"g1": {"single_auroc": 0.8}}}
        row = {"id": 7, "created_at": None, "dataset_desc": "d",
               "metrics": {"ablation": {}, "trad_groups": tg}}
        with patch.object(routes_learning, "_latest_run_dict", return_value=row):
            out = routes_learning.api_learning_contribution_latest("cat")
        self.assertEqual(out["trad_groups"], tg)


if __name__ == "__main__":
    unittest.main()
