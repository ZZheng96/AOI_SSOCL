"""AC-058 存档统计页口径（离屏）：推理耗时与 e2e 分列、复用工作台口径函数、
缺数据显示“—/不可用”不补 0、口径行/tooltip 可见、批次表与导出不以 0 冒充缺失。"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_stats as rs
from backend.db import database
from backend.db.models import Base, Detection
from ui.pages import dashboard_page as D
from ui.pages import stats_page as S

_app = QApplication.instance() or QApplication([])
N = 7
_T0 = time.perf_counter()


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)


def _page():
    # 构造时 reload() 走后台线程；离屏测试直接驱动 _fill_*，故屏蔽 run_async
    with patch.object(S, "run_async", lambda *a, **k: None):
        return S.StatsPage(FakeClient(), lambda: "", lambda: 200.0)


def _cells(p, row):
    return [p.table_batch.item(row, c).text() for c in range(p.table_batch.columnCount())]


class StatsPageCaliberTest(unittest.TestCase):
    def test_reuses_dashboard_caliber(self):
        _log(1, "统计页复用工作台口径函数/常量（同一对象，非复制）")
        self.assertIs(S.scope_caliber_texts, D.scope_caliber_texts)
        self.assertIs(S.CALIBER_TIP, D.CALIBER_TIP)
        self.assertIs(S.NA, D.NA)

    def test_initial_widgets_and_tooltip(self):
        _log(2, "离屏实例化：耗时卡片初始“—”，说明文字清晰可见")
        p = _page()
        self.assertEqual(p.kpi_latency._title.text(), "模型判断耗时（均值）")
        self.assertEqual(p.kpi_latency_e2e._title.text(), "检测处理耗时（均值）")
        self.assertEqual(p.kpi_latency._value.text(), D.NA)
        self.assertEqual(p.kpi_latency_e2e._value.text(), D.NA)
        heads = [p.table_batch.horizontalHeaderItem(c).text()
                 for c in range(p.table_batch.columnCount())]
        self.assertEqual(heads[-2:], ["模型判断耗时", "检测处理耗时"])
        for w in (p.kpi_latency, p.kpi_latency_e2e, p.lbl_caliber, p.table_batch):
            self.assertIn("当前工单的全部检测记录", w.toolTip())
            self.assertIn("不是相机拍到的帧数", w.toolTip())
            self.assertIn("显示不可用", w.toolTip())

    def test_legacy_and_empty_payloads(self):
        _log(3, "旧后端/空范围（avg_latency_ms=0.0）：显示 —/不可用，不出现 0 ms / 0.0%")
        p = _page()
        for data in (None, {}, {"scope_stats": {"n_inspected": 0, "avg_latency_ms": 0.0}}):
            p._fill_caliber(data)
            self.assertEqual(p.kpi_latency._value.text(), D.NA)
            self.assertEqual(p.kpi_latency_e2e._value.text(), "不可用")
            self.assertIn("接收→落库率 不可用", p.lbl_caliber.text())
            self.assertIn("共 0 条已保存检测结果", p.lbl_caliber.text())
        p._fill_quality({"n_detections": 0, "n_reviewed": 0, "coverage": 0.0,
                         "avg_latency_ms": 0.0, "fp_rate": 0.0, "fn_rate": 0.0})
        for k in (p.kpi_coverage, p.kpi_true_rate, p.kpi_fp, p.kpi_fn, p.kpi_latency):
            self.assertEqual(k._value.text(), D.NA)
        p._fill_health({"scanned": 0, "gray_rate": 0.0, "open_rate": 0.0, "align_rate": 0.0})
        for k in (p.kpi_gray, p.kpi_open, p.kpi_align):
            self.assertEqual(k._value.text(), D.NA)

    def test_full_scope_split(self):
        _log(4, "完整统计数据：模型判断耗时与完整处理耗时分列，说明工单和来源")
        p = _page()
        p._fill_caliber({"scope_stats": {
            "n_inspected": 4, "avg_latency_ms": 25.0, "latency_basis": "inference_only",
            "avg_latency_e2e_ms": 40.0, "p95_latency_e2e_ms": 50.0, "n_e2e_missing": 2,
            "range": "all", "window_start": None, "denominator": "persisted_detections",
            "workorder_id": 7, "by_source": {"pipeline": 3, "upload": 1},
            "receipt_rate": None}})
        self.assertEqual(p.kpi_latency._value.text(), "25 ms")
        self.assertIn("模型判断耗时", p.kpi_latency._sub.text())
        self.assertIn("参考上限 <200 ms", p.kpi_latency._sub.text())
        self.assertEqual(p.kpi_latency_e2e._value.text(), "40 ms")
        self.assertIn("2 / 4", p.kpi_latency_e2e._sub.text())
        self.assertIn("没有该数据", p.kpi_latency_e2e._sub.text())
        txt = p.lbl_caliber.text()
        for frag in ("当前工单 #7", "共 4 条已保存检测结果",
                     "产线 3 条", "上传 1 条", "接收→落库率 不可用"): 
            self.assertIn(frag, txt)

    def test_batch_table_missing(self):
        _log(5, "批次表：e2e 后端补 0.0 → “—”，推理/e2e 分列，无复核比率 “—”")
        p = _page()
        p._fill_batch({"items": [
            {"name": "B1", "n_detections": 3, "n_anomaly": 1, "n_reviewed": 0,
             "fp_rate": 0.0, "fn_rate": 0.0, "avg_latency_ms": 12.0, "avg_latency_e2e_ms": 0.0},
            {"name": "B2", "n_detections": 2, "n_anomaly": 0, "n_reviewed": 2,
             "fp_rate": 0.5, "fn_rate": 0.0, "avg_latency_ms": 20.0, "avg_latency_e2e_ms": 35.0}]})
        self.assertEqual(_cells(p, 0)[4:], [D.NA, D.NA, "12 ms", D.NA])
        self.assertEqual(_cells(p, 1)[4:], ["50.0%", "0.0%", "20 ms", "35 ms"])

    def test_export_blank_not_zero(self):
        _log(6, "导出 CSV：分母为 0 的比率/缺失 e2e 留空，不写 0")
        self.assertEqual(S._csv_num(0.0, 0), "")
        self.assertEqual(S._csv_num(None, 5), "")
        self.assertEqual(S._csv_num(0.0, 5), 0.0)
        self.assertEqual(S._csv_num(12.345, 3, 1), 12.3)
        self.assertEqual(S._ms_pos(0.0), D.NA)
        self.assertEqual(S._pct_if(0.0, 0), D.NA)
        self.assertEqual(S._pct_if(0.0, 4), "0.0%")


class WorkorderTrendUiTest(unittest.TestCase):
    def test_dashboard_all_without_range_combo(self):
        from PySide6.QtWidgets import QComboBox
        p = D.DashboardPage(FakeClient())
        self.assertEqual(p._range_name(), "all")
        self.assertEqual(p.findChildren(QComboBox), [])
        p.close()

    def test_latency_labels_missing_and_stale_response(self):
        import math
        p = _page()
        p._fill_latency({"basis": "detection_order", "items": [
            {"label": "检测 1", "n_inspected": 1, "avg_latency_ms": 10},
            {"label": "检测 2", "n_inspected": 1, "avg_latency_ms": None}]})
        self.assertIn("按检测顺序", p.lbl_latency_head.text())
        values = p.latency_chart.plot.listDataItems()[0].yData
        self.assertEqual(values[0], 10)
        self.assertTrue(math.isnan(values[1]))
        calls = []
        selected = [1]
        p._selected_wo_id = lambda: selected[0]
        with patch.object(S, "run_async", lambda owner, job, cb: calls.append((job, cb))):
            p._load_stats(1, {})
            old_callback = calls[-1][1]
            selected[0] = 2
            p._load_stats(2, {})
        old_callback({"basis": "batch", "items": [
            {"label": "wrong", "n_inspected": 1, "avg_latency_ms": 900}]})
        self.assertEqual(p.latency_chart.plot.listDataItems(), [])
        p.close()


class BackendToStatsPageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_stats_caliber_")
        self.addCleanup(tmp.cleanup)
        db = create_engine("sqlite:///" + (Path(tmp.name) / "a.db").as_posix(),
                           connect_args={"check_same_thread": False})
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        for p in (patch.object(database, "_engine", db),
                  patch.object(database, "_SessionLocal", sessionmaker(bind=db, expire_on_commit=False))):
            p.start()
            self.addCleanup(p.stop)
        with database.session_scope() as s:
            for tt, lat, e2e in (("pipeline", 10.0, 30.0), ("pipeline", 20.0, None),
                                 ("upload", 30.0, 50.0), (None, 40.0, None)):
                s.add(Detection(target_type=tt, category="A", image_path="x.png",
                                final_score=0.1, is_anomaly=False,
                                latency_ms=lat, latency_e2e_ms=e2e))

    def test_real_overview_and_quality(self):
        _log(7, "真实 overview/quality_overview（临时 SQLite）→ 统计页卡片/口径行")
        p = _page()
        p._fill_caliber(rs.api_stats_overview(range="all"))
        p._fill_quality(rs.api_stats_quality_overview(range_name="all"))
        self.assertEqual(p.kpi_det._value.text(), "4")
        self.assertIn("检测", p.kpi_det._sub.text())
        self.assertEqual(p.kpi_latency._value.text(), "25 ms")
        self.assertEqual(p.kpi_latency_e2e._value.text(), "40 ms")
        self.assertIn("2 / 4", p.kpi_latency_e2e._sub.text())
        self.assertEqual(p.kpi_true_rate._value.text(), D.NA)  # 无复核
        txt = p.lbl_caliber.text()
        self.assertIn("当前未选工单（全局）", txt)
        self.assertIn("共 4 条已保存检测结果", txt)
        self.assertIn("产线 2", txt)
        self.assertIn("图片 1", txt)
        with database.session_scope() as s:
            s.query(Detection).delete()
        p._fill_caliber(rs.api_stats_overview(range="all"))  # 空范围：旧口径 avg=0.0
        self.assertEqual(p.kpi_latency._value.text(), D.NA)
        self.assertEqual(p.kpi_latency_e2e._value.text(), "不可用")
        self.assertIn("来源：暂无检测记录", p.lbl_caliber.text())


if __name__ == "__main__":
    unittest.main()
