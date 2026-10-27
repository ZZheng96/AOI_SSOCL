"""AC-058 UI 统计口径（离屏）：工作台展示推理/e2e 耗时、接收→落库率、窗口/分母/工单/来源，
缺数据显示“—/不可用”不补 0；口径定义在 tooltip 可见。"""
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

_app = QApplication.instance() or QApplication([])
N = 5
_T0 = time.perf_counter()


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)


def _page():
    return D.DashboardPage(FakeClient())


class CaliberTextTest(unittest.TestCase):
    def test_full_fields(self):
        _log(1, "完整统计字段 → 面向用户的文案（耗时/缺失数/工单/来源）")
        t = D.scope_caliber_texts({
            "n_inspected": 4, "avg_latency_ms": 25.0, "latency_basis": "inference_only",
            "avg_latency_e2e_ms": 40.0, "p95_latency_e2e_ms": 50.0, "n_e2e_missing": 2,
            "range": "7d", "window_start": "2026-10-06T08:00:00.123",
            "denominator": "persisted_detections", "workorder_id": None,
            "excludes_archived_workorders": True,
            "by_source": {"pipeline": 2, "upload": 1, "image": 1},
            "receipt_rate": None, "receipt_rate_availability": "unavailable"})
        self.assertEqual(t["infer"], "25 ms")
        self.assertIn("模型判断耗时", t["infer_sub"])
        self.assertEqual(t["e2e"], "40 ms")
        self.assertIn("2 / 4", t["e2e_sub"])
        self.assertEqual(t["receipt"], "不可用")
        self.assertIn("当前未选工单（全局）", t["basis"])
        self.assertIn("共 4 条已保存检测结果", t["basis"])
        self.assertIn("产线 2 条", t["basis"])
        self.assertIn("上传 1 条", t["basis"])

    def test_missing_not_zero(self):
        _log(2, "缺数据/旧后端：显示 —/不可用，不出现 0 ms 或 0.0%")
        for scope in ({}, None, {"n_inspected": 0, "avg_latency_ms": 0.0, "by_source": {}},
                      {"n_inspected": 3, "avg_latency_ms": 12.0}):
            t = D.scope_caliber_texts(scope)
            self.assertEqual(t["e2e"], "不可用")
            self.assertEqual(t["receipt"], "不可用")
            for v in t.values():
                self.assertNotRegex(v, r"(?<!\d)0 ms")
                self.assertNotIn("0.0%", v)
        self.assertEqual(D.scope_caliber_texts({})["infer"], D.NA)
        self.assertEqual(D.scope_caliber_texts({"n_inspected": 0, "avg_latency_ms": 0.0})["infer"], D.NA)
        self.assertIn("口径未声明", D.scope_caliber_texts({"n_inspected": 3, "avg_latency_ms": 12.0})["infer_sub"])
        self.assertIn("工单 #7", D.scope_caliber_texts({"workorder_id": 7})["basis"])
        # 后端若将来提供真实回执率，按真实值展示
        self.assertEqual(D.scope_caliber_texts({"receipt_rate": 0.985})["receipt"], "98.5%")


class DashboardWidgetTest(unittest.TestCase):
    def test_widgets_and_tooltip(self):
        _log(3, "工作台离屏实例化：口径卡片初始“—”、tooltip 含口径定义")
        p = _page()
        self.assertEqual(p.kpi_latency_e2e._value.text(), D.NA)
        self.assertEqual(p.kpi_receipt._value.text(), D.NA)
        for w in (p.kpi_latency_infer, p.kpi_latency_e2e, p.kpi_receipt, p.lbl_caliber):
            self.assertIn("当前工单的全部检测记录", w.toolTip())
            self.assertIn("不是相机拍到的帧数", w.toolTip())
            self.assertIn("显示不可用", w.toolTip())

    def test_fill_kpi_legacy_payload(self):
        _log(4, "旧后端 payload（无新口径字段）：_fill_kpi 不崩、不冒充 0")
        p = _page()
        p._fill_kpi({"scope_stats": {"n_inspected": 0, "n_anomaly": 0, "rate": 0.0,
                                     "avg_latency_ms": 0.0}})
        self.assertEqual(p.kpi_latency_infer._value.text(), D.NA)
        self.assertEqual(p.kpi_latency_e2e._value.text(), "不可用")
        self.assertEqual(p.kpi_receipt._value.text(), "不可用")
        self.assertIn("共 0 条已保存检测结果", p.lbl_caliber.text())


class BackendToUiTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_ui_caliber_")
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
            for tt, lat, e2e, an in (("pipeline", 10.0, 30.0, True), ("pipeline", 20.0, None, False),
                                     ("upload", 30.0, 50.0, False), (None, 40.0, None, False)):
                s.add(Detection(target_type=tt, category="A", image_path="x.png", final_score=0.1,
                                is_anomaly=an, latency_ms=lat, latency_e2e_ms=e2e))

    def test_real_overview_to_dashboard(self):
        _log(5, "真实 api_stats_overview（临时 SQLite）→ 工作台卡片/口径行")
        p = _page()
        p._fill_kpi(rs.api_stats_overview(range="all"))
        self.assertEqual(p.kpi_inspected._value.text(), "4")
        self.assertEqual(p.kpi_latency_infer._value.text(), "25 ms")
        self.assertIn("模型判断耗时", p.kpi_latency_infer._sub.text())
        self.assertEqual(p.kpi_latency_e2e._value.text(), "40 ms")
        self.assertIn("2 / 4", p.kpi_latency_e2e._sub.text())
        self.assertEqual(p.kpi_receipt._value.text(), "不可用")
        txt = p.lbl_caliber.text()
        self.assertIn("当前未选工单（全局）", txt)
        self.assertIn("共 4 条已保存检测结果", txt)
        self.assertNotIn("P95", txt)
        self.assertNotIn("persisted_detections", txt)
        self.assertIn("产线 2", txt)
        self.assertIn("图片 1", txt)
        with database.session_scope() as s:
            s.query(Detection).update({Detection.latency_e2e_ms: None})
        p._fill_kpi(rs.api_stats_overview(range="7d"))
        self.assertEqual(p.kpi_latency_e2e._value.text(), "不可用")
        self.assertIn("4 / 4", p.kpi_latency_e2e._sub.text())
        self.assertIn("近 7 天（起点 ", p.lbl_caliber.text())


if __name__ == "__main__":
    unittest.main()
