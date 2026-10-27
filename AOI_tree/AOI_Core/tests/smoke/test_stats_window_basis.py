"""AC-058：/stats/overview 口径——e2e 缺失不以推理耗时冒充，返回窗口/分母/来源分布。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_stats as rs
from backend.db import database
from backend.db.models import Base, Detection


class StatsWindowBasisTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_stats_basis_")
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

    def test_e2e_not_faked_and_basis_fields(self):
        st = rs.api_stats_overview(range="all")["scope_stats"]
        self.assertEqual(st["n_inspected"], 4)
        self.assertAlmostEqual(st["avg_latency_ms"], 25.0)
        self.assertEqual(st["latency_basis"], "inference_only")
        self.assertAlmostEqual(st["avg_latency_e2e_ms"], 40.0)  # 仅 30/50，缺失不回退
        self.assertEqual(st["p95_latency_e2e_ms"], 50.0)
        self.assertEqual(st["n_e2e_missing"], 2)
        self.assertIsNone(st["window_start"])
        self.assertEqual(st["denominator"], "persisted_detections")
        self.assertEqual(st["by_source"], {"pipeline": 2, "upload": 1, "image": 1})  # 模型默认 image
        self.assertIsNone(st["receipt_rate"])
        self.assertEqual(st["receipt_rate_availability"], "unavailable")

    def test_window_start_and_all_missing_e2e(self):
        with database.session_scope() as s:
            s.query(Detection).update({Detection.latency_e2e_ms: None})
        st = rs.api_stats_overview(range="7d")["scope_stats"]
        self.assertIsNotNone(st["window_start"])
        self.assertIsNone(st["avg_latency_e2e_ms"])
        self.assertEqual(st["n_e2e_missing"], 4)


class WorkorderIsolationTest(unittest.TestCase):
    def setUp(self):
        StatsWindowBasisTest.setUp(self)
        from datetime import datetime
        from backend.db.models import Dataset, Feedback, Image, WorkOrder
        with database.session_scope() as s:
            s.query(Detection).delete()
            s.add_all([WorkOrder(id=1, name="one"),
                       WorkOrder(id=2, name="two", archived=True)])
            s.add_all([Dataset(id=1, name="first"), Dataset(id=2, name="second")])
            s.flush()
            s.add_all([Image(id=1, path="1.png", dataset_id=1),
                       Image(id=2, path="2.png", dataset_id=2)])
            s.flush()
            for did, wid, iid, lat in ((1, 1, 2, 10), (2, 2, 2, 900),
                                       (3, 1, 1, 30), (4, None, 1, 800)):
                s.add(Detection(id=did, workorder_id=wid, image_id=iid,
                                category="shared", image_path="x.png", final_score=0.1,
                                is_anomaly=True, latency_ms=lat,
                                created_at=datetime(2020, 1, 1),
                                target_type="pipeline", n_tiles={"decision": "gray"}))
            s.flush()
            for did in (1, 2, 3, 4):
                s.add(Feedback(detection_id=did, feedback_type="confirmed",
                               operator_label=1, defect_type=str(did),
                               created_at=datetime(2020, 1, 1)))

    def test_workorder_all_interfaces_isolated(self):
        print("[isolation] shared category, archived workorder and orphan records", flush=True)
        scope = rs.api_stats_overview(workorder_id=1)["scope_stats"]
        self.assertEqual(scope["n_inspected"], 2)
        self.assertEqual(scope["avg_latency_ms"], 20)
        self.assertEqual(rs.api_stats_quality_overview(workorder_id=1)["n_reviewed"], 2)
        self.assertEqual(rs.api_stats_defect_types(workorder_id=1)["total"], 2)
        trend_items = rs.api_stats_mistake_trend(workorder_id=1, range_name="all")
        self.assertEqual(trend_items["basis"], "batch")
        self.assertEqual([item["batch_id"] for item in trend_items["items"]], [1, 2])
        self.assertEqual(sum(item["reviewed"] for item in trend_items["items"]), 2)
        self.assertEqual(rs.api_stats_health(workorder_id=1, range_name="all", limit=1)
                         ["scanned"], 2)
        batches = rs.api_stats_batch_summary(workorder_id=1)["items"]
        self.assertEqual(sum(b["n_detections"] for b in batches), 2)
        trend = rs.api_stats_latency_trend(1)
        self.assertEqual(trend["basis"], "batch")
        self.assertEqual([i["dataset_id"] for i in trend["items"]], [2, 1])
        self.assertEqual([i["avg_latency_ms"] for i in trend["items"]], [10, 30])
        self.assertEqual(rs.api_stats_latency_trend(2)["items"][0]["avg_latency_ms"], 900)
        self.assertEqual(rs.api_stats_latency_trend(999)["items"], [])

    def test_mistake_trend_uses_actual_workorder_groups(self):
        from backend.api import _wo_shared
        from backend.db.models import Image
        with database.session_scope() as s:
            s.query(Image).update({Image.dataset_id: 1})
        groups = [
            {"batch_key": "6:shared:2", "source": "source", "batch_index": 2,
             "image_ids": [1]},
            {"batch_key": "6:shared:10", "source": "source", "batch_index": 10,
             "image_ids": [2]},
        ]
        with patch.object(_wo_shared, "_workorder_detect_batches", return_value=groups):
            for scope in ({"workorder_id": 1}, {"category": "shared"}):
                items = rs.api_stats_mistake_trend(**scope)["items"]
                actual = [i for i in items if isinstance(i["batch_id"], str)]
                self.assertEqual([i["batch_id"] for i in actual],
                                 ["wo1:6:shared:2", "wo1:6:shared:10"])
                self.assertEqual([i["reviewed"] for i in actual], [1, 1])
                self.assertTrue(all(i["fp_rate"] is None for i in actual))

    def test_unbatched_detection_order(self):
        with database.session_scope() as s:
            s.add(Detection(workorder_id=1, category="shared", image_path="stream",
                            final_score=0, is_anomaly=False, latency_ms=50))
        trend = rs.api_stats_latency_trend(1)
        self.assertEqual(trend["basis"], "detection_order")
        self.assertEqual([i["avg_latency_ms"] for i in trend["items"]], [10, 30, 50])

if __name__ == "__main__":
    unittest.main()
