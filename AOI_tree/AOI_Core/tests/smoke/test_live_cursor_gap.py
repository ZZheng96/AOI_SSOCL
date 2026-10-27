"""AC-058：/detections/live 游标——过滤造成的 ID 洞不算缺失，真实删除计 gap，库重置要求重订阅。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_detect as rd
from backend.db import database
from backend.db.models import Base, Detection


class LiveCursorGapTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_live_gap_")
        self.addCleanup(tmp.cleanup)
        db = create_engine("sqlite:///" + (Path(tmp.name) / "a.db").as_posix(),
                           connect_args={"check_same_thread": False})
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        for p in (patch.object(database, "_engine", db),
                  patch.object(database, "_SessionLocal", sessionmaker(bind=db, expire_on_commit=False))):
            p.start()
            self.addCleanup(p.stop)
        # id1 pipeline/A, id2 image(非产线), id3 pipeline/B, id4 pipeline/A
        with database.session_scope() as s:
            for tt, cat in (("pipeline", "A"), ("image", "A"), ("pipeline", "B"), ("pipeline", "A")):
                s.add(Detection(target_type=tt, category=cat, image_path="x.png",
                                final_score=0.1, is_anomaly=False, latency_ms=1.0))

    def live(self, after_id, **kw):
        with patch("backend.core.alarm.get_alarm_monitor") as m:
            m.return_value.status.return_value = {}
            return rd.api_detections_live(after_id=after_id, **kw)

    def test_filtered_hole_is_not_gap(self):
        r = self.live(1, category="A")  # id2(非产线)、id3(品类B)被过滤
        self.assertEqual([it["id"] for it in r["items"]], [4])
        self.assertFalse(r["gap"])
        self.assertEqual((r["dropped_count"], r["filtered_out"]), (0, 2))
        self.assertFalse(r["reset_required"])

    def test_deleted_rows_count_as_gap(self):
        with database.session_scope() as s:
            s.query(Detection).filter(Detection.id.in_([2, 3])).delete(synchronize_session=False)
        r = self.live(1)
        self.assertTrue(r["gap"])
        self.assertEqual((r["dropped_count"], r["filtered_out"]), (2, 0))
        self.assertFalse(r["reset_required"])
        self.assertEqual(r["next_after_id"], 4)

    def test_cursor_beyond_max_requires_reset(self):
        r = self.live(99)
        self.assertEqual(r["items"], [])
        self.assertTrue(r["reset_required"])
        self.assertEqual(r["next_after_id"], 0)
        r = self.live(4)  # 已追平：无新数据但不重置
        self.assertFalse(r["reset_required"])
        self.assertEqual(r["next_after_id"], 4)


if __name__ == "__main__":
    unittest.main()
