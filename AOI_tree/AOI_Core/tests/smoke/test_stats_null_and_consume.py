"""统计缺失值 None（AC-058 e2e 不回退）+ 即时学习 region consumption 回写（AC-063/064）
+ 复核对话框“学习消费”标签 + 已消费框删除/撤销 → 发布前 REGION_STALE。
TEMP SQLite + offscreen；engine 为替身，不证明真实在线学习效果。"""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision import localization as L
from backend.db import database
from backend.db.models import Base, Detection, Feedback, Image, WorkOrder

_app = QApplication.instance() or QApplication([])
N = 7


def _regions():
    return [{"region_id": f"r{i}", "bbox": [i * 10, 0, i * 10 + 8, 8], "area": 64,
             "source": "engine_box", "score_peak": 0.9 - i * 0.1, "postproc": "pp-x",
             "status": "candidate", "confirmed": False} for i in range(3)]


def _op(rv, who, op, rid=None, ev="e", **kw):
    return L.apply_op(rv, {"op": op, "region_id": rid, "operator": who, "event_id": ev,
                           "expected_revision": rv["revision"], **kw}, size=[100, 100])


def _reviewed():
    rv = L.new_review(_regions(), "pp-x")
    rv = _op(rv, "甲", "confirm", "r0", "e1")
    rv = _op(rv, "甲", "revise", "r1", "e2", bbox=[11, 1, 19, 9])
    return _op(rv, "乙", "add", ev="e3", bbox=[50, 50, 60, 60])


class _TempDb(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_null_consume_")
        self.addCleanup(tmp.cleanup)
        self.eng = create_engine("sqlite:///" + (Path(tmp.name) / "a.db").as_posix(),
                                 connect_args={"check_same_thread": False})
        self.addCleanup(self.eng.dispose)
        Base.metadata.create_all(self.eng)
        for p in (patch.object(database, "_engine", self.eng),
                  patch.object(database, "_SessionLocal",
                               sessionmaker(bind=self.eng, expire_on_commit=False))):
            p.start()
            self.addCleanup(p.stop)


class StatsNullTest(_TempDb):
    def test_empty_overview_health_ratios_none(self):
        from backend.api import routes_stats as rs
        print(f"[1/{N}] 空库 quality_overview / health / batch_summary 比率为 None", flush=True)
        q = rs.api_stats_quality_overview()
        for k in ("system_defect_rate", "avg_latency_ms", "coverage", "true_defect_rate",
                  "fp_rate", "recall_defect", "fn_rate", "review_accuracy"): 
            self.assertIsNone(q[k], k)
        h = rs.api_stats_health()
        self.assertEqual(h["scanned"], 0)
        for k in ("gray_rate", "open_rate", "align_rate"):
            self.assertIsNone(h[k], k)
        self.assertEqual(rs.api_stats_batch_summary()["items"], [])

    def test_timeseries_category_e2e_not_faked(self):
        from backend.api import routes_stats as rs
        print(f"[2/{N}] timeseries(category)：e2e 全缺 → None，推理耗时照常", flush=True)
        with database.session_scope() as s:
            for lat in (10.0, 30.0):
                s.add(Detection(category="A", image_path="x.png", final_score=0.1,
                                is_anomaly=False, latency_ms=lat, latency_e2e_ms=None))
        items = rs.api_stats_timeseries(days=3, category="A")["items"]
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]["avg_latency_ms"], 20.0)
        self.assertIsNone(items[0]["avg_latency_e2e_ms"])
        with database.session_scope() as s:
            s.add(Detection(category="A", image_path="y.png", final_score=0.1,
                            is_anomaly=False, latency_ms=5.0, latency_e2e_ms=70.0))
        it = rs.api_stats_timeseries(days=3, category="A")["items"][0]
        self.assertAlmostEqual(it["avg_latency_e2e_ms"], 70.0)   # AVG 忽略 NULL，不平均进推理耗时

    def test_workorder_batch_report_none(self):
        from backend.api import _wo_shared as ws
        from backend.api import routes_wo_stats as rw
        print(f"[3/{N}] 工单 bad_rate / _p95 / report：缺失为 None，e2e 不回退推理耗时", flush=True)
        self.assertIsNone(rw._p95([]))
        self.assertEqual(rw._p95([3, 1, 2]), 3.0)
        with database.session_scope() as s:
            self.assertIsNone(ws._batch_stats_by_ids(s, [])["bad_rate"])
            s.add(WorkOrder(id=7, name="wo-empty"))
        sm = rw._workorder_report(7)["summary"]
        self.assertEqual(sm["n_total"], 0)
        for k in ("defect_rate", "avg_latency_ms", "p95_latency_ms"):
            self.assertIsNone(sm[k], k)
        csv_text = rw.api_workorder_report_csv(7).body.decode("utf-8-sig")
        self.assertIn("平均延迟ms(e2e),,", csv_text)               # None → 空串，不写 0
        with database.session_scope() as s:
            for lat, e2e in ((10.0, None), (20.0, 40.0)):
                s.add(Detection(workorder_id=7, category="A", image_path="z.png", final_score=0.1,
                                is_anomaly=False, latency_ms=lat, latency_e2e_ms=e2e))
        sm = rw._workorder_report(7)["summary"]
        self.assertEqual((sm["n_total"], sm["n_e2e_missing"]), (2, 1))
        self.assertEqual((sm["avg_latency_ms"], sm["p95_latency_ms"]), (40.0, 40.0))
        with database.session_scope() as s:
            s.query(Detection).update({Detection.latency_e2e_ms: None})
        sm = rw._workorder_report(7)["summary"]
        self.assertIsNone(sm["avg_latency_ms"])
        self.assertIsNone(sm["p95_latency_ms"])
        self.assertEqual(sm["defect_rate"], 0.0)                   # 有分母时真实 0 保持 0


class InstantConsumptionTest(_TempDb):
    """legacy 即时学习（engine.submit_feedback）成功后同样回写 region_review.consumption。"""

    def setUp(self):
        super().setUp()
        with database.session_scope() as s:
            s.add(Image(path="x.png", category="test", split="feedback", label="anomaly"))
            s.add(Detection(id=1, image_path="x.png", category="test", final_score=0.9,
                            is_anomaly=True, latency_ms=1.0,
                            n_tiles={"region_review": _reviewed()}))
        self.update = {"action": "difficult_queued"}
        self.engine = SimpleNamespace(
            current_version=Mock(return_value=3),
            predict_image_path=Mock(return_value=SimpleNamespace(
                score=.9, decision="anomaly", slot_scores={})))
        from backend.self_learning import service
        self.service = service
        for p in (patch.object(service, "get_engine", return_value=self.engine),
                  patch.object(service, "notify_event"), patch.object(service, "log_action")):
            p.start()
            self.addCleanup(p.stop)

    def _submit(self, ev):
        return self.service.submit_feedback_record(
            1, "false_negative", 1, source_event_id=ev,
            candidate_only=False, allow_legacy_online=True)

    def _review(self):
        with database.session_scope() as s:
            return s.get(Detection, 1).n_tiles["region_review"]

    def test_instant_success_records_and_failure_does_not(self):
        print(f"[4/{N}] 即时学习：成功回写 consumption(kind=instant)；失败不回写", flush=True)
        before = self._review()

        def fail(*a, **kw):
            kw["on_result"](None, RuntimeError("learn failed"))
            return {"async": True}
        self.engine.submit_feedback = Mock(side_effect=fail)
        out = self._submit("bad")
        self.assertEqual(out["learning_status"], "failed")
        self.assertNotIn("consumption", self._review())
        self.assertNotIn("region_consumption", out["learning_trace"])

        def ok(*a, **kw):
            kw["on_result"](self.update, None)
            return self.update
        self.engine.submit_feedback = Mock(side_effect=ok)
        out = self._submit("good")
        fb_id = out["feedback_id"]
        self.assertEqual(out["learning_status"], "applied")
        self.assertEqual(out["learning_trace"]["region_consumption"]["status"], "recorded")
        self.assertEqual(out["learning_trace"]["region_consumption"]["count"], 3)
        rv = self._review()
        self.assertEqual(rv["revision"], before["revision"])        # 消费不是人工修订
        recs = {c["region_id"]: c for c in rv["consumption"]}
        self.assertEqual(set(recs), {"r0", "r1", "h0"})
        self.assertEqual((recs["r1"]["kind"], recs["r1"]["candidate_id"], recs["r1"]["feedback_id"],
                          recs["r1"]["region_revision"], recs["r1"]["version"]),
                         ("instant", f"instant:feedback:{fb_id}", fb_id, 2, 3))
        view = {r["region_id"]: r for r in L.annotate_review(rv)["regions"]}
        self.assertTrue(view["h0"]["used"] and not view["r2"]["used"])
        # 幂等重放不重复回写
        self.assertEqual(self._submit("good")["learning_trace"], out["learning_trace"])
        self.assertEqual(len(self._review()["consumption"]), 3)
        with database.session_scope() as s:
            self.assertTrue(s.get(Feedback, fb_id).consumed)


def _consume(rv):
    return L.record_consumption(rv, [{**s, "candidate_id": "candidate:feedback:1", "version": 2,
                                      "feedback_id": 1, "at": "t"}
                                     for s in L.consumable_regions(rv)])


class _Client:
    def __init__(self, rv):
        self.rv = rv

    def get(self, path, **kw):
        return {"detection_id": 1, "coord_space": [100, 100], **L.annotate_review(self.rv)}


class DialogConsumptionLabelTest(unittest.TestCase):
    def test_used_and_stale_labels(self):
        from ui.widgets import region_review_dialog as R
        print(f"[5/{N}] 复核对话框“学习消费”列：未用于学习 / 已用于学习 / 已用后变更", flush=True)
        self.assertEqual(R.COLUMNS[8], "学习消费")
        self.assertEqual(R.consumption_text({}), "未用于学习")
        self.assertEqual(R.consumption_text({"used": True}), "已用于学习")
        self.assertEqual(R.consumption_text({"used": True, "stale_after_use": True}), "已用后变更")
        used = _consume(_reviewed())
        c = _Client(_op(used, "甲", "revise", "r0", "e9", bbox=[0, 0, 6, 6]))
        d = R.RegionReviewDialog(c, 1)
        self.addCleanup(d.deleteLater)
        d.refresh()
        col = {d.table.item(i, 0).text(): d.table.item(i, 8).text()
               for i in range(d.table.rowCount())}
        self.assertEqual(col, {"r0": "已用后变更", "r1": "已用于学习",
                               "r2": "未用于学习", "h0": "已用于学习"})


class ConsumedDeleteUndoBlocksReleaseTest(_TempDb):
    """候选构建消费 r0/r1/h0 → 删除 / 撤销已消费框 → 发布前 REGION_STALE。"""

    def setUp(self):
        super().setUp()
        loc = {"regions": _regions(), "postproc": {"version_id": "pp-x"},
               "coord_space": {"size": [100, 100]}}
        with database.session_scope() as s:
            s.add(Detection(id=1, image_path="x.png", category="test", final_score=0.9,
                            is_anomaly=True, latency_ms=1.0,
                            n_tiles={"localization": loc, "region_review": _reviewed()}))
            s.add(Feedback(id=1, detection_id=1, feedback_type="false_negative",
                           operator_label=1, learning_trace={"candidate_only": True}))
        from backend.self_learning import service
        engine = Mock()
        engine.learn_candidate.return_value = {"event_id": "feedback:1", "version_num": 2,
                                               "parent_version": 1}
        with patch.object(service, "get_engine", return_value=engine):
            service.build_feedback_candidate(1, 1)
        with database.session_scope() as s:
            self.base = s.get(Detection, 1).n_tiles["region_review"]
        self.assertEqual(len(self.base["consumption"]), 3)

    def _write(self, rv):
        with database.session_scope() as s:
            det = s.get(Detection, 1)
            det.n_tiles = {**det.n_tiles, "region_review": rv}

    def _blocked(self, rv):
        from backend.self_learning.candidate_release import (CandidateRelease,
                                                              CandidateReleaseError)
        self._write(rv)
        with self.assertRaises(CandidateReleaseError) as caught:
            CandidateRelease(Mock())._validate_feedback_guard({"result": {"event_id": "feedback:1"}})
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_REGION_STALE")
        return {v["region_id"]: v["reason"]
                for v in caught.exception.violations[0]["details"]["regions"]}

    def test_delete_consumed_box_blocks(self):
        print(f"[6/{N}] 删除已消费人工框 h0 → REGION_STALE(deleted)", flush=True)
        self.assertEqual(self._blocked(_op(self.base, "乙", "delete", "h0", "d1", reason="误标")),
                         {"h0": "deleted"})

    def test_undo_consumed_box_blocks(self):
        print(f"[7/{N}] 撤销已消费框（undo add h0 → retracted；undo revise r1 → candidate）→ REGION_STALE",
              flush=True)
        rv = _op(self.base, "乙", "undo", ev="u1")
        self.assertEqual({r["region_id"]: r["status"] for r in rv["regions"]}["h0"], "retracted")
        self.assertEqual(self._blocked(rv), {"h0": "deleted"})
        rv = _op(self.base, "甲", "undo", ev="u2")                 # 甲最近一次为 revise r1
        self.assertEqual({r["region_id"]: r["status"] for r in rv["regions"]}["r1"], "candidate")
        self.assertEqual(self._blocked(rv), {"r1": "excluded"})
        self._write(self.base)                                       # 恢复后放行
        from backend.self_learning.candidate_release import CandidateRelease
        CandidateRelease(Mock())._validate_feedback_guard({"result": {"event_id": "feedback:1"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
