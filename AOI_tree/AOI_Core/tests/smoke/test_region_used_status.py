"""AC-063/064：region 级训练消费（used）回写 + 消费后修订/删除/作废 → 发布前 fail-closed。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision import localization as L
from backend.db import database
from backend.db.models import Base, Detection, Feedback


def _regions():
    return [{"region_id": f"r{i}", "bbox": [i * 10, 0, i * 10 + 8, 8], "area": 64,
             "source": "engine_box", "score_peak": 0.9 - i * 0.1, "postproc": "pp-x",
             "status": "candidate", "confirmed": False} for i in range(3)]


def _op(rv, who, op, rid=None, ev="e", **kw):
    return L.apply_op(rv, {"op": op, "region_id": rid, "operator": who, "event_id": ev,
                           "expected_revision": rv["revision"], **kw}, size=[100, 100])


def _by_id(rv):
    return {r["region_id"]: r for r in rv["regions"]}


def _reviewed():
    rv = L.new_review(_regions(), "pp-x")
    rv = _op(rv, "甲", "confirm", "r0", "e1")
    rv = _op(rv, "甲", "revise", "r1", "e2", bbox=[11, 1, 19, 9])
    return _op(rv, "乙", "add", ev="e3", bbox=[50, 50, 60, 60])


def _consume(rv, cid="candidate:feedback:1", version=2):
    return L.record_consumption(rv, [{**s, "candidate_id": cid, "version": version,
                                      "feedback_id": 1, "at": "t"}
                                     for s in L.consumable_regions(rv)])


class RegionUsedPureTest(unittest.TestCase):
    def test_snapshot_and_used(self):
        print("[1/5] 消费快照 + used/used_by 派生，不改 revision/events", flush=True)
        rv = _reviewed()
        snap = L.consumable_regions(rv)
        self.assertEqual([(s["region_id"], s["region_revision"]) for s in snap],
                         [("r0", 1), ("r1", 2), ("h0", 3)])
        self.assertEqual([s["bbox"] for s in snap], L.trainable_region_boxes(rv))
        view = _by_id(L.annotate_review(rv))
        self.assertTrue(all(r["used"] is False and r["used_by"] == []
                            and r["stale_after_use"] is False for r in view.values()))
        used = _consume(rv)
        self.assertEqual((used["revision"], len(used["events"])), (rv["revision"], len(rv["events"])))
        again = _consume(used)                       # 同候选重复回写幂等
        self.assertEqual(len(again["consumption"]), 3)
        view = L.annotate_review(used)
        by = _by_id(view)
        self.assertTrue(by["r0"]["used"] and by["h0"]["used"])
        self.assertEqual(by["r1"]["used_by"][0]["candidate_id"], "candidate:feedback:1")
        self.assertEqual(by["r1"]["used_by"][0]["version"], 2)
        self.assertFalse(by["r2"]["used"])           # 拒绝/候选框不被消费
        self.assertEqual(view["stale_consumption"], [])
        self.assertNotIn("used", _by_id(used)["r0"])  # 派生字段不落存储

    def test_stale_after_use(self):
        print("[2/5] 消费后修订/作废/删除 → stale_after_use + reason", flush=True)
        used = _consume(_reviewed())
        snap = L.consumable_regions(used)
        rv = _op(used, "甲", "revise", "r0", "e4", bbox=[0, 0, 7, 7])
        rv = _op(rv, "丙", "dispute", "r1", "e5")
        rv = _op(rv, "乙", "delete", "h0", "e6", reason="误标")
        view = L.annotate_review(rv)
        by = _by_id(view)
        self.assertEqual({k: (by[k]["stale_after_use"], by[k]["stale_reason"]) for k in ("r0", "r1", "h0")},
                         {"r0": (True, "revised"), "r1": (True, "excluded"), "h0": (True, "deleted")})
        self.assertTrue(by["r0"]["used"])             # 仍可追溯“曾被消费”
        self.assertFalse(by["r2"]["stale_after_use"])
        self.assertEqual({s["region_id"] for s in view["stale_consumption"]}, {"r0", "r1", "h0"})
        self.assertEqual({s["reason"] for s in L.stale_regions(rv, snap)}, {"revised", "excluded", "deleted"})
        # 区域从 review 中消失也视为删除；未消费框的后续修订不算 stale
        gone = {**used, "regions": [r for r in used["regions"] if r["region_id"] != "h0"]}
        self.assertEqual(L.stale_regions(gone, snap), [{"region_id": "h0", "reason": "deleted"}])
        rv2 = _op(used, "甲", "confirm", "r2", "e7")
        self.assertEqual(L.annotate_review(rv2)["stale_consumption"], [])
        # 新候选在新修订上重新消费 → 以最近消费为准不再 stale
        re_used = _consume(rv, cid="candidate:feedback:2", version=3)
        self.assertFalse(_by_id(L.annotate_review(re_used))["r0"]["stale_after_use"])
        self.assertEqual(len(_by_id(L.annotate_review(re_used))["r0"]["used_by"]), 2)


class RegionUsedDbTest(unittest.TestCase):
    """候选构建回写消费记录 → API/to_dict 透出 → 消费后修订/删除/作废发布前 fail-closed。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = create_engine(f"sqlite:///{Path(self.tmp.name) / 'u.db'}",
                                 connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.eng)
        self.SL = sessionmaker(bind=self.eng, expire_on_commit=False)
        loc = {"regions": _regions(), "postproc": {"version_id": "pp-x"},
               "coord_space": {"size": [100, 100]}}
        with self.SL() as s:
            s.add(Detection(id=1, image_path="x.png", category="test", final_score=0.9,
                            is_anomaly=True, latency_ms=1.0,
                            n_tiles={"localization": loc, "region_review": _reviewed()}))
            s.add(Feedback(id=1, detection_id=1, feedback_type="false_negative",
                           operator_label=1, learning_trace={"candidate_only": True}))
            s.commit()
        self.patches = [patch.object(database, "_engine", self.eng),
                        patch.object(database, "_SessionLocal", self.SL)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.eng.dispose()
        self.tmp.cleanup()

    def _build(self):
        from backend.self_learning import service
        engine = Mock()
        engine.learn_candidate.return_value = {"event_id": "feedback:1", "version_num": 2,
                                               "parent_version": 1}
        with patch.object(service, "get_engine", return_value=engine):
            out = service.build_feedback_candidate(1, 1)
        self.assertEqual(engine.learn_candidate.call_args.kwargs["event_id"], "feedback:1")
        return out

    def _review(self):
        with database.session_scope() as s:
            return s.get(Detection, 1).n_tiles["region_review"]

    def _write_review(self, rv):
        with database.session_scope() as s:
            det = s.get(Detection, 1)
            det.n_tiles = {**det.n_tiles, "region_review": rv}

    def _guard(self):
        from backend.self_learning.candidate_release import CandidateRelease
        CandidateRelease(Mock())._validate_feedback_guard({"result": {"event_id": "feedback:1"}})

    def test_build_records_consumption_and_api_exposes(self):
        from backend.api.region_review import register
        from backend.api.schemas import to_dict
        from backend.core.security import ApiKeyMiddleware
        print("[3/5] 候选构建 → guard.regions + review.consumption 回写 + API/to_dict 透出", flush=True)
        before = self._review()
        self._build()
        with database.session_scope() as s:
            trace = s.get(Feedback, 1).learning_trace
        guard = trace["candidate_guard"]
        self.assertEqual([g["region_id"] for g in guard["regions"]], ["r0", "r1", "h0"])
        self.assertEqual(trace["region_consumption"]["status"], "recorded")
        rv = self._review()
        self.assertEqual(rv["revision"], before["revision"])       # 消费不是人工修订
        recs = {c["region_id"]: c for c in rv["consumption"]}
        self.assertEqual(set(recs), {"r0", "r1", "h0"})
        self.assertEqual((recs["r1"]["region_revision"], recs["r1"]["candidate_id"],
                          recs["r1"]["version"], recs["r1"]["parent_version"], recs["r1"]["feedback_id"]),
                         (2, "candidate:feedback:1", 2, 1, 1))
        self.assertTrue(recs["r1"]["at"])
        self._guard()                                              # 未修订：放行
        # to_dict 透出
        with database.session_scope() as s:
            d = to_dict(s.get(Detection, 1))
        by = _by_id(d["region_review"])
        self.assertTrue(by["r0"]["used"] and not by["r2"]["used"])
        self.assertFalse(by["r0"]["stale_after_use"])
        # API：GET 透出 used；POST 修订后立即显示 stale_after_use
        sec = Mock()
        sec.section.return_value = {"keys": {"t-eng": "engineer"}}
        with patch("backend.core.security.get_settings", return_value=sec), \
                patch("backend.core.security.get_configured_api_key", return_value=""):
            app = FastAPI()
            app.add_middleware(ApiKeyMiddleware)
            router = APIRouter()
            register(router)
            app.include_router(router, prefix="/api")
            url, h = "/api/detections/1/regions", {"X-API-Key": "t-eng"}
            with TestClient(app) as c:
                got = _by_id(c.get(url, headers=h).json())
                self.assertEqual((got["h0"]["used"], got["h0"]["used_by"][0]["candidate_id"],
                                  got["h0"]["stale_after_use"]), (True, "candidate:feedback:1", False))
                body = {"op": "revise", "region_id": "r0", "bbox": [0, 0, 6, 6], "operator": "甲",
                        "event_id": "ev-rev", "expected_revision": rv["revision"]}
                resp = c.post(url, headers=h, json=body)
                self.assertEqual(resp.status_code, 200, resp.text)
                r0 = _by_id(resp.json()["review"])["r0"]
                self.assertEqual((r0["used"], r0["stale_after_use"], r0["stale_reason"]),
                                 (True, True, "revised"))
                self.assertEqual(self._review()["consumption"], rv["consumption"])   # 复核保留消费记录

    def _assert_blocked(self, reason):
        from backend.self_learning.candidate_release import CandidateReleaseError
        with self.assertRaisesRegex(CandidateReleaseError, "人工修订") as caught:
            self._guard()
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_REGION_STALE")
        details = caught.exception.violations[0]["details"]["regions"]
        self.assertIn(reason, {v["reason"] for v in details})
        return details

    def test_stale_blocks_release(self):
        print("[4/5] 消费后修订/作废/删除 → 发布前 fail-closed（REGION_STALE）", flush=True)
        self._build()
        base = self._review()
        cases = [("revised", lambda rv: _op(rv, "甲", "revise", "r1", "x1", bbox=[12, 1, 19, 9])),
                 ("excluded", lambda rv: _op(rv, "丙", "dispute", "r0", "x2")),
                 ("deleted", lambda rv: _op(rv, "乙", "delete", "h0", "x3", reason="误标"))]
        for reason, mutate in cases:
            self._write_review(mutate(base))
            self.assertEqual(self._assert_blocked(reason)[0]["reason"], reason)
        # 未被消费框（r2）的变化不报 REGION_STALE，但整体 revision 校验仍 fail-closed
        from backend.self_learning.candidate_release import CandidateReleaseError
        self._write_review(_op(base, "甲", "confirm", "r2", "x4"))
        with self.assertRaises(CandidateReleaseError) as caught:
            self._guard()
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_REVIEW_REVISION")
        self._write_review(base)
        self._guard()

    def test_legacy_guard_without_regions(self):
        print("[5/5] 旧 guard（无 regions）行为不变；无 eligible 框不回写", flush=True)
        from backend.self_learning.candidate_release import CandidateReleaseError
        self._write_review(L.new_review(_regions(), "pp-x"))     # 全部候选，无 eligible
        self._build()
        with database.session_scope() as s:
            trace = s.get(Feedback, 1).learning_trace
        self.assertEqual(trace["candidate_guard"]["regions"], [])
        self.assertEqual(trace["region_consumption"], {"status": "none", "count": 0})
        self.assertNotIn("consumption", self._review())
        with database.session_scope() as s:
            fb = s.get(Feedback, 1)
            g = dict(fb.learning_trace["candidate_guard"])
            g.pop("regions")
            fb.learning_trace = {**fb.learning_trace, "candidate_guard": g}
        self._guard()
        self._write_review({**self._review(), "revision": 9})
        with self.assertRaises(CandidateReleaseError) as caught:
            self._guard()
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_REVIEW_REVISION")


if __name__ == "__main__":
    unittest.main(verbosity=2)
