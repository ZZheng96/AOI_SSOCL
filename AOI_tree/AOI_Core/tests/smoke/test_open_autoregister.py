"""E_open 自动注册（候选化）：new_defect 处置 → 候选 draft，归并计数，永不自动上线。
TEMP SQLite + TEMP 注册表，不触碰生产 DB/快照。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from algo.feature_revision import FeatureSetRevision
from backend.db import database
from backend.db.models import Base, Detection, Feedback
from backend.api.open_alerts import register, new_item


class OpenAutoRegisterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.engine = create_engine(f"sqlite:///{tmp / 'autoreg.db'}",
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.store = FeatureSetRevision(tmp / "rev" / "registry.json")
        self._patches = [patch.object(database, "_engine", self.engine),
                         patch.object(database, "_SessionLocal", factory),
                         patch("backend.api.open_alerts._revision_store", return_value=self.store)]
        for p in self._patches:
            p.start()
        with database.session_scope() as s:
            # 1-3: 有模型版本；4: 无模型版本（无法绑定快照血缘）
            for i, nt in enumerate(({"model_version": 7}, {"model_version": 7},
                                    {"model_version": 7}, {}), 1):
                s.add(Detection(image_path=f"d{i}.png", category="cat", final_score=.1,
                                is_anomaly=False, latency_ms=1,
                                n_tiles={"open_alert": True, **nt, "open_alert_item": new_item(3.0, 2.5)}))
        router = APIRouter()
        register(router)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        self.client = TestClient(app)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.engine.dispose()
        self._tmp.cleanup()

    def _dispose(self, did, event_id, revision=0, **kw):
        body = {"status": "confirmed_defect", "operator": "tester", "reason": "人工确认新缺陷",
                "event_id": event_id, "expected_revision": revision, "cause": "new_defect", **kw}
        r = self.client.post(f"/api/review/open_alerts/{did}/disposition", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_autoregister_merge_and_gates(self):
        print("[1/5] new_defect 处置 → 候选 draft", flush=True)
        j = self._dispose(1, "e1", defect_type="scratch")
        a1 = j["autoregister"]
        self.assertEqual((a1["action"], a1["count"], a1["key"]), ("created", 1, "cat/scratch"))
        rid = a1["revision_id"]
        rows = self.store.list()
        self.assertIn(rid, rows)
        self.assertEqual(j["item"]["admission"]["revision_id"], rid)
        self.assertFalse(j["item"]["admission"]["auto_online"])
        m = rows[rid]["manifest"]
        self.assertEqual((m["origin"], m["admission_required"]), ("open_autoregister", True))
        self.assertEqual(m["dataset_split"]["defect_type"], "scratch")
        with database.session_scope() as s:
            fb = s.query(Feedback).one()
            self.assertEqual(fb.learning_trace["admission"]["revision_id"], rid)
            self.assertFalse(fb.consumed)
        ledger = self.store.open_candidates()["cat/scratch"]
        self.assertEqual(ledger["samples"][0]["detection_id"], 1)
        self.assertEqual(ledger["samples"][0]["image_path"], "d1.png")

        print("[2/5] 同类型归并计数 + 幂等不重复计数", flush=True)
        idem = self._dispose(1, "e1", defect_type="scratch")
        self.assertTrue(idem["idempotent"])
        self.assertEqual(idem["autoregister"]["revision_id"], rid)
        a2 = self._dispose(2, "e2", defect_type="scratch")["autoregister"]
        self.assertEqual((a2["action"], a2["revision_id"], a2["count"]), ("merged", rid, 2))
        # 同一检测再次（新事件）处置为同类型：去重，不计数
        a3 = self._dispose(1, "e1b", revision=1, defect_type="scratch")["autoregister"]
        self.assertEqual((a3["action"], a3["count"]), ("duplicate", 2))
        other = self._dispose(3, "e3", defect_type="bubble")["autoregister"]
        self.assertEqual(other["action"], "created")
        self.assertNotEqual(other["revision_id"], rid)
        self.assertEqual(len(self.store.list()), 2)

        print("[3/5] 非 new_defect 不注册；缺模型版本 skipped", flush=True)
        r = self.client.post("/api/review/open_alerts/2/disposition", json={
            "status": "confirmed_normal", "operator": "tester", "reason": "漂移",
            "event_id": "n1", "expected_revision": 1, "cause": "drift"}).json()
        self.assertIsNone(r["autoregister"])
        self.assertIsNone(r["item"]["admission"])
        sk = self._dispose(4, "e4", defect_type="scratch")["autoregister"]
        self.assertEqual((sk["action"], sk["revision_id"]), ("skipped", None))
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(self.store.open_candidates()["cat/scratch"]["count"], 2)

        print("[4/5] 草案未激活 + admission 门禁仍需人工", flush=True)
        for row in self.store.list().values():
            self.assertEqual((row["state"], row["history"]), ("draft", []))
        self.assertIsNone(self.store.production_pointer())
        with self.assertRaises(ValueError):          # 无评价证据不得进入 evaluated
            self.store.transition(rid, "evaluated", actor="tester")
        with self.assertRaises(ValueError):          # 禁止越级
            self.store.transition(rid, "active", actor="tester")
        ev = Path(self._tmp.name) / "eval.json"
        ev.write_text(json.dumps({"revision_id": rid, "dataset_split": m["dataset_split"],
                                  "model_snapshot": m["model_snapshot"], "seed": m["seed"],
                                  "eval_run_id": "run1", "status": "completed", "independent": True,
                                  "acceptance": "passed", "auroc": .9}), encoding="utf-8")
        self.store.transition(rid, "evaluated", actor="tester", evidence_path=str(ev))
        with self.assertRaisesRegex(ValueError, "难例准入试算缺失"):
            self.store.transition(rid, "approved", actor="tester")
        self.assertEqual(self.store.list()[rid]["state"], "evaluated")
        self.assertIsNone(self.store.production_pointer())

        print("[5/5] GET 列出自动候选 + CDE 报告口径", flush=True)
        lst = self.client.get("/api/review/open_alerts/autoregistered").json()
        self.assertEqual(lst["total"], 2)
        by_key = {i["key"]: i for i in lst["items"]}
        self.assertEqual((by_key["cat/scratch"]["count"], by_key["cat/scratch"]["state"]), (2, "evaluated"))
        self.assertEqual(by_key["cat/bubble"]["state"], "draft")
        self.assertTrue(all(i["auto_online"] is False and i["admission_required"] for i in lst["items"]))
        self.assertEqual(self.client.get("/api/review/open_alerts/autoregistered?category=nope").json()["total"], 0)
        from algo.fusion.cde_report import MECHANISM_FACTS
        self.assertEqual(MECHANISM_FACTS["auto_feature_registration"]["status"], "candidate_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
