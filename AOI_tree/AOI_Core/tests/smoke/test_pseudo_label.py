"""teacher 伪标签：只出高置信 / 人工覆盖伪标签 / 伪标签不激活模型（只进候选）。

TEMP sqlite + mock engine，不触碰生产 DB 与快照。
"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Detection, Feedback
from backend.self_learning import service as svc
from backend.self_learning.label_audit import (LabelRecord, generate_pseudo_labels,
                                               merge_pseudo_with_human)

T0 = datetime(2026, 1, 1, 8, 0, 0)


class TestGeneratePseudo(unittest.TestCase):
    def test_only_high_confidence(self):
        print("[+] 伪标签只出高置信：远低→0，远高→1，中间不标", flush=True)
        samples = [{"sample_id": "a", "score": 0.1, "threshold": 1.0},
                   {"sample_id": "b", "score": 0.9, "threshold": 1.0},   # 不确定区
                   {"sample_id": "c", "score": 1.2, "threshold": 1.0},   # 不确定区
                   {"sample_id": "d", "score": 2.0, "threshold": 1.0},
                   {"sample_id": "e", "score": 0.1, "threshold": None},  # 无阈值跳过
                   {"sample_id": "f", "score": 0.5, "threshold": 1.0}]   # 边界=ok
        recs = generate_pseudo_labels(samples, 3, now=T0)
        self.assertEqual({r.sample_id: r.label for r in recs}, {"a": 0, "d": 1, "f": 0})
        self.assertTrue(all(r.kind == "pseudo" and r.teacher_version == "3" for r in recs))
        tighter = generate_pseudo_labels(samples, 3, ok_ratio=0.2, ng_ratio=3.0, now=T0)
        self.assertEqual([r.sample_id for r in tighter], ["a"])
        with self.assertRaises(ValueError):
            generate_pseudo_labels(samples, 3, ok_ratio=1.2)
        with self.assertRaises(ValueError):
            generate_pseudo_labels(samples, None)

    def test_human_overrides_pseudo(self):
        print("[+] 人工标签覆盖伪标签；teacher 版本不符的伪标签作废", flush=True)
        pseudo = generate_pseudo_labels(
            [{"sample_id": "x", "score": 0.1, "threshold": 1.0},
             {"sample_id": "y", "score": 3.0, "threshold": 1.0}], 2, now=T0)
        human = [LabelRecord(sample_id="x", label=1, kind="human", revision=1)]
        res = merge_pseudo_with_human(pseudo, human, T0, 2)
        self.assertEqual((res["x"]["source"], res["x"]["label"]), ("human", 1))
        self.assertEqual(res["x"]["replaced_pseudo"], 1)
        self.assertEqual((res["y"]["source"], res["y"]["label"]), ("pseudo", 1))
        stale = merge_pseudo_with_human(pseudo, [], T0, 5)
        self.assertEqual(stale["y"]["status"], "stale")


class TestPseudoCandidates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = create_engine(f"sqlite:///{Path(self.tmp.name) / 'pl.db'}",
                                 connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.eng)
        sl = sessionmaker(bind=self.eng, expire_on_commit=False)
        self.engine = MagicMock()
        self.engine.current_version.return_value = 2
        self.engine.consolidate.return_value = 3
        self.patches = [patch.object(database, "_engine", self.eng),
                        patch.object(database, "_SessionLocal", sl),
                        patch.object(svc, "get_engine", return_value=self.engine),
                        patch.object(svc, "notify_event")]
        for p in self.patches:
            p.start()
        self.ids = {}
        with database.session_scope() as s:
            # (名称, 分数, 推理版本)；阈值统一 1.0
            for name, score, mv in [("ok_far", 0.1, 2), ("mid", 0.95, 2), ("ng_far", 2.5, 2),
                                    ("old_teacher", 0.05, 1), ("human_fix", 0.1, 2)]:
                d = Detection(image_path=f"{name}.png", category="cat", final_score=score,
                              is_anomaly=score >= 1.0, latency_ms=1.0,
                              n_tiles={"model_version": mv, "threshold": 1.0})
                s.add(d)
                s.flush()
                self.ids[name] = d.id
            s.add(Feedback(detection_id=self.ids["human_fix"], feedback_type="false_negative",
                           operator_label=1, created_at=T0))

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.eng.dispose()
        self.tmp.cleanup()

    def _snapshot_db(self):
        with database.session_scope() as s:
            return (sorted((d.id, d.final_score, d.is_anomaly) for d in s.query(Detection)),
                    sorted((f.id, f.consumed, f.feedback_type) for f in s.query(Feedback)))

    def test_candidates_merge_and_no_activation(self):
        print("[+] 伪标签候选：teacher=激活版本、人工优先、不写库不激活", flush=True)
        before = self._snapshot_db()
        out = svc.pseudo_label_candidates("cat", now=T0, cfg={})
        self.assertTrue(out["candidate_only"])
        self.assertEqual(out["teacher_version"], 2)
        got = {x["detection_id"]: (x["label"], x["source"]) for x in out["items"]}
        self.assertEqual(got[self.ids["ok_far"]], (0, "pseudo"))
        self.assertEqual(got[self.ids["ng_far"]], (1, "pseudo"))
        self.assertEqual(got[self.ids["human_fix"]], (1, "human"))   # 人工覆盖伪 OK
        self.assertNotIn(self.ids["mid"], got)                        # 不确定区
        self.assertNotIn(self.ids["old_teacher"], got)                # 非当前 teacher
        self.assertEqual(out["counts"]["pseudo_used"], 2)
        # 不激活、不学习、不写库
        self.engine.activate.assert_not_called()
        self.engine.learn_candidate.assert_not_called()
        self.engine.submit_feedback.assert_not_called()
        self.engine.consolidate.assert_not_called()
        self.assertEqual(self._snapshot_db(), before)

    def test_disabled(self):
        out = svc.pseudo_label_candidates("cat", cfg={"enabled": False})
        self.assertFalse(out["enabled"])
        self.assertEqual(out["items"], [])

    def test_consolidate_candidate_carries_pseudo_only(self):
        print("[+] consolidate(activate=False) 附带伪标签候选；activate=True 不生成", flush=True)
        out = svc.consolidate("cat", note="t", activate=False)
        self.engine.consolidate.assert_called_once()
        self.assertFalse(self.engine.consolidate.call_args.kwargs["activate"])
        self.engine.activate.assert_not_called()
        pc = out["pseudo_candidates"]
        self.assertTrue(pc["candidate_only"])
        self.assertEqual(pc["teacher_version"], 2)
        self.assertGreaterEqual(pc["counts"]["pseudo_used"], 2)
        out2 = svc.consolidate("cat", note="t2", activate=True)
        self.assertIsNone(out2["pseudo_candidates"])


if __name__ == "__main__":
    unittest.main()
