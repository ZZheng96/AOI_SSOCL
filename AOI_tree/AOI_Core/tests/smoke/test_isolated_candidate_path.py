"""TEMP候选底座：真实持久化/SQLite/WAL；反馈动作替身，非算法验收。"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from algo.decide import Decider
from algo.eval.offline import Pipeline
from algo.persist import load_snapshot, save_snapshot, set_current_version
from algo.slots.trad import TradSlot
from backend.engine import DetectionEngine


class IsolatedCandidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_ssa_slice_")
        self.addCleanup(self.tmp.cleanup)
        self.engine = DetectionEngine(self.tmp.name, "configs/engine_fast.yaml")
        self.engine.device = "cpu"
        import copy
        self.engine.ssocl_cfg = copy.deepcopy(self.engine.ssocl_cfg)
        self.root = Path(self.engine._cat_dir("test"))
        self.root.mkdir()
        self.image = str(self.root / "asset.png")
        Path(self.image).write_bytes(b"fixture")
        self.pipe = Pipeline({"tiling": {}, "backbone": {"enabled": False}}, None,
                             [TradSlot({"topk": 3, "zclip": 8.0})])
        self.addCleanup(self.close_engine_pipes)
        self.pipe.slot_sanity = {}
        self.pipe.weights = {"trad": 1.0}
        self.pipe.slots["trad"].bank_w = np.ones((2, 200))
        self.pipe.decider = Decider(0.9, 0.7)
        self.pipe.router_gate = self.pipe.consensus_gate = None
        save_snapshot(self.pipe, str(self.root / "v1"), version=1)
        set_current_version(str(self.root), 1)
        self.engine._cache_put("test", self.pipe, 1)
        self.loads = []
        def build(*args):
            return None, [TradSlot({"topk": 3, "zclip": 8.0})]
        self.patches = [patch("algo.factory.build", side_effect=build),
                        patch.object(Pipeline, "feedback", self.feedback)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def feedback(pipe, *args, **kwargs):
        pipe.weights["trad"] += 1
        pipe.slots["trad"].bank_w += 1
        pipe.decider.tau_high += 0.1
        return {"action": "stub_mutation"}

    def run_candidate(self, event="one", recover=False):
        return self.engine.learn_candidate("test", self.image, verdict="wrong", label=1,
                                           event_id=event, parent_version=1, recover=recover)

    def state(self):
        conn = self.engine._feedback_receipts()
        try:
            return conn.execute("SELECT state,result FROM candidate_events WHERE event_key='candidate:one'").fetchone()
        finally:
            conn.close()

    def close_engine_pipes(self):
        for pipe, _ in getattr(self.engine, "_pipes", {}).values():
            executor = getattr(pipe, "_cpu_exec", None)
            if executor is not None:
                executor.shutdown(wait=True)

    def restart(self):
        self.close_engine_pipes()
        self.engine = DetectionEngine(self.tmp.name, "configs/engine_fast.yaml")
        self.engine.device = "cpu"

    def test_fault_checkpoints_and_explicit_restart_recovery(self):
        for point in ("before_save", "after_save", "before_prepared", "after_prepared",
                      "before_rename", "after_rename", "before_completed", "after_completed"):
            with self.subTest(point=point):
                event = point
                print("[TEMP recovery] " + point, flush=True)
                current = (self.root / "current.json").read_bytes()
                wal = Path(self.engine._journal_path("test"))
                before = wal.read_bytes() if wal.exists() else None
                def fail(p):
                    if p == point:
                        raise RuntimeError("checkpoint injected")
                with patch.object(self.engine, "_candidate_checkpoint", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "checkpoint injected"):
                        self.run_candidate(event)
                self.restart()
                if point in ("before_save", "after_save", "before_prepared"):
                    with self.assertRaisesRegex(ValueError, "未完成或失败"):
                        self.run_candidate(event, recover=True)
                else:
                    if point != "after_completed":
                        with self.assertRaisesRegex(ValueError, "显式recover"):
                            self.run_candidate(event)
                    out = self.run_candidate(event, recover=True)
                    self.assertTrue(out["idempotent"])
                    self.assertTrue(self.run_candidate(event)["idempotent"])
                self.assertEqual((self.root / "current.json").read_bytes(), current)
                self.assertEqual(wal.read_bytes() if wal.exists() else None, before)
                self.assertEqual(self.pipe.weights, {"trad": 1.0})

    def test_training_intent_not_guessed_failed(self):
        class Crash(BaseException):
            pass
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=Crash):
            with self.assertRaises(Crash):
                self.run_candidate()
        self.restart()
        self.assertEqual(self.state()[0], "training")
        with self.assertRaisesRegex(ValueError, "未完成或失败"):
            self.run_candidate()
        self.assertEqual(self.state()[0], "training")
        with self.assertRaisesRegex(ValueError, "未恢复候选"):
            self.run_candidate("two")
        with self.assertRaisesRegex(ValueError, "未完成或失败"):
            self.run_candidate(recover=True)
        self.assertEqual(self.state()[0], "rejected")
        self.run_candidate("two")
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_prepared_integrity_parent_and_wal_rejection(self):
        for change in ("candidate", "parent", "wal", "current"):
            with self.subTest(change=change):
                print("[TEMP integrity] " + change, flush=True)
                def fail(point):
                    if point == "after_prepared":
                        raise RuntimeError("prepared injected")
                with patch.object(self.engine, "_candidate_checkpoint", side_effect=fail):
                    with self.assertRaises(RuntimeError):
                        self.run_candidate(change)
                conn = self.engine._feedback_receipts()
                intent = json.loads(conn.execute("SELECT result FROM candidate_events WHERE event_key=?",
                                                 ("candidate:" + change,)).fetchone()[0])
                conn.close()
                target = Path(intent["stage"]) / "meta.json"
                parent = self.root / "v1" / "meta.json"
                original = parent.read_bytes()
                if change == "candidate":
                    target.write_bytes(target.read_bytes() + b" ")
                elif change == "parent":
                    parent.write_bytes(original + b" ")
                elif change == "wal":
                    self.engine._append_journal("test", {"base": 1})
                else:
                    set_current_version(str(self.root), 9)
                self.restart()
                current = (self.root / "current.json").read_bytes()
                wal = Path(self.engine._journal_path("test"))
                before = wal.read_bytes() if wal.exists() else None
                with self.assertRaises(ValueError):
                    self.run_candidate(change, recover=True)
                self.assertEqual((self.root / "current.json").read_bytes(), current)
                self.assertEqual(wal.read_bytes() if wal.exists() else None, before)
                parent.write_bytes(original)
                if wal.exists():
                    wal.unlink()
                set_current_version(str(self.root), 1)

    def test_second_engine_cannot_reject_live_training(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        other = DetectionEngine(self.tmp.name, "configs/engine_fast.yaml")
        def slow(pipe, *a, **kw):
            entered.set()
            release.wait(20)
            return self.feedback(pipe, *a, **kw)
        def worker():
            try:
                self.run_candidate()
            except Exception as e:
                errors.append(e)
        with patch.object(Pipeline, "feedback", slow):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(ValueError, "持锁"):
                    other.learn_candidate("test", self.image, verdict="wrong", label=1,
                                          event_id="one", parent_version=1, recover=True)
                self.assertEqual(self.state()[0], "training")
                import subprocess
                import sys
                print("[TEMP mutex] child process recover while training", flush=True)
                code = ("import sys; from backend.engine import DetectionEngine; "
                        "e=DetectionEngine(sys.argv[1], 'configs/engine_fast.yaml'); "
                        "e.learn_candidate('test',sys.argv[2],verdict='wrong',label=1,"
                        "event_id='one',parent_version=1,recover=True)")
                child = subprocess.run([sys.executable, "-u", "-c", code,
                                        self.tmp.name, self.image], capture_output=True,
                                       text=True, timeout=15)
                self.assertNotEqual(child.returncode, 0)
                self.assertIn("其他执行者持锁", child.stderr)
                self.assertEqual(self.state()[0], "training")
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_production()

    def test_budget_shared_round_and_retry(self):
        from algo.ssocl.candidate_budget import CandidateDeferred, charge
        wal = Path(self.engine._journal_path("test"))
        wal.write_text(json.dumps({"base": 1, "path": self.image, "verdict": "wrong",
                                   "label": 1, "event_id": "parent"}) + "\n")
        before = wal.read_bytes()
        self.engine.ssocl_cfg["candidate_budget"] = {"max_steps": 3}
        def work(pipe, *a, **kw):
            charge("steps", 2, "fixture_step")
            return self.feedback(pipe, *a, **kw)
        with patch.object(Pipeline, "feedback", work):
            with self.assertRaises(CandidateDeferred):
                self.run_candidate()
        state, evidence = self.state()
        evidence = json.loads(evidence)
        self.assertEqual(state, "deferred")
        self.assertEqual(evidence["candidate_budget"]["used"]["events"], 2)
        self.assertEqual(evidence["candidate_budget"]["used"]["steps"], 4)
        with self.assertRaisesRegex(ValueError, "未完成或失败"):
            self.run_candidate()
        self.engine.ssocl_cfg["candidate_budget"]["max_steps"] = 4
        with patch.object(Pipeline, "feedback", work):
            result = self.run_candidate("retry")
        self.assertNotEqual(result["candidate_budget"]["config_hash"],
                            evidence["candidate_budget"]["config_hash"])
        self.assertEqual(result["candidate_budget"]["used"]["reloads"], 2)
        self.assertEqual(wal.read_bytes(), before)
        self.assert_production()

    def test_budget_limits_fail_closed(self):
        from algo.ssocl.candidate_budget import CandidateDeferred, charge
        for name in ("events", "loo", "search", "steps", "reloads", "snapshot_bytes", "patch_bytes"):
            with self.subTest(limit=name):
                print("[TEMP budget] " + name, flush=True)
                self.engine.ssocl_cfg["candidate_budget"] = {"max_" + name: 0}
                def work(pipe, *a, **kw):
                    if name in ("loo", "search", "steps"):
                        charge(name)
                    return self.feedback(pipe, *a, **kw)
                with patch.object(Pipeline, "feedback", work):
                    with self.assertRaises(CandidateDeferred):
                        self.run_candidate("limit_" + name)
                self.assert_production()
                self.assertFalse(list(self.root.glob(".candidate-*")))

    def test_budget_wall_clock(self):
        from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred
        now = [100.0]
        self.engine.ssocl_cfg["candidate_budget"] = {"max_wall_ms": 500}
        budget = CandidateBudget(self.engine, category="test", event_id="wall",
                                 clock=lambda: now[0])
        now[0] += 0.4
        budget.check("within")
        now[0] += 0.05
        snap = budget.snapshot()
        self.assertEqual((snap["used"]["wall_ms"], snap["elapsed_ms"]), (400, 450))
        with self.assertRaises(ValueError):
            budget.add("wall_ms", 1)
        now[0] += 0.2
        with self.assertRaisesRegex(CandidateDeferred, "墙钟超限") as caught:
            budget.add("steps", 1, "slow_step")
        self.assertEqual(caught.exception.state, "deferred")
        # 端到端：零墙钟额度在首个耗时≥1ms的检查点拒绝，生产与临时目录不受影响。
        self.engine.ssocl_cfg["candidate_budget"] = {"max_wall_ms": 0}
        with self.assertRaisesRegex(CandidateDeferred, "墙钟超限"):
            self.run_candidate("wall_e2e")
        self.assert_production()
        self.assertFalse(list(self.root.glob(".candidate-*")))

    def test_sweep_orphan_candidate_stages(self):
        # 崩溃残留（无 prepared 意图引用）在下一次候选取得 OS 锁后被清理；
        # prepared 暂存保留供显式 recover（见 fault 测试 after_prepared 分支）。
        orphan = self.root / ".candidate-orphan"
        orphan.mkdir()
        (orphan / "meta.json").write_text("{}")
        kept = self.root / ".candidate-kept"
        kept.mkdir()
        conn = self.engine._feedback_receipts()
        try:
            conn.execute("INSERT INTO candidate_events VALUES (?,?,?,'prepared',?)",
                         ("candidate:other", "[]", 1, json.dumps({"stage": str(kept)})))
            conn.commit()
            removed = self.engine._sweep_candidate_stages("test", conn)
        finally:
            conn.close()
        self.assertEqual([Path(p).name for p in removed], [".candidate-orphan"])
        self.assertFalse(orphan.exists())
        self.assertTrue(kept.exists())

    def test_learning_yields_to_foreground(self):
        import time
        from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred
        eng = self.engine
        eng.learning_idle_ms = 50
        eng.learning_parallel = False
        # 空闲：不等待
        self.assertLess(eng._learning_yield(), 0.02)
        # 前台活跃：学习阻塞，前台结束 + idle 后继续
        started, released = threading.Event(), threading.Event()
        def fg():
            with eng._foreground():
                started.set()
                released.wait(5)
        t = threading.Thread(target=fg)
        t.start()
        started.wait(5)
        waited = []
        w = threading.Thread(target=lambda: waited.append(eng._learning_yield()))
        w.start()
        time.sleep(0.15)
        self.assertTrue(w.is_alive(), "前台推理期间学习应让行")
        released.set()
        t.join(5)
        w.join(5)
        self.assertGreaterEqual(waited[0], 0.15 + 0.04)
        # 持品类锁时不得等待（避免与等锁的前台推理死锁）
        with eng._foreground():
            with eng._pipe_lock("test"):
                self.assertEqual(eng._learning_yield(), 0.0)
            # 并行模式：不等待
            eng.learning_parallel = True
            self.assertEqual(eng._learning_yield(), 0.0)
            eng.learning_parallel = False
            # 取消可打断等待
            cancel = threading.Event()
            threading.Timer(0.1, cancel.set).start()
            t0 = time.monotonic()
            eng._learning_yield(cancel)
            self.assertLess(time.monotonic() - t0, 1.0)
        # 让行时长从候选墙钟预算扣除
        now = [0.0]
        eng.ssocl_cfg["candidate_budget"] = {"max_wall_ms": 100}
        budget = CandidateBudget(eng, category="test", event_id="yield", clock=lambda: now[0])
        def fake_yield(cancel_event=None):
            now[0] += 5.0
            return 5.0
        with patch.object(eng, "_learning_yield", side_effect=fake_yield):
            budget.check("paused")
        snap = budget.snapshot()
        self.assertEqual((snap["used"]["wall_ms"], snap["yield_ms"]), (0, 5000))
        now[0] += 0.2
        with self.assertRaises(CandidateDeferred):
            budget.check("slow")

    def test_cooperative_cancel_commit_boundary(self):
        from algo.ssocl.candidate_budget import CandidateDeferred
        for point in ("budget:parent_events", "budget:after_reload", "before_prepared", "after_prepared"):
            cancel = threading.Event()
            def checkpoint(p):
                if p == point:
                    cancel.set()
            print("[TEMP cancel] " + point, flush=True)
            with patch.object(self.engine, "_candidate_checkpoint", side_effect=checkpoint):
                call = lambda: self.engine.learn_candidate("test", self.image, verdict="wrong",
                    label=1, event_id=point, parent_version=1, cancel_event=cancel)
                if point == "after_prepared":
                    self.assertEqual(call()["candidate_state"], "completed")
                else:
                    with self.assertRaises(CandidateDeferred) as caught:
                        call()
                    self.assertEqual(caught.exception.state, "cancelled")
            self.assert_production()

    def test_real_router_and_loo_propagate_budget(self):
        from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred, candidate_scope
        from algo.fusion.router import LinearRouter, train_router_on_matrix
        from algo.ssocl.feedback import FeedbackHandler
        self.engine.ssocl_cfg["candidate_budget"] = {"max_steps": 1, "max_loo": 0}
        budget = CandidateBudget(self.engine, category="test", event_id="unit")
        with candidate_scope(budget):
            with self.assertRaises(CandidateDeferred):
                train_router_on_matrix(LinearRouter(2), np.ones((2, 2)),
                                       np.zeros((2, 2)), epochs=2)
            # 零额度在访问bank/item前拒绝，不可能破坏LOO临时剔除状态。
            with self.assertRaises(CandidateDeferred):
                with FeedbackHandler._loo(object(), {}):
                    self.fail("不得进入LOO")
        from types import SimpleNamespace
        from algo.ssocl.incubate import HeadManager, IncubatedHead
        with candidate_scope(budget), patch.object(IncubatedHead, "incubate",
                side_effect=CandidateDeferred("incubate budget")):
            with self.assertRaises(CandidateDeferred):
                HeadManager({"min_samples": 1}).maybe_incubate(
                    SimpleNamespace(samples=[object()]), None)
        from PIL import Image
        from algo.ssocl.candidate_budget import charge_image
        Image.new("RGB", (3, 2)).save(self.image)
        self.engine.ssocl_cfg["candidate_budget"] = {"max_input_pixels": 5}
        with candidate_scope(CandidateBudget(self.engine, category="test", event_id="pixels")):
            with self.assertRaises(CandidateDeferred):
                charge_image(self.image)
        self.assertEqual(budget.used["steps"], 2)
        self.assertEqual(budget.used["loo"], 1)
        self.assert_production()

    def assert_production(self):
        self.assertIs(self.engine._pipes["test"][0], self.pipe)
        self.assertEqual(self.engine.current_version("test"), 1)
        self.assertEqual(self.pipe.weights, {"trad": 1.0})
        np.testing.assert_array_equal(self.pipe.slots["trad"].bank_w, np.ones((2, 200)))
        self.assertEqual(self.pipe.decider.tau_high, 0.9)

    def test_save_reload_duplicate_wal_and_activation_block(self):
        rec = {"base": 1, "path": self.image, "verdict": "wrong", "label": 0, "feedback_id": 7}
        self.engine._append_journal("test", rec)
        self.engine._append_journal("test", rec)
        before = Path(self.engine._journal_path("test")).read_bytes()
        out = self.run_candidate()
        self.assertEqual(out["candidate_state"], "completed")
        restored = load_snapshot(str(self.root / "v2"), "cpu")
        self.addCleanup(restored._cpu_exec.shutdown, wait=True)
        self.assertEqual(restored.weights, {"trad": 3.0})
        self.assertAlmostEqual(restored.decider.tau_high, 1.1)
        np.testing.assert_array_equal(restored.slots["trad"].bank_w, np.full((2, 200), 3))
        self.assertTrue(self.run_candidate()["idempotent"])
        self.assertEqual(self.engine.list_versions("test"), [1, 2])
        self.assertEqual(Path(self.engine._journal_path("test")).read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "验收"):
            self.engine.activate("test", 2)
        self.assert_production()

    def test_failure_after_mutation_no_production_pollution(self):
        def fail(pipe, *a, **kw):
            self.feedback(pipe, *a, **kw)
            raise RuntimeError("injected")
        with patch.object(Pipeline, "feedback", fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.run_candidate()
        with self.assertRaisesRegex(ValueError, "未完成或失败"):
            self.run_candidate()
        self.assertEqual(self.engine.list_versions("test"), [1])
        self.assert_production()

    def test_parent_changes_during_training_rejected(self):
        def change(pipe, *a, **kw):
            self.feedback(pipe, *a, **kw)
            set_current_version(str(self.root), 9)
            return {}
        with patch.object(Pipeline, "feedback", change):
            with self.assertRaisesRegex(ValueError, "父版本已变化"):
                self.run_candidate()
        self.assertEqual(self.engine.list_versions("test"), [1])
        self.assertEqual(self.pipe.weights, {"trad": 1.0})

    def test_prediction_does_not_wait_for_candidate_training(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        def slow(pipe, *a, **kw):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timeout")
            return self.feedback(pipe, *a, **kw)
        def worker():
            try:
                self.run_candidate()
            except Exception as e:
                errors.append(e)
        with patch.object(Pipeline, "feedback", slow), patch.object(
                self.pipe, "predict", return_value={"fused": 0.2, "decision": "normal"}):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertTrue(self.engine._pipe_lock("test").acquire(timeout=0.2))
                self.engine._pipe_lock("test").release()
                self.assertEqual(self.engine.predict_image_path("test", self.image).score, 0.2)
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assert_production()

    def test_real_service_candidate_queue_and_trace(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.db import database
        from backend.db.models import Base, Detection, Feedback, Image
        from backend.self_learning import service
        from backend.api.routes_feedback import CandidateFeedbackRequest, api_candidate_feedback
        db = create_engine("sqlite:///" + str(self.root / "business.sqlite3"))
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        sessions = sessionmaker(bind=db, expire_on_commit=False)
        with sessions.begin() as s:
            s.add(Image(path=self.image, category="test", split="feedback", label="anomaly"))
            s.add(Detection(id=1, image_path=self.image, category="test", final_score=0.2,
                            is_anomaly=False, latency_ms=1))
        outputs = []
        def task(name, payload, fn):
            outputs.append(fn(lambda *args: None))
            return "temp-task"
        with patch.object(database, "get_session", sessions), \
             patch.object(service, "get_engine", return_value=self.engine), \
             patch.object(self.engine, "predict_image_path", side_effect=RuntimeError("no pre")), \
             patch("backend.api.routes_feedback.task_manager.submit", side_effect=task), \
             patch("backend.api.routes_models.register_engine_snapshots"):
            req = CandidateFeedbackRequest(detection_id=1, feedback_type="false_negative",
                    operator_label=1, source_event_id="source", parent_version=1)
            queued = service.submit_feedback_record(
                1, "false_negative", 1, source_event_id="source", candidate_only=True)
            self.assertEqual(queued["learning_status"], "candidate_queued")
            response = api_candidate_feedback(req)
            self.assertEqual(response["learning_status"], "candidate_queued")
            self.assertEqual(outputs[0]["learning_status"], "candidate")
            duplicate = api_candidate_feedback(req)
            self.assertTrue(duplicate["idempotent"])
            self.assertEqual(len(outputs), 1)
            from backend.api.routes_feedback import CandidateBuildRequest, api_build_feedback_candidate
            from fastapi import HTTPException
            same = api_build_feedback_candidate(
                queued["feedback_id"], CandidateBuildRequest(parent_version=1))
            self.assertTrue(same["idempotent"])
            self.assertEqual(same["candidate"]["version_num"], outputs[0]["version_num"])
            with self.assertRaises(HTTPException) as conflict:
                api_build_feedback_candidate(
                    queued["feedback_id"], CandidateBuildRequest(parent_version=2))
            self.assertEqual(conflict.exception.status_code, 409)
            with self.assertRaises(HTTPException) as missing:
                api_build_feedback_candidate(99999, CandidateBuildRequest(parent_version=1))
            self.assertEqual(missing.exception.status_code, 404)
            self.assertEqual(len(outputs), 1)
            with sessions() as s:
                fb = s.query(Feedback).one()
                self.assertFalse(fb.consumed)
                self.assertEqual(fb.learning_trace["candidate"]["parent_version"], 1)
            with self.assertRaises(service.FeedbackConflict):
                service.submit_feedback_record(1, "false_negative", 1, source_event_id="source",
                                               candidate_only=False, allow_legacy_online=True)
            self.engine.ssocl_cfg["candidate_budget"] = {"max_events": 0}
            failed_req = CandidateFeedbackRequest(detection_id=1, feedback_type="false_negative",
                    operator_label=1, source_event_id="budget-source", parent_version=1)
            from algo.ssocl.candidate_budget import CandidateDeferred
            with self.assertRaises(CandidateDeferred):
                api_candidate_feedback(failed_req)
            with sessions() as s:
                failed = s.query(Feedback).filter(Feedback.source_event_id == "budget-source").one()
                self.assertFalse(failed.consumed)
                self.assertEqual(failed.learning_trace["learning_status"], "candidate_failed")
        self.assert_production()

    def test_wal_changes_during_training_rejected(self):
        def change(pipe, *a, **kw):
            self.engine._append_journal("test", {"base": 1, "path": self.image, "label": 0})
            return self.feedback(pipe, *a, **kw)
        with patch.object(Pipeline, "feedback", change):
            with self.assertRaisesRegex(ValueError, "父WAL已变化"):
                self.run_candidate()
        self.assert_production()

    def test_save_failure_and_event_conflict(self):
        with patch("algo.persist.save_snapshot", side_effect=OSError("save injected")):
            with self.assertRaisesRegex(OSError, "save injected"):
                self.run_candidate()
        with self.assertRaisesRegex(ValueError, "payload"):
            self.engine.learn_candidate("test", self.image, verdict="correct", label=0,
                                       event_id="one", parent_version=1)
        self.assert_production()

    def test_wal_missing_asset_fails_closed(self):
        self.engine._append_journal("test", {"base": 1, "path": "missing", "label": 0})
        with self.assertRaisesRegex(ValueError, "资产缺失"):
            self.run_candidate()
        self.assert_production()


if __name__ == "__main__":
    unittest.main(verbosity=2)
