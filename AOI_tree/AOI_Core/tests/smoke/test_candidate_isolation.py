"""候选巩固不得污染生产激活指针，且保留未巩固反馈的缓存/WAL一致性。"""
import json
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, mock_open

from backend.engine import DetectionEngine


class CandidateIsolationTest(unittest.TestCase):
    def make_engine(self):
        engine = DetectionEngine.__new__(DetectionEngine)
        live = (SimpleNamespace(), 3)
        engine._pipes = {"cat": live}
        engine._write_lock = threading.RLock()
        engine._cat_lock = threading.RLock()
        engine._pipe_lock = Mock(return_value=engine._cat_lock)
        engine._ensure_loaded = Mock()
        engine._pipe_entry = Mock(return_value=live)
        tmp = tempfile.TemporaryDirectory(prefix="aoi_candidate_cat_")
        self.addCleanup(tmp.cleanup)
        self.cat_dir = tmp.name
        engine._cat_dir = Mock(return_value=self.cat_dir)
        engine._v_dir = Mock(side_effect=lambda cat, v: os.path.join(self.cat_dir, f"v{v}"))
        # 候选巩固拆为 写状态(锁内) / 哈希+meta(锁外) / 分配版本+rename(_write_lock)
        for name, kw in (("write_snapshot_state",
                          {"return_value": {"production_feature_metadata": {}}}),
                         ("finalize_snapshot", {}), ("set_snapshot_version", {})):
            p = patch(f"algo.persist.{name}", **kw)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        engine._cache_put = Mock()
        engine._clear_journal = Mock()
        engine.device = "cpu"
        engine.ssocl_cfg = {}
        return engine

    def test_candidate_consolidate_restores_active_cache(self):
        engine = self.make_engine()
        engine._replay_journal = Mock()
        active_pipe = SimpleNamespace()
        with patch("algo.persist.list_versions", return_value=[1, 3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot", return_value=active_pipe), \
             patch("algo.persist.set_current_version") as set_current, \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            version = engine.consolidate("cat", activate=False)
        self.assertEqual(version, 4)
        set_current.assert_not_called()
        engine._clear_journal.assert_not_called()
        engine._replay_journal.assert_called_once_with("cat", active_pipe, 3)
        engine._cache_put.assert_called_once_with("cat", active_pipe, 3)

    def test_candidate_replays_real_wal_without_appending_or_clearing(self):
        engine = self.make_engine()
        with tempfile.TemporaryDirectory(prefix="aoi_candidate_wal_") as root:
            image = os.path.join(root, "image.png")
            with open(image, "wb") as f:
                f.write(b"test asset")
            journal = os.path.join(root, "learning_journal.jsonl")
            records = [{"base": 3, "path": image, "verdict": "correct", "label": 1,
                        "box": [0.1, 0.1, 0.5, 0.5], "feedback_id": 17},
                       {"base": 2, "path": image, "verdict": "correct", "label": 0}]
            with open(journal, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            with open(journal, "rb") as f:
                before = f.read()
            engine._journal_path = Mock(return_value=journal)
            snap = os.path.join(root, "v4")
            os.mkdir(snap)
            with open(os.path.join(snap, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"production_feature_metadata": {}}, f)
            engine._v_dir = Mock(side_effect=lambda cat, v: snap if v == 3
                                 else os.path.join(self.cat_dir, f"v{v}"))
            active_pipe = SimpleNamespace(feedback=Mock())
            engine._cache_put.side_effect = lambda *args: self.assertEqual(
                active_pipe.feedback.call_count, 1)
            with patch("algo.persist.list_versions", return_value=[1, 3]), \
                 patch("algo.persist.save_snapshot"), \
                 patch("algo.persist.load_snapshot", return_value=active_pipe), \
                 patch("algo.persist.set_current_version") as set_current:
                self.assertEqual(engine.consolidate("cat", activate=False), 4)
            active_pipe.feedback.assert_called_once_with(
                image, verdict="correct", label=1, box=[0.1, 0.1, 0.5, 0.5])
            set_current.assert_not_called()
            engine._clear_journal.assert_not_called()
            with open(journal, "rb") as f:
                self.assertEqual(f.read(), before)

    def test_replay_failure_does_not_replace_live_cache(self):
        engine = self.make_engine()
        engine._replay_journal = Mock(side_effect=OSError("journal unreadable"))
        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot", return_value=SimpleNamespace()), \
             patch("algo.persist.set_current_version") as set_current, \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            with self.assertRaisesRegex(OSError, "journal unreadable"):
                engine.consolidate("cat", activate=False)
        engine._cache_put.assert_not_called()
        engine._clear_journal.assert_not_called()
        set_current.assert_not_called()

    def test_activated_consolidate_keeps_existing_commit_semantics(self):
        engine = self.make_engine()
        engine._replay_journal = Mock()
        pipe = engine._pipe_entry.return_value[0]
        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot") as load, \
             patch("algo.persist.set_current_version") as set_current, \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            self.assertEqual(engine.consolidate("cat", activate=True), 4)
        set_current.assert_called_once_with(self.cat_dir, 4)
        self.write_snapshot_state.assert_not_called()
        engine._clear_journal.assert_called_once_with("cat")
        engine._cache_put.assert_called_once_with("cat", pipe, 4)
        engine._replay_journal.assert_not_called()
        load.assert_not_called()

    def _other_thread_can_lock(self, lock):
        got = []
        t = threading.Thread(target=lambda: got.append(lock.acquire(timeout=1)) or (got[0] and lock.release()))
        t.start()
        t.join(2)
        return bool(got and got[0])

    def test_candidate_load_replay_runs_outside_category_lock(self):
        engine = self.make_engine()
        seen = {}

        def load(*_a):
            seen["load"] = self._other_thread_can_lock(engine._cat_lock)
            return SimpleNamespace()

        engine._replay_journal = Mock(side_effect=lambda *_a: seen.setdefault(
            "replay", self._other_thread_can_lock(engine._cat_lock)))
        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot", side_effect=load), \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            engine.consolidate("cat", activate=False)
        self.assertEqual(seen, {"load": True, "replay": True})
        engine._cache_put.assert_called_once()

    def test_candidate_hash_finalize_runs_outside_category_lock(self):
        engine = self.make_engine()
        engine._replay_journal = Mock()
        seen = {}
        self.write_snapshot_state.side_effect = lambda *_a, **_k: (
            seen.setdefault("write", self._other_thread_can_lock(engine._cat_lock))
            and None) or {"production_feature_metadata": {"x": 1}}
        self.finalize_snapshot.side_effect = lambda stage, *_a, **_k: seen.update(
            finalize=self._other_thread_can_lock(engine._cat_lock),
            finalize_write=self._other_thread_can_lock(engine._write_lock),
            stage_hidden=os.path.basename(stage).startswith("."))
        self.set_snapshot_version.side_effect = lambda _s, v: seen.update(
            version=v, version_locked=not self._other_thread_can_lock(engine._write_lock))
        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.load_snapshot", return_value=SimpleNamespace()):
            self.assertEqual(engine.consolidate("cat", activate=False), 4)
        self.assertEqual(seen, {"write": False, "finalize": True, "finalize_write": True,
                                "stage_hidden": True, "version": 4, "version_locked": True})
        self.assertTrue(os.path.isdir(os.path.join(self.cat_dir, "v4")))
        self.assertEqual([n for n in os.listdir(self.cat_dir) if n.startswith(".")], [])
        self.assertEqual(engine._pipe_entry.return_value[0].production_feature_metadata, {"x": 1})

    def test_candidate_stage_removed_on_failure(self):
        engine = self.make_engine()
        self.finalize_snapshot.side_effect = OSError("disk full")
        with patch("algo.persist.list_versions", return_value=[3]):
            with self.assertRaisesRegex(OSError, "disk full"):
                engine.consolidate("cat", activate=False)
        self.assertEqual(os.listdir(self.cat_dir), [])

    def test_stale_consolidate_stage_swept_fresh_kept(self):
        engine = self.make_engine()
        stale, fresh = (os.path.join(self.cat_dir, n) for n in (".consolidate-old", ".consolidate-live"))
        for d in (stale, fresh):
            os.makedirs(d)
        os.utime(stale, (0, 0))
        self.finalize_snapshot.side_effect = OSError("stop")
        with patch("algo.persist.list_versions", return_value=[3]):
            with self.assertRaises(OSError):
                engine.consolidate("cat", activate=False)
        self.assertEqual(os.listdir(self.cat_dir), [".consolidate-live"])

    def test_concurrent_learning_during_replay_keeps_live_cache(self):
        engine = self.make_engine()
        marks = iter([(10, 1), (20, 2)])  # 发布前 WAL 增长 = 期间有新学习
        engine._journal_mark = Mock(side_effect=lambda _c: next(marks))
        engine._replay_journal = Mock()
        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot", return_value=SimpleNamespace()), \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            self.assertEqual(engine.consolidate("cat", activate=False), 4)
        engine._cache_put.assert_not_called()

    def test_evicted_or_replaced_cache_is_not_overwritten(self):
        engine = self.make_engine()
        engine._replay_journal = Mock()

        def load(*_a):
            engine._pipes["cat"] = (SimpleNamespace(), 3)  # 期间被重载/替换
            return SimpleNamespace()

        with patch("algo.persist.list_versions", return_value=[3]), \
             patch("algo.persist.save_snapshot"), \
             patch("algo.persist.load_snapshot", side_effect=load), \
             patch("builtins.open", mock_open(read_data='{"production_feature_metadata": {}}')):
            engine.consolidate("cat", activate=False)
        engine._cache_put.assert_not_called()


if __name__ == "__main__":
    unittest.main()
