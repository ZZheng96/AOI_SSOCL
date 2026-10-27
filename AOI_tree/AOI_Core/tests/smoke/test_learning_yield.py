"""AC-069 学习让行/锁粒度/锁等待可观测：桩 pipe 模拟“学习持续 + 前台推理”，TEMP 目录，不碰生产资产。"""
import os
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace

from backend.engine import DetectionEngine
from algo.eval.offline import Pipeline as OfflinePipeline

EXTRACT_S = 0.25   # 模拟 DINO 特征提取（改动前在品类锁内）
UPDATE_S = 0.03    # 模拟锁内打分 + 状态更新


class _StubPipe:
    decider = None
    handler = None
    feature_representation = {}

    def __init__(self):
        self.backbone = object()
        self._pre_cfg = object()
        self._item_cache = {}
        self.a = 0
        self.b = 0
        self.learned = []
        self.extract_calls = 0

    def build_item(self, path):
        self.extract_calls += 1
        time.sleep(EXTRACT_S)
        return {"path": path, "feats": len(path)}

    def adopt_item(self, item):
        return self._item_cache.setdefault(item["path"], item)

    def feedback(self, path, verdict=None, label=None, box=None):
        item = self._item_cache.get(path) or self.adopt_item(self.build_item(path))
        self.a += item["feats"]
        time.sleep(UPDATE_S)          # 半更新窗口：predict 若不持锁会读到 a != b
        self.b += item["feats"]
        self.learned.append((path, verdict, label))
        return {}

    def predict(self, path):
        return {"fused": 0.1, "torn": self.a != self.b}


def _engine(tmp, pipe):
    e = DetectionEngine.__new__(DetectionEngine)
    e.storage_dir = tmp
    e.snap_root = os.path.join(tmp, "snapshots")
    e._pipes = OrderedDict(cat=(pipe, 1))
    e._pipe_locks, e._prepare_locks = {}, {}
    e._locks_guard = threading.Lock()
    e._write_lock = threading.RLock()
    e._fg_cond = threading.Condition()
    e._lock_stats = {}
    e.learning_parallel = False
    e.learning_idle_ms = 50
    e.feedback_yield_max_ms = 200
    return e


class LearningYieldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_ac069_")
        self.addCleanup(self.tmp.cleanup)
        self.paths = [os.path.join(self.tmp.name, f"img_{i}.png") for i in range(4)]

    def _run_concurrent(self, pipe):
        e = _engine(self.tmp.name, pipe)
        stop = threading.Event()
        waits = []

        def fg():
            while not stop.is_set():
                r = e.predict_image_path("cat", "probe.png")
                waits.append(r.extra["lock_wait_ms"])
                time.sleep(0.01)

        th = threading.Thread(target=fg, daemon=True)
        th.start()
        infos = [e.submit_feedback("cat", p, "correct", label=0, timeout_ms=5000)
                 for p in self.paths]
        stop.set()
        th.join(2)
        return e, infos, waits

    def test_foreground_lock_wait_bounded_and_prefetched(self):
        pipe = _StubPipe()
        e, infos, waits = self._run_concurrent(pipe)
        self.assertGreater(len(waits), 10)
        # 改动前前台最坏需等 EXTRACT_S+UPDATE_S（280ms）；现锁内只剩 UPDATE_S
        self.assertLess(max(waits), 150.0, waits)
        for info in infos:
            self.assertTrue(info["lock"]["prefetched"])
            self.assertLess(info["lock"]["lock_hold_ms"], 150.0)
            self.assertLessEqual(info["lock"]["yield_ms"], e.feedback_yield_max_ms + 100)
        self.assertEqual(pipe.extract_calls, len(self.paths))   # 锁外提取后锁内不重算
        st = e.lock_stats()["feedback"]
        self.assertEqual(st["n"], len(self.paths))
        self.assertLess(st["max_hold_ms"], 150.0)

    def test_no_torn_read_and_matches_serial(self):
        pipe = _StubPipe()
        e = _engine(self.tmp.name, pipe)
        e._to_result = lambda p, rec, dt, v=None, lock_wait_ms=None: SimpleNamespace(
            extra={"lock_wait_ms": lock_wait_ms, "torn": rec["torn"]})
        stop, torn = threading.Event(), []

        def fg():
            while not stop.is_set():
                torn.append(e.predict_image_path("cat", "probe.png").extra["torn"])

        th = threading.Thread(target=fg, daemon=True)
        th.start()
        for p in self.paths:
            e.submit_feedback("cat", p, "correct", label=0, timeout_ms=5000)
        stop.set()
        th.join(2)
        self.assertTrue(torn)
        self.assertFalse(any(torn), "前台读到半更新状态")

        serial = _StubPipe()
        for p in self.paths:
            serial.feedback(p, verdict="correct", label=0)
        self.assertEqual((pipe.a, pipe.b, pipe.learned), (serial.a, serial.b, serial.learned))
        self.assertEqual(sorted(pipe._item_cache), sorted(serial._item_cache))

    def test_pipe_swapped_discards_prefetch(self):
        old, new = _StubPipe(), _StubPipe()
        e = _engine(self.tmp.name, old)
        orig = old.build_item

        def build_then_swap(path):
            item = orig(path)
            e._pipes["cat"] = (new, 2)   # 预提取期间巩固/激活换了 pipe
            return item

        old.build_item = build_then_swap
        info = e.submit_feedback("cat", self.paths[0], "correct", label=0, timeout_ms=5000)
        self.assertFalse(info["lock"]["prefetched"])
        self.assertEqual(new.extract_calls, 1)        # 新 pipe 锁内按自身主干现算
        self.assertEqual(old.a, 0)
        self.assertEqual(new.a, new.b)

    def test_yield_bounded_under_continuous_foreground(self):
        e = _engine(self.tmp.name, _StubPipe())
        with e._foreground():
            t0 = time.monotonic()
            waited = e._learning_yield(max_wait_s=0.1)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertGreaterEqual(waited, 0.09)

    def test_consolidate_yield_bounded_under_continuous_foreground(self):
        """持续前台负载下巩固（API 同步调用）的锁外让行有上界，不被无限饿死。"""
        from unittest.mock import Mock, patch
        live = (SimpleNamespace(), 3)
        e = _engine(self.tmp.name, live[0])
        e._pipes = OrderedDict(cat=live)
        e._ensure_loaded = Mock()
        e._pipe_entry = Mock(return_value=live)
        e._cat_dir = Mock(return_value=self.tmp.name)
        e._v_dir = Mock(side_effect=lambda c, v: os.path.join(self.tmp.name, f"v{v}"))
        e._cache_put, e._replay_journal = Mock(), Mock()
        e.device, e.ssocl_cfg = "cpu", {}
        e.feedback_yield_max_ms = 100
        stop, out = threading.Event(), {}

        def fg():   # 前台推理永不空闲（_fg_active 常驻 1）
            with e._foreground():
                stop.wait(10)

        th_fg = threading.Thread(target=fg, daemon=True)
        th_fg.start()
        self.addCleanup(stop.set)
        with patch("algo.persist.write_snapshot_state",
                   return_value={"production_feature_metadata": {}}), \
             patch("algo.persist.finalize_snapshot"), patch("algo.persist.set_snapshot_version"), \
             patch("algo.persist.list_versions", return_value=[1, 3]), \
             patch("algo.persist.load_snapshot", return_value=SimpleNamespace()):
            th = threading.Thread(target=lambda: out.update(
                v=e.consolidate("cat", activate=False)), daemon=True)
            t0 = time.monotonic()
            th.start()
            th.join(5)
        stop.set()
        self.assertFalse(th.is_alive(), "巩固被持续前台无限饿死")
        self.assertEqual(out.get("v"), 4)
        self.assertLess(time.monotonic() - t0, 2.0)   # 两次让行各 ≤100ms

    def test_learning_parallel_default_false(self):
        self.assertIsNone(DetectionEngine.learning_parallel)
        e = DetectionEngine.__new__(DetectionEngine)
        self.assertFalse(e._learning_parallel())
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for name in ("default.yaml", "docker.yaml"):
            with open(os.path.join(root, "configs", name), encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertIs(cfg["self_learning"]["learning_parallel"], False, name)

    def test_offline_build_adopt_split(self):
        p = OfflinePipeline.__new__(OfflinePipeline)
        p._item_cache, p._item_cache_max = {}, 1
        a = p.adopt_item({"path": "a"})
        self.assertIs(p.adopt_item({"path": "a"}), a)   # 已存在保留旧条目
        p.adopt_item({"path": "b"})
        p.adopt_item({"path": "c"})                      # 超上限 FIFO 逐出 a
        self.assertNotIn("a", p._item_cache)
        self.assertIs(p._get_item("c"), p._item_cache["c"])


if __name__ == "__main__":
    unittest.main()
