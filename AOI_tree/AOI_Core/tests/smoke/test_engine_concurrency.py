"""AC-069 前台推理锁外预提取：多前台请求并发结果 == 串行，临界区缩短，学习让行不破坏。
真实 Pipeline 的 predict/predict_frame/prepare_input/_predict_prepared/_timed_finalize，
桩化读图后处理/主干/打分（TEMP 目录，不碰生产资产）。"""
import os
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from backend.engine import DetectionEngine
from algo.eval.offline import Pipeline

PRE_S, FW_S, REC_S = 0.05, 0.06, 0.04   # 预处理 / 主干前向 / 锁内打分


class _Backbone:
    def __init__(self):
        self.active, self.max_active, self.calls = 0, 0, 0
        self._g = threading.Lock()

    def extract_tiles(self, tile_imgs, batch=16):
        with self._g:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(FW_S)
        with self._g:
            self.active -= 1
        return torch.tensor([[float(t.mean()), float(t.std())] for t in tile_imgs])


def _pipe():
    p = Pipeline.__new__(Pipeline)
    p.cfg = {"backbone": {}}
    p.tiling = {"mode": "single", "tile_size": 64, "stride": 56}
    p.backbone = _Backbone()
    p._pre_cfg = {}
    p._item_cache, p._item_cache_max = {}, 128
    p.slots, p._score_slots = {}, []
    p.defect_clf = p._align_ref = p.trace_api = None
    p.tta_enabled = False
    p._cpu_exec = ThreadPoolExecutor(max_workers=2)
    p.n_scored = 0
    p.in_record = 0

    def _preprocess(img):
        time.sleep(PRE_S)
        return img.copy()

    def _record(item, cpu_futures=None):
        p.in_record += 1                  # 同品类锁内串行：并发进入即 >1
        assert p.in_record == 1, "锁内打分被并发进入"
        time.sleep(REC_S)
        p.n_scored += 1
        p.in_record -= 1
        f = item["feats"]
        return {"fused": float(f[:, 0].sum() / 255.0), "spread": float(f[:, 1].sum()),
                "decision": "normal", "path": item["path"]}

    p._preprocess = _preprocess
    p._record = _record
    p._finalize = lambda item, rec: rec
    return p


def _engine(pipe):
    e = DetectionEngine.__new__(DetectionEngine)
    e._pipes = OrderedDict(cat=(pipe, 1))
    e._pipe_locks, e._prepare_locks = {}, {}
    e._locks_guard = threading.Lock()
    e._write_lock = threading.RLock()
    e._fg_cond = threading.Condition()
    e._fg_active, e._fg_last = 0, 0.0
    e._lock_stats = {}
    e._gpu_sem = threading.BoundedSemaphore(1)
    e.learning_parallel = False
    e.learning_idle_ms = 50
    e._to_result = lambda p, rec, dt, v=None, lock_wait_ms=None: dict(
        rec, dt_ms=dt, lock_wait_ms=lock_wait_ms)
    return e


class EngineConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_ac070_")
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(7)
        self.frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
        self.paths = []
        for i, f in enumerate(self.frames):
            p = os.path.join(self.tmp.name, f"img_{i}.png")
            cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            self.paths.append(p)

    def _run(self, prefetch, mode, conc):
        pipe = _pipe()
        self.addCleanup(pipe._cpu_exec.shutdown)
        e = _engine(pipe)
        e.predict_prefetch = prefetch

        def one(i):
            if mode == "path":
                return e.predict_image_path("cat", self.paths[i])
            return e.predict_ndarray("cat", self.frames[i], path_hint=f"f{i}")

        t0 = time.perf_counter()
        if conc == 1:
            out = [one(i) for i in range(len(self.frames))]
        else:
            with ThreadPoolExecutor(conc) as ex:
                out = list(ex.map(one, range(len(self.frames))))
        return out, time.perf_counter() - t0, pipe, e

    def _keys(self, out):
        return [(r["fused"], r["spread"], r["decision"], r["path"]) for r in out]

    def test_concurrent_equals_serial(self):
        for mode in ("path", "frame"):
            with self.subTest(mode=mode):
                ref, _, _, _ = self._run(False, mode, 1)
                for prefetch in (False, True):
                    out, _, pipe, _ = self._run(prefetch, mode, 4)
                    self.assertEqual(self._keys(out), self._keys(ref))
                    self.assertEqual(pipe.n_scored, len(self.frames))
                    self.assertEqual(pipe.backbone.calls, len(self.frames))   # 不重算
                    self.assertEqual(pipe.backbone.max_active, 1)            # GPU 串行
                    if mode == "path":
                        self.assertEqual(sorted(pipe._item_cache), sorted(self.paths))

    def test_critical_section_shrinks(self):
        _, t_old, _, e_old = self._run(False, "frame", 4)
        out, t_new, _, _ = self._run(True, "frame", 4)
        # 原路径整段持锁 8×(PRE+FW+REC)=1.2s；新路径预处理并行、前向与锁内打分重叠
        self.assertLess(t_new, 0.8 * t_old, (t_new, t_old))
        for r in out:
            tm = r["timing"]
            self.assertGreaterEqual(tm["forward"], FW_S * 1000 * 0.8)
            # total = 锁外预提取（forward 含排队 GPU 信号量，最多 3 个他人前向）+ 锁内（不含等品类锁）
            self.assertLess(tm["total"], (PRE_S + FW_S + REC_S) * 1000 + 150 + 3 * FW_S * 1000, tm)
            self.assertLess(r["dt_ms"], (PRE_S + FW_S + REC_S) * 1000 + 150 + 4 * FW_S * 1000)

    def test_cached_path_skips_prefetch(self):
        out1, _, pipe, e = self._run(True, "path", 1)
        calls = pipe.backbone.calls
        r = e.predict_image_path("cat", self.paths[0])
        self.assertEqual(pipe.backbone.calls, calls)       # 缓存命中不再前向
        self.assertEqual(r["fused"], out1[0]["fused"])

    def test_pipe_swapped_discards_prefetch(self):
        old, new = _pipe(), _pipe()
        for p in (old, new):
            self.addCleanup(p._cpu_exec.shutdown)
        e = _engine(old)
        orig = old.prepare_input

        def prep_then_swap(**kw):
            r = orig(**kw)
            e._pipes["cat"] = (new, 2)          # 预提取期间巩固/激活换了 pipe
            return r

        old.prepare_input = prep_then_swap
        type(old)  # prepare_input 仍按类型可调用判定
        r = e.predict_ndarray("cat", self.frames[0], path_hint="x")
        self.assertEqual(old.n_scored, 0)
        self.assertEqual(new.n_scored, 1)
        self.assertEqual(new.backbone.calls, 1)  # 新 pipe 锁内按自身主干现算
        ref = _pipe()
        self.addCleanup(ref._cpu_exec.shutdown)
        self.assertEqual(r["fused"], ref.predict_frame(self.frames[0], path="x")["fused"])

    def test_backbone_changed_discards_prefetch(self):
        pipe = _pipe()
        self.addCleanup(pipe._cpu_exec.shutdown)
        e = _engine(pipe)
        orig = pipe.prepare_input
        bb2 = _Backbone()

        def prep_then_swap_bb(**kw):
            r = orig(**kw)
            pipe.backbone = bb2
            return r

        pipe.prepare_input = prep_then_swap_bb
        e.predict_ndarray("cat", self.frames[1])
        self.assertEqual(bb2.calls, 1)

    def test_foreground_marked_during_prefetch(self):
        pipe = _pipe()
        self.addCleanup(pipe._cpu_exec.shutdown)
        e = _engine(pipe)
        seen = []
        orig = pipe.prepare_input

        def prep(**kw):
            seen.append(e._fg_active)
            return orig(**kw)

        pipe.prepare_input = prep
        r = e.predict_ndarray("cat", self.frames[2])
        self.assertEqual(seen, [1])              # 学习让行可见锁外预提取
        self.assertEqual(e._fg_active, 0)
        self.assertLess(r["lock_wait_ms"], 50.0)

    def test_learning_yield_waits_for_prefetch(self):
        pipe = _pipe()
        self.addCleanup(pipe._cpu_exec.shutdown)
        e = _engine(pipe)
        th = threading.Thread(target=lambda: e.predict_ndarray("cat", self.frames[3]))
        th.start()
        time.sleep(0.02)                         # 处于锁外预处理阶段
        waited = e._learning_yield(max_wait_s=2.0)
        th.join(2)
        self.assertGreater(waited, PRE_S * 0.5)


if __name__ == "__main__":
    unittest.main()
