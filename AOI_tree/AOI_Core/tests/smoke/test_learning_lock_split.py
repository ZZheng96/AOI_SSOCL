"""§78 学习锁内持有缩短：RegressGate check→recenter 锚定分复用（状态指纹单次消费）。
真实 Pipeline._gate_anchor_pass/_gate_state_token/_gate_anchor_replay + 真实 RegressGate，
桩化单张 predict（确定性、依赖权重/校准 edges/sem 库/路由参数/正常库版本，带耗时）。
TEMP 目录，不碰生产资产。"""
import os
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from algo.eval.offline import Pipeline
from algo.ssocl.feedback import FeedbackHandler, RegressGate
from backend.engine import DetectionEngine

N_ANCHOR = 10
PRED_S = 0.02      # 桩化单张锚定 predict 耗时（真实约 30～50ms/张）


class _NB:
    def __init__(self):
        self.version, self.ext, self.observed = 0, [], []

    def _observe_decision(self, is_anomaly):
        self.observed.append(bool(is_anomaly))


class _Trace:
    def __init__(self):
        self.rows = []

    def record(self, rec):
        self.rows.append((rec["path"], rec["fused"], rec["decision"]))


class _GatePipe(Pipeline):
    """只桩化 predict；门控接口走 Pipeline 真实实现。"""

    def __init__(self, delay=0.0, trace=True):   # 不调 Pipeline.__init__
        self.delay = delay
        self.n_predict = 0
        self.decider = SimpleNamespace(tau_gray=0.35, tau_high=0.45)
        self.weights = {"sem": 0.5, "trad": 0.5}
        self._score_slots = ["sem", "trad"]
        self.calibrators = {"sem": SimpleNamespace(edges=np.linspace(0, 1, 5)),
                            "trad": SimpleNamespace(edges=np.linspace(0, 1, 5))}
        self.slots = {"sem": SimpleNamespace(bank=torch.zeros(3, 2))}
        self.handler = SimpleNamespace(normal_bank=_NB(),
                                       defect_bank=SimpleNamespace(version=0, items=[]))
        self.head_mgr = self.local_branch = self.defect_clf = None
        self.router = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.router.weight.fill_(0.0)
        self.trace_api = _Trace() if trace else None
        self._item_cache = {}

    def score(self, path):
        i = int(str(path).rsplit("_", 1)[1])
        return (0.3 * self.weights["sem"]
                + 0.2 * float(np.mean(self.calibrators["sem"].edges))
                + 0.1 * float(self.slots["sem"].bank.sum())
                + 0.05 * float(self.router.weight.sum())
                + 0.01 * self.handler.normal_bank.version + 0.03 * i)

    def predict(self, path):
        time.sleep(self.delay)
        self.n_predict += 1
        fused = self.score(path)
        dec = "anomaly" if fused >= self.decider.tau_high else "normal"
        self._last_observed = dec == "anomaly"          # 同 _finalize
        self.handler.normal_bank._observe_decision(dec == "anomaly")
        rec = {"path": path, "fused": fused, "decision": dec}
        if self.trace_api is not None:
            self.trace_api.record(rec)
        return rec


class _NoReusePipe(_GatePipe):
    """改动前行为：无 §78 门控接口 → RegressGate 逐张 _predict_final_fused。"""
    _gate_anchor_pass = None
    _gate_state_token = None


ANCHORS = [f"anchor_{i}" for i in range(N_ANCHOR)]


def _gate(pipe):
    return RegressGate(tol=10.0).setup(pipe, ANCHORS)


def _decide(gate, decision):
    FeedbackHandler._audit_decide(SimpleNamespace(gate=gate), decision)


def _learn_step(pipe, gate, k):
    """模拟 reflow：改 sem 库（换张量）+ CDF edges → check → accepted → recenter；
    k 为奇数时模拟 rolled_back（回滚状态，不 recenter）。"""
    old_bank, old_edges = pipe.slots["sem"].bank, pipe.calibrators["sem"].edges.copy()
    pipe.slots["sem"].bank = old_bank + 0.01
    pipe.calibrators["sem"].edges = old_edges * 0.99
    gate.check(pipe, f"reflow_{k}")
    if k % 2:
        pipe.slots["sem"].bank = old_bank.clone()
        pipe.calibrators["sem"].edges = old_edges
        _decide(gate, "rolled_back")
        return
    _decide(gate, "accepted")
    gate.recenter(pipe)


def _state(pipe, gate):
    return (gate.base_scores, gate.base_mean, gate.base_std,
            [(h["tag"], h["drift"], h["new_mean"], h["ok"]) for h in gate.history],
            pipe.handler.normal_bank.observed,
            pipe.trace_api.rows if pipe.trace_api is not None else None,
            float(pipe.slots["sem"].bank.sum()), list(pipe.calibrators["sem"].edges))


class GateReuseEquivalenceTest(unittest.TestCase):
    def test_reuse_equals_recompute_including_side_effects(self):
        for trace in (True, False):
            with self.subTest(trace=trace):
                a, b = _GatePipe(trace=trace), _NoReusePipe(trace=trace)
                ga, gb = _gate(a), _gate(b)
                for k in range(6):
                    _learn_step(a, ga, k)
                    _learn_step(b, gb, k)
                self.assertEqual(_state(a, ga), _state(b, gb))
                # accepted 3 次各省一整轮锚定前向
                self.assertEqual(b.n_predict - a.n_predict, 3 * N_ANCHOR)
                self.assertEqual(ga.reuse_stats, {"reused": 3, "recomputed": 0, "mismatch": 0})
                self.assertIsNone(ga._pending)

    def test_state_change_between_check_and_recenter_detected(self):
        def w(p): p.weights["sem"] = 0.6
        def edges_inplace(p): p.calibrators["trad"].edges[1] += 0.05
        def bank_swap(p): p.slots["sem"].bank = p.slots["sem"].bank + 1.0
        def router_inplace(p):
            with torch.no_grad():
                p.router.weight.add_(0.1)
        def nb_version(p): p.handler.normal_bank.version += 1
        def tau(p): p.decider.tau_high = 0.40
        for name, mut in [("weights", w), ("edges_inplace", edges_inplace),
                          ("sem_bank", bank_swap), ("router_inplace", router_inplace),
                          ("normal_bank_version", nb_version), ("tau", tau)]:
            with self.subTest(mutation=name):
                p = _GatePipe()
                g = _gate(p)
                g.check(p, "x")
                mut(p)
                n0 = p.n_predict
                g.recenter(p)
                self.assertEqual(p.n_predict - n0, N_ANCHOR)          # 回退重算
                self.assertEqual(g.base_scores, [p.score(x) for x in ANCHORS])
                self.assertEqual(g.reuse_stats["mismatch"], 1)
                self.assertEqual(g.reuse_stats["reused"], 0)

    def test_pending_single_use_and_scoped(self):
        p, q = _GatePipe(), _GatePipe()
        g = _gate(p)
        g.check(p, "x")
        g.recenter(q)                               # 不同 pipeline → 失配
        self.assertEqual(g.reuse_stats["mismatch"], 1)
        n0 = p.n_predict
        g.recenter(p)                               # 已被消费 → 重算（无 check 留存）
        self.assertEqual(p.n_predict - n0, N_ANCHOR)
        g.check(p, "y")
        g.anchor_paths = ANCHORS[:-1]               # 锚定集变了 → 失配
        g.recenter(p)
        self.assertEqual(g.reuse_stats["mismatch"], 2)
        g.anchor_paths = list(ANCHORS)
        g.check(p, "z")
        _decide(g, "rolled_back")                   # 回滚显式作废留存
        self.assertIsNone(g._pending)
        g.unfreeze(p, rebase=True)                  # rebase 不受影响（照旧现算）

    def test_fake_pipeline_untouched(self):
        fake = SimpleNamespace(_predict_final_fused=lambda path: 0.1)
        g = RegressGate().setup(fake, ANCHORS)
        g.check(fake)
        self.assertIsNone(g._pending)
        self.assertAlmostEqual(g.recenter(fake), 0.1)


class _EnginePipe:
    """引擎侧桩：feedback 在品类锁内执行 _learn_step（真实门控 + _GatePipe 打分）。"""
    feature_representation = {}

    def __init__(self, gp):
        self.gp = gp
        self.gate = _gate(gp)
        self.decider = gp.decider
        self.handler = None
        self.backbone = object()
        self._pre_cfg = object()
        self._item_cache = {}
        self.k = 0
        self.half = False

    def build_item(self, path):
        return {"path": path}

    def adopt_item(self, item):
        return self._item_cache.setdefault(item["path"], item)

    def feedback(self, path, verdict=None, label=None, box=None):
        self.half = True
        _learn_step(self.gp, self.gate, self.k)
        self.k += 1
        self.half = False
        return {}

    def predict(self, path):
        # 前台：只读打分（不计缺陷率窗/追溯，保证与串行可比）
        return {"fused": self.gp.score("anchor_0"), "torn": self.half}


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
    e._to_result = lambda p, rec, dt, v=None, lock_wait_ms=None: dict(
        rec, dt_ms=dt, lock_wait_ms=lock_wait_ms)
    return e


class EngineLockHoldTest(unittest.TestCase):
    N_FB = 4

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_s78_")
        self.addCleanup(self.tmp.cleanup)
        self.paths = [os.path.join(self.tmp.name, f"img_{i}.png") for i in range(self.N_FB)]

    def _run(self, cls, with_fg):
        ep = _EnginePipe(cls(delay=PRED_S))
        e = _engine(self.tmp.name, ep)
        stop, fg = threading.Event(), []

        def loop():
            while not stop.is_set():
                fg.append(e.predict_image_path("cat", "probe.png"))
                time.sleep(0.005)

        th = threading.Thread(target=loop, daemon=True)
        if with_fg:
            th.start()
        infos = [e.submit_feedback("cat", p, "correct", label=0, timeout_ms=10000)
                 for p in self.paths]
        stop.set()
        if with_fg:
            th.join(2)
        return ep, infos, fg

    def test_concurrent_learning_matches_serial(self):
        ep, _, fg = self._run(_GatePipe, with_fg=True)
        ser = _NoReusePipe()
        gs = _gate(ser)
        for k in range(self.N_FB):
            _learn_step(ser, gs, k)
        self.assertEqual(_state(ep.gp, ep.gate), _state(ser, gs))
        self.assertTrue(fg)
        self.assertFalse(any(r["torn"] for r in fg), "前台读到半更新状态")

    def test_accepted_lock_hold_shrinks(self):
        _, old, _ = self._run(_NoReusePipe, with_fg=False)
        _, new, _ = self._run(_GatePipe, with_fg=False)
        # 偶数步 accepted（check+recenter 两轮 → 一轮），奇数步 rolled_back（一轮，不变）
        for k in range(0, self.N_FB, 2):
            ho, hn = old[k]["lock"]["lock_hold_ms"], new[k]["lock"]["lock_hold_ms"]
            self.assertGreater(ho, 2 * N_ANCHOR * PRED_S * 1000 * 0.9, ho)
            self.assertLess(hn, 0.7 * ho, (hn, ho))
        for k in range(1, self.N_FB, 2):
            hn = new[k]["lock"]["lock_hold_ms"]
            self.assertLess(hn, 1.6 * N_ANCHOR * PRED_S * 1000, hn)


if __name__ == "__main__":
    unittest.main()
