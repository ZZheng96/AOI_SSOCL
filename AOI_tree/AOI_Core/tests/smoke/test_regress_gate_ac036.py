"""AC-036：RegressGate 固定初始尾部 + 累计上限 + 连续违规冻结/恢复 + 持久化兼容。
假 pipeline，不碰生产快照/DB。"""
import json
import unittest
from types import SimpleNamespace

from algo.persist import _restore_gate_ac036
from algo.ssocl.feedback import RegressGate

BASE = [.10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0]


def _pipe(scores):
    it = iter(scores)
    return SimpleNamespace(decider=SimpleNamespace(tau_high=.95, tau_gray=.85),
                           _predict_final_fused=lambda p: next(it))


def _gate(tol=.5):
    g = RegressGate(tol=tol)
    g.setup(_pipe(BASE), [f"x{i}" for i in range(len(BASE))])
    return g


class RegressGateAC036Test(unittest.TestCase):
    def test_slow_drift_capped(self):
        print("[1/4] 小步上行漂移：单步过门，累计越限回滚", flush=True)
        g = _gate()
        self.assertAlmostEqual(g.init_tail, .1)
        cur, rolled = list(BASE), None
        for step in range(1, 30):
            cand = [s + .02 for s in cur]
            ok, drift, _ = g.check(_pipe(cand), tag="reflow")
            if not ok:
                rolled = step
                break
            g.recenter(_pipe(cand))     # 调用点：过门后基准跟随
            cur = cand
        self.assertIsNotNone(rolled, "累计漂移未被拦截")
        self.assertGreater(rolled, 1)   # 前几步单步门控放行
        self.assertTrue(g.history[-1]["cum_drift"] > 0)
        self.assertEqual(g.init_mean, sum(BASE) / len(BASE))   # recenter 不改初始尾部

    def test_downward_not_cum_blocked(self):
        print("[2/4] 下行（recenter 意图方向）不触发累计拦截", flush=True)
        g = _gate()
        ok, _, _ = g.check(_pipe([s - .05 for s in BASE]))
        self.assertTrue(ok)
        self.assertLess(g.history[-1]["cum_drift"], 0)

    def test_freeze_and_unfreeze(self):
        print("[3/4] 连续 3 次违规冻结；冻结期正常更新也回滚；unfreeze 恢复", flush=True)
        g = _gate()
        bad = [s + .5 for s in BASE]
        for i in range(3):
            self.assertFalse(g.check(_pipe(bad))[0])
            self.assertEqual(g.frozen, i == 2)
        ok, drift, new_mean = g.check(_pipe(BASE))
        self.assertFalse(ok)
        self.assertGreaterEqual(new_mean, g.base_mean)   # 定向门控也走回滚
        self.assertTrue(g.audit[-1]["frozen"])
        g.unfreeze()
        self.assertEqual((g.frozen, g.violations), (False, 0))
        self.assertTrue(g.check(_pipe(BASE))[0])
        # 中间有一次通过 → 计数清零，不冻结
        g.check(_pipe(bad)); g.check(_pipe(bad)); g.check(_pipe(BASE)); g.check(_pipe(bad))
        self.assertFalse(g.frozen)
        g.unfreeze(_pipe([s + .1 for s in BASE]), rebase=True)
        self.assertAlmostEqual(g.init_mean, sum(BASE) / len(BASE) + .1)

    def test_persist_and_legacy(self):
        print("[4/4] status 可 JSON 化回填；旧快照无 init_* 不判累计", flush=True)
        g = _gate()
        for _ in range(3):
            g.check(_pipe([s + .5 for s in BASE]))
        gs = json.loads(json.dumps(g.status()))
        g2 = RegressGate(tol=.5)
        g2.anchor_paths, g2.base_mean, g2.base_std = g.anchor_paths, g.base_mean, g.base_std
        _restore_gate_ac036(g2, gs)
        self.assertTrue(g2.frozen)
        self.assertEqual(g2.init_p95, g.init_p95)
        legacy = RegressGate(tol=.5)
        legacy.anchor_paths, legacy.base_mean, legacy.base_std = g.anchor_paths, g.base_mean, g.base_std
        _restore_gate_ac036(legacy, {"tol": .5})
        self.assertIsNone(legacy.init_mean)
        self.assertEqual(legacy.cumulative(BASE), (0.0, None))
        self.assertTrue(legacy.check(_pipe(BASE))[0])
        # 旧 pickle 对象无新属性
        old = RegressGate.__new__(RegressGate)
        old.__dict__.update({k: v for k, v in legacy.__dict__.items()
                             if k not in RegressGate.PERSIST_FIELDS})
        old.violations = 0
        self.assertTrue(old.check(_pipe(BASE))[0])


if __name__ == "__main__":
    unittest.main()
