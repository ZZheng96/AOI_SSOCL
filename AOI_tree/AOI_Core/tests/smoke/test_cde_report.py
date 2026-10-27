"""C/D/E（AC-018~027）：校准契约/融合基线/路由准入/域差/槽位健康/更新审计/权重语义。
TEMP 合成数据 + 假 pipeline，不碰生产快照/DB。"""
import contextlib
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from algo.fusion import cde_report as cr
from algo.ssocl.feedback import FeedbackHandler, RegressGate


class _Cal:
    def __init__(self, s):
        self.s = np.sort(np.asarray(s, float))

    def transform(self, x):
        return np.searchsorted(self.s, np.asarray(x, float), side="right") / max(self.s.size, 1)


class PureFunctionsTest(unittest.TestCase):
    def test_contract_baselines_split(self):
        print("[1/8] 校准契约 / 融合基线 / 分层切分", flush=True)
        rng = np.random.default_rng(0)
        fit = {"sem": rng.normal(size=40).tolist(), "blob": rng.normal(size=10).tolist()}
        cals = {n: _Cal(v) for n, v in fit.items()}
        rep = cr.calibration_contract(fit, cals, 256, isolation="self",
                                      holdout_scores={"sem": rng.normal(-.5, 1, 20)}, n_boot=20)
        self.assertTrue(rep["self_match_risk"])
        sem, blob = rep["slots"]["sem"], rep["slots"]["blob"]
        for k in ("method", "n", "lookup", "extrapolation", "clip", "semantics", "tail"):
            self.assertIn(k, sem)
        self.assertFalse(sem["small_sample"])
        self.assertTrue(blob["small_sample"])
        self.assertIn("self_match_bias", sem)
        self.assertNotIn("self_match_bias", blob)
        self.assertFalse(cr.calibration_contract(fit, cals, 256, isolation="holdout")["self_match_risk"])

        cn = {"a": rng.uniform(0, .5, 30), "b": rng.uniform(0, 1, 30)}
        cd = {"a": rng.uniform(.5, 1, 10), "b": rng.uniform(0, 1, 10)}
        fb = cr.fusion_baselines(cn, cd, ["a", "b"], {"equal": {"a": 1, "b": 1}, "sanity": {}},
                                 "equal", slot_sanity={"a": .9, "b": .5})
        self.assertEqual(fb["slots_used"], ["a"])
        self.assertEqual(fb["unverified_slots"], ["b"])
        self.assertNotIn("sanity", fb["baselines"])            # 空权重不报
        self.assertEqual(fb["baselines"]["equal"]["auc"], 1.0)
        self.assertIn("hard_rule_max", fb["baselines"])

        sp = cr.stratified_split(10, 5, seed=1)
        allneg = np.concatenate([sp[k][0] for k in ("train", "val", "test")])
        allpos = np.concatenate([sp[k][1] for k in ("train", "val", "test")])
        self.assertEqual(sorted(allneg.tolist()), list(range(10)))
        self.assertEqual(sorted(allpos.tolist()), list(range(5)))
        self.assertEqual(len(sp["train"][0]), 6)

    def test_router_admission(self):
        print("[2/8] router 准入：样本不足 / trace_only（不胜固定权重）", flush=True)
        names = ["a", "b"]
        bw = {"a": .5, "b": .5}
        rep = cr.router_admission(np.zeros((4, 2)), np.ones((4, 2)), names, bw, {})
        self.assertTrue(rep["insufficient"])
        self.assertEqual(rep["decision"], "trace_only")
        self.assertEqual(rep["n_slots"], 2)
        rng = np.random.default_rng(0)
        neg = rng.uniform(0, .4, (30, 2))
        pos = rng.uniform(.6, 1, (30, 2))                       # 固定权重已完美可分
        rep = cr.router_admission(neg, pos, names, bw, {"epochs": 3})
        self.assertNotIn("insufficient", rep)
        self.assertEqual(rep["decision"], "trace_only")
        self.assertFalse(rep["permitted"])
        for k in ("train", "val", "test"):
            self.assertEqual(set(rep["auc"][k]), {"router", "fixed", "hard_rule"})
        self.assertIn("overfit_gap", rep)
        self.assertIn("norm_entropy", rep)

    def test_domain_shift(self):
        print("[3/8] 域差：insufficient / normal_drift / shift", flush=True)
        ref = np.linspace(0, 1, 50)
        self.assertEqual(cr.domain_shift(ref, None, .9, .05)["status"], "insufficient")
        self.assertEqual(cr.domain_shift(ref, [.5] * 3, .9, .05)["status"], "insufficient")
        r = cr.domain_shift(ref, np.linspace(0, .85, 40), .9, .05,
                            slot_sanity={"a": .8, "b": .6, "c": None})
        self.assertEqual(r["status"], "normal_drift")
        self.assertEqual(r["slot_reliability"], {"a": "reliable", "b": "weak", "c": "unreliable"})
        self.assertTrue(r["mechanisms"]["dynamic_routing"].startswith("not_integrated"))
        self.assertEqual(cr.domain_shift(ref, np.linspace(.8, 1.2, 40), .9, .05)["status"], "shift")

    def test_slot_health_pure(self):
        print("[4/8] slot_health：能力表 + 运行时状态 + open 不进融合", flush=True)
        pipe = SimpleNamespace(
            slots={"sem": 1, "blob": 1}, open_det=object(),
            calibrators={"sem": 1}, slot_sanity={"sem": .9, "blob": .4},
            _cal_normal_hist={"sem": [0] * 7}, weights={"sem": .7, "blob": .3},
            cde_report={"calibration": {"slots": {"sem": {"small_sample": True}}}})
        h = cr.slot_health(pipe)
        self.assertEqual(set(h["slots"]), {"sem", "blob", "open"})
        self.assertEqual(h["slots"]["sem"]["status"], "ok")
        self.assertEqual(h["slots"]["blob"]["status"], "reverse")
        self.assertEqual(h["slots"]["open"]["status"], "unverified")
        self.assertFalse(h["slots"]["open"]["in_fusion"])
        self.assertTrue(h["slots"]["open"]["fitted"])
        self.assertFalse(h["slots"]["blob"]["fitted"])
        self.assertEqual(h["slots"]["sem"]["cal_n"], 7)
        self.assertTrue(h["slots"]["sem"]["small_sample"])
        self.assertIn("gate", h["weights_semantics"])


def _fake_pipe(scores):
    it = iter(scores)
    return SimpleNamespace(decider=SimpleNamespace(tau_high=.9, tau_gray=.7),
                           weights={"a": 1.0}, slots={},
                           _predict_final_fused=lambda p: next(it))


def _handler(gate):
    h = FeedbackHandler.__new__(FeedbackHandler)
    h.p = _fake_pipe([])
    h.gate = gate
    h.rollback_log = []
    h._recent_fb = [({"a": .95}, 1), ({"a": .2}, 0), ({"a": .92}, 0)]
    h._sem_bank_base = None
    h.weight_gate_tol = .1
    gate.context_fn = h._audit_context
    return h


class AuditTest(unittest.TestCase):
    def test_record_decide_report(self):
        print("[5/8] RegressGate 审计条目 / _audit_decide / 汇总（先验/门控前否决）", flush=True)
        g = RegressGate(tol=.5)
        g.setup(_fake_pipe([.1, .8, .95, .2]), ["x1", "x2", "x3", "x4"])
        self.assertEqual(g.base_scores, [.1, .8, .95, .2])
        h = _handler(g)
        ok, drift, _ = g.check(_fake_pipe([.95, .95, .95, .8]), tag="reflow")
        self.assertFalse(ok)
        e = g.audit[-1]
        self.assertEqual(e["snapshot_id"], "upd-00001")
        self.assertEqual(e["op"], "reflow")
        self.assertEqual(e["n_anchor"], 4)
        self.assertEqual(e["anchor_fpr"], {"before": .25, "after": .75})
        self.assertEqual(e["anchor_gray_rate"], {"before": .25, "after": .25})
        self.assertEqual(e["recall_cached"], 1.0)
        self.assertEqual(e["ood_fpr_cached"], .5)
        self.assertEqual(e["n_recent_fb"], 3)
        self.assertIn("g1_estimate", e)
        h._audit_decide("rolled_back", "gate_drift")
        # 门控点外直接 check（无 decide）→ 按 gate_ok
        g.check(_fake_pipe([.1, .8, .95, .2]), tag="defect_bank")
        h._audit_weight([.1, .8, .95, .2], 0.0, {"a": .5}, {"a": 1.0}, auc=.8)
        h._audit_decide("accepted")
        g.audit.insert(0, {"snapshot_id": "upd-00000", "op": "weight_prior",
                           "gate": "not_applicable_pre_setup"})
        h.rollback_log += [{"reason": "reject_pairwise"}, {"reason": "gate_drift"}]
        rep = h.update_audit_report()
        self.assertEqual([x["status"] for x in rep["entries"]],
                         ["applied_prior", "rolled_back", "accepted", "accepted"])
        self.assertEqual(rep["entries"][1]["rollback_reason"], "gate_drift")
        self.assertEqual(rep["entries"][3]["op"], "weight_learn")
        self.assertEqual(rep["entries"][3]["feedback_pair_auc"], .8)
        self.assertEqual(rep["counts"], {"accepted": 2, "rolled_back": 1, "rejected": 0,
                                         "applied_prior": 1})
        self.assertEqual(rep["pre_gate_rejects"], 1)
        # context_fn 异常不阻断
        g.context_fn = lambda: 1 / 0
        self.assertIn("context_error", g.record_audit(_fake_pipe([]), "x", [], 0.0))
        g.context_fn = None
        for _ in range(510):
            g.record_audit(_fake_pipe([]), "x", [], 0.0)
        self.assertEqual(len(g.audit), 500)

    def test_trace_update_log(self):
        print("[6/8] trace.update_log 带 audit；无 handler → None", flush=True)
        from algo.api.trace import TraceAPI
        api = TraceAPI.__new__(TraceAPI)
        api.p = SimpleNamespace(handler=None)
        self.assertIsNone(api.update_log()["audit"])
        g = RegressGate()
        h = _handler(g)
        h.feedback_log = []
        g.record_audit(_fake_pipe([]), "reflow", [np.float32(.95)], np.float64(0.0))
        api.p = SimpleNamespace(handler=h)
        out = api.update_log()
        self.assertEqual(out["audit"]["counts"]["accepted"], 1)
        json.dumps(out)


class PersistAndRouterTest(unittest.TestCase):
    def test_persist_roundtrip_and_legacy(self):
        print("[7/8] persist：regress_gate 审计/基准分往返 + 旧快照兼容 + cde_report.json", flush=True)
        from algo.common.safe_pickle import register_pickle_root
        from algo.decide import Decider
        from algo.eval.offline import Pipeline
        from algo.persist import load_snapshot, save_snapshot
        from algo.slots.shead import SheadHead

        def slot():
            return SimpleNamespace(name="shead", head=SheadHead(4, hidden=8, dropout=.3),
                                   in_dim=4, device="cpu", chan_stats=False, patch_stats=False)
        cfg = {"seed": 1, "backbone": {"enabled": False}, "tiling": {}}
        p = Pipeline(cfg, None, [slot()])
        self.addCleanup(p._cpu_exec.shutdown)
        p.weights, p.slot_sanity = {"shead": 1.}, {}
        p.decider = Decider(.9, .7)
        p.router_gate = p.consensus_gate = None
        p.handler = FeedbackHandler(p, {"height_recal": False})
        p.cde_report = {"router_admission": {"decision": "trace_only"}}
        g = p.handler.gate
        p._anchor_items = [{"path": "a", "feats": None}]
        g.anchor_paths, g.base_mean, g.base_std = ["a"], .1, 1.
        g.base_scores = [.1, .2]
        g.record_audit(p, "reflow", [.3], .2)
        p.handler._audit_decide("rolled_back", "gate_drift")
        with tempfile.TemporaryDirectory() as tmp:
            register_pickle_root(tmp)
            save_snapshot(p, tmp, version=1)
            with open(Path(tmp) / "cde_report.json", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["router_admission"]["decision"], "trace_only")

            def load():
                with patch("algo.factory.build", return_value=(None, [slot()])):
                    r = load_snapshot(tmp, "cpu")
                self.addCleanup(r._cpu_exec.shutdown)
                return r
            r = load()
            rg = r.handler.gate
            self.assertEqual(rg.base_scores, [.1, .2])
            self.assertEqual(rg._seq, 1)
            self.assertEqual(rg.audit[0]["decision"], "rolled_back")
            self.assertIs(rg.context_fn.__self__, r.handler)
            self.assertEqual(r.cde_report["router_admission"]["decision"], "trace_only")
            gp = next(Path(tmp).rglob("regress_gate.json"))
            d = json.loads(gp.read_text(encoding="utf-8"))
            for k in ("audit", "audit_seq", "base_scores"):
                d.pop(k)
            gp.write_text(json.dumps(d), encoding="utf-8")
            (Path(tmp) / "cde_report.json").unlink()
            from algo.persist import _artifact_hashes   # 模拟旧快照：重写清单
            mp = Path(tmp) / "meta.json"
            meta = json.loads(mp.read_text(encoding="utf-8"))
            meta["artifact_hashes"] = _artifact_hashes(tmp)
            mp.write_text(json.dumps(meta), encoding="utf-8")
            r2 = load()
            self.assertEqual((r2.handler.gate.audit, r2.handler.gate.base_scores,
                              r2.handler.gate._seq), ([], [], 0))
            self.assertIsNone(r2.cde_report)
            self.assertEqual(r2.handler.update_audit_report()["counts"]["accepted"], 0)

    def test_retrain_router_dims_and_effective_weights(self):
        print("[8/8] _retrain_router 维度签名 + effective_weights/weights_semantics", flush=True)
        from algo.eval.offline import Pipeline
        from algo.fusion.router import LinearRouter
        fake = SimpleNamespace(router=LinearRouter(2), router_slot_names=["a", "b"],
                               _score_slots=["a", "b"], _anchor_items=[{"path": "n"}],
                               cfg={"seed": 1}, _router_snapshot=None,
                               _get_item=lambda p: p,
                               _record=lambda it: {"slot_scores": {"a": .1, "b": .2, "open": 9}})
        diff = [{"slots": {"a": .9, "b": .8, "open": 1}}] * 2
        fake.router_slot_names = ["a", "b", "open"]
        self.assertFalse(Pipeline._retrain_router(fake, diff, epochs=1))   # 维度不符
        fake.router_slot_names = ["a", "b"]
        self.assertFalse(Pipeline._retrain_router(fake, [{"slots": {"a": 1}}] * 2, epochs=1))
        fake._anchor_items = [{"path": "n1"}, {"path": "n2"}]
        self.assertTrue(Pipeline._retrain_router(fake, diff, epochs=1))
        self.assertIsNotNone(fake._router_snapshot)

        from backend.engine import DetectionEngine
        eng = DetectionEngine.__new__(DetectionEngine)
        pipe = SimpleNamespace(weights={"a": .5, "b": .5}, decider=None,
                               router_gate={"decision": "trace_only"})
        rec = {"fused": .3, "slot_scores": {"a": .9, "b": .1},
               "decision_trace": [{"stage": "fusion", "effective_weights": {"a": 1.0, "b": 0.0},
                                   "router_candidate_weights": {"a": .2, "b": .8}}]}
        r = eng._to_result(pipe, rec, 1.0)
        self.assertEqual(r.weights, {"a": 1.0, "b": 0.0})
        self.assertFalse(r.extra["candidate_router"]["applied"])
        self.assertEqual(r.extra["candidate_router"]["candidate_weights"], {"a": .2, "b": .8})
        self.assertEqual(set(r.extra["weights_semantics"]), set(cr.WEIGHTS_SEMANTICS))
        rec.pop("decision_trace")
        self.assertEqual(eng._to_result(pipe, rec, 1.0).weights, {"a": .5, "b": .5})


class SlotHealthRouteTest(unittest.TestCase):
    def test_route_roles_and_disabled(self):
        print("[+] slot-health API：engineer 可读 / operator 403 / 未准备 enabled=False", flush=True)
        from backend.api import routes_models
        from backend.engine import DetectionEngine
        eng = DetectionEngine.__new__(DetectionEngine)
        h = _handler(RegressGate())
        h.p = _fake_pipe([])
        h.gate.record_audit(h.p, "reflow", [.95], 0.0)
        pipe = SimpleNamespace(slots={"sem": 1}, open_det=None, calibrators={"sem": 1},
                               slot_sanity={"sem": np.float64(.8)}, _cal_normal_hist={},
                               weights={"sem": 1.0}, handler=h,
                               cde_report={"domain_shift": {"status": "insufficient"}})
        eng._ensure_loaded = lambda c: None
        eng._pipe_lock = lambda c: contextlib.nullcontext()
        eng._pipe_entry = lambda c: (pipe, 2)
        app = FastAPI()

        @app.middleware("http")
        async def _role(request: Request, call_next):
            request.state.role = request.headers.get("X-Role", "admin")
            return await call_next(request)
        app.include_router(routes_models.router, prefix="/api")
        c = TestClient(app)
        with patch("backend.engine.get_engine", return_value=eng):
            self.assertEqual(c.get("/api/models/c/slot-health",
                                   headers={"X-Role": "operator"}).status_code, 403)
            d = c.get("/api/models/c/slot-health", headers={"X-Role": "engineer"}).json()
            self.assertTrue(d["enabled"])
            self.assertEqual(d["version"], "v2")
            self.assertEqual(d["slots"]["sem"]["status"], "ok")
            self.assertEqual(d["domain_shift"]["status"], "insufficient")
            self.assertIsNone(d["router_admission"])
            self.assertEqual(d["update_audit"]["counts"]["accepted"], 1)
            eng._ensure_loaded = lambda c: (_ for _ in ()).throw(RuntimeError("none"))
            self.assertEqual(c.get("/api/models/c/slot-health").json(),
                             {"category": "c", "enabled": False})


class MechanismGainTest(unittest.TestCase):
    """AC-037 分机制收益对照。"""

    def _sets(self, seed=0):
        r = np.random.default_rng(seed)
        neg = r.normal(0.3, 0.1, 60)
        return neg, r

    def test_variants_ablation_verdicts(self):
        print("[+] AC-037 variants/ablations：gain/harm/inconclusive + 分组副作用", flush=True)
        neg, r = self._sets()
        pos_good = r.normal(0.8, 0.1, 30)
        pos_bad = r.normal(0.4, 0.1, 30)
        variants = {"patchcore": {"neg": neg, "pos": pos_bad},
                    "current": {"neg": neg, "pos": pos_good}}
        ab = {"memory": {"on": {"neg": neg, "pos": pos_good}, "off": {"neg": neg, "pos": pos_bad}},
              "open": {"on": {"neg": neg, "pos": pos_bad}, "off": {"neg": neg, "pos": pos_good}},
              "rollback": {"on": {"neg": neg, "pos": pos_good}, "off": {"neg": neg, "pos": pos_good}}}
        groups = {"lineA": {"neg_idx": list(range(30)), "pos_idx": list(range(15))},
                  "lineB": {"neg_idx": list(range(30, 60)), "pos_idx": list(range(15, 30))}}
        rep = cr.mechanism_gain_report(variants, 0.6, ablations=ab, groups=groups, n_boot=100)
        self.assertEqual(rep["missing_variants"], ["fixed_fusion", "new_head"])
        self.assertGreater(rep["variants"]["patchcore"]["current_minus_this"], 0)
        self.assertGreater(rep["variants"]["patchcore"]["delta_ci95"][0], 0)
        self.assertEqual(rep["mechanisms"]["memory"]["verdict"], "gain")
        self.assertEqual(rep["mechanisms"]["open"]["verdict"], "harm")
        self.assertEqual(rep["mechanisms"]["rollback"]["verdict"], "inconclusive")
        self.assertGreater(rep["mechanisms"]["memory"]["side_effect"]["recall_delta"], 0)
        self.assertEqual(set(rep["mechanisms"]["memory"]["side_effect"]["by_group"]), {"lineA", "lineB"})
        self.assertFalse(rep["disclosure"]["below_patchcore"])
        self.assertIsNone(rep["disclosure"]["note"])
        self.assertEqual(rep["facts"]["multi_head"]["status"], "implemented_limited")
        json.dumps(rep)

    def test_disclosure_and_guard(self):
        print("[+] AC-037 低于 PatchCore 且耗缺陷样本时披露；缺 current 报错", flush=True)
        neg, r = self._sets(1)
        good, bad = r.normal(0.8, 0.1, 20), r.normal(0.4, 0.1, 20)
        rep = cr.mechanism_gain_report({"patchcore": {"neg": neg, "pos": good},
                                        "current": {"neg": neg, "pos": bad}}, 0.6,
                                       n_defect_used=12, n_boot=50)
        d = rep["disclosure"]
        self.assertTrue(d["below_patchcore"])
        self.assertEqual(d["current_defects_used"], 12)
        self.assertIsNotNone(d["note"])
        self.assertFalse(d["patchcore_uses_defects"])
        # 长度不同 → 不做配对 CI
        rep2 = cr.mechanism_gain_report({"current": {"neg": neg, "pos": bad},
                                         "new_head": {"neg": neg[:10], "pos": bad}}, 0.6, n_boot=20)
        self.assertNotIn("delta_ci95", rep2["variants"]["new_head"])
        self.assertIsNone(rep2["disclosure"]["below_patchcore"])
        with self.assertRaises(ValueError):
            cr.mechanism_gain_report({"patchcore": {"neg": neg, "pos": good}}, 0.6)


if __name__ == "__main__":
    unittest.main()
