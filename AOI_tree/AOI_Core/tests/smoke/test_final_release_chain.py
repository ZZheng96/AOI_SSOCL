"""AC-067/073/074：EvalRun强门禁、allow_stale恢复授权、发布审计链（TEMP合成数据，预测替身）。"""
import json
import time
from pathlib import Path
from unittest.mock import patch

from tests.smoke import test_candidate_release as base
from backend.self_learning import candidate_release as cr
from backend.self_learning.candidate_release import CandidateReleaseError, digest, encode

N = 9
BUSINESS = {"max_fpr": 0.05, "min_recall": 0.95, "max_gray": 0.15, "old_recall_drop": 0.02}


def progress(i, name):
    print(f"[{i}/{N}] {name}", flush=True)


class FinalReleaseChainTest(base.CandidateReleaseTest):
    def setUp(self):
        super().setUp()
        p = patch.object(cr, "business_policy", lambda: dict(BUSINESS))
        p.start()
        self.addCleanup(p.stop)

    def stored_report(self):
        with self.release.connection() as c:
            return json.loads(c.execute("SELECT report FROM candidate_release").fetchone()[0])

    def store_report(self, report):
        report.pop("report_hash", None)
        report["report_hash"] = digest(report)
        with self.release.connection() as c:
            c.execute("UPDATE candidate_release SET report=?", (encode(report),))
        return report["report_hash"]

    def plain_snapshot(self, version):
        from algo.persist import save_snapshot
        save_snapshot(self.pipe, str(self.root / f"v{version}"), version=version)

    def test_policy_looser_than_business_rejected(self):
        progress(1, "policy宽于业务冻结阈值→拒绝")
        self.policy["max_fpr"] = 0.5
        with self.assertRaises(CandidateReleaseError) as ctx:
            self.evaluate()
        self.assertEqual(ctx.exception.reason_code, "POLICY_LOOSER_THAN_BUSINESS")

    def test_business_policy_changed_after_eval(self):
        progress(2, "验收后业务阈值变化→审批/激活拒绝")
        r = self.evaluate()
        self.assertEqual(r["eval_run"]["business_policy"], BUSINESS)
        changed = {**BUSINESS, "max_fpr": 0.04}
        with patch.object(cr, "business_policy", lambda: dict(changed)):
            with self.assertRaises(CandidateReleaseError) as ctx:
                self.release.decide("test", 2, True, r["report_hash"], "admin", "x")
            self.assertEqual(ctx.exception.reason_code, "EVAL_RUN_POLICY_CHANGED")
        self.release.decide("test", 2, True, r["report_hash"], "admin", "ok")
        with patch.object(cr, "business_policy", lambda: dict(changed)):
            with self.assertRaises(CandidateReleaseError) as ctx:
                self.engine.activate("test", 2)
            self.assertEqual(ctx.exception.reason_code, "EVAL_RUN_POLICY_CHANGED")
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_eval_run_binding_mismatch_and_missing(self):
        progress(3, "EvalRun绑定篡改/缺失→拒绝")
        self.evaluate()
        report = self.stored_report()
        report["eval_run"]["data_hash"] = "forged"
        h = self.store_report(report)
        with self.assertRaises(CandidateReleaseError) as ctx:
            self.release.decide("test", 2, True, h, "admin", "x")
        self.assertEqual(ctx.exception.reason_code, "EVAL_RUN_BINDING_MISMATCH")
        self.assertIn("data", ctx.exception.violations[0]["details"]["fields"])
        report = self.stored_report()
        report.pop("eval_run")
        h = self.store_report(report)
        with self.assertRaises(CandidateReleaseError) as ctx:
            self.release.decide("test", 2, True, h, "admin", "x")
        self.assertEqual(ctx.exception.reason_code, "EVAL_RUN_MISSING")

    def test_parent_stale_rejected(self):
        progress(4, "父版本过期→审批/激活拒绝")
        from algo.persist import set_current_version
        self.approve()
        self.plain_snapshot(3)
        set_current_version(str(self.root), 3)
        with self.assertRaisesRegex(ValueError, "父版本"):
            self.engine.activate("test", 2)
        r = self.stored_report()
        _, intent = self.release.intent("test", 2)
        with self.assertRaises(CandidateReleaseError) as ctx:
            self.release._check_eval_run(r, "test", 2, intent)
        self.assertEqual(ctx.exception.reason_code, "EVAL_RUN_PARENT_STALE")

    def synthetic(self, fp_in_new=0):
        from algo.eval.offline import Pipeline
        self.policy.update(BUSINESS)
        domains = {}
        for d in ("old", "new"):
            rows = []
            for label in (0, 1):
                for i in range(80):
                    p = self.root / f"syn_{d}_{label}_{i}.png"
                    flip = d == "new" and label == 0 and i < fp_in_new
                    p.write_bytes(bytes([2 if flip else label]))
                    rows.append({"id": len(rows), "path": str(p), "label": label})
            domains[d] = rows

        def predict(pipe, path):
            b = Path(path).read_bytes()[0]
            return {"fused": float(b), "decision": "anomaly" if b else "normal"}
        with patch.object(Pipeline, "predict", predict):
            return self.release.evaluate("test", 2, domains, self.policy, 99)

    def test_wilson_ci_gate_passes_clean_synthetic(self):
        progress(5, "合成标注集 80/类 0错→CI门禁通过")
        r = self.synthetic(0)
        self.assertEqual(r["acceptance"], "passed")
        for m in r["domains"].values():
            self.assertEqual(m["ci_gate"], "enforced")
            self.assertLessEqual(m["candidate"]["fpr_ci"][1], 0.05)

    def test_wilson_ci_gate_rejects_point_pass(self):
        progress(6, "合成标注集 1FP：点估计通过但CI上界超限→拒绝")
        r = self.synthetic(1)
        self.assertLessEqual(r["domains"]["new"]["candidate"]["fpr"], 0.05)
        self.assertEqual(r["acceptance"], "rejected")
        self.assertIn("POLICY_MAX_FPR_CI", [v["code"] for v in r["violations"]])
        with self.assertRaisesRegex(ValueError, "未通过"):
            self.release.decide("test", 2, True, r["report_hash"], "admin", "x")

    def test_audit_chain_same_transaction_and_tamper(self):
        progress(7, "审计链：evaluate→approve→activate，篡改可检出")
        self.approve()
        self.engine.activate("test", 2)
        chain = self.release.audit_chain("test")
        self.assertTrue(chain["verified"])
        actions = [e["action"] for e in chain["entries"]]
        for a in ("evaluate", "approve", "activate_prepared", "activate_committed"):
            self.assertIn(a, actions)
        approve = next(e for e in chain["entries"] if e["action"] == "approve")
        self.assertEqual(approve["actor"], "admin")
        self.assertTrue(approve["payload"]["eval_run_id"])
        with self.release.connection() as c:
            c.execute("UPDATE release_audit SET payload='{}' WHERE id=?", (approve["id"],))
        chain = self.release.audit_chain("test")
        self.assertFalse(chain["verified"])
        self.assertEqual(chain["broken_at"], approve["id"])

    def test_allow_stale_requires_bound_authorization(self):
        progress(8, "allow_stale：无/过期/越权授权拒绝，合法授权放行且入审计")
        self.plain_snapshot(3)
        auth = {"actor": "ops", "reason": "rollback drill", "ticket": "INC-1",
                "category": "test", "version": 3, "expires_at": time.time() + 60}
        cases = ((None, "STALE_AUTHORIZATION_REQUIRED"),
                 ({**auth, "ticket": ""}, "STALE_AUTHORIZATION_INVALID"),
                 ({**auth, "version": 1}, "STALE_AUTHORIZATION_SCOPE"),
                 ({**auth, "expires_at": time.time() - 1}, "STALE_AUTHORIZATION_EXPIRED"))
        for bad, code in cases:
            with self.assertRaises(CandidateReleaseError) as ctx:
                self.engine.activate("test", 3, allow_stale=True, recovery_authorization=bad)
            self.assertEqual(ctx.exception.reason_code, code)
            self.assertEqual(self.engine.current_version("test"), 1)
        self.engine.activate("test", 3, allow_stale=True, recovery_authorization=auth)
        self.assertEqual(self.engine.current_version("test"), 3)
        entry = next(e for e in self.release.audit_chain("test")["entries"]
                     if e["action"] == "snapshot_activate_prepared")
        self.assertEqual(entry["actor"], "ops")
        self.assertEqual(entry["payload"]["authorization"]["ticket"], "INC-1")
        self.assertTrue(entry["payload"]["authorization"]["authorization_hash"])

    def test_plain_snapshot_activation_ledger_crash_recover(self):
        progress(9, "普通快照激活走账本；指针后崩溃→prepared阻断→recover")
        import algo.persist as persist
        self.plain_snapshot(3)
        self.engine.activate("test", 3)
        with self.release.connection() as c:
            rows = c.execute("SELECT action,state FROM release_events WHERE action='snapshot_activate'").fetchall()
        self.assertEqual(rows, [("snapshot_activate", "committed")])
        real = persist.set_current_version

        def crash(*a, **k):
            real(*a, **k)
            raise RuntimeError("crash after pointer")
        with patch.object(persist, "set_current_version", crash):
            with self.assertRaisesRegex(RuntimeError, "crash"):
                self.engine.activate("test", 1)
        self.assertEqual(self.engine.current_version("test"), 1)
        with self.assertRaisesRegex(ValueError, "未完成发布"):
            self.engine.activate("test", 3)
        result = self.release.recover("test")
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["model_mirror"]["version"], 1)
        self.assert_mirror(1)
        with self.release.connection() as c:
            states = [r[0] for r in c.execute("SELECT state FROM release_events WHERE action='snapshot_activate' ORDER BY id")]
        self.assertEqual(states, ["committed", "committed"])
        actions = [e["action"] for e in self.release.audit_chain("test")["entries"]]
        self.assertIn("snapshot_activate_recovered_committed", actions)
        self.assertTrue(self.release.audit_chain()["verified"])


for _cls in (base.IsolatedCandidateTest, base.CandidateReleaseTest):
    for _name in list(_cls.__dict__):
        if _name.startswith("test_") and _name not in FinalReleaseChainTest.__dict__:
            setattr(FinalReleaseChainTest, _name, None)
del _cls, _name
