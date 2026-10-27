"""标准库/TEMP 前门禁验证；不运行生产 prepare、不写生产存储。"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from algo.feature_revision import FeatureSetRevision, PARITY_CHECKS, digest, production_hashes


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aoi_revision_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = FeatureSetRevision(self.root / "registry.json")
        self.manifest = {
            "schema_version": 1, "groups": [{"key": "a", "dim": 2,
            "slice": [0, 2], "params": {"bins": 2}}], "dimension": 2,
            "preprocess": {"max_side": 256}, "implementation_hashes": {"test": "a" * 64},
            "seed": 42, "dataset_split": {"test": [{"sha256": "b" * 64, "label": 0}]},
            "model_snapshot": {"category": "temp", "version": 1, "artifact_hashes": {"bank": "c" * 64}}}
        self.rid = self.store.register(self.manifest)
        self.evidence = {"revision_id": self.rid, "eval_run_id": "test-run", "status": "completed",
                         "independent": True, "acceptance": "passed", "auroc": 0.9,
                         **{k: deepcopy(self.manifest[k]) for k in ("seed", "dataset_split", "model_snapshot")}}
        self.parity = {**self.evidence, "status": "passed", "checks": dict.fromkeys(PARITY_CHECKS, True),
                       "baseline_implementation_hashes": production_hashes(), "protocol_hash": "d" * 64,
                       "numeric": {"n_samples": 3, "max_abs_diff": 0.0, "atol": 1e-6}}

    def save(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def evaluated(self):
        self.store.transition(self.rid, "evaluated", actor="test", evidence_path=self.save("eval.json", self.evidence))

    def approved(self):
        self.evaluated()
        self.store.transition(self.rid, "approved", actor="test")

    def test_identity_immutability_and_validation(self):
        self.assertEqual(self.rid, self.store.register(deepcopy(self.manifest)))
        self.manifest["groups"][0]["params"]["bins"] = 3
        self.assertNotEqual(self.rid, self.store.register(self.manifest))
        view = self.store.list()
        view[self.rid]["manifest"]["seed"] = 0
        self.assertEqual(self.store.list()[self.rid]["manifest"]["seed"], 42)
        for key in ("seed", "dimension", "schema_version"):
            bad = deepcopy(self.manifest)
            bad[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.register(bad)
        bad = deepcopy(self.manifest)
        bad["groups"][0]["slice"] = [1, 3]
        with self.assertRaises(ValueError):
            self.store.register(bad)
        rows = self.store.list()
        rows[self.rid]["manifest"]["seed"] = 0
        self.save("registry.json", rows)
        with self.assertRaises(ValueError):
            self.store.list()

    def test_full_chain_and_model_binding(self):
        before = production_hashes()
        self.approved()
        parity_path = self.save("parity.json", self.parity)
        self.store.transition(self.rid, "active", actor="test", parity_path=parity_path)
        self.store.check_activation(self.rid, self.manifest["model_snapshot"])
        with self.assertRaises(ValueError):
            self.store.check_activation(self.rid, {"unknown": True})
        self.assertEqual(before, production_hashes())
        self.assertFalse((self.root / "current.json").exists())
        self.store.transition(self.rid, "retired", actor="test")
        with self.assertRaises(ValueError):
            self.store.transition(self.rid, "active", actor="test", parity_path=parity_path)
        self.assertEqual(self.store.list()[self.rid]["state"], "retired")

    def test_no_jump_missing_evidence_and_state_tamper(self):
        for target in ("approved", "active", "retired"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.store.transition(self.rid, target, actor="test")
        with self.assertRaises(ValueError):
            self.store.transition(self.rid, "evaluated", actor="test")
        rows = self.store.list()
        rows[self.rid]["state"] = "active"
        self.save("registry.json", rows)
        with self.assertRaises(ValueError):
            self.store.list()

    def test_evaluation_lineage(self):
        for key in ("revision_id", "seed", "dataset_split", "model_snapshot", "status", "eval_run_id"):
            bad = deepcopy(self.evidence)
            bad[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.transition(self.rid, "evaluated", actor="test",
                                      evidence_path=self.save("bad.json", bad))
        self.assertEqual(self.store.list()[self.rid]["state"], "draft")

    def test_independent_evaluation_and_frozen_evidence(self):
        self.evidence.update(independent=False, acceptance="unavailable")
        self.evaluated()
        with self.assertRaises(ValueError):
            self.store.transition(self.rid, "approved", actor="test")
        independent = {**self.evidence, "independent": True, "acceptance": "passed"}
        for auc in (None, True, float("nan"), float("inf"), -1, 2):
            with self.subTest(auc=auc), self.assertRaises(ValueError):
                self.store.transition(self.rid, "approved", actor="test",
                    evidence_path=self.save("independent.json", {**independent, "auroc": auc}))
        self.store.transition(self.rid, "approved", actor="test",
                              evidence_path=self.save("independent.json", independent))
        self.save("eval.json", {**self.evidence, "auroc": 0.1})
        with self.assertRaises(ValueError):
            self.store.transition(self.rid, "active", actor="test", parity_path=self.save("p.json", self.parity))

    def test_parity_fail_closed(self):
        self.approved()
        variants = []
        for key in PARITY_CHECKS:
            bad = deepcopy(self.parity)
            bad["checks"][key] = False
            variants.append(bad)
        for key, value in (("n_samples", True), ("n_samples", 0), ("max_abs_diff", True),
                           ("max_abs_diff", float("nan")), ("max_abs_diff", 1), ("atol", -1)):
            bad = deepcopy(self.parity)
            bad["numeric"][key] = value
            variants.append(bad)
        for key in ("baseline_implementation_hashes", "protocol_hash"):
            bad = deepcopy(self.parity)
            bad[key] = "unknown"
            variants.append(bad)
        for i, bad in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(ValueError):
                self.store.transition(self.rid, "active", actor="test", parity_path=self.save("bad.json", bad))
        self.assertEqual(self.store.list()[self.rid]["state"], "approved")

    def test_active_evidence_change_and_lock(self):
        self.approved()
        path = self.save("p.json", self.parity)
        self.store.transition(self.rid, "active", actor="test", parity_path=path)
        self.save("p.json", {**self.parity, "numeric": {"n_samples": 3, "max_abs_diff": 0, "atol": 0}})
        with self.assertRaises(ValueError):
            self.store.check_activation(self.rid, self.manifest["model_snapshot"])
        self.save("registry.json.lock", {})
        with self.assertRaises(ValueError):
            self.store.register(self.manifest)
        self.assertFalse(list(self.root.glob(".revision_*")))

    def test_production_and_frontdoor_contract_static(self):
        core = Path(__file__).resolve().parents[2]
        trad = (core / "algo/slots/trad.py").read_text(encoding="utf-8")
        self.assertNotIn("AOI_feature", trad)
        self.assertNotIn("feature_revision", trad)
        self.assertIn("np.random.seed(0)", trad)
        routes = (core / "backend/api/routes_models.py").read_text(encoding="utf-8")
        tree = ast.parse(routes)
        activation = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "api_activate_model")
        source = ast.get_source_segment(routes, activation)
        self.assertLess(source.index("check_activation"), source.index("if gate:"))
        self.assertIn('"force_allowed": False', source)
        ui = (core / "ui/pages/model_page.py").read_text(encoding="utf-8")
        self.assertIn('detail.get("force_allowed") is False', ui)

    def test_detection_stamp_lineage(self):
        self.approved()
        parity_path = self.save("parity.json", self.parity)
        self.store.transition(self.rid, "active", actor="test", parity_path=parity_path)
        self.store.bind_production(
            self.rid,
            actor="test",
            model_snapshot=self.manifest["model_snapshot"],
        )
        stamp = self.store.detection_stamp(self.manifest["model_snapshot"])
        self.assertEqual(stamp["status"], "bound")
        self.assertEqual(stamp["feature_revision"], self.rid)
        self.assertEqual(stamp["model_snapshot"], self.manifest["model_snapshot"])
        self.assertEqual(stamp["lineage"], [self.rid])
        self.assertEqual(self.store.detection_stamp({"unknown": True})["status"], "snapshot_mismatch")
        self.assertEqual(
            FeatureSetRevision(self.root / "other.json").detection_stamp()["status"],
            "unbound",
        )


if __name__ == "__main__":
    print("[进度] TEMP revision 状态、血缘、等价与生产不变契约验证", flush=True)
    unittest.main(verbosity=2)
