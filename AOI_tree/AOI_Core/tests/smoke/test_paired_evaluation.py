"""AC-049 最小配对评价 smoke：纯预测记录，不访问生产存储。"""
import json
import unittest
from copy import deepcopy

from backend.evaluation import paired_prediction_delta


class PairedEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.a = [
            {"image_id": "n1", "label": 0, "score": 0.1, "decision": "normal"},
            {"image_id": "n2", "label": 0, "score": 0.8, "decision": "gray"},
            {"image_id": "p1", "label": 1, "score": 0.7, "decision": "normal"},
            {"image_id": "p2", "label": 1, "score": 0.9, "decision": "anomaly"},
        ]
        self.b = deepcopy(self.a)
        self.b[1].update(score=0.2, decision="normal")
        self.b[2].update(score=0.8, decision="anomaly")

    def compare(self, a=None, b=None, **kwargs):
        result = paired_prediction_delta(self.a if a is None else a,
                                         self.b if b is None else b,
                                         seed_a=42, seed_b=42, **kwargs)
        json.dumps(result, allow_nan=False)
        return result

    def test_reordered_pairs_and_exact_deltas(self):
        print("[AC-049] 按 ID 配对/工作决策/gray 口径/输入不变", flush=True)
        original = deepcopy(self.a)
        r = self.compare(b=list(reversed(self.b)))
        self.assertEqual(r["status"], "available")
        self.assertEqual(r["n_pairs"], 4)
        for key, delta in (("auroc", 0.25), ("fpr", -0.5), ("recall", 0.5), ("gray", -0.25)):
            self.assertAlmostEqual(r["metrics"][key]["delta"], delta)
        self.assertEqual(self.a, original)
        self.assertEqual(r["uncertainty"]["status"], "unavailable")
        identity = self.compare(b=self.a)
        self.assertTrue(all(m["delta"] == 0 for m in identity["metrics"].values()))

    def test_fail_closed_pairing(self):
        print("[AC-049] 缺种子/种子不一致/缺 ID/重复/缺样本/标签冲突拒绝配对", flush=True)
        changed_label = deepcopy(self.b)
        changed_label[0]["label"] = 1
        missing_id = deepcopy(self.b)
        missing_id[0].pop("image_id")
        for b in (self.b[:-1], self.b + [self.b[0]], changed_label, missing_id, []):
            with self.subTest(b=b):
                r = self.compare(b=b)
                self.assertEqual(r["status"], "unavailable")
                self.assertTrue(r["gaps"])
                self.assertTrue(all(m["delta"] is None for m in r["metrics"].values()))
        for seeds in ({}, {"seed_a": 42, "seed_b": 7}, {"seed_a": float("nan"), "seed_b": 42}):
            r = paired_prediction_delta(self.a, self.b, **seeds)
            self.assertEqual(r["status"], "unavailable")
            json.dumps(r, allow_nan=False)

    def test_nonfinite_and_missing_decisions_are_not_fabricated(self):
        print("[AC-049] NaN/Inf/缺分数不虚构 AUROC，二值预测不虚构 gray", flush=True)
        for score in (float("nan"), float("inf"), None):
            b = deepcopy(self.b)
            b[0]["score"] = score
            r = self.compare(b=b)
            self.assertEqual(r["status"], "partial")
            self.assertIsNone(r["metrics"]["auroc"]["delta"])
            self.assertEqual(r["metrics"]["fpr"]["delta"], -0.5)
        b = deepcopy(self.b)
        for row in b:
            row["pred"] = int(row.pop("decision") != "normal")
        r = self.compare(b=b)
        self.assertIsNone(r["metrics"]["gray"]["delta"])
        self.assertEqual(r["metrics"]["recall"]["delta"], 0.5)
        b[0]["pred"] = float("nan")
        self.assertIsNone(self.compare(b=b)["metrics"]["fpr"]["delta"])

    def test_empty_and_single_class(self):
        print("[AC-049] 空集/单样本/单类别安全，未定义指标为 unavailable", flush=True)
        self.assertEqual(self.compare(a=[], b=[])["status"], "unavailable")
        for row, undefined, defined in ((self.a[0], "recall", "fpr"), (self.a[2], "fpr", "recall")):
            r = self.compare(a=[row], b=[row])
            self.assertEqual(r["status"], "partial")
            self.assertIsNone(r["metrics"]["auroc"]["delta"])
            self.assertIsNone(r["metrics"][undefined]["delta"])
            self.assertEqual(r["metrics"][defined]["delta"], 0)
            self.assertEqual(r["uncertainty"]["status"], "unavailable")

    def test_protocol_audit_record_aliases(self):
        rows = [{"path": "n.png", "label": 0, "fused": 0.1, "decision": "normal"},
                {"path": "p.png", "label": 1, "fused": 0.9, "decision": "gray"}]
        r = self.compare(a=rows, b=rows[::-1], id_key="path")
        self.assertEqual(r["status"], "available")
        self.assertEqual(r["metrics"]["gray"]["baseline"], 0.5)


if __name__ == "__main__":
    unittest.main()
