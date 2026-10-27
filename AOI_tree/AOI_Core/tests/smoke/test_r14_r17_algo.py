"""R14-R17 小型内存测试，不加载模型或访问生产资产。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from algo.common.image_identity import image_signature, near_duplicate
from algo.fusion.fixed import fuse, effective_weights
from algo.slots.base import SlotOutput
from algo.ssocl.banks import NormalBank
from algo.ssocl.feedback import FeedbackHandler


class ImageIdentityTests(unittest.TestCase):
    def test_reencoded_image_is_near_duplicate(self):
        y, x = np.indices((96, 96))
        image = np.stack((x * 2, y * 2, (x + y)), axis=-1).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.assertTrue(ok)
        copy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertTrue(near_duplicate(image_signature(image), image_signature(copy)))
        other = np.random.default_rng(2).integers(0, 256, image.shape, dtype=np.uint8)
        self.assertFalse(near_duplicate(image_signature(image), image_signature(other)))

    def test_feedback_skips_before_scoring_or_self_matching(self):
        image = np.full((32, 32, 3), 120, np.uint8)
        handler = FeedbackHandler.__new__(FeedbackHandler)
        handler.normal_bank = NormalBank()
        handler.normal_bank.feedback_identities = [
            {"path": "original.png", "signature": image_signature(image)}]
        handler.p = SimpleNamespace(_get_item=Mock(return_value={"img": image, "feats": None}),
                                    _record=Mock(side_effect=AssertionError("must not score")))
        result = handler.feedback("different.jpg", "correct", label=0)
        self.assertEqual(result["action"], "duplicate_feedback_skipped")
        handler.p._record.assert_not_called()


class IndependentClusterTests(unittest.TestCase):
    def setUp(self):
        self.bank = NormalBank(cluster_k=3, sim_thresh=-1)
        self.bank.seed_core(self.feats(0))

    @staticmethod
    def feats(seed):
        rng = np.random.default_rng(seed)
        return (np.ones((1, 8, 3, 3)) + rng.normal(0, 0.02, (1, 8, 3, 3))).astype(np.float32)

    def test_same_id_and_same_content_do_not_count_twice(self):
        self.assertEqual(self.bank.reflow_candidate(self.feats(1), image_id="a"), "pending")
        self.bank.reflow_candidate(self.feats(2), image_id="a")
        self.bank.reflow_candidate(self.feats(1), image_id="renamed")
        self.assertEqual(list(self.bank.cluster_hits.values()), [1])
        self.assertEqual(self.bank.reflow_candidate(self.feats(3), image_id="b"), "pending")
        self.assertEqual(self.bank.reflow_candidate(self.feats(4), image_id="c"), "accepted")

    def test_near_duplicate_and_snapshot_rollback(self):
        sig = image_signature(np.full((32, 32, 3), 100, np.uint8))
        self.bank.reflow_candidate(self.feats(1), image_id="a", signature=sig)
        self.bank.snapshot()
        self.bank.reflow_candidate(self.feats(2), image_id="copy", signature=sig)
        self.assertEqual(list(self.bank.cluster_hits.values()), [1])
        self.bank.reflow_candidate(self.feats(3), image_id="b")
        self.bank.rollback()
        self.assertEqual(len(self.bank._cluster_identities), 1)
        self.assertEqual(list(self.bank.cluster_hits.values()), [1])


class FusionConfidenceTests(unittest.TestCase):
    def test_legacy_and_disabled_slot(self):
        self.assertAlmostEqual(fuse({"a": .2, "b": .8}, {"a": .25, "b": .75}), .65)
        self.assertAlmostEqual(fuse({"a": .2}, {"a": .25, "disabled": .75}), .2)

    def test_confidence_and_zero_confidence(self):
        scores = {"a": .2, "b": .8}
        self.assertAlmostEqual(fuse(scores, confidences={"a": 1, "b": .5}), .4)
        self.assertAlmostEqual(fuse(scores, confidences={"a": 1, "b": 0}), .2)
        self.assertAlmostEqual(sum(effective_weights(scores).values()), 1)

    def test_invalid_or_no_active_weight_fails_explicitly(self):
        for confidence in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                fuse({"a": .3}, confidences={"a": confidence})
        with self.assertRaises(ValueError):
            fuse({"a": .3}, {"a": 0})
        with self.assertRaises(ValueError):
            fuse({})

    def test_record_consumes_slot_output_confidence(self):
        from algo.eval.offline import Pipeline
        pipe = Pipeline.__new__(Pipeline)
        pipe.slots = {n: SimpleNamespace(needs_dino=True) for n in ("a", "b")}
        pipe._score_slots = ["a", "b"]
        pipe.calibrators = {n: SimpleNamespace(transform=lambda x: x) for n in pipe.slots}
        pipe.weights = {"a": .5, "b": .5, "disabled": .8}
        pipe.router = None
        pipe.open_det = None
        pipe._slot_image_score = lambda slot, item: (
            SlotOutput(.2, np.array([.2]), confidence=1) if slot is pipe.slots["a"]
            else SlotOutput(.8, np.array([.8]), confidence=.5))
        result = pipe._record({"path": "memory", "feats": None})
        self.assertAlmostEqual(result["fused"], .4)
        self.assertAlmostEqual(result["decision_trace"][0]["effective_weights"]["a"], 2 / 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
