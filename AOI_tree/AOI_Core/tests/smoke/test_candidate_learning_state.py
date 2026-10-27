"""TEMP真实小型CPU头/Handler续训；不使用生产资产，不代表生产精度验收。"""
import copy
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from algo.common.safe_pickle import register_pickle_root
from algo.decide import Decider
from algo.eval.offline import Pipeline
from algo.persist import save_snapshot, load_snapshot, _artifact_hashes
from algo.slots.shead import SheadHead
from algo.ssocl.feedback import FeedbackHandler
from algo.ssocl.incubate import HeadManager
from algo.ssocl.learning_state import capture, candidate_random, restore


class LearningStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_learning_state_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        register_pickle_root(self.tmp.name)
        self.cfg = {"seed": 13, "backbone": {"enabled": False}, "tiling": {}}
        self.online = {"height_recal": False, "weight_min_samples": 999,
                       "head_ft": {"enabled": True, "min_pos": 1, "min_neg": 1,
                                   "epochs": 2, "anchor_n": 0, "lr": .001,
                                   "tol_up": 100, "tol_down": 100},
                       "disc_ft": {"enabled": False},
                       "ssocl_fix": {"pair_thresh": False, "labeled_per_class": 2}}
        self.p = self.make()
        self.addCleanup(self.p._cpu_exec.shutdown)
        self.p._candidate_mode = True
        self.h = self.p.handler
        self.h.gate.anchor_paths = ["anchor"]
        self.h.gate.base_mean, self.h.gate.base_std = 0., 1.
        self.p._anchor_items = [self.item("anchor")]
        self.p._train_normal_paths = ["anchor"]
        self.h._n_labeled_fb = 30
        self.h._normal_fb_raw["shead"] = [.1, .2]
        self.h._normal_fb_cal["shead"] = [.3, .4]
        self.h._hr_cooldown = {"shead": 12}
        self.h.normal_bank._observe_decision(True)
        self.h.normal_bank.seed_core(self.item("seed")["feats"])
        self.h.defect_bank.add(self.item("defect")["feats"])
        self.h._preq = [{"ss": {"shead": .7}, "boost": .1, "label": 1}]
        self.add_buffers(self.h)
        for i in range(8):
            self.h._add_labeled(self.item(str(i)), i % 2)
        self.p.head_mgr.last_sample_n = 3
        self.patch = patch("algo.factory.build", side_effect=lambda *a: (None, [self.slot()]))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    @staticmethod
    def slot():
        return SimpleNamespace(name="shead", head=SheadHead(4, hidden=8, dropout=.3),
                               in_dim=4, device="cpu", chan_stats=False, patch_stats=False)

    def make(self):
        p = Pipeline(self.cfg, None, [self.slot()])
        p.weights, p.slot_sanity = {"shead": 1.}, {}
        p.decider = Decider(.9, .7)
        p.router_gate = p.consensus_gate = None
        p.handler = FeedbackHandler(p, self.online)
        p.head_mgr = HeadManager({"min_samples": 999})
        return p

    @staticmethod
    def item(path):
        return {"path": path, "feats": torch.arange(64).float().reshape(1, 4, 4, 4) / 64,
                "img": np.zeros((4, 4, 3), dtype=np.uint8), "tiles": [(0, 0, 4, 4)],
                "tile_imgs": [np.zeros((4, 4, 3), dtype=np.uint8)]}

    @staticmethod
    def add_buffers(h):
        h.box_pos_feats = [torch.arange(32).float().reshape(8, 4) / 32]
        h.normal_fb_feats = [torch.arange(32).float().reshape(8, 4) / 64]
        h._head_ft_counts = {"pos": 8, "neg": 8}

    def load(self, directory):
        p = load_snapshot(str(directory), "cpu", candidate=True)
        self.addCleanup(p._cpu_exec.shutdown)
        return p

    def train(self, p):
        h = p.handler
        h._anchor_shead_mean = lambda: 0.
        p._predict_fused_only = lambda _: float(p.slots["shead"].head(torch.ones(1, 4)).item())
        losses, gradients = [], []
        backward, step = torch.Tensor.backward, torch.optim.Adam.step
        def capture_backward(t, *args, **kwargs):
            losses.append(t.detach().item())
            return backward(t, *args, **kwargs)
        def capture_step(opt, *args, **kwargs):
            gradients.append([w.grad.clone() for group in opt.param_groups for w in group["params"]])
            return step(opt, *args, **kwargs)
        rng = torch.get_rng_state().clone()
        with patch.object(torch.Tensor, "backward", capture_backward), patch.object(torch.optim.Adam, "step", capture_step):
            h._maybe_head_ft()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        return losses, gradients

    def test_continuous_save_load_next_optimizer_and_reservoir(self):
        original = self.p
        for round_id in range(3):
            print("[TEMP learning trajectory] round", round_id, flush=True)
            directory = self.root / ("v" + str(round_id))
            save_snapshot(original, str(directory), version=round_id)
            restored = self.load(directory)
            self.assertEqual(restored.head_mgr.last_sample_n, 3)
            self.assertEqual(capture(original)["handler"]["_hr_cooldown"], capture(restored)["handler"]["_hr_cooldown"])
            self.assertEqual(restored.handler.normal_bank.recent_decisions, [1])
            self.assertEqual(restored.handler.defect_bank.version, original.handler.defect_bank.version)
            self.assertTrue(torch.equal(restored.handler.defect_bank.samples[0][0], original.handler.defect_bank.samples[0][0]))
            for i in range(8, 20):
                for p in (original, restored):
                    p.handler._add_labeled(self.item(str(i)), i % 2)
            self.assertEqual([it["path"] for it, _ in original.handler._labeled_items],
                             [it["path"] for it, _ in restored.handler._labeled_items])
            left, right = self.train(original), self.train(restored)
            self.assertEqual(left[0], right[0])
            self.assertGreater(len(left[0]), 0)
            for lg, rg in zip(left[1], right[1]):
                for a, b in zip(lg, rg):
                    self.assertTrue(torch.equal(a, b))
            for a, b in zip(original.slots["shead"].head.parameters(), restored.slots["shead"].head.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(original.handler._head_ft_cooldown, restored.handler._head_ft_cooldown)
            self.assertEqual(original.handler.head_ft_log[-1]["signed"], restored.handler.head_ft_log[-1]["signed"])
            self.add_buffers(restored.handler)
            original = restored

    def test_real_feedback_continuation(self):
        self.p.handler.hf_min_pos = 999
        self.p.handler._learning_cfg["head_ft"]["min_pos"] = 999
        directory = self.root / "feedback"
        save_snapshot(self.p, str(directory), version=1)
        restored = self.load(directory)
        for p in (self.p, restored):
            p._ablate_reflow = True
            p._get_item = lambda path, feats=None: self.item(path)
            p._record = lambda item: {"fused": .2, "slot_scores": {"shead": .2},
                                     "raw_scores": {"shead": .2}}
            p._predict_fused_only = lambda item: .0
            p._predict_final_fused = lambda path: .0
        for i in range(6):
            print("[TEMP real feedback]", i, flush=True)
            a = self.p.handler.feedback(str(i), "correct", label=i % 2)
            b = restored.handler.feedback(str(i), "correct", label=i % 2)
            a.pop("t")
            b.pop("t")
            self.assertEqual(a, b)
            self.assertEqual(self.p.handler.weight_learner.n, restored.handler.weight_learner.n)
            self.assertEqual(self.p.handler._n_labeled_fb, restored.handler._n_labeled_fb)
            self.assertEqual(self.p.handler.defect_bank.version, restored.handler.defect_bank.version)
            self.assertEqual(self.p.handler._preq, restored.handler._preq)
            for left, right in zip(self.p.handler.defect_bank.samples, restored.handler.defect_bank.samples):
                self.assertTrue(torch.equal(left[0], right[0]))
        self.assertEqual([it["path"] for it, _ in self.p.handler._labeled_items],
                         [it["path"] for it, _ in restored.handler._labeled_items])

    def rewrite_meta(self, directory, change=None):
        file = directory / "meta.json"
        meta = json.loads(file.read_text(encoding="utf-8"))
        if change:
            change(meta)
        meta["artifact_hashes"] = _artifact_hashes(str(directory))
        file.write_text(json.dumps(meta), encoding="utf-8")

    def test_unknown_and_legacy_fail_closed(self):
        directory = self.root / "bad"
        save_snapshot(self.p, str(directory), version=1)
        self.rewrite_meta(directory, lambda m: m["learning_state"].update(version=999))
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.load(directory)
        self.rewrite_meta(directory, lambda m: m.pop("learning_state"))
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.load(directory)

    def test_missing_malicious_field_and_wrong_shape(self):
        for variant in ("missing", "field", "shape", "version", "method"):
            print("[TEMP learning validation]", variant, flush=True)
            directory = self.root / variant
            save_snapshot(self.p, str(directory), version=1)
            file = directory / "ssocl" / "learning_state.pkl"
            state = capture(self.p)
            if variant == "missing":
                state["handler"].pop("_preq")
            elif variant == "field":
                state["handler"]["p"] = "overwrite"
            elif variant == "shape":
                state["handler"]["box_pos_feats"] = [torch.ones(1, 99)]
            elif variant == "version":
                state["version"] = 99
            else:
                state["weight_learner"]["weights"] = "overwrite method"
            file.write_bytes(pickle.dumps(state))
            self.rewrite_meta(directory)
            with self.assertRaises(ValueError):
                self.load(directory)

    def test_missing_required_artifact_and_pickle_whitelist(self):
        from algo.common.safe_pickle import safe_pickle_load
        for artifact in ("learning_state.pkl", "normal_bank.pkl", "defect_bank.pkl", "head_mgr.pkl"):
            directory = self.root / artifact
            save_snapshot(self.p, str(directory), version=1)
            (directory / "ssocl" / artifact).unlink()
            self.rewrite_meta(directory)
            with self.assertRaises(ValueError):
                self.load(directory)
        with tempfile.TemporaryDirectory(prefix="untrusted_learning_") as outside:
            file = Path(outside) / "state.pkl"
            file.write_bytes(pickle.dumps({"version": 1}))
            with self.assertRaises(Exception):
                safe_pickle_load(str(file))

    def test_gate_reject_cooldown_resume(self):
        self.p.handler.hf_tol_up = -100
        self.p.handler._learning_cfg["head_ft"]["tol_up"] = -100
        directory = self.root / "reject"
        save_snapshot(self.p, str(directory), version=1)
        restored = self.load(directory)
        a, b = self.train(self.p), self.train(restored)
        self.assertEqual(a[0], b[0])
        self.assertEqual(restored.handler._head_ft_cooldown, 30)
        self.assertEqual(self.p.handler._head_ft_cooldown, 30)
        save_snapshot(restored, str(self.root / "cooldown"), version=2)
        reloaded = self.load(self.root / "cooldown")
        self.assertEqual(self.train(reloaded)[0], [])

    def test_global_rng_continuation_and_isolated_default(self):
        directory = self.root / "global_rng"
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        state = capture(self.p)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        restore(self.p, state, restore_global_rng=True)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(expected, actual)
        save_snapshot(self.p, str(directory), version=1)
        random.seed(401)
        np.random.seed(402)
        torch.manual_seed(403)
        before = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        restored = self.load(directory)
        after = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertNotEqual(before, after)

    def test_invalid_process_rng_rejected_before_restore(self):
        variants = {"python_rng": (3, (), None),
                    "numpy_rng": ("unknown", np.zeros(624, dtype=np.uint32), 0, 0, 0.),
                    "torch_global_rng": torch.zeros(1, dtype=torch.uint8)}
        for key, value in variants.items():
            for global_restore in (False, True):
                with self.subTest(key=key, global_restore=global_restore):
                    state = capture(self.p)
                    state[key] = value
                    state["handler"]["_n_labeled_fb"] = 999
                    python_before = random.getstate()
                    numpy_before = np.random.get_state()
                    torch_before = torch.get_rng_state().clone()
                    local_before = self.h._candidate_torch_gen.get_state().clone()
                    with self.assertRaisesRegex(ValueError, "RNG"):
                        restore(self.p, state, restore_global_rng=global_restore)
                    self.assertEqual(self.h._n_labeled_fb, 30)
                    self.assertEqual(python_before, random.getstate())
                    numpy_after = np.random.get_state()
                    self.assertEqual(numpy_before[0], numpy_after[0])
                    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
                    self.assertEqual(numpy_before[2:], numpy_after[2:])
                    self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
                    self.assertTrue(torch.equal(local_before, self.h._candidate_torch_gen.get_state()))

    def test_corrupt_process_rng_snapshot_fail_closed(self):
        directory = self.root / "corrupt_rng"
        save_snapshot(self.p, str(directory), version=1)
        state = capture(self.p)
        state["torch_global_rng"] = torch.zeros(1, dtype=torch.uint8)
        (directory / "ssocl" / "learning_state.pkl").write_bytes(pickle.dumps(state))
        self.rewrite_meta(directory)
        with self.assertRaisesRegex(ValueError, "RNG"):
            self.load(directory)

    def test_cuda_rng_device_count_fail_closed(self):
        state = capture(self.p)
        state["cuda_rng"] = []
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                restore(self.p, state)
        state["cuda_rng"] = [torch.zeros(1, dtype=torch.uint8)]
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                restore(self.p, state)

    def test_default_restore_does_not_rewind_process_rng(self):
        state = capture(self.p)
        random.random()
        np.random.rand()
        torch.rand(1)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        restore(self.p, state)
        self.assertEqual(python_before, random.getstate())
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_router_candidate_does_not_seed_global_rng(self):
        from algo.fusion.router import LinearRouter, train_router_on_matrix
        @candidate_random
        def train(h):
            train_router_on_matrix(LinearRouter(1), np.ones((2, 1)), np.zeros((2, 1)), epochs=1)
        rng = torch.get_rng_state().clone()
        train(self.h)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
