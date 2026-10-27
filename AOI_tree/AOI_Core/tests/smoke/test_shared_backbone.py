"""冻结 DINO 主干进程级共享：同实例复用、RNG 轨迹与重建一致、特征不变。"""
import unittest

import numpy as np
import torch

from algo.backbone import dino


class SharedBackboneTest(unittest.TestCase):
    def setUp(self):
        dino._SHARED_MODELS.clear()

    def tearDown(self):
        dino._SHARED_MODELS.clear()

    def test_rng_trajectory_matches_fresh_build(self):
        print("[1/3] fresh build reference", flush=True)
        torch.manual_seed(42)
        a = dino.FrozenDINO(device="cpu")
        ref = torch.rand(8)
        print("[2/3] cache hit after same seed", flush=True)
        torch.manual_seed(42)
        b = dino.FrozenDINO(device="cpu")
        self.assertIs(a.model, b.model)
        self.assertTrue(torch.equal(torch.rand(8), ref))
        print("[3/3] unseen RNG state rebuilds mapping", flush=True)
        torch.manual_seed(7)
        expected_state = None
        dino._SHARED_MODELS.clear()
        torch.manual_seed(7)
        dino.FrozenDINO(device="cpu")
        expected_state = torch.rand(4)
        torch.manual_seed(42)
        dino.FrozenDINO(device="cpu")      # 新映射 42
        torch.manual_seed(7)
        dino.FrozenDINO(device="cpu")      # 命中映射 7
        self.assertTrue(torch.equal(torch.rand(4), expected_state))

    def test_features_identical_between_instances(self):
        torch.manual_seed(42)
        a = dino.FrozenDINO(device="cpu")
        torch.manual_seed(42)
        b = dino.FrozenDINO(device="cpu")
        img = np.random.default_rng(0).integers(0, 255, (448, 448, 3), dtype=np.uint8)
        self.assertTrue(torch.equal(a.extract_tiles([img]), b.extract_tiles([img])))
        self.assertFalse(any(p.requires_grad for p in b.parameters()))


if __name__ == "__main__":
    unittest.main()
