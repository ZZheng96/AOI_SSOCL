"""CPU-only numerical regression tests; no pipeline/model weights required."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "algo/ssocl/feedback.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
HANDLER = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "FeedbackHandler")
HEAD_SOURCE = (ROOT / "algo/slots/shead.py").read_text(encoding="utf-8")
HEAD_TREE = ast.parse(HEAD_SOURCE)
HEAD_GLOBALS = {"torch": torch, "nn": nn}
for node in HEAD_TREE.body:
    if isinstance(node, ast.ClassDef) and node.name in ("DeviationLoss", "SheadHead"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT), "exec"), HEAD_GLOBALS)


def method(name, legacy=False):
    node = next(n for n in HANDLER.body if isinstance(n, ast.FunctionDef) and n.name == name)
    source = ast.get_source_segment(SOURCE, node)
    if legacy:
        # Reconstruct the original per-epoch transfer/per-batch anchor materialization.
        source = source.replace("        pos = pos.to(slot.device)\n        neg = neg.to(slot.device)\n", "")
        source = source.replace("Xb = torch.cat([pos[pi], neg[ni]])", "Xb = torch.cat([pos[pi], neg[ni]]).to(slot.device)")
        if name == "_maybe_head_ft":
            start = source.index("        if anchor_feats:\n            anchor_feats = torch.cat")
            end = source.index("        # 均衡批采样", start)
            source = source[:start] + source[end:]
            source = source.replace(
                "                if anchor_feats is not None:\n",
                "                if anchor_feats:\n                    A = torch.cat(anchor_feats).to(slot.device)\n                    ref = torch.tensor(anchor_ref, device=slot.device)\n")
            source = source.replace("(slot.head(anchor_feats) - anchor_ref)", "(slot.head(A) - ref)")
    env = {"torch": torch, "np": np, "F": F, "time": time,
           "DeviationLoss": HEAD_GLOBALS["DeviationLoss"],
           "box_patch_feats": lambda feats, *_: feats}
    exec(compile(source, str(ROOT / "algo/ssocl/feedback.py"), "exec"), env)
    return env[name]


def fixture(kind, head, anchors, signed, auc_drop=False):
    gen = torch.Generator().manual_seed(12)
    pos = [torch.randn(20, 4, generator=gen)]
    neg = [torch.randn(20, 4, generator=gen)]
    slot = SimpleNamespace(head=head, device="cpu", chan_stats=False, patch_stats=False)
    h = SimpleNamespace(
        p=SimpleNamespace(slots={kind: slot}, cfg={"seed": 42},
                          _anchor_items=anchors, _predict_fused_only=lambda _: signed),
        gate=SimpleNamespace(anchor_paths=["a", "b"], base_mean=0., base_std=1.),
        head_ft_on=True, disc_ft_on=True, hf_min_pos=1, hf_min_neg=1,
        df_min_pos=1, df_min_neg=1, _n_labeled_fb=30,
        _head_ft_cooldown=0, _disc_ft_cooldown=0,
        box_pos_feats=copy.deepcopy(pos), normal_fb_feats=copy.deepcopy(neg),
        disc_pos_patch_feats=copy.deepcopy(pos), disc_neg_patch_feats=copy.deepcopy(neg),
        hf_opt="sgd", hf_lr=.001, df_lr=.001, hf_anchor_n=8,
        hf_anchor_lambda=.7, hf_epochs=3, df_epochs=3, df_batch=16,
        hf_tol_down=.2, df_tol_down=.2, hf_tol_up=.05,
        stats={"gate_reject": 0}, rollback_log=[], head_ft_log=[], disc_ft_log=[],
        _anchor_shead_mean=lambda: 0., _labeled_items=[("p", 1), ("n", 0)],
        pair_min_each=1)
    aucs = iter([.8, .7 if auc_drop else .8])
    h._labeled_auc = lambda: next(aucs)
    def restore():
        snapshot = getattr(h, "_head_ft_snapshot" if kind == "shead" else "_disc_ft_snapshot")
        with torch.no_grad():
            for w, saved in zip(head.parameters(), snapshot):
                w.copy_(saved)
    h._restore_head_ft = restore
    h._restore_disc_ft = restore
    return h


def run(h, kind, legacy):
    losses, grads, transfers, cats = [], [], [], []
    original_to, original_cat = torch.Tensor.to, torch.cat
    optimizer = torch.optim.SGD if kind == "shead" else torch.optim.Adam
    original_step = optimizer.step
    def to(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_to(tensor, *args, **kwargs)
    def cat(tensors, *args, **kwargs):
        result = original_cat(tensors, *args, **kwargs)
        cats.append(tuple(result.shape))
        return result
    def step(opt, *args, **kwargs):
        grads.append([p.grad.clone() for group in opt.param_groups for p in group["params"]])
        return original_step(opt, *args, **kwargs)
    original_backward = torch.Tensor.backward
    def backward(tensor, *args, **kwargs):
        losses.append(tensor.detach().clone())
        return original_backward(tensor, *args, **kwargs)
    torch.manual_seed(99)
    with contextlib.redirect_stdout(io.StringIO()), patch.object(torch.Tensor, "to", to), \
            patch.object(torch, "cat", cat), patch.object(optimizer, "step", step), \
            patch.object(torch.Tensor, "backward", backward):
        method("_maybe_head_ft" if kind == "shead" else "_maybe_disc_ft", legacy)(h)
    return losses, grads, transfers, cats


class FeedbackFineTuneEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def compare(self, kind, head, anchors, signed=0., auc_drop=False):
        before = [p.detach().clone() for p in head.parameters()]
        old = fixture(kind, copy.deepcopy(head), anchors, signed, auc_drop)
        new = fixture(kind, copy.deepcopy(head), anchors, signed, auc_drop)
        old_run, new_run = run(old, kind, True), run(new, kind, False)
        self.assertEqual(len(old_run[0]), len(new_run[0]))
        for left, right in zip(old_run[0], new_run[0]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(old_run[1], new_run[1]):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key, value in old.p.slots[kind].head.state_dict().items():
            torch.testing.assert_close(value, new.p.slots[kind].head.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(old.stats, new.stats)
        log_name = "head_ft_log" if kind == "shead" else "disc_ft_log"
        clean = lambda logs: [{k: v for k, v in row.items() if k != "t"} for row in logs]
        self.assertEqual(clean(getattr(old, log_name)), clean(getattr(new, log_name)))
        self.assertEqual(clean(old.rollback_log), clean(new.rollback_log))
        for attr in ("box_pos_feats", "normal_fb_feats", "disc_pos_patch_feats", "disc_neg_patch_feats"):
            self.assertEqual(len(getattr(old, attr)), len(getattr(new, attr)))
        self.assertFalse(new.p.slots[kind].head.training)
        if new.stats["gate_reject"]:
            for expected, actual in zip(before, new.p.slots[kind].head.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            cooldown = "_head_ft_cooldown" if kind == "shead" else "_disc_ft_cooldown"
            self.assertEqual(getattr(new, cooldown), 30)
        self.assertEqual(old_run[2].count((40, 4)), 3)
        self.assertEqual(new_run[2].count((40, 4)), 0)
        self.assertEqual(new_run[2].count((20, 4)), 2)
        valid_anchors = [x for x in (anchors or []) if x["feats"] is not None]
        if valid_anchors and kind == "shead":
            shape = (sum(x["feats"].shape[0] for x in valid_anchors), 4)
            # Reference forward transfers remain per image; concatenated anchors only once.
            self.assertEqual(new_run[3].count(shape), 1)
            self.assertEqual(old_run[3].count(shape), len(old_run[0]))
        return new

    def test_shead_dropout_loss_gradient_transfers_and_gate(self):
        anchors = [{"feats": torch.ones(1, 4)}, {"feats": None}, {"feats": torch.full((1, 4), 2.)}]
        for signed in (0., .06, -.21, .05, -.2):
            with self.subTest(signed=signed):
                self.compare("shead", HEAD_GLOBALS["SheadHead"](4, hidden=8, dropout=.3), anchors, signed)

    def test_shead_empty_anchors(self):
        for anchors in (None, [], [{"feats": None}]):
            with self.subTest(anchors=anchors):
                # None-only anchors must not trigger Tensor truth-value evaluation.
                h = self.compare("shead", HEAD_GLOBALS["SheadHead"](4, hidden=8), anchors)
                self.assertEqual(h.stats["head_ft"], 1)

    def test_shead_batchnorm_and_original_first_output_broadcast(self):
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(.3), nn.Linear(4, 1))
            def forward(self, x):
                return self.net(x).squeeze(-1)
        # Original reference is one scalar per image, not a score per tile or mean.
        self.compare("shead", Head(), [{"feats": torch.arange(8.).reshape(2, 4)}])

    def test_disc_loss_gradient_transfers_and_both_gates(self):
        class Head(nn.Linear):
            def forward(self, x):
                return super().forward(x).squeeze(-1)
        for signed, auc_drop in ((0., False), (.06, False), (-.21, False), (0., True)):
            with self.subTest(signed=signed, auc_drop=auc_drop):
                self.compare("disc", Head(4, 1), None, signed, auc_drop)


if __name__ == "__main__":
    unittest.main(verbosity=2)
