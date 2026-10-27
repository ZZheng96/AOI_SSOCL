"""贡献分析边界：未定义指标必须保持 JSON-safe 且明确原因。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from algo.eval.contribution import contribution_profile, trad_group_profile


class ContributionBoundaryTest(unittest.TestCase):
    def test_constant_scores_are_none_with_insufficient_reason(self):
        profile = contribution_profile(
            {"sem": [0.5, 0.5, 0.5], "trad": [0.5, 0.5, 0.5]},
            [0, 0, 0],
            [None, None, None],
        )

        self.assertIsNone(profile["univariate"]["overall"]["sem"])
        self.assertEqual(
            profile["univariate"]["availability"]["overall"]["sem"]["status"],
            "insufficient",
        )
        self.assertIsNone(profile["redundancy"]["corr"][0][1])
        self.assertEqual(
            profile["redundancy"]["availability"][0][1]["status"],
            "insufficient",
        )
        self.assertIsNone(profile["ablation"]["full"])
        self.assertEqual(
            profile["ablation"]["availability"]["full"]["status"],
            "insufficient",
        )
        json.dumps(profile, allow_nan=False)


class PerImageContributionTest(unittest.TestCase):
    def _profile(self, groups):
        from algo.features import registry
        trad = SimpleNamespace(
            bank_w=np.array([[0., 0.], [1., 1.]]), norm_w=None,
            scoring="legacy", group_weight={}, group_scale={}, groups=groups,
            ext=SimpleNamespace(last_trace=[]),
            _vec=lambda x: x, _z=lambda x, _: x)
        with patch.object(registry, "GROUP_KEYS", ["a", "b"]), \
                patch.object(registry, "dim_index", side_effect=lambda keys, dim: [0 if keys[0] == "a" else 1]), \
                patch.object(registry, "group", return_value={"name": "test", "category": "test", "version": 1}):
            return trad_group_profile(trad, [np.array([0., 0.]), np.array([3., 4.])],
                                      [0, 1], [None, "bad"], n_boot=0)

    def test_real_distance_delta(self):
        profile = self._profile(["a", "b"])
        self.assertEqual(profile["per_image_status"], "available")
        rows = profile["per_image"]
        self.assertEqual(len(rows), 4)
        row = next(r for r in rows if r["image_index"] == 1 and r["feature_group"] == "a")
        self.assertAlmostEqual(row["baseline"], 5.)
        self.assertAlmostEqual(row["ablated"], 4.)
        self.assertAlmostEqual(row["contribution"], 1.)
        json.dumps(profile, allow_nan=False)

    def test_only_group_has_no_fabricated_ablation(self):
        profile = self._profile(["a"])
        self.assertEqual(profile["per_image"], [])
        self.assertEqual(profile["per_image_status"], "unavailable")

    def test_service_persists_image_ids_and_rejects_ambiguous_paths(self):
        from backend.evaluation import service
        for ambiguous in (False, True):
            profile = {"trad_groups": {"meta": {}, "per_image": [
                {"path": "normal.png", "feature_group": "a", "baseline": 2.,
                 "ablated": 1., "contribution": 1.}]}}
            engine = MagicMock()
            engine.contribution_report.return_value = profile
            engine.current_version.return_value = 3
            session = MagicMock()
            images = [SimpleNamespace(path="normal.png", id=7)]
            if ambiguous:
                images.append(SimpleNamespace(path="normal.png", id=8))
            session.query.return_value.filter.return_value.all.return_value = images
            context = MagicMock()
            context.__enter__.return_value = session
            with patch.object(service, "_fetch_eval_items", return_value=[("normal.png", 0), ("bad.png", 1)]), \
                    patch("backend.engine.get_engine", return_value=engine), \
                    patch.object(service, "session_scope", return_value=context), \
                    patch.object(service, "log_action"):
                result = service.evaluate_contribution("test")
            tg = result["profile"]["trad_groups"]
            if ambiguous:
                self.assertEqual(tg["per_image_status"], "unavailable")
                self.assertEqual(tg["per_image"], [])
            else:
                self.assertEqual(tg["per_image"][0]["image_id"], 7)
            self.assertIs(session.add.call_args.args[0].metrics, profile)

    def test_augmentation_boundary_has_no_metrics(self):
        from backend.api.routes_data import api_augment_ablation
        from backend.api.schemas import AugmentConfigUpdateRequest
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            api_augment_ablation(AugmentConfigUpdateRequest(enhance={"flip": True}))
        self.assertEqual(ctx.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
