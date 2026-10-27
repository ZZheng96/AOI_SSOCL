"""真实增强消融的配置、指标与任务提交契约测试。"""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from backend.augmentation import ablation
from backend.api.schemas import AugmentAblationRequest


class AblationContractTest(unittest.TestCase):
    def _cfg(self):
        return {
            'seed': 42,
            'backbone': {'enabled': True},
            'slots': {'disc': {'enabled': True, 'freeze': False,
                               'epochs': 1, 'finetune_epochs': 1,
                               'pseudo_per_image': 2}},
            'fusion': {},
            'augment': {
                'pseudo': {'methods': ['cutpaste']},
                'preprocess': {'enabled': True, 'gray': True, 'clahe': 2.0, 'median': 3},
                'enhance': {'flip': True, 'rot90': True, 'brightness': True},
            },
        }

    def test_schema_defaults_and_limits(self):
        req = AugmentAblationRequest(category='part', factor='enhance.flip')
        self.assertEqual(req.split, 'test')
        self.assertEqual(req.train_limit, 8)
        with self.assertRaises(ValueError):
            AugmentAblationRequest(category='*', factor='enhance.flip')
        with self.assertRaises(ValueError):
            AugmentAblationRequest(category='part', factor='unknown')

    def test_only_selected_factor_is_disabled(self):
        payload = {'factor': 'preprocess.gray', 'preprocess': {'enabled': True, 'gray': True}}
        baseline, ablated = ablation.configs(self._cfg(), payload)
        self.assertTrue(baseline['augment']['preprocess']['gray'])
        self.assertEqual(ablated['augment']['preprocess']['gray'], 0)
        self.assertTrue(ablated['augment']['preprocess']['clahe'])
        self.assertTrue(ablated['augment']['enhance']['flip'])

    def test_training_only_factor_rejects_frozen_disc(self):
        cfg = self._cfg()
        cfg['slots']['disc']['freeze'] = True
        with self.assertRaisesRegex(ValueError, 'disc 未参与训练'):
            ablation.configs(cfg, {'factor': 'enhance.flip'})

    def test_metrics_use_real_scores_and_gray_is_anomaly(self):
        rows = [
            {'label': 0, 'score': 0.1, 'decision': 'normal'},
            {'label': 0, 'score': 0.8, 'decision': 'gray'},
            {'label': 1, 'score': 0.9, 'decision': 'anomaly'},
            {'label': 1, 'score': 0.2, 'decision': 'normal'},
        ]
        metrics = ablation._metrics(rows)
        self.assertTrue(np.isfinite(metrics['auroc']))
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']), (1, 1, 1, 1))
        self.assertEqual(metrics['fpr'], 0.5)

    def test_single_task_submission_is_mutually_exclusive(self):
        manager = Mock()
        manager.submit.return_value = 7
        manager.get.return_value = {'status': 'running'}
        with patch.object(ablation, 'task_manager', manager), patch.object(ablation, '_active_task', 6):
            with self.assertRaisesRegex(ValueError, '已有增强消融任务'):
                ablation.submit_ablation({'category': 'part', 'factor': 'enhance.flip'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
