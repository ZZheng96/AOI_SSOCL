"""Small, isolated traditional visualization contract checks (no production DB)."""
import base64
import os
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.augmentation import feature_space as fs
from backend.api.schemas import FeatureSpaceRequest
from backend.db.models import Base, Dataset, Image


def points(kinds):
    return [{'kind': kind, 'id': i, 'content_hash': str(i), 'category': 'part',
             'base_id': 0 if kind == 'pseudo' else None} for i, kind in enumerate(kinds)]


class ProjectionTest(unittest.TestCase):
    def test_low_rank_distance_and_no_missing_class(self):
        p = points(['normal'] * 3 + ['pseudo'])
        out = fs.project_features([[0, 1], [1, 1], [2, 1], [0, 1]], p)
        self.assertEqual(out['rank'], 1)
        self.assertEqual(out['effective_dimensions'], 1)
        self.assertEqual(p[3]['nearest_normal_id'], 0)
        self.assertEqual(p[3]['distance'], 0)
        self.assertNotEqual(p[0]['nearest_normal_id'], 0)
        self.assertTrue(all(s['xyz'][1:] == [0, 0] for s in p))
        self.assertFalse(any(s['kind'] == 'defect' for s in p))
        self.assertAlmostEqual(p[0]['distance'], 1 / np.std([0, 1, 2]))

    def test_duplicate_reference_and_no_variance_fail(self):
        p = points(['normal'] * 3)
        p[2]['content_hash'] = p[0]['content_hash']
        with self.assertRaisesRegex(ValueError, '独立'):
            fs.project_features([[0], [1], [2]], p)
        with self.assertRaisesRegex(ValueError, '方差'):
            fs.project_features([[1], [1], [1]], points(['normal'] * 3))

    def test_request_limits(self):
        for payload in ({'category': ''}, {'category': '  '}, {'category': '*'},
                        {'category': 'part', 'sample_limit': 13},
                        {'category': 'part', 'preprocess': {'median': 1001}}):
            with self.assertRaises(ValueError):
                FeatureSpaceRequest(**payload)

    def test_typed_api_submission_and_busy(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes_data import router
        app = FastAPI()
        app.include_router(router, prefix='/api')
        with TestClient(app) as client, patch.object(fs, 'submit_analysis', return_value=42) as submit:
            response = client.post('/api/augment/feature-space', json={'category': 'part'})
            self.assertEqual(response.json(), {'task_id': 42})
            self.assertEqual(submit.call_args.args[0]['sample_limit'], 8)
            self.assertEqual(client.post('/api/augment/feature-space', json={'category': ''}).status_code, 422)
            submit.side_effect = ValueError('已有分析')
            self.assertEqual(client.post('/api/augment/feature-space', json={'category': 'part'}).status_code, 409)

    def test_nonblocking_single_task(self):
        manager = Mock()
        manager.submit.return_value = 9
        manager.get.return_value = {'status': 'running'}
        with patch.object(fs, 'task_manager', manager), patch.object(fs, '_active_task', None):
            self.assertEqual(fs.submit_analysis({'category': 'part'}), 9)
            with self.assertRaisesRegex(ValueError, '已有'):
                fs.submit_analysis({'category': 'part'})
            self.assertEqual(manager.submit.call_count, 1)


class ExtractionTest(unittest.TestCase):
    def test_real_extractor_bounded_db_and_cancel(self):
        print('feature-space: isolated DB + 3 real extractor images + 3 memory pseudo', flush=True)
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        with TemporaryDirectory() as tmp:
            with Session(engine) as session:
                session.add_all([Dataset(id=1, name='raw', category='part'),
                                 Dataset(id=2, name='generated', category='part', source_type='augment')])
                for i in range(3):
                    path = str(Path(tmp) / f'{i}.png')
                    rgb = np.random.default_rng(i).integers(0, 256, (80, 90, 3), dtype=np.uint8)
                    PILImage.fromarray(rgb).save(path)
                    session.add(Image(path=path, category='part', label='normal', dataset_id=1))
                session.add_all([Image(path='not-read-aug', category='part', label='normal', dataset_id=2),
                                 Image(path='not-read-pseudo', category='part', label='normal', source='pseudo', dataset_id=1),
                                 Image(path='not-read-split', category='part', label='anomaly', split='pseudo', dataset_id=1),
                                 Image(path='not-read-other', category='other', label='anomaly', dataset_id=1)])
                session.commit()

            @contextmanager
            def isolated():
                with Session(engine) as session:
                    yield session

            payload = FeatureSpaceRequest(category='part', sample_limit=3, pseudo={'methods': ['color_blot']}).model_dump()
            progress = Mock()
            with patch.object(fs, 'session_scope', isolated):
                result = fs.analyze(payload, progress)
                self.assertEqual(result['actual_counts'], {'normal': 3, 'defect': 0, 'pseudo': 3})
                self.assertEqual(result['candidate_counts'], {'normal': 3, 'defect': 0})
                self.assertEqual(result['original_dimensions'], 200)
                self.assertEqual(progress.call_count, 6)
                for sample in result['samples']:
                    png = cv2.imdecode(np.frombuffer(base64.b64decode(sample['thumbnail']), np.uint8), 1)
                    self.assertLessEqual(max(png.shape[:2]), 160)
                event = threading.Event()
                event.set()
                with self.assertRaises(fs.TaskCancelled):
                    fs.analyze(payload, progress, event)
            # Store only for independent GUI smoke checks.
            self.assertTrue(any('无有效真实缺陷' in w for w in result['warnings']))
        engine.dispose()

    def test_datasource_scope_and_masked_donor(self):
        print('feature-space: same-source donor/mask + deterministic repeat (small fixtures)', flush=True)
        from backend.db.models import DataSource
        from algo.vendor.traditional import TraditionalFeatureExtractor
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        with TemporaryDirectory() as tmp, Session(engine) as session:
            session.add(DataSource(id=1, name='scope'))
            session.add_all([Dataset(id=1, name='scope-batch', datasource_id=1),
                             Dataset(id=2, name='outside')])
            for i in range(4):
                path = str(Path(tmp) / f'{i}.png')
                PILImage.fromarray(np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
                session.add(Image(path=path, category='part', dataset_id=1,
                                  label='normal' if i < 3 else 'anomaly'))
            mask = np.zeros((64, 64), np.uint8)
            mask[20:40, 20:40] = 255
            PILImage.fromarray(mask).save(str(Path(tmp) / '3_mask.png'))
            session.add(Image(path='outside-must-not-read', category='part', label='anomaly', dataset_id=2))
            session.commit()
            @contextmanager
            def isolated():
                yield session
            payload = FeatureSpaceRequest(category='part', datasource_id=1, sample_limit=3,
                                          pseudo={'methods': ['defect_transplant']}).model_dump()
            def cheap_extract(_self, rgb, rng=None):
                return rgb.mean(axis=(0, 1))
            with patch.object(fs, 'session_scope', isolated), patch.object(TraditionalFeatureExtractor, 'extract', cheap_extract):
                result = fs.analyze(payload, Mock())
                repeat = fs.analyze(payload, Mock())
            self.assertEqual(result, repeat)
            self.assertEqual(result['actual_counts'], {'normal': 3, 'defect': 1, 'pseudo': 3})
            self.assertEqual(result['candidate_counts']['defect'], 1)
            self.assertFalse(any('回退' in warning for warning in result['warnings']))
        engine.dispose()

    def test_reject_extreme_shape(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'wide.png')
            PILImage.new('RGB', (1024, 32)).save(path)
            with self.assertRaisesRegex(ValueError, '尺寸超限'):
                fs._read_rgb(path)


class DialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_draw_pick_and_closed_callback(self):
        from ui.widgets.feature_space_dialog import FeatureSpaceDialog
        dialog = FeatureSpaceDialog(Mock(), {'category': 'part', 'dataset_id': 1}, {}, {})
        p = points(['normal'] * 3 + ['pseudo'])
        _, encoded = cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))
        for sample in p:
            sample['thumbnail'] = base64.b64encode(encoded).decode('ascii')
        result = fs.project_features([[0], [1], [2], [0]], p)
        result.update(actual_counts={'normal': 3, 'defect': 0, 'pseudo': 1},
                      candidate_counts={'normal': 3, 'defect': 0}, warnings=[],
                      skipped={'normal': [], 'defect': [], 'pseudo': []}, config={})
        dialog.render_result(result)
        self.assertEqual(len(dialog._artists), 2)
        artist = next(iter(dialog._artists))
        dialog._pick(SimpleNamespace(artist=artist, ind=[0]))
        self.assertIn('高维最近正常距离', dialog.detail.text())
        self.assertFalse(dialog.image.pixmap().isNull())
        dialog.close()
        self.assertFalse(dialog.timer.isActive())
        self.assertTrue(dialog._closed)

    def test_fast_completion_poll_and_late_submit_cancel(self):
        from ui.widgets import feature_space_dialog as module
        callbacks = []
        def immediate(parent, fn, ok=None, fail=None):
            callbacks.append((fn, ok, fail))
        client = Mock()
        dialog = module.FeatureSpaceDialog(client, {'category': 'part'}, {}, {})
        with patch.object(module, 'run_async', immediate):
            dialog._start()
            callbacks.pop(0)[1]({'task_id': 33})
            self.assertEqual(len(callbacks), 1)  # immediate polling without WS
            callbacks.pop(0)[1]({'status': 'cancelled', 'progress': 0})
            self.assertTrue(dialog.start.isEnabled())
            dialog._start()
            submitted = callbacks.pop(0)[1]
            dialog.close()
            submitted({'task_id': 34})
            callbacks.pop(0)[0]()
            client.cancel_task.assert_called_once_with(34)


if __name__ == '__main__':
    unittest.main(verbosity=2)
