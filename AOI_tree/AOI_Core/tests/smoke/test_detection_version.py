"""轻量验证实际推理版本追溯与不完备报告消费，不加载模型或数据集。"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api.schemas import to_dict
from backend.db import database
from backend.db.models import Base, Detection, Model
from backend.engine import DetectionEngine, DetectionResult
from backend.pipeline.service import DetectionService
from ui.pages.dashboard_page import DashboardPage
from ui.theme import DANGER, TEXT_SUB


class EngineVersionTest(unittest.TestCase):
    def _predict(self, entry):
        engine = DetectionEngine.__new__(DetectionEngine)
        lock = threading.RLock()
        engine._ensure_loaded = Mock()
        engine._pipe_lock = Mock(return_value=lock)
        current = [7]
        pipe = SimpleNamespace(weights={"sem": 1.0},
                               decider=SimpleNamespace(tau_high=0.9, tau_gray=0.7))

        def pipe_entry(category):
            self.assertTrue(lock._is_owned())
            return pipe, current[0]

        def predict(*args, **kwargs):
            current[0] = 8  # 模拟取得旧 pipeline 后激活新版本。
            return {"fused": 0.2, "decision": "normal", "slot_scores": {"sem": 0.2}}

        engine._pipe_entry = Mock(side_effect=pipe_entry)
        engine.current_version = Mock(side_effect=AssertionError("不能重读当前版本"))
        pipe.predict = pipe.predict_frame = Mock(side_effect=predict)
        if entry == "path":
            result = engine.predict_image_path("test", "fake.jpg")
        else:
            result = engine.predict_ndarray("test", np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(current[0], 8)
        self.assertEqual(result.model_version, 7)
        self.assertEqual((result.score, result.threshold, result.gray_threshold), (0.2, 0.9, 0.7))
        engine.current_version.assert_not_called()

    def test_path_keeps_inference_version(self):
        self._predict("path")

    def test_frame_keeps_inference_version(self):
        self._predict("frame")

    def test_service_response_keeps_version_without_persistence(self):
        service = DetectionService.__new__(DetectionService)
        service._record_alarm = Mock(return_value={"active": False})
        service._persist = Mock()
        result = DetectionResult(model_version=7, score=0.2, threshold=0.9)
        out = service._finish_detect(result, None, "fake.jpg", "test", time.perf_counter(),
                                     persist=False, with_heatmap=False)
        self.assertEqual(out["model_version"], 7)
        self.assertFalse(out["persisted"])
        service._persist.assert_not_called()


class PersistenceVersionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = create_engine(f"sqlite:///{Path(tmp.name) / 'test.db'}")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        patches = (patch.object(database, "_engine", engine),
                   patch.object(database, "_SessionLocal", factory),
                   patch("backend.pipeline.service.get_engine",
                         side_effect=AssertionError("落库不能读取当前版本")))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with database.session_scope() as s:
            old = Model(name="old", category="test", version="v7", path="old.pkl")
            active = Model(name="active", category="test", version="v8", path="new.pkl", is_active=True)
            s.add_all([old, active])
            s.flush()
            self.old_id = old.id

    def _persist(self, version):
        det = {"category": "test", "image_path": "fake.jpg", "final_score": 0.2,
               "is_anomaly": False, "latency_ms": 1.0}
        if version is not None:
            det["model_version"] = version
        det_id = DetectionService._persist(Mock(), det, None, "image")
        with database.session_scope() as s:
            row = s.get(Detection, det_id)
            return row.model_id, row.n_tiles, to_dict(row)

    def test_persist_links_actual_not_active_model(self):
        model_id, nt, out = self._persist(7)
        self.assertEqual(model_id, self.old_id)
        self.assertEqual(nt["model_version"], 7)
        self.assertEqual(out["model_version"], 7)

    def test_missing_model_keeps_actual_version(self):
        model_id, nt, out = self._persist(6)
        self.assertIsNone(model_id)
        self.assertEqual(nt["model_version"], 6)
        self.assertEqual(out["model_version"], 6)

    def test_unknown_and_legacy_versions_remain_unknown(self):
        model_id, nt, out = self._persist(None)
        self.assertIsNone(model_id)
        self.assertIsNone(out["model_version"])
        legacy = Detection(n_tiles={"decision": "normal"}, is_anomaly=False)
        self.assertIsNone(to_dict(legacy)["model_version"])


class IncompleteReportTest(unittest.TestCase):
    def test_alert_contract_and_clear_tooltip(self):
        label = Mock()
        page = SimpleNamespace(todo_incomplete=(None, label))
        DashboardPage._fill_incomplete(page, {"reports": [
            {"category": "a", "alert": True},
            {"category": "b", "alert": False, "warn": True}]})
        label.setText.assert_called_with("1")
        label.setStyleSheet.assert_called_with(f"color: {DANGER};")
        label.setToolTip.assert_called_with("预警品类：a")
        DashboardPage._fill_incomplete(page, {"reports": []})
        label.setText.assert_called_with("0")
        label.setStyleSheet.assert_called_with(f"color: {TEXT_SUB};")
        label.setToolTip.assert_called_with("")


class ProductionFeatureMetadataTest(unittest.TestCase):
    def _pipe(self):
        from algo.eval.offline import Pipeline
        from algo.slots.trad import TradSlot
        pipe = Pipeline({"tiling": {}, "backbone": {"enabled": False}}, None,
                        [TradSlot({"topk": 3, "zclip": 8.0})])
        self.addCleanup(pipe._cpu_exec.shutdown)
        pipe.weights = {"trad": 1.0}
        pipe.slot_sanity = {}
        pipe.consensus_gate = None
        pipe.router_gate = None
        return pipe

    def _load(self, path):
        from algo.persist import load_snapshot
        from algo.slots.trad import TradSlot
        with patch("algo.factory.build", return_value=(None, [TradSlot({"topk": 3, "zclip": 8.0})])), \
                patch.dict("os.environ", {"AOI_PICKLE_EXTRA_ROOTS": str(path)}):
            pipe = load_snapshot(str(path), "cpu")
        self.addCleanup(pipe._cpu_exec.shutdown)
        return pipe

    def test_save_load_metadata_and_unknown_fields(self):
        import hashlib
        import json
        from algo.persist import save_snapshot
        from algo.vendor import traditional
        with tempfile.TemporaryDirectory() as tmp:
            pipe = self._pipe()
            save_snapshot(pipe, tmp, version=4)
            path = Path(tmp) / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            feature = meta["production_feature_metadata"]
            self.assertEqual(feature["schema_version"], 1)
            self.assertEqual(feature["status"], "production_active")
            self.assertEqual(feature["config"], {"topk": 3, "zclip": 8.0})
            self.assertEqual(feature["dimension"], 200)
            self.assertEqual(feature["extractor"], "TraditionalFeatureExtractor")
            self.assertEqual(feature["implementation_sha256"],
                             hashlib.sha256(Path(traditional.__file__).read_bytes()).hexdigest())
            canonical = json.dumps(feature["config"], sort_keys=True, separators=(",", ":"))
            self.assertEqual(feature["config_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
            feature["future_field"] = {"ignored": True}
            path.write_text(json.dumps(meta), encoding="utf-8")
            restored = self._load(tmp)
            self.assertEqual(restored.production_feature_metadata, feature)
            self.assertEqual(restored.weights, pipe.weights)
            self.assertFalse(hasattr(pipe, "production_feature_metadata"))  # save 不修改冷对象

    def test_legacy_missing_metadata_and_meta_file(self):
        import json
        from algo.persist import save_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            save_snapshot(self._pipe(), tmp, version=1)
            path = Path(tmp) / "meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            del meta["production_feature_metadata"]
            path.write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(self._load(tmp).production_feature_metadata,
                             {"status": "legacy_unknown", "reason": "field_missing"})
            path.unlink()
            with self.assertWarns(RuntimeWarning):
                restored = self._load(tmp)
            self.assertEqual(restored.production_feature_metadata,
                             {"status": "legacy_unknown", "reason": "meta_missing"})

    def test_config_keys_and_fixed_dimension(self):
        from algo.persist import _production_feature_metadata, _restore_slot_state
        pipe = self._pipe()
        for key in ("cfg", "config"):
            with self.subTest(key=key):
                _restore_slot_state(pipe.slots["trad"], {key: {"legacy": True}})
                self.assertEqual(pipe.slots["trad"].cfg, {"legacy": True})
        pipe.slots["trad"].ext._fixed_dim = 172
        self.assertEqual(_production_feature_metadata(pipe)["dimension"], 172)
        pipe.slots.clear()
        self.assertEqual(_production_feature_metadata(pipe)["status"], "unknown")

    def test_detection_metadata_contract(self):
        service = DetectionService.__new__(DetectionService)
        service._record_alarm = Mock(return_value={"active": False})
        feature = {"status": "production_active", "implementation_sha256": "a" * 64}
        result = DetectionResult(model_version=4, raw_scores={"trad": 2.0},
                                 slot_scores={"trad": 0.4}, weights={"trad": 1.0},
                                 router_w={"trad": 0.8},
                                 extra={"feature_metadata": feature,
                                        "candidate_router": {"enabled": False, "router_auc": 0.7}})
        out = service._finish_detect(result, None, "fake.jpg", "test", time.perf_counter(),
                                     persist=False, with_heatmap=False)
        self.assertEqual(out["raw_scores"], {"trad": 2.0})
        self.assertEqual(out["cal_scores"], {"trad": 0.4})
        self.assertEqual(out["feature_metadata"], feature)
        self.assertEqual(out["actual_router"], {"trad": 0.8})
        self.assertFalse(out["candidate_router"]["enabled"])
        row = Detection(n_tiles=out["n_tiles"], is_anomaly=False)
        historical = to_dict(row)
        for key in ("raw_scores", "cal_scores", "feature_metadata", "actual_router", "candidate_router", "weights"):
            self.assertEqual(historical[key], out[key])
        legacy = to_dict(Detection(n_tiles={}, is_anomaly=False))
        self.assertIsNone(legacy["feature_metadata"])
        self.assertEqual(legacy["raw_scores"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
