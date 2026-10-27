"""PCB-DUAL-04: bounded synthetic branch and uncancellable capacity regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from app.detect.adapters.smt_adapter import _load_smt, SmtAdapter
from app.detect.adapters.through_hole_adapter import ThroughHoleAdapter, DEFECT_ID_BY_ALG
from app.detect.contract import Roi
from app.detect.types import DetectRequest
from app.detect.scheduler import _run_with_timeout, inflight_snapshot, DetectScheduler
from app.recipe.store import RecipeStore
from app.recipe.param_store import ParamStore


class PcbDual04Tests(unittest.TestCase):
    def wait_empty(self):
        deadline = time.monotonic() + 2
        while inflight_snapshot()["count"] and time.monotonic() < deadline:
            time.sleep(0.002)
        self.assertEqual(inflight_snapshot()["count"], 0)

    def test_four_slots_reject_release_recover_request_isolation(self):
        self.wait_empty()
        releases = [threading.Event() for _ in range(4)]
        calls = []
        try:
            with self.assertLogs("app.detect.scheduler", level="WARNING") as logs:
                for i, release in enumerate(releases):
                    def task(i=i, release=release):
                        calls.append(i)
                        release.wait(3)
                    with self.assertRaises(TimeoutError):
                        _run_with_timeout(task, 0.01, tries=9, key=f"dual04:{i}")
                snap = inflight_snapshot()
                self.assertEqual((snap["count"], snap["limit"]), (4, 4))
                self.assertEqual({t["key"] for t in snap["tasks"]}, {f"dual04:{i}" for i in range(4)})
                self.assertTrue(all(t["timed_out"] and not t["cancellable"] and t["age_ms"] >= 0 for t in snap["tasks"]))
                with self.assertRaises(TimeoutError):
                    _run_with_timeout(lambda: calls.append(99), 0.01, key="dual04:fifth")
                with self.assertRaises(TimeoutError):
                    _run_with_timeout(lambda: calls.append(98), 0.01, key="dual04:1")
            self.assertTrue(any("容量已满" in text for text in logs.output))
            releases[0].set()
            deadline = time.monotonic() + 1
            while inflight_snapshot()["count"] == 4 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertEqual(_run_with_timeout(lambda: "recovered", 0.2, key="dual04:new"), "recovered")
            self.assertEqual(sorted(calls), list(range(4)))
            print("capacity: 4 occupied, fifth rejected, independent keys, released slot recovered", flush=True)
        finally:
            for event in releases:
                event.set()
            self.wait_empty()

    def test_scheduler_batch_timeout_once_and_cleanup(self):
        release = threading.Event()
        adapter = SmtAdapter()
        calls = []
        def batch(requests):
            calls.append(len(requests))
            release.wait(3)
            return []
        image = np.zeros((64, 64, 3), np.uint8)
        try:
            with tempfile.TemporaryDirectory() as temp:
                scheduler = DetectScheduler(RecipeStore(Path(temp) / "recipes"), ParamStore(Path(temp)))
                with patch("app.detect.scheduler.get_adapter_for", return_value=adapter), patch.object(adapter, "run_many", side_effect=batch), patch("app.detect.scheduler._ALGO_TIMEOUT_S", 0.01):
                    result = scheduler.run(image, image, None, None, ["solder_少锡", "solder_虚焊"], region_sets={"smt_pads": [{"x": 10, "y": 10, "w": 30, "h": 30}]}, gate_enabled=False, allow_stub=False)
                self.assertEqual(calls, [2])
                self.assertFalse(result.overall_ok)
                self.assertTrue(all(r.error_code == "RUN_TIMEOUT" for r in result.results))
                self.assertTrue(all(r.metadata["inflight"]["count"] == 1 for r in result.results))
        finally:
            release.set()
            self.wait_empty()

    def test_rim_minus_toe_rule1_and_crack_unavailable(self):
        core = _load_smt()["klass"].run.__globals__["judge_cold_solder"]
        image = np.zeros((80, 100, 3), np.uint8)
        diagnostics = {}
        result = core(image, np.zeros(image.shape[:2], np.uint8), [(10, 10, 60, 40)], image.shape, {"_diagnostics": diagnostics}, toe_rects={0: (10, 10, 20, 40)}, rim_rects={0: (10, 10, 60, 40)})
        mask = diagnostics["rim_toe"][0]
        globals_ = core.__globals__
        expected = cv2.subtract(globals_["_rect_mask"](10, 10, 60, 40, image.shape), globals_["_rect_mask"](10, 10, 20, 40, image.shape))
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(result[0][6], ["rule1"])
        self.assertFalse(diagnostics["crack"][0]["executed"])
        self.assertTrue(diagnostics["crack"][0]["unavailable_reason"])

    def test_crack_real_hough_execution_and_missing_inputs(self):
        core = _load_smt()["klass"].run.__globals__["judge_cold_solder"]
        image = np.full((80, 120, 3), 200, np.uint8)
        image[20:61, 55:59] = 0
        diagnostics = {}
        params = {"_diagnostics": diagnostics, "cold_diff_ratio_thresh": 0.1}
        with patch.dict(core.__globals__, _USE_CPP=False), patch.object(cv2, "HoughLinesP", wraps=cv2.HoughLinesP) as hough:
            core(image, np.full(image.shape[:2], 255, np.uint8), [(10, 10, 90, 60)], image.shape, params, diff_mask_full=np.full(image.shape[:2], 255, np.uint8))
        self.assertEqual(hough.call_count, 1)
        self.assertTrue(diagnostics["crack"][0]["executed"])
        self.assertIn("roi_origin", diagnostics["crack"][0]["hough"])
        diagnostics.clear()
        core(image, np.zeros(image.shape[:2], np.uint8), [(10, 10, 90, 60)], image.shape, params)
        self.assertFalse(diagnostics["crack"][0]["executed"])

    def test_resize_roi_offset_output_original_coordinates(self):
        algorithm = _load_smt()["instance"]
        image = np.zeros((300, 400, 3), np.uint8)
        cfg = {"pad_frames": [[20, 20, 60, 40]], "detect_insufficient": True, "detect_excess": False, "detect_bridge": False, "detect_cold_solder": False, "enable_auto_resize": True, "auto_resize_max_size": 100, "auto_resize_threshold": 110, "diagnostics": True}
        roi = SimpleNamespace(x=40, y=60, width=200, height=160)
        result = algorithm.run(image, cfg, roi_bbox=roi)
        self.assertEqual(result.code, 0, result.message)
        self.assertEqual(len(result.parts), 1)
        part = result.parts[0]
        self.assertEqual((part.x, part.y, part.width, part.height), (60, 80, 60, 40))
        output = result.metadata["output_image"]
        self.assertEqual(output.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(output)), 0)
        self.assertFalse(np.array_equal(output, image))
        self.assertTrue(np.any(output[part.y, part.x] != image[part.y, part.x]))
        self.assertEqual(result.metadata["diagnostics"]["coordinates"]["scale"], 0.5)
        self.assertEqual(result.metadata["diagnostics"]["solder_mask"].shape, (80, 100))

    def test_alignment_exception_not_ok(self):
        image = np.zeros((64, 64, 3), np.uint8)
        roi = Roi(id="pad", kind="smt", layer="smt_pads", shape={"x": 10, "y": 10, "w": 30, "h": 30})
        req = DetectRequest(image_test=image, image_std=image.copy(), rois=[roi])
        globals_ = _load_smt()["klass"].run.__globals__
        with patch.dict(globals_, align_template=lambda *args: (_ for _ in ()).throw(RuntimeError("boom"))):
            result = SmtAdapter().run("solder_虚焊", req, {})
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "ERROR")

    def test_alignment_failure_not_ok(self):
        image = np.zeros((64, 64, 3), np.uint8)
        roi = Roi(id="pad", kind="smt", layer="smt_pads", shape={"x": 10, "y": 10, "w": 30, "h": 30})
        req = DetectRequest(image_test=image, image_std=image.copy(), rois=[roi])
        globals_ = _load_smt()["klass"].run.__globals__
        with patch.dict(globals_, align_template=lambda *args: None):
            result = SmtAdapter().run("solder_虚焊", req, {})
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "ERROR")
        self.assertIn("alignment failed", result.message)

    def test_th_checkbox_modes_offscreen(self):
        from PySide6.QtWidgets import QApplication, QCheckBox
        from app.ui.detect_page import DetectPage
        app = QApplication.instance() or QApplication([])
        checks = {alg: QCheckBox(alg) for alg in DEFECT_ID_BY_ALG}
        widget = SimpleNamespace(bound_template=None, template_store=SimpleNamespace(sync_calibration_to_legacy=lambda tpl: None), scheduler=SimpleNamespace(param_store=SimpleNamespace(import_from_template=lambda *args: None)), canvas_std=SimpleNamespace(set_image=lambda image: None, set_placeholder_text=lambda text: None), _all_checkboxes_map=lambda: checks, _drawer_frames=[], _refresh_all_drawer_headers=lambda: None, _update_calib_button=lambda: None, _refresh_open_param_panels=lambda: None)
        for count in (0, 1, 2, 1):
            tpl = SimpleNamespace(id="dual04-fixture", standard_image=SimpleNamespace(path=""), region_sets={"th": [{"x": 5 + i * 20, "y": 5, "w": 15, "h": 15} for i in range(count)]}, params_map=lambda: {}, enabled_algorithm_ids=lambda: list(checks))
            DetectPage._on_template_changed(widget, tpl)
            for alg, checkbox in checks.items():
                ready = count >= 2 if alg == "tht_连锡" else count < 2
                self.assertEqual(checkbox.isEnabled(), ready)
                self.assertEqual(checkbox.isChecked(), ready)
                if not ready:
                    self.assertTrue(checkbox.toolTip())
        for checkbox in checks.values():
            checkbox.close()

    def test_th_auto_single_multi_readiness_matches_scheduler(self):
        adapter = ThroughHoleAdapter()
        for count in (0, 1, 2):
            rois = [Roi(id=str(i), kind="th", layer="th", shape={"x": i * 20, "y": 10, "w": 15, "h": 15}) for i in range(count)]
            for alg in DEFECT_ID_BY_ALG:
                ready, reason = adapter.is_ready(alg, None, rois=rois)
                expected = count >= 2 if alg == "tht_连锡" else count < 2
                self.assertEqual(ready, expected, (count, alg, reason))
                if not ready:
                    self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
