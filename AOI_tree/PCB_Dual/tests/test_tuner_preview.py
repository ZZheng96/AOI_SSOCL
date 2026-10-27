"""TEMP/合成小图与离屏Qt定向验证，不执行数据集或生产DB。"""
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
from PySide6.QtWidgets import QApplication

from app.detect.types import AlgorithmResult, DetectRequest, DetectSummary
from app.detect.contract import Roi
from app.detect.adapters.shift_adapter import ShiftAdapter
from app.detect.adapters.smt_adapter import SmtAdapter, _smt_pads
from app.ui.tuner_preview import PreviewController
from app.ui.tuner_dialog import TunerDialog
from app.tools.tuner import HsvObjectSpec


class TunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        self.fail("等待异步结果超时")

    def test_real_shift_parameters_and_coordinates(self):
        image = np.full((72, 96, 3), 255, np.uint8)
        image[20:40, 25:45] = 0
        roi = Roi(id="base", kind="shift", layer="shift_base", shape={"x": 25, "y": 20, "w": 20, "h": 20})
        req = DetectRequest(image_test=image, rois=[roi], preprocess_meta={"diagnostics": True})
        adapter = ShiftAdapter()
        params = {"feature_h_low": 0, "feature_h_high": 179, "feature_s_low": 0,
                  "feature_s_high": 255, "feature_v_low": 0, "feature_v_high": 50,
                  "feature_thresh": 0.2, "use_rotation": False}
        first = adapter.run("shift_缺件", req, params)
        second = adapter.run("shift_缺件", req, {**params, "feature_v_low": 250, "feature_v_high": 255})
        self.assertIn(first.status, ("OK", "NG"))
        self.assertIn(second.status, ("OK", "NG"))
        a, b = first.metadata["regions"][0], second.metadata["regions"][0]
        self.assertFalse(np.array_equal(a["base_mask"], b["base_mask"]))
        self.assertEqual(a["roi"], [25, 20, 20, 20])
        self.assertEqual(a["feature_origin"], [0, 0])
        self.assertEqual(a["feature_map"].shape, (53, 77))
        self.assertIn("不可用", a["GT/IoU"])

    def test_smt_pad_layers_and_real_mask(self):
        shapes = [{"x": 20, "y": 20, "w": 20, "h": 20}, {"x": 24, "y": 24, "w": 10, "h": 10}]
        rois = [Roi(id=layer, kind="smt", layer=layer, shape=shape) for layer, shape in zip(("smt_pads", "smt_toe"), shapes)]
        self.assertEqual(_smt_pads(None, rois), [(20, 20, 20, 20)])
        image = np.full((80, 80, 3), 180, np.uint8)
        req = DetectRequest(image_test=image, image_std=image.copy(), rois=rois,
                            preprocess_meta={"diagnostics": True})
        result = SmtAdapter().run("solder_少锡", req, {"auto_resize": False, "template_mode": "grow"})
        self.assertNotEqual(result.status, "ERROR", result.message)
        debug = result.metadata["diagnostics"]
        self.assertEqual(debug["solder_mask"].shape, (80, 80))
        self.assertEqual(len(debug["pad_coverage"]), 1)
        self.assertIn("不可用", debug["crack"])
        excess = SmtAdapter().run("solder_多锡", req, {"auto_resize": False, "template_mode": "grow"})
        self.assertNotEqual(excess.status, "ERROR", excess.message)
        self.assertEqual(excess.metadata["diagnostics"]["ring_mask"].shape, (80, 80))

    def test_through_hole_real_diagnostics(self):
        from app.detect.adapters.through_hole_adapter import ThroughHoleAdapter
        image = np.zeros((96, 96, 3), np.uint8)
        cv2.circle(image, (48, 48), 16, (200, 200, 200), -1)
        req = DetectRequest(image_test=image, image_std=image.copy(),
                            rois=[Roi(id="th", kind="th", layer="th", shape={"shape": "rect", "x": 28, "y": 28, "w": 40, "h": 40})],
                            preprocess_meta={"diagnostics": True})
        result = ThroughHoleAdapter().run("tht_少锡", req, {})
        self.assertNotEqual(result.status, "ERROR", result.message)
        self.assertIn(result.status, ("OK", "NG", "REVIEW"))
        debug = result.metadata["diagnostics"]
        self.assertIn("contour_view", debug)
        self.assertIn("difference", debug)
        self.assertIn("alignment", debug)
        self.assertIn("不是焊锡填充率", debug["coverage_meaning"])
        if result.status == "REVIEW":
            self.assertFalse(result.ok)
            self.assertFalse(result.skipped)
            self.assertTrue(result.skip_reason)

    def test_apply_freeze_scheduler_equivalence(self):
        from app.ui.template_studio_page import TemplateStudioPage
        from app.template.model import InspectionTemplate, TemplateAlgorithm
        from app.recipe.param_store import ParamStore
        from app.recipe.store import RecipeStore
        from app.detect.scheduler import DetectScheduler
        from app.detect.registry import get_adapter_for
        from app.tools.tuner import discover_hsv_objects
        with tempfile.TemporaryDirectory() as directory:
            store = ParamStore(Path(directory))
            shared, item = "shift_共用参数", "shift_缺件"
            tpl = InspectionTemplate(id="tuner-test", detection_items=[
                TemplateAlgorithm(algorithm_id=item, params={"feature_v_high": 5}),
                TemplateAlgorithm(algorithm_id=shared, enabled=False, params={})],
                region_sets={"shift": [{"shape": "rect", "x": 25, "y": 20, "w": 20, "h": 20}]})
            store.save("template_algorithm", item, {"feature_v_high": 5}, template_name=tpl.id)
            host = SimpleNamespace(param_store=store, _param_panels={},
                                   _selected_ids=lambda: [item], _refresh_param_panels=lambda: None)
            spec = next(s for s in discover_hsv_objects([item]) if "feature_v_high" in s.keys.values())
            hsv = {"h_lo": 0, "h_hi": 179, "s_lo": 0, "s_hi": 255, "v_lo": 0, "v_hi": 50}
            with patch("app.ui.template_studio_page.QMessageBox.information"):
                TemplateStudioPage._apply_hsv_params_impl(host, tpl, spec, hsv)
            self.assertNotIn("feature_v_high", tpl.algorithms[0].params)
            TemplateStudioPage._freeze_params(host, tpl)
            overrides = {a.algorithm_id: a.params for a in tpl.algorithms}
            scheduler = DetectScheduler(recipe_store=RecipeStore(Path(directory) / "recipes"), param_store=store)
            resolved, _ = scheduler._resolve_params(get_adapter_for(item), item, tpl.id, None, overrides, True)
            self.assertEqual(resolved["feature_v_high"], 50)
            image = np.full((72, 96, 3), 255, np.uint8)
            image[20:40, 25:45] = 0
            kwargs = dict(param_overrides=overrides, allow_stub=False, prefer_frozen_params=True,
                          region_sets=tpl.region_sets, template_id=tpl.id, template_version=tpl.version)
            preview = scheduler.run(image, image, None, tpl.id, [item], diagnostics=True, **kwargs)
            production = scheduler.run(image, image, None, tpl.id, [item], **kwargs)
            self.assertEqual(preview.results[0].status, production.results[0].status)
            debug = preview.results[0].metadata["preview_request"]
            self.assertEqual(debug["params"]["feature_v_high"], 50)
            self.assertEqual(debug["rois"][0]["layer"], "shift")
            self.assertIn("stub disabled", debug["mode"])
            self.assertNotIn("preview_request", production.results[0].metadata)
            missing = scheduler.run(image, image, None, tpl.id, ["unknown-tuner"], allow_stub=False)
            self.assertFalse(missing.overall_ok)
            self.assertEqual(missing.results[0].status, "ERROR")

    def test_non_hsv_live_edit_and_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            cv2.imwrite(str(path), np.zeros((48, 48, 3), np.uint8))
            calls, applied = [], []
            shared = "shift_共用参数"
            def run(payload):
                calls.append(payload)
                return DetectSummary(overall_ok=True, vis_image=payload[1])
            dialog = TunerDialog(str(path), algorithm_ids=["shift_移位", "shift_缺件"],
                                 algorithm_params={shared: {"use_rotation": False}},
                                 preview_runner=run,
                                 params_apply_callback=lambda edits, hsv: applied.append((edits, hsv)))
            panel = dialog._param_panels[shared]
            self.assertEqual(panel.current_values()["use_rotation"], "false")
            panel.sliders["rotation_range"].set_value(15)
            panel.sliders["feature_thresh"].set_value(0.3)
            panel.sliders["use_rotation"].set_value("true")
            self.wait(lambda: calls and "__parameter_layers__" in calls[-1][2]
                      and calls[-1][2]["__parameter_layers__"].get(shared, {}).get("feature_thresh") == 0.3)
            dialog.cmb_algorithm.setCurrentIndex(1)
            self.assertEqual(dialog._param_panels[shared].current_values()["rotation_range"], 15)
            dialog._apply()
            self.assertEqual(applied[-1][0][shared]["use_rotation"], "true")
            self.assertEqual(applied[-1][0][shared]["feature_thresh"], 0.3)
            dialog.reject()

    def test_latest_cancel_close(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        def run(value):
            calls.append(value)
            if value == 1:
                entered.set()
                release.wait(3)
            return value
        controller = PreviewController(run, debounce_ms=5)
        results = []
        controller.finished.connect(results.append)
        controller.submit(1)
        self.wait(entered.is_set)
        controller.submit(2)
        controller.submit(3)
        release.set()
        self.wait(lambda: results == [3])
        self.assertEqual(calls, [1, 3])
        controller.submit(4)
        controller.cancel()
        controller.submit(5)
        controller.close()
        for _ in range(20):
            self.app.processEvents()
            time.sleep(0.005)
        self.assertEqual(results, [3])

    def test_dialog_changes_apply_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            cv2.imwrite(str(path), np.zeros((48, 48, 3), np.uint8))
            spec = HsvObjectSpec("feature", "底座", (0, 255, 0),
                                 {k: "feature_" + k for k in ("h_lo", "h_hi", "s_lo", "s_hi", "v_lo", "v_hi")})
            calls, applied = [], []
            def run(payload):
                calls.append(payload)
                return DetectSummary(overall_ok=False, results=[AlgorithmResult(
                    algorithm=payload[0], ok=False, status="REVIEW", message="真实状态",
                    metadata={"preview_request": {"params": payload[2]}, "mask": np.zeros((12, 12), np.uint8),
                              "rim_toe": {0: np.zeros((6, 6), np.uint8)}})],
                    vis_image=payload[1])
            dialog = TunerDialog(str(path), hsv_objects=[spec], algorithm_ids=["shift_移位"],
                                 preview_runner=run, apply_callback=lambda s, hsv: applied.append(hsv))
            dialog._sliders["v_hi"].setValue(123)
            self.wait(lambda: "REVIEW" in dialog.lbl_run.text())
            self.assertEqual(calls[-1][2]["feature_v_hi"], 123)
            dialog._apply()
            self.assertEqual(applied[-1]["v_hi"], 123)
            dialog.cmb_aux.setCurrentText("shift_移位/mask")
            self.assertEqual(dialog.canvas._roi_boxes, [])
            dialog.cmb_aux.setCurrentText("H通道")
            dialog.cmb_pre.setCurrentText("灰度")
            self.assertEqual(dialog.canvas._pixmap.width(), 48)
            dialog.reject()
            self.assertTrue(dialog._preview.closed)


if __name__ == "__main__":
    print("[tuner] 开始合成小图/离屏Qt定向验证", flush=True)
    unittest.main(verbosity=2)
