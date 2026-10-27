"""请求内批量的精确等价与真实调度定向验证（小合成图）。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import dataclasses
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from app.detect.adapters.smt_adapter import SmtAdapter, DETECT_FLAG_BY_ALG, _load_smt
from app.detect.adapters.through_hole_adapter import ThroughHoleAdapter, DEFECT_ID_BY_ALG, _th_runtime
from app.detect.contract import Roi, job_status
from app.detect.types import DetectRequest, AlgorithmResult
from app.detect.scheduler import DetectScheduler, _run_with_timeout, inflight_snapshot
from app.recipe.store import RecipeStore
from app.recipe.param_store import ParamStore


class BatchTests(unittest.TestCase):
    def equal(self, a, b, path=""):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=path)
        elif dataclasses.is_dataclass(a):
            for field in dataclasses.fields(a):
                if field.name != "elapsed_ms":
                    self.equal(getattr(a, field.name), getattr(b, field.name), path + "." + field.name)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys(), path)
            for key in a:
                if key not in {"total_ms"}:
                    self.equal(a[key], b[key], path + "." + str(key))
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), path)
            for x, y in zip(a, b):
                self.equal(x, y, path)
        elif isinstance(a, float) and np.isnan(a):
            self.assertTrue(np.isnan(b))
        else:
            self.assertEqual(a, b, path)

    def request(self, kind="smt", multi=False, diagnostics=True):
        image = np.zeros((96, 96, 3), np.uint8)
        cv2.circle(image, (48, 48), 16, (200, 200, 200), -1)
        roi = Roi(id="pad", kind=kind, layer="smt_pads" if kind == "smt" else "th",
                  shape={"shape": "rect", "x": 28, "y": 28, "w": 40, "h": 40})
        rois = [roi]
        if multi:
            rois.append(Roi(id="pad2", kind=kind, layer=roi.layer,
                            shape={"shape": "rect", "x": 4, "y": 4, "w": 18, "h": 18}))
        return DetectRequest(image_test=image, image_std=image.copy(), rois=rois,
                             preprocess_meta={"diagnostics": diagnostics})

    def test_smt_exact_flags_toe_rim_and_shared_extraction(self):
        adapter = SmtAdapter()
        req = self.request()
        req.rois.extend([Roi(id=layer, kind="smt", layer=layer,
                            shape={"x": 30, "y": 30, "w": 10, "h": 10})
                         for layer in ("smt_toe", "smt_rim")])
        calls = [(alg, req, {}) for alg in DETECT_FLAG_BY_ALG]
        expected = [adapter.run(*call) for call in calls]
        from unittest.mock import Mock
        globals_ = _load_smt()["klass"].run.__globals__
        extract = Mock(wraps=globals_["extract_solder"])
        with patch.dict(globals_, extract_solder=extract):
            actual = adapter.run_many(calls)
        self.equal(expected, actual)
        self.assertEqual(extract.call_count, 1)
        flags = [(alg, req, {"detect_insufficient": True, "detect_cold_solder": False})
                 for alg in DETECT_FLAG_BY_ALG]
        self.equal([adapter.run(*c) for c in flags], adapter.run_many(flags))

    def test_smt_configuration_and_input_isolation(self):
        adapter = SmtAdapter()
        req = self.request()
        other = self.request()
        other.rois[0].shape["x"] = 20
        calls = [("solder_多锡", req, {}), ("solder_少锡", req, {"solder_blue_v_min": 250}),
                 ("solder_虚焊", other, {})]
        from unittest.mock import Mock
        globals_ = _load_smt()["klass"].run.__globals__
        extract = Mock(wraps=globals_["extract_solder"])
        with patch.dict(globals_, extract_solder=extract):
            actual = adapter.run_many(calls)
        self.assertEqual(extract.call_count, 3)
        self.equal([adapter.run(*c) for c in calls], actual)
        req.image_std = None
        self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))

    def test_th_exact_single_multi_review_and_configuration(self):
        adapter = ThroughHoleAdapter()
        for multi in (False, True):
            req = self.request("th", multi)
            calls = [(alg, req, {}) for alg in DEFECT_ID_BY_ALG]
            expected = [adapter.run(*c) for c in calls]
            with _th_runtime():
                import importlib
                module = importlib.import_module("algorithms.pcb_through_hole.pcb_through_hole_solder_alg")
                import contextlib
                with patch("app.detect.adapters.through_hole_adapter._th_runtime", side_effect=contextlib.nullcontext), \
                     patch.object(module, "run_inspect", wraps=module.run_inspect) as inspect_run:
                    actual = adapter.run_many(calls)
                self.assertEqual(inspect_run.call_count, 1)
            self.equal(expected, actual)
        req = self.request("th")
        calls = [("tht_孔洞", req, {}), ("tht_少锡", req, {"void_preset": "HIGH"})]
        with _th_runtime():
            import importlib
            module = importlib.import_module("algorithms.pcb_through_hole.pcb_through_hole_solder_alg")
            import contextlib
            with patch("app.detect.adapters.through_hole_adapter._th_runtime", side_effect=contextlib.nullcontext), \
                 patch.object(module, "run_inspect", wraps=module.run_inspect) as inspect_run:
                actual = adapter.run_many(calls)
            self.assertEqual(inspect_run.call_count, 2)
        self.equal([adapter.run(*c) for c in calls], actual)
        req.image_std = None
        self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))

    def test_errors_and_unready_are_identical(self):
        adapter = SmtAdapter()
        req = self.request()
        calls = [(alg, req, {}) for alg in DETECT_FLAG_BY_ALG]
        with patch.object(_load_smt()["instance"], "run", side_effect=RuntimeError("synthetic failure")):
            self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))
        req.rois = []
        self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))
        import contextlib
        import importlib
        from types import SimpleNamespace
        adapter = ThroughHoleAdapter()
        req = self.request("th")
        calls = [(alg, req, {}) for alg in DEFECT_ID_BY_ALG]
        with _th_runtime():
            module = importlib.import_module("algorithms.pcb_through_hole.pcb_through_hole_solder_alg")
            with patch("app.detect.adapters.through_hole_adapter._th_runtime", side_effect=contextlib.nullcontext):
                for payload in (SimpleNamespace(code=1, message="bad config"),
                                SimpleNamespace(code=0, message="review", parts=[],
                                    metadata={"status": "REVIEW", "review_reason": "low coverage", "output_image": req.image_test})):
                    with patch.object(module.PcbThroughHoleSolderAlg, "run", return_value=payload):
                        self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))
                with patch.object(module.PcbThroughHoleSolderAlg, "run", side_effect=RuntimeError("synthetic failure")):
                    self.equal([adapter.run(*c) for c in calls], adapter.run_many(calls))

    def test_scheduler_uses_batch_and_real_shared_computation(self):
        with tempfile.TemporaryDirectory() as temp:
            scheduler = DetectScheduler(RecipeStore(Path(temp) / "recipes"), ParamStore(Path(temp)))
            for kind, adapter, ids in (("smt", SmtAdapter(), list(DETECT_FLAG_BY_ALG)),
                                       ("th", ThroughHoleAdapter(), list(DEFECT_ID_BY_ALG))):
                req = self.request(kind)
                regions = {"smt_pads" if kind == "smt" else "th": [r.shape for r in req.rois]}
                import contextlib
                import importlib
                from unittest.mock import Mock
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch("app.detect.scheduler.get_adapter_for", return_value=adapter))
                    many = stack.enter_context(patch.object(adapter, "run_many", wraps=adapter.run_many))
                    if kind == "smt":
                        globals_ = _load_smt()["klass"].run.__globals__
                        work = Mock(wraps=globals_["extract_solder"])
                        stack.enter_context(patch.dict(globals_, extract_solder=work))
                    else:
                        stack.enter_context(_th_runtime())
                        module = importlib.import_module("algorithms.pcb_through_hole.pcb_through_hole_solder_alg")
                        stack.enter_context(patch("app.detect.adapters.through_hole_adapter._th_runtime", side_effect=contextlib.nullcontext))
                        work = stack.enter_context(patch.object(module, "run_inspect", wraps=module.run_inspect))
                    kwargs = dict(region_sets=regions, gate_enabled=False, diagnostics=True)
                    scalar = scheduler.run(req.image_std, req.image_test, None, None, ids,
                                           batch_enabled=False, **kwargs)
                    self.assertEqual(work.call_count, 4)
                    work.reset_mock()
                    result = scheduler.run(req.image_std, req.image_test, None, None, ids, **kwargs)
                    self.assertEqual(work.call_count, 1 if kind == "smt" else 2)
                    print(f"scheduler {kind}: scalar shared work=4, batch shared work={work.call_count}, adapter.run_many={many.call_count}", flush=True)
                    for before, after in zip(scalar.results, result.results):
                        self.assertEqual(before.status, after.status)
                        self.equal(before.boxes, after.boxes)
                        if before.diff_image is not None:
                            np.testing.assert_array_equal(before.diff_image, after.diff_image)
                self.assertEqual(many.call_count, 1)
                ready_results = [r for r in result.results if r.metadata.get("execution_mode") == "production_adapter_batch"]
                self.assertEqual(len(ready_results), 4)
                self.assertTrue(all(r.metadata["request_snapshot"]["version_id"] for r in ready_results))
                self.assertEqual(result.overall_ok, all(r.status == "OK" for r in result.results))
                with patch("app.detect.scheduler.get_adapter_for", return_value=adapter), \
                     patch.object(adapter, "run_many", side_effect=RuntimeError("batch failure")):
                    failed = scheduler.run(req.image_std, req.image_test, None, None, ids,
                        region_sets=regions, gate_enabled=False)
                self.assertFalse(failed.overall_ok)
                self.assertTrue(all(r.error_code == "RUN_EXCEPTION" for r in failed.results
                    if r.metadata.get("execution_mode") == "production_adapter_batch"))
                with patch("app.detect.scheduler.get_adapter_for", return_value=adapter), \
                     patch.object(adapter, "run_many") as stopped_many, \
                     patch.object(adapter, "run", return_value=AlgorithmResult(ids[0], False, "failure", status="ERROR")) as one:
                    stopped = scheduler.run(req.image_std, req.image_test, None, None, ids,
                        region_sets=regions, gate_enabled=False, stop_on_error=True)
                self.assertEqual(stopped_many.call_count, 0)
                self.assertEqual(one.call_count, 1)
                self.assertEqual(len(stopped.results), 1)
                with patch("app.detect.scheduler.get_adapter_for", return_value=adapter), \
                     patch.object(adapter, "run_many", wraps=adapter.run_many) as many:
                    scheduler.run(req.image_std, req.image_test, None, None, ids,
                                  region_sets=regions, gate_enabled=False, batch_enabled=False)
                self.assertEqual(many.call_count, 0)

    def test_frozen_shared_override_and_non_ok_states(self):
        with tempfile.TemporaryDirectory() as temp:
            scheduler = DetectScheduler(RecipeStore(Path(temp) / "recipes"), ParamStore(Path(temp)))
            adapter = SmtAdapter()
            params, _ = scheduler._resolve_params(adapter, "solder_少锡", None, None,
                {"solder_共用参数": {"solder_blue_v_min": 120},
                 "solder_少锡": {"solder_blue_v_min": 160, "insufficient_thresh": 0.2}}, True)
            self.assertEqual(params["solder_blue_v_min"], 160)
            self.assertEqual(params["insufficient_thresh"], 0.2)
        for state in ("REVIEW", "SKIP", "ERROR"):
            self.assertNotEqual(job_status(gate_blocked=False,
                items=[AlgorithmResult("a", False, "reason", status=state)]), "OK")

    def test_status_trace_table(self):
        from PySide6.QtWidgets import QApplication
        from app.ui.defect_table import DefectTable
        from app.detect.types import DetectSummary, DefectBox
        app = QApplication.instance() or QApplication([])
        results = [AlgorithmResult("solder_少锡", False, "NG reason", status="NG", elapsed_ms=7,
                    item_id="solder_少锡", boxes=[DefectBox(1, 2, 3, 4, "少锡", item_id="solder_少锡", roi_id="pad")])]
        results.extend(AlgorithmResult(state, False, "复核原因", status=state, item_id=state,
                       skip_reason="复核原因", elapsed_ms=3,
                       metadata={"request_snapshot": {"rois": [{"id": "pad"}]}})
                       for state in ("REVIEW", "SKIP", "ERROR"))
        summary = DetectSummary(overall_ok=True, results=results)
        self.assertFalse(summary.overall_ok)
        table = DefectTable()
        table.bind_summary(summary)
        self.assertEqual(table.table.rowCount(), 4)
        self.assertEqual(table.table.item(0, 4).text(), "solder_少锡")
        self.assertEqual(table.table.item(0, 5).text(), "pad")
        self.assertEqual(table.table.item(0, 7).text(), "7")
        for row, state in enumerate(("REVIEW", "SKIP", "ERROR"), 1):
            self.assertEqual(table.table.item(row, 6).text(), state)
            self.assertIn("复核原因", table.table.item(row, 9).text())
        table.close()

    def test_short_timeout_snapshot_release_and_no_retry(self):
        release = threading.Event()
        finished = threading.Event()
        calls = []
        def task():
            calls.append(1)
            release.wait(1)
            finished.set()
        with self.assertLogs("app.detect.scheduler", level="WARNING"):
            with self.assertRaises(TimeoutError):
                _run_with_timeout(task, 0.01, tries=3, key="batch-test-timeout")
        snapshot = inflight_snapshot()
        item = next(t for t in snapshot["tasks"] if t["key"] == "batch-test-timeout")
        self.assertTrue(item["timed_out"])
        self.assertFalse(item["cancellable"])
        with self.assertRaises(TimeoutError):
            _run_with_timeout(task, 0.01, key="batch-test-timeout")
        release.set()
        self.assertTrue(finished.wait(1))
        for _ in range(100):
            if not any(t["key"] == "batch-test-timeout" for t in inflight_snapshot()["tasks"]):
                break
            time.sleep(0.001)
        self.assertEqual(calls, [1])
        self.assertFalse(any(t["key"] == "batch-test-timeout" for t in inflight_snapshot()["tasks"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
