"""评估看板定向离屏测试：不加载真实模型、不连接生产数据库。"""
import ast
from contextlib import contextmanager
from pathlib import Path
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from ui.pages import eval_page as ep
from ui.widgets.charts import LineChart


class Client(QObject):
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.list_eval_runs = Mock(return_value=[])
        self.get_config = Mock(return_value={"items": [
            {"section": "evaluation", "key": "benchmark_n_images", "value": 37},
            {"section": "evaluation", "key": "benchmark_warmup", "value": 7}]})
        self.eval_accuracy = Mock(return_value={"task_id": 42})
        self.eval_benchmark = Mock(return_value={"task_id": 43})
        self.eval_robustness = Mock(return_value={"task_id": 44})
        self.get_task = Mock(return_value={"id": 42, "status": "running", "progress": 100})


def immediate(parent, fn, on_ok=None, on_fail=None):
    try:
        result = fn()
    except Exception as exc:
        if on_fail:
            on_fail(str(exc))
        else:
            raise
    else:
        if on_ok:
            on_ok(result)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        print(f"[eval UI] {self._testMethodName}", flush=True)
        self.async_patch = patch.object(ep, "run_async", immediate)
        self.async_patch.start()
        self.warn_patch = patch.object(ep, "warn")
        self.warn_patch.start()
        self.client = Client()
        self.page = ep.EvalPage(self.client, lambda: "demo")

    def tearDown(self):
        self.page._poll_timer.stop()
        self.page.close()
        self.page.deleteLater()
        self.async_patch.stop()
        self.warn_patch.stop()

    def test_three_entries_task_mutex_poll_and_terminal(self):
        self.assertEqual(len([b for b in self.page.findChildren(QPushButton) if b.text().startswith("运行")]), 3)
        self.assertEqual([s.value() for s in self.page.sample_limits.values()], [20, 37, 12])
        self.client.list_eval_runs.assert_called_with(all_pages=True)
        self.page._on_run("accuracy")
        self.assertEqual(self.page._task_id, "42")
        self.assertTrue(all(not b.isEnabled() for b in self.page.run_buttons.values()))
        self.page._on_run("benchmark")
        self.client.eval_benchmark.assert_not_called()
        self.page._on_task({"id": 99, "status": "done"})
        self.assertEqual(self.page._task_id, "42")
        self.client.get_task.return_value = {"id": 42, "status": "done"}
        self.page._poll_task()
        self.assertIsNone(self.page._task_id)
        self.assertTrue(all(b.isEnabled() for b in self.page.run_buttons.values()))
        self.assertFalse(self.page._poll_timer.isActive())

    def test_submission_pending_and_failure(self):
        callbacks = []
        def delayed(parent, fn, on_ok=None, on_fail=None):
            callbacks.append((fn, on_ok, on_fail))
        with patch.object(ep, "run_async", delayed):
            self.page._on_run("accuracy")
            self.assertTrue(self.page._submitting)
            self.assertTrue(all(not b.isEnabled() for b in self.page.run_buttons.values()))
            self.page._on_run("benchmark")
            self.assertEqual(len(callbacks), 1)
            callbacks[0][2]("test failure")
        self.assertFalse(self.page._submitting)
        self.assertTrue(all(b.isEnabled() for b in self.page.run_buttons.values()))

    def test_other_entries_and_robust_cost_confirmation(self):
        self.page._on_run("benchmark")
        self.client.eval_benchmark.assert_called_once_with("demo", n_images=37)
        self.page._on_task({"id": 43, "status": "failed"})
        with patch.object(ep.QMessageBox, "question", return_value=ep.QMessageBox.Yes) as question:
            self.page._on_run("robustness")
            self.assertIn("192", question.call_args.args[2])
        self.client.eval_robustness.assert_called_once_with("demo", split="test", limit=12)
        self.page._on_task({"id": 44, "status": "cancelled"})
        self.assertIsNone(self.page._task_id)

    def test_benchmark_refresh_preserves_explicit_sample_count(self):
        spin = self.page.sample_limits["benchmark"]
        self.assertIn("7 次预热", spin.toolTip())
        spin.setValue(11)
        self.page.reload()
        self.assertEqual(spin.value(), 11)
        self.page._on_run("benchmark")
        self.client.eval_benchmark.assert_called_once_with("demo", n_images=11)

    def test_records_unknown_rates_and_missing_latency(self):
        run = {"id": 1, "run_type": "accuracy", "n_images": 20,
               "metrics": {"model_version": 7, "auroc": .99, "tp": 6, "fn": 2, "fp": 1, "tn": 11}}
        self.page._fill_runs([run])
        table = self.page.table
        self.assertEqual(table.columnCount(), 12)
        self.assertEqual(table.item(0, 8).text(), "v7")
        self.assertEqual(table.item(0, 9).text(), "20")
        self.assertEqual(table.item(0, 10).text(), "75.0%")
        self.assertEqual(table.item(0, 11).text(), "8.3%")
        self.assertNotIn("通过", table.item(0, 7).text())
        self.assertNotIn("漏检率", self.page.lbl_metrics_detail.text())
        self.assertEqual(ep._rate_text({"tp": 0, "fn": 0, "recall_defect": 0}, "tp", "fn"), "未知")
        self.assertEqual(ep._rate_text({}, "fp", "tn"), "未知")
        self.page._fill_runs([])
        self.assertEqual(table.columnSpan(0, 0), 12)

    def test_embedded_curves_positions_missing_and_history(self):
        run = {"id": 2, "run_type": "robustness", "metrics": {
            "levels": [0, .2, .8], "curves": {"gauss": {"auc_levels": [1, None, .6], "a_rob": None}}}}
        self.page._fill_runs([run])
        self.assertFalse(isinstance(self.page.robustness_panel, QDialog))
        self.assertFalse(self.page.robustness_panel.isHidden())
        chart = self.page.robustness_panel.findChild(LineChart)
        np.testing.assert_array_equal(chart.plot.listDataItems()[0].xData, [0, 1, 2])
        self.assertTrue(np.isnan(chart.plot.listDataItems()[0].yData[1]))
        self.assertEqual(chart.plot.getAxis("bottom")._tickLevels[0][2], (2, "0.8"))
        for metrics in ({"curves": {"blur": {"auc_levels": [1, .8]}}}, {"curves": {"blur": {}}}, {"curves": {}}):
            self.page.robustness_panel.set_run({"metrics": metrics})
        self.assertIn("历史记录", self.page.robustness_panel._layout.itemAt(3).widget().text())


def load_service_functions():
    """仅提取目标服务函数，杜绝导入期生产存储初始化。"""
    path = Path(__file__).resolve().parents[2] / "backend/evaluation/service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ("run_accuracy_eval", "run_robustness_eval", "run_latency_benchmark", "_latency_stats")]
    namespace = {"__name__": "backend.evaluation.service", "__package__": "backend.evaluation",
                 "np": np, "Path": Path}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + funcs,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class ServiceTests(unittest.TestCase):
    def setUp(self):
        print(f"[eval service] {self._testMethodName}", flush=True)
        self.ns = load_service_functions()
        self.query = Mock()
        self.query.filter.return_value = self.query
        self.query.order_by.return_value = self.query
        self.query.limit.return_value = self.query
        self.session = Mock()
        self.session.query.return_value = self.query
        @contextmanager
        def scope():
            yield self.session
        self.ns.update(session_scope=scope, ImageRow=SimpleNamespace(
            id=Mock(), category=Mock(), split=Mock(), label=Mock()),
            EvalRun=lambda **kwargs: SimpleNamespace(id=1, **kwargs), log_action=Mock())

    def test_benchmark_config_defaults_and_explicit_overrides(self):
        from datetime import datetime
        from backend.api import schemas
        from ui.api_client import ApiClient
        settings = SimpleNamespace(get=lambda section, key, default=None: {
            "benchmark_n_images": 4, "benchmark_warmup": 2}.get(key, default))
        with patch.object(schemas, "get_settings", return_value=settings):
            req = schemas.EvalBenchmarkRequest(category="demo")
            self.assertEqual((req.n_images, req.warmup), (4, 2))
            req = schemas.EvalBenchmarkRequest(category="demo", n_images=3, warmup=0)
            self.assertEqual((req.n_images, req.warmup), (3, 0))
        client = SimpleNamespace(post=Mock())
        ApiClient.eval_benchmark(client, "demo")
        client.post.assert_called_with("/api/eval/benchmark", json={"category": "demo"})
        ApiClient.eval_benchmark(client, "demo", n_images=3, warmup=0)
        client.post.assert_called_with("/api/eval/benchmark", json={
            "category": "demo", "n_images": 3, "warmup": 0})
        self.ns["ImageRow"].width = Mock()
        self.ns["ImageRow"].width.__ge__ = Mock(return_value=True)
        self.query.all.return_value = [SimpleNamespace(id=1, path="fake.png")]
        detector = Mock()
        detector.detect_image.return_value = {"latency_ms": 1}
        self.ns.update(get_settings=lambda: settings, datetime=datetime,
                       get_detection_service=lambda: detector)
        result = self.ns["run_latency_benchmark"]("demo")
        self.assertEqual(len(result["per_image_ms"]), 4)
        self.assertEqual(detector.detect_image.call_count, 6)
        detector.detect_image.reset_mock()
        result = self.ns["run_latency_benchmark"]("demo", n_images=3, warmup=0)
        self.assertEqual(len(result["per_image_ms"]), 3)
        self.assertEqual(detector.detect_image.call_count, 3)

    def test_accuracy_limit_one_zero_quota(self):
        self.query.all.side_effect = [[], []]
        config = SimpleNamespace(get_settings=lambda: SimpleNamespace(governance_decisions=lambda: {"approved_extra_good_batches": ""}))
        self.ns["get_settings"] = config.get_settings
        with self.assertRaisesRegex(ValueError, "无有标注图像"):
            self.ns["run_accuracy_eval"]("demo", limit=1)
        self.assertEqual([c.args[0] for c in self.query.limit.call_args_list], [1, 0])

    def test_robustness_full_progress_levels_and_class_validation(self):
        import sys
        from datetime import datetime
        rows = [SimpleNamespace(id=i, path=f"fake{i}.png", label="anomaly" if i < 4 else "normal") for i in range(8)]
        self.query.all.side_effect = [rows[:4], rows[4:]]
        engine = SimpleNamespace(list_versions=lambda c: [3], current_version=lambda c: 3,
                                 build_variant=lambda c, v: SimpleNamespace(predict=lambda p: {"fused": .5}))
        config = SimpleNamespace(get_settings=lambda: SimpleNamespace(storage=lambda name: Path("fake-eval-tmp")))
        cv = SimpleNamespace(IMREAD_COLOR=1, COLOR_BGR2RGB=1, COLOR_RGB2BGR=2,
                             imread=lambda *a: np.zeros((2, 2, 3), dtype=np.uint8),
                             cvtColor=lambda a, *b: a, imwrite=lambda *a: True)
        self.ns["datetime"] = datetime
        progress = []
        with patch.dict(sys.modules, {"backend.engine": SimpleNamespace(get_engine=lambda: engine),
                                     "backend.core.config": config, "cv2": cv}), patch.object(Path, "unlink"), patch.object(Path, "mkdir"):
            result = self.ns["run_robustness_eval"]("demo", limit=8, noise_kinds=["brightness"],
                                                    levels=[0, .8], progress_cb=lambda d, t: progress.append((d, t)))
            self.assertEqual(progress, [(i, 24) for i in range(1, 25)])
            self.assertEqual(result["levels"], [0, .8])
            saved = self.session.add.call_args.args[0]
            self.assertEqual(saved.metrics["levels"], [0, .8])
            self.query.all.side_effect = [rows[:4] * 2, []]
            with self.assertRaisesRegex(ValueError, "两类齐全"):
                self.ns["run_robustness_eval"]("demo", limit=32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
