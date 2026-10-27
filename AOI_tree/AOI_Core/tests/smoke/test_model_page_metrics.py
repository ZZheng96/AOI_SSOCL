"""模型管理：真实版本评估绑定与离屏展示，不训练、不访问生产数据。"""
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from ui.pages import model_page as M
from ui.api_client import ApiClient

_app = QApplication.instance() or QApplication([])


class Client(QObject):
    error_occurred = Signal(str)


def run(i, version=2, **extra):
    metrics = {"model_version": version, "status": "completed", "auroc": 0.8,
               "f1": 0.75, "fp": 1, "fn": 2, "threshold": 0.3}
    metrics.update(extra)
    return {"id": i, "category": "board", "run_type": "accuracy",
            "n_images": 8, "dataset_desc": "split=test", "metrics": metrics}


class ModelMetricsTest(unittest.TestCase):
    def setUp(self):
        print(f"[模型管理测试] {self._testMethodName}", flush=True)
        self.model = {"id": 7, "category": "board", "version": "v2",
                      "is_active": True, "metrics": {}}

    def test_binding_excludes_unbound_wrong_and_incomplete(self):
        valid = run(1)
        unbound = run(8)
        del unbound["metrics"]["model_version"]
        other = run(9)
        other["category"] = "other"
        benchmark = run(10)
        benchmark["run_type"] = "benchmark"
        cases = [valid, run(2, 3), run(3, status="failed"), run(4, status="running"),
                 run(5, "bad"), run(6, model_id=99), unbound, other, benchmark]
        self.assertEqual(M._latest_bound_eval(cases, self.model)["id"], 1)
        self.assertEqual(M._latest_bound_eval([valid, run(11, "v2")], self.model)["id"], 11)
        self.assertIsNone(M._latest_bound_eval([unbound], self.model))
        by_id = run(12, None, model_id=7)
        self.assertEqual(M._latest_bound_eval([by_id], self.model)["id"], 12)

    def test_offscreen_metrics_and_missing_slots(self):
        p = M.ModelPage(Client(), lambda: "board", None)
        p._cur = {"category": "board", "datasource_name": "A", "groups": {}}
        p._stack.setCurrentIndex(1)
        p._fill_models([self.model])
        p._fill_evals([run(1), run(2, 3, auroc=0.99)])
        self.assertIn("EvalRun #1", p.lbl_eval.text())
        self.assertIn("AUROC=0.8000", p.lbl_eval.text())
        self.assertIn("非生产阈值", p.lbl_eval.text())
        self.assertIn("未知", p.lbl_slots.text())
        self.model["metrics"]["slot_names"] = ["sem", "disc"]
        p._refresh_panel()
        self.assertIn("正常语义记忆", p.lbl_slots.text())
        self.assertNotIn("模板", p.lbl_slots.text())
        p._fill_evals([run(3, 1)])
        self.assertIn("未评估", p.lbl_eval.text())
        self.assertEqual(p.lbl_eval.toolTip(), "")
        p._fill_evals([run(4, auroc=None)])
        self.assertIn("AUROC=—", p.lbl_eval.text())
        p._cur["datasource_name"] = "B"
        p._refresh_panel()
        self.assertIn("EvalRun #4", p.lbl_eval.text())
        p._fill_evals(None)
        self.assertIn("暂不可用", p.lbl_eval.text())
        self.assertEqual(p.btn_learn.text(), "固化学习快照")
        p.close()

    def test_all_pages_and_failure_not_partial_success(self):
        client = ApiClient()
        client.get = Mock(side_effect=[{"total": 2, "items": [run(1)]},
                                       {"total": 2, "items": [run(2)]}])
        self.assertEqual(len(client.list_eval_runs(all_pages=True)), 2)
        self.assertEqual(client.get.call_args.kwargs["params"]["page"], 2)
        client.get = Mock(side_effect=[{"total": 2, "items": [run(1)]}, None])
        self.assertIsNone(client.list_eval_runs(all_pages=True))

    def test_help_dialog_opens_offscreen(self):
        p = M.ModelPage(Client(), lambda: "board", None)
        with patch.object(M.QDialog, "exec", return_value=0):
            p._show_model_help()
        self.assertIn("并行", M.MODEL_HELP)
        self.assertIn("auto_activate", M.MODEL_HELP)
        p.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
