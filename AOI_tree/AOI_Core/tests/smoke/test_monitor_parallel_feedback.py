"""离屏监控定向 smoke：仅模拟客户端与异步回调，不使用生产库或模型前向。"""
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QHeaderView, QMessageBox

from ui.pages import monitor_page as M
from ui.widgets import feedback_dialog as F

_app = QApplication.instance() or QApplication([])


class FakeClient(QObject):
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.learned = []
        self.polls = []

    def workorder_learn(self, wid):
        self.learned.append(wid)
        return {"message": "学习提升完成"}

    def get_workorder(self, wid):
        return {"pipeline_status": "stopped", "auto_resume": True}

    def detections_live(self, **kwargs):
        self.polls.append(kwargs)
        return {"items": [frame(2)]}

    def file_url(self, path):
        return path


def frame(did):
    return {"id": did, "final_score": 1.25, "feedback_supported": True,
            "n_tiles": {"decision": "normal"}, "category": "测试品类"}


def immediate(_owner, work, done, fail=None):
    try:
        result = work()
    except Exception as exc:
        if fail:
            fail(str(exc))
        else:
            raise
    else:
        done(result)


class DeferredLoader(QObject):
    loaded = Signal(bytes)
    finished = Signal()
    instances = []

    def __init__(self, client, url, parent=None):
        super().__init__(parent)
        self.instances.append(self)

    def start(self):
        pass


class MonitorParallelSmoke(unittest.TestCase):
    def setUp(self):
        print(f"[monitor smoke] {self._testMethodName}", flush=True)
        self.async_patch = patch.object(M, "run_async", immediate)
        self.async_patch.start()
        self.addCleanup(self.async_patch.stop)
        self.client = FakeClient()
        self.page = M.MonitorPage(self.client, lambda: "测试品类")
        self.page._get_workorder_id = lambda: 7
        self.page._pipe_timer.stop()
        self.page._live_timer.stop()
        self.addCleanup(self.page.close)
        self.addCleanup(self.page.deleteLater)

    def test_queue_empty_to_full_layout_and_caliber(self):
        p = self.page
        p.resize(1440, 1100)
        p.show()
        p._fill_queue({})
        self.assertEqual(p.queue_table.columnSpan(0, 0), 9)
        name = "很长的检测批次身份名称" * 8
        p._fill_queue({"items": [{"name": name, "source": name, "category": name,
                                 "batch_key": "batch-key", "n_images": 8,
                                 "n_detected": 12, "n_anomaly": 6,
                                 "n_feedback": 3, "bad_batch": True}]})
        _app.processEvents()
        table = p.queue_table
        self.assertEqual(table.columnSpan(0, 0), 1)
        self.assertEqual([table.item(0, c).text() for c in range(3, 8)],
                         ["8", "12", "6", "3", "待关注"])
        self.assertEqual(p.monitor_splitter.orientation(), Qt.Vertical)
        self.assertTrue(p.monitor_splitter.widget(1).isAncestorOf(table))
        self.assertEqual(table.horizontalScrollBar().maximum(), 0)
        for c in range(9):
            self.assertEqual(table.horizontalHeader().sectionResizeMode(c), QHeaderView.Stretch)
            self.assertGreater(table.columnWidth(c),
                               table.fontMetrics().horizontalAdvance(table.horizontalHeaderItem(c).text()))
        self.assertTrue(table.cellWidget(0, 0).wordWrap())
        self.assertEqual(table.cellWidget(0, 0).toolTip(), name)
        self.assertIn("跨工单", table.toolTip())
        self.assertIn("不能证明错检", table.toolTip())
        table.selectRow(0)
        self.assertEqual(p._selected_queue_keys(), ["batch-key"])
        p._fill_queue({})
        self.assertIsNone(table.cellWidget(0, 0))

    def test_lock_poll_and_independent_fixed_feedback(self):
        p = self.page
        p.show()
        p._on_live_batch({"items": [frame(1)]})
        p._current_detection["nested"] = {"value": [1]}
        p._open_feedback("false_positive")
        dlg = p._feedback_dialog
        self.addCleanup(dlg.reject)
        self.assertFalse(dlg.isModal())
        self.assertFalse(p.chk_follow.isChecked())
        p._current_detection["nested"]["value"].append(2)
        self.assertEqual(dlg._det["nested"]["value"], [1])
        p._poll_live()
        self.assertEqual(len(self.client.polls), 1)
        self.assertEqual(p._last_id, 2)
        self.assertEqual(p._current_detection["detection_id"], 1)
        p.chk_follow.setChecked(True)
        self.assertEqual(p._current_detection["detection_id"], 2)
        self.assertEqual(dlg._det["detection_id"], 1)
        p._open_feedback("false_negative")
        self.assertIs(p._feedback_dialog, dlg)
        dlg.reject()
        self.assertIsNone(p._feedback_dialog)
        p._open_feedback("confirmed")
        self.assertEqual(p._feedback_dialog._det["detection_id"], 2)
        p._feedback_dialog.reject()

    def test_locked_frame_survives_window_rollover(self):
        p = self.page
        p._on_live_batch({"items": [frame(1)]})
        p.btn_lock_frame.click()
        p._on_live_batch({"items": [frame(i) for i in range(2, 54)]})
        self.assertEqual(len(p._frames), 50)
        self.assertEqual(p._selected_frame, -1)
        self.assertEqual(p._current_detection["detection_id"], 1)
        p.btn_latest.click()
        self.assertEqual(p._current_detection["detection_id"], 53)

    def test_snapshot_confirmation_and_lifecycle(self):
        p = self.page
        with patch.object(M.QMessageBox, "question", return_value=QMessageBox.No) as question:
            p._pipeline_learn()
        self.assertEqual(self.client.learned, [])
        confirmation = question.call_args.args[2]
        for text in ("pending", "已停止", "auto_resume", "品类", "不自动"):
            self.assertIn(text, confirmation)
        with patch.object(M.QMessageBox, "question", return_value=QMessageBox.Yes), \
                patch.object(M, "notify") as notify:
            p._pipeline_learn()
        self.assertEqual(self.client.learned, [7])
        self.assertIn("不代表", notify.call_args.args[1])
        self.assertTrue(p.btn_pp_learn.isEnabled())
        self.assertIn("不强杀", p.btn_pp_pause.toolTip())
        self.assertIn("保留普通数据", p.btn_pp_stop.toolTip())
        for text in ("Image", "Detection", "Feedback", "候选构建", "验收", "激活", "50 帧", "CSV"):
            self.assertIn(text, p.lbl_dataflow.text())
        targets = []
        p.goto_page.connect(targets.append)
        p.btn_go_feedback.click()
        self.assertEqual(targets, ["标注反馈"])
        p._update_slot_bars({"sem": 1.25})
        self.assertEqual(p._slot_bars["sem"][1].format(), "1.250")
        self.assertIn("不是缺陷概率", p.gauge.toolTip())

    def test_pair_requires_explicit_confirmation_and_scopes_image(self):
        p = self.page
        p._pair_settings = Mock()
        saved = {}
        p._pair_settings.value.side_effect = lambda key, default: saved.get(key, default)
        p._pair_settings.setValue.side_effect = saved.__setitem__
        det = {"detection_id": 1, "category": "测试品类", "target_type": "pipeline",
               "image_path": "frame.png"}
        identity = {"sha256": "real-content", "size": [964, 653]}
        self.assertIsNone(p._pair_binding(det))
        p._current_detection = det
        with patch.object(p, "_pair_identity", return_value=identity), \
                patch.object(M.QFileDialog, "getOpenFileName", return_value=("template.png", "")), \
                patch.object(M.QMessageBox, "question", return_value=QMessageBox.No), \
                patch.object(p, "_invalidate_pair_layers"):
            p._confirm_pair()
            self.assertEqual(saved, {})
        with patch.object(p, "_pair_identity", return_value=identity), \
                patch.object(M.QFileDialog, "getOpenFileName", return_value=("template.png", "")), \
                patch.object(M.QMessageBox, "question", return_value=QMessageBox.Yes), \
                patch.object(p, "_invalidate_pair_layers"):
            p._confirm_pair()
            self.assertEqual(p._pair_binding(det), ("frame.png", "template.png"))
            self.assertIsNone(p._pair_binding(dict(det, image_path="other.png")))
            p._get_workorder_id = lambda: 8
            self.assertIsNone(p._pair_binding(det))
        self.assertEqual(len(saved), 1)
        p._pair_settings.sync.assert_called_once()

    def test_pair_rejects_different_dimensions(self):
        p = self.page
        p._current_detection = {"detection_id": 1, "category": "测试品类",
                                "target_type": "pipeline", "image_path": "frame.png"}
        p._pair_settings = Mock()
        for size in ([712, 704], [746, 740]):
            with patch.object(p, "_pair_identity", side_effect=[
                    {"sha256": "a", "size": size},
                    {"sha256": "b", "size": [964, 653]}]), \
                    patch.object(M.QFileDialog, "getOpenFileName", return_value=("template.png", "")), \
                    patch.object(M, "warn") as warning, \
                    patch.object(M.QMessageBox, "question") as question:
                p._confirm_pair()
                self.assertIn("禁止自动缩放", warning.call_args.args[1])
                question.assert_not_called()
        p._pair_settings.setValue.assert_not_called()

    def test_switch_to_unpaired_frame_discards_late_pair(self):
        p = self.page
        callbacks = []
        first = {"detection_id": 1, "image_path": "first.png", "pair_overlay_path": "stale.png"}
        second = {"detection_id": 2, "image_path": "second.png"}
        with patch.object(p, "_pair_binding", side_effect=lambda det:
                          ("first.png", "template.png") if det["detection_id"] == 1 else None), \
                patch.object(M, "run_async", side_effect=lambda owner, work, done, fail: callbacks.append(done)), \
                patch.object(p.viewer, "set_image_url"), \
                patch.object(p.viewer, "clear_images") as clear:
            p._show_detection(first)
            p._show_detection(second)
            callbacks[0]({"overlay_path": "late.png"})
            self.assertEqual(p._current_detection["detection_id"], 2)
            self.assertNotIn("pair_overlay_path", second)
            self.assertEqual(clear.call_count, 2)
            self.assertEqual(p._pair_cache, {})

    def test_pair_request_survives_repeat_show_and_uses_single_inflight(self):
        p = self.page
        callbacks = []
        det = {"detection_id": 1077, "image_path": "frame.png"}
        with patch.object(p, "_pair_binding", return_value=("frame.png", "template.png")), \
                patch.object(M, "run_async", side_effect=lambda owner, work, done, fail: callbacks.append(done)), \
                patch.object(p.viewer, "set_image_url"), \
                patch.object(p.viewer, "clear_images"):
            p._show_detection(det)
            p._show_detection(det)
            self.assertEqual(len(callbacks), 1)
            callbacks[0]({"heatmap_path": "heat.jpg", "overlay_path": "overlay.jpg"})
        self.assertEqual(p._current_detection["pair_overlay_path"], "overlay.jpg")
        self.assertEqual(p._pair_cache[("frame.png", "template.png")]["pair_overlay_path"], "overlay.jpg")
        self.assertFalse(p._pair_inflight)

    def test_feedback_images_ignore_late_or_wrong_id(self):
        DeferredLoader.instances = []
        with patch.object(F, "_ThumbLoader", DeferredLoader):
            dlg = F.FeedbackDialog(self.client, {"detection_id": 1})
            apply = Mock()
            dlg._load_image("image-a", apply)
            first = DeferredLoader.instances[-1]
            first.loaded.emit(b"current")
            apply.assert_called_once_with(b"current")
            dlg._det["detection_id"] = 2
            first.loaded.emit(b"wrong-id")
            self.assertEqual(apply.call_count, 1)
            dlg._load_image("image-b", apply)
            second = DeferredLoader.instances[-1]
            dlg.reject()
            second.loaded.emit(b"closed")
            self.assertEqual(apply.call_count, 1)
            for loader in DeferredLoader.instances:
                loader.finished.emit()
            dlg.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
