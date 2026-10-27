"""PCB 配对热图请求上下文与持久化回归测试。"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.detect.types import DetectSummary
from app.result.store import ResultStore
from app.template.model import InspectionTemplate, StandardImageRef
from app.ui.inspect_page import InspectPage


class PairHeatmapPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.standard = self.root / "standard.png"
        self.standard.write_bytes(b"standard")
        self.test_image = self.root / "board.png"
        self.test_image.write_bytes(b"board")
        self.tpl = InspectionTemplate(
            id="tpl-a", version=3,
            standard_image=StandardImageRef(path=str(self.standard)),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_does_not_call_pair_engine(self):
        page = InspectPage(template_store=Mock(), result_store=ResultStore(self.root / "out"))
        page.current_template = self.tpl
        page.queue = [self.test_image]
        page.current_index = 0
        with patch("app.engines.feature.get_engine") as get_engine:
            page._refresh_current_view()
            get_engine.assert_not_called()
        self.assertFalse(page.inspect_canvas.chk_pair.isChecked())
        page.deleteLater()

    def test_checkbox_sends_user_confirmation_and_context(self):
        page = InspectPage(template_store=Mock(), result_store=ResultStore(self.root / "out"))
        page.current_template = self.tpl
        page.queue = [self.test_image]
        page.current_index = 0
        with patch("app.engines.feature.get_engine", return_value=Mock()) as get_engine, \
                patch.object(page._pair_pool, "start") as start:
            page._pair_toggled(True)
            get_engine.assert_called_once()
            start.assert_called_once()
        self.assertEqual(page._pair_context, (str(self.test_image), "tpl-a", 3, str(self.standard)))
        page.deleteLater()

    def test_checkbox_signal_and_callback_enable_heatmap_buttons(self):
        page = InspectPage(template_store=Mock(), result_store=ResultStore(self.root / "out"))
        page.current_template = self.tpl
        page.queue = [self.test_image]
        page.current_index = 0
        with patch("app.engines.feature.get_engine", return_value=Mock()), patch.object(page._pair_pool, "start") as start:
            page.inspect_canvas.chk_pair.setChecked(True)
            start.assert_called_once()
        heatmap = self.root / "heatmap.png"
        overlay = self.root / "overlay.png"
        heatmap.write_bytes(b"heat")
        overlay.write_bytes(b"overlay")
        page._pair_done({"heatmap_path": str(heatmap), "overlay_path": str(overlay)},
                        str(self.test_image), page._pair_token)
        self.assertTrue(page.inspect_canvas.btn_heatmap.isEnabled())
        self.assertTrue(page.inspect_canvas.btn_overlay.isEnabled())
        self.assertEqual(page.inspect_canvas._heatmap_paths["heatmap"], str(heatmap))
        self.assertEqual(page.inspect_canvas._heatmap_paths["overlay"], str(overlay))
        page.deleteLater()

    def test_late_result_and_same_path_re_request_are_filtered(self):
        page = InspectPage(template_store=Mock(), result_store=ResultStore(self.root / "out"))
        page.current_template = self.tpl
        page.queue = [self.test_image]
        page.current_index = 0
        with patch("app.engines.feature.get_engine", return_value=Mock()), patch.object(page._pair_pool, "start"):
            page._pair_toggled(True)
        old_token = page._pair_token
        page._pair_toggled(False)
        page._pair_toggled(True)
        new_token = page._pair_token
        self.assertNotEqual(old_token, new_token)
        result = {"heatmap_path": str(self.root / "old.png")}
        page._pair_done(result, str(self.test_image), old_token)
        self.assertEqual(page._pair_paths, {})
        page.deleteLater()

    def test_archive_copy_and_history_payload_restore(self):
        store = ResultStore(self.root / "out")
        summary = DetectSummary(overall_ok=True, overall="OK", metadata={"keep": "value"})
        folder = store.save("tpl", str(self.standard), str(self.test_image), None, [], summary,
                            template_id="tpl-a", template_version=3)
        heatmap = self.root / "heatmap.png"
        overlay = self.root / "overlay.png"
        heatmap.write_bytes(b"heat")
        overlay.write_bytes(b"overlay")
        archived = store.attach_aux_pair(folder, {
            "heatmap_path": str(heatmap), "overlay_path": str(overlay),
        })
        payload = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["template_id"], "tpl-a")
        self.assertEqual(payload["template_version"], 3)
        self.assertEqual(payload["overall"], "OK")
        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["vis_file"], "result_vis.png")
        self.assertEqual(payload["metadata"]["keep"], "value")
        self.assertEqual(payload["metadata"]["aux_pair"], archived)
        self.assertEqual((folder / archived["archived_paths"]["heatmap_path"]).read_bytes(), b"heat")
        records = store.list_records()
        self.assertEqual(records[0]["metadata"]["aux_pair"]["archived_paths"], archived["archived_paths"])


if __name__ == "__main__":
    unittest.main()
