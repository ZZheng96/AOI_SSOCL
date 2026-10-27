"""PCB-ACCEPT-03 指标配置的定向测试。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from app import config
from app.template.model import InspectionTemplate
from app.ui.settings_dialog import SettingsDialog


class AcceptanceSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "settings.json"
        self.settings_patch = patch.object(config, "SETTINGS_FILE", self.settings_path)
        self.settings_patch.start()

    def tearDown(self):
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    def test_default_values_and_legacy_settings(self):
        self.assertEqual(config.get_acceptance_settings(), {
            "recall_min": 0.99,
            "false_positive_rate_max": 0.01,
            "iou_min": 0.5,
            "wilson_confidence": 0.95,
            "report_indeterminate_separately": True,
        })
        config.save_settings({"gate_enabled": False, "size_gate_max_diff_px": 3})
        self.assertEqual(config.get_acceptance_settings()["recall_min"], 0.99)
        self.assertFalse(config.get_gate_settings()["gate_enabled"])

    def test_save_and_validate(self):
        config.set_acceptance_settings(0.98, 0.02, 0.6, 0.9, False)
        self.assertEqual(config.get_acceptance_settings()["iou_min"], 0.6)
        self.assertFalse(config.get_acceptance_settings()["report_indeterminate_separately"])
        with self.assertRaises(ValueError):
            config.set_acceptance_settings(1.1, 0.02, 0.6, 0.9)
        with self.assertRaises(ValueError):
            config.set_acceptance_settings(0.98, 0.02, 0.6, 0.4)

    def test_rule_snapshot_contains_runtime_policy(self):
        config.set_acceptance_settings(0.98, 0.02, 0.6, 0.9)
        template = InspectionTemplate(id="acceptance-test")
        self.assertEqual(template.rule_snapshot()["acceptance_policy"]["recall_min"], 0.98)

    def test_dialog_exposes_and_saves_acceptance_values(self):
        app = QApplication.instance() or QApplication([])
        dialog = SettingsDialog()
        self.assertEqual(dialog.spin_recall.value(), 0.99)
        self.assertEqual(dialog.spin_fpr.value(), 0.01)
        self.assertEqual(dialog.spin_iou.value(), 0.5)
        self.assertEqual(dialog.spin_confidence.value(), 0.95)
        dialog.spin_recall.setValue(0.97)
        dialog.spin_fpr.setValue(0.03)
        dialog.spin_iou.setValue(0.7)
        dialog.spin_confidence.setValue(0.9)
        dialog.chk_indeterminate.setChecked(False)
        dialog._save()
        self.assertEqual(config.get_acceptance_settings()["recall_min"], 0.97)
        self.assertFalse(config.get_acceptance_settings()["report_indeterminate_separately"])
        dialog.deleteLater()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
