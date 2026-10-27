"""ResultHud 可见性与双引擎展示回归测试。"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.detect.types import DetectSummary
from app.ui.result_hud import ResultHud


class ResultHudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_lifecycle_with_traditional_summary(self):
        hud = ResultHud()
        hud.clear()
        hud.set_running()
        summary = DetectSummary(overall_ok=True, overall="OK", elapsed_ms=12)
        hud.bind_summary(summary)
        self.assertTrue(hud.dual_panel.isHidden())
        hud.deleteLater()

    def test_lifecycle_with_dual_summary(self):
        hud = ResultHud()
        hud.clear()
        hud.set_running()
        summary = DetectSummary(
            overall_ok=False,
            overall="NG",
            elapsed_ms=24,
            metadata={
                "dual": True,
                "traditional": {"overall": "OK", "elapsed_ms": 10},
                "feature": {"overall": "NG", "score": 0.8123, "elapsed_ms": 14},
                "fusion_overall": "NG",
            },
        )
        hud.bind_summary(summary)
        self.assertFalse(hud.dual_panel.isHidden())
        self.assertEqual(hud.dual_labels[0].text().split(" · ", 1)[0], "OK")
        self.assertEqual(hud.dual_labels[1].text().split(" · ", 1)[0], "NG")
        hud.deleteLater()


if __name__ == "__main__":
    unittest.main()
