"""Gauge 判定契约离屏单元测试：不启动后端、不读取生产模型。"""
from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from ui.pages.monitor_page import MonitorPage, _row_to_det
from ui.theme import DANGER, SUCCESS, TEXT_SUB, WARNING
from ui.widgets.gauge import ScoreGauge


class GaugeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.gauge = ScoreGauge()
        self.gauge.resize(220, 240)

    def tearDown(self):
        self.gauge.close()

    def test_three_states_follow_decision_not_score(self):
        for decision, score, text, color, anomaly in (
            ("normal", 0.99, "正 常", SUCCESS, False),
            ("gray", 0.99, "待复核（灰区）", WARNING, False),
            ("anomaly", 0.01, "异 常", DANGER, True),
        ):
            with self.subTest(decision=decision):
                self.gauge.set_score(score, 0.5, decision=decision)
                self.assertEqual(self.gauge.verdict_text, text)
                self.assertEqual(self.gauge.verdict_color, color)
                self.assertIs(self.gauge.is_anomaly, anomaly)
                self.assertFalse(self.gauge.grab().isNull())

    def test_out_of_range_values_preserved_and_only_geometry_clipped(self):
        for score, threshold, decision in ((1.2, 1.5, "normal"),
                                            (-0.2, -0.1, "gray"),
                                            (1.8, 2.0, "anomaly")):
            with self.subTest(score=score):
                self.gauge.set_score(score, threshold, decision=decision)
                with patch("ui.widgets.gauge.QPainter") as painter_class:
                    self.gauge.paintEvent(None)
                    painter = painter_class.return_value
                    self.assertEqual(painter.drawArc.call_args_list[1].args[2],
                                     int(-max(0, min(1, score)) * 360 * 16))
                    self.assertEqual(painter.drawText.call_args_list[0].args[2],
                                     f"{score:.3f}")
                self.assertEqual(self.gauge._score, score)
                self.assertEqual(self.gauge._threshold, threshold)
                self.assertEqual(self.gauge._decision, decision)

    def test_threshold_updates_never_change_decision(self):
        self.gauge.set_score(0.75, 0.5, decision="normal")
        for threshold in (-1.0, 0.0, 2.5):
            self.gauge.set_threshold(threshold)
            self.assertEqual(self.gauge._threshold, threshold)
            self.assertIs(self.gauge.is_anomaly, False)

    def test_missing_or_unrecognized_decision_clears_previous_verdict(self):
        for decision in (None, "", "unexpected"):
            with self.subTest(decision=decision):
                self.gauge.set_score(1, 0.5, decision="anomaly")
                self.gauge.set_score(1, 0.5, decision=decision)
                self.assertIsNone(self.gauge.is_anomaly)
                self.assertEqual(self.gauge.verdict_text, "未知 / 未判定")
                self.assertEqual(self.gauge.verdict_color, TEXT_SUB)

    def test_missing_threshold_clears_previous_tick(self):
        self.gauge.set_score(0.4, 0.6, decision="normal")
        self.gauge.set_score(0.4, decision="gray")
        self.assertIsNone(self.gauge._threshold)
        with patch("ui.widgets.gauge.QPainter") as painter_class:
            self.gauge.paintEvent(None)
            painter_class.return_value.drawLine.assert_not_called()


class MonitorGaugeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = SimpleNamespace(
            gauge=ScoreGauge(), viewer=Mock(), _client=Mock(),
            _repolish=Mock(), _update_slot_bars=Mock(), _get_budget=lambda: 200,
        )
        for name in ("badge_verdict", "lbl_review_hint", "badge_open", "badge_attr",
                     "badge_align", "lbl_alarm", "lbl_latency", "lbl_threshold",
                     "lbl_model", "lbl_scores", "lbl_weights", "lbl_router",
                     "lbl_features", "lbl_boxes", "lbl_trigger"):
            setattr(self.page, name, QLabel())
        for name in ("btn_fp", "btn_fn", "btn_ok"):
            setattr(self.page, name, QPushButton())

    def tearDown(self):
        for value in vars(self.page).values():
            if isinstance(value, (ScoreGauge, QLabel, QPushButton)):
                value.close()

    def test_caller_passes_all_decisions_and_open_is_independent(self):
        for decision, text, badge in (("normal", "正 常", "success"),
                                       ("gray", "待复核（灰区）", "warning"),
                                       ("anomaly", "异 常", "danger"),
                                       (None, "未知 / 未判定", "info")):
            for open_alert in (False, True):
                with self.subTest(decision=decision, open_alert=open_alert):
                    det = {"final_score": 1.2, "threshold": 1.5,
                           "decision": decision, "is_anomaly": True,
                           "open_alert": open_alert}
                    MonitorPage._show_detection(self.page, det)
                    self.assertEqual(self.page.gauge._decision, decision)
                    self.assertEqual(self.page.gauge._score, 1.2)
                    self.assertEqual(self.page.gauge._threshold, 1.5)
                    self.assertEqual(self.page.badge_verdict.text(), text)
                    self.assertEqual(self.page.badge_verdict.property("badge"), badge)
                    self.assertEqual(self.page.badge_open.isHidden(), not open_alert)
                    self.assertEqual(self.page.lbl_review_hint.isHidden(), decision != "gray")

    def test_legacy_rows_without_decision_stay_unknown(self):
        for is_anomaly in (False, True):
            for nt in ({}, json.dumps({"threshold": 1.5, "open_alert": True})):
                with self.subTest(is_anomaly=is_anomaly, n_tiles=nt):
                    det = _row_to_det({"final_score": 1.2,
                                       "is_anomaly": is_anomaly, "n_tiles": nt})
                    self.assertIsNone(det["decision"])
                    MonitorPage._show_detection(self.page, det)
                    self.assertIsNone(self.page.gauge.is_anomaly)
                    self.assertEqual(self.page.badge_verdict.text(), "未知 / 未判定")

    def test_row_adapter_preserves_gray_with_open_alert(self):
        det = _row_to_det({"final_score": -0.2, "is_anomaly": False,
                           "n_tiles": {"decision": "gray", "threshold": -0.1,
                                       "open_alert": True}})
        MonitorPage._show_detection(self.page, det)
        self.assertEqual(self.page.gauge.verdict_text, "待复核（灰区）")
        self.assertIs(self.page.gauge.is_anomaly, False)
        self.assertFalse(self.page.badge_open.isHidden())


if __name__ == "__main__":
    unittest.main(verbosity=2)
