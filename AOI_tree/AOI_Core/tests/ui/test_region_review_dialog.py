"""AC-064 多框复核面板（离屏）：列框、确认进训练、无理由推翻→争议隔离、乐观锁冲突提示。"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from algo.decision import localization as L
from ui.widgets.region_review_dialog import RegionReviewDialog

_app = QApplication.instance() or QApplication([])


class FakeClient:
    """内存版 /detections/{id}/regions，复用真实 apply_op 规则与修订号乐观锁。"""

    def __init__(self):
        regs = [{"region_id": f"r{i}", "bbox": [i * 10, 0, i * 10 + 8, 8], "area": 64,
                 "source": "engine_box", "score_peak": 0.9} for i in range(2)]
        self.rv = L.new_review(regs, "pp-x")

    def get(self, path, **kw):
        return {"detection_id": 1, "coord_space": None, **self.rv}

    def post(self, path, json=None, conflict_ok=False, **kw):
        if json["expected_revision"] != self.rv.get("revision", 0):
            return {"_status": 409, "detail": "区域复核已被修改，请刷新后重试"}
        try:
            self.rv = L.apply_op(self.rv, json)
        except ValueError as e:
            return {"_status": 409, "detail": str(e)} if conflict_ok else None
        return {"review": self.rv}


class RegionReviewDialogTest(unittest.TestCase):
    def setUp(self):
        self.c = FakeClient()
        self.d = RegionReviewDialog(self.c, 1)
        self.addCleanup(self.d.deleteLater)

    def act(self, row, op, who, reason=""):
        self.d.table.selectRow(row)
        self.d.edit_operator.setText(who)
        self.d.edit_reason.setText(reason)
        return self.d.apply(op)

    def test_lists_candidates_not_trainable(self):
        self.assertEqual(self.d.table.rowCount(), 2)
        self.assertEqual(self.d.table.item(0, 4).text(), "候选(不训)")
        self.assertIn("可训 0 框", self.d.lbl_status.text())

    def test_confirm_then_unreasoned_flip_disputed(self):
        self.assertTrue(self.act(0, "confirm", "甲"))
        self.assertEqual(L.trainable_region_boxes(self.c.rv), [[0, 0, 8, 8]])
        self.assertTrue(self.act(0, "reject", "乙"))  # 无理由推翻
        self.assertEqual(self.d.table.item(0, 4).text(), "争议(隔离)")
        self.assertEqual(L.trainable_region_boxes(self.c.rv), [])

    def test_requires_selection_and_operator(self):
        self.d.edit_operator.setText("")
        self.assertFalse(self.d.apply("confirm"))
        self.assertIn("操作员", self.d.lbl_status.text())

    def test_stale_revision_shows_conflict_and_refreshes(self):
        self.c.rv = L.apply_op(self.c.rv, {"op": "confirm", "region_id": "r1", "operator": "丙",
                                           "event_id": "x", "expected_revision": 0})
        self.assertFalse(self.act(0, "confirm", "甲"))  # 面板仍持有修订号 0
        self.assertIn("已被修改", self.d.lbl_status.text())
        self.assertTrue(self.act(0, "confirm", "甲"))  # 刷新后成功


if __name__ == "__main__":
    unittest.main()
