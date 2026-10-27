"""AC-063/AC-064 多框复核绘框（离屏）：坐标换算、add/revise payload、叠加框颜色与高亮、
表格 train_status/reviewer 列、to_dict 派生视图。"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from algo.decision import localization as L
from ui.widgets import region_review_dialog as R

_app = QApplication.instance() or QApplication([])
N = 7


def _png(w, h):
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor("#808080"))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


class FakeClient:
    """内存版 /api/detections/{id}/regions：GET 返回 annotate_review 视图，POST 走真实 apply_op。"""

    def __init__(self, size=(200, 100)):
        regs = [{"region_id": f"r{i}", "bbox": [i * 50, 10, i * 50 + 40, 50], "area": 1600,
                 "source": "engine_box", "score_peak": 0.9} for i in range(2)]
        self.rv = L.new_review(regs, "pp-x")
        self.size = list(size)
        self.paths = []

    def get(self, path, **kw):
        self.paths.append(path)
        return {"detection_id": 1, "coord_space": self.size, **L.annotate_review(self.rv)}

    def post(self, path, json=None, conflict_ok=False, **kw):
        self.paths.append(path)
        if json["expected_revision"] != self.rv.get("revision", 0):
            return {"_status": 409, "detail": "区域复核已被修改，请刷新后重试"}
        try:
            self.rv = L.apply_op(self.rv, json, size=self.size)
        except ValueError as e:
            return {"_status": 409, "detail": str(e)} if conflict_ok else None
        return {"review": L.annotate_review(self.rv)}


class PureFunctionTest(unittest.TestCase):
    def test_coord_scale_and_clip(self):
        print(f"[1/{N}] 坐标换算：缩放/反向拖拽/越界裁剪/往返", flush=True)
        # 显示图 100x50，原图 200x100 → ×2
        self.assertEqual(R.scene_rect_to_image((10, 5, 30, 25), (100, 50), (200, 100)), [20, 10, 60, 50])
        self.assertEqual(R.scene_rect_to_image((30, 25, 10, 5), (100, 50), (200, 100)), [20, 10, 60, 50])
        self.assertEqual(R.scene_rect_to_image((-5, -5, 120, 60), (100, 50), (200, 100)), [0, 0, 200, 100])
        self.assertEqual(R.scene_rect_to_image((1, 2, 3, 4), (100, 50), None), [1, 2, 3, 4])
        x, y, w, h = R.image_to_scene_rect([20, 10, 60, 50], (100, 50), (200, 100))
        self.assertEqual((x, y, w, h), (10.0, 5.0, 20.0, 20.0))
        self.assertEqual(R.scene_rect_to_image((x, y, x + w, y + h), (100, 50), (200, 100)), [20, 10, 60, 50])
        self.assertEqual(R.hit_region([{"region_id": "a", "bbox": [0, 0, 100, 100]},
                                       {"region_id": "b", "bbox": [10, 10, 20, 20]}], 15, 15), "b")
        self.assertIsNone(R.hit_region([{"region_id": "a", "bbox": [0, 0, 5, 5]}], 50, 50))

    def test_add_payload(self):
        print(f"[2/{N}] add payload：reason/operator/revision/越界校验", flush=True)
        body, err = R.build_box_payload("add", [1, 2, 30, 40], " 甲 ", "漏检补框", 3,
                                        image_wh=[200, 100], event_id="e1")
        self.assertIsNone(err)
        self.assertEqual(body, {"op": "add", "bbox": [1, 2, 30, 40], "operator": "甲",
                                "event_id": "e1", "expected_revision": 3, "reason": "漏检补框"})
        self.assertNotIn("region_id", body)
        self.assertIn("理由", R.build_box_payload("add", [1, 2, 30, 40], "甲", " ", 0)[1])
        self.assertIn("操作员", R.build_box_payload("add", [1, 2, 30, 40], "", "r", 0)[1])
        self.assertIn("修订号", R.build_box_payload("add", [1, 2, 30, 40], "甲", "r", None)[1])
        self.assertIn("拖拽", R.build_box_payload("add", None, "甲", "r", 0)[1])
        self.assertIsNotNone(R.build_box_payload("add", [0, 0, 1, 1], "甲", "r", 0)[1])
        self.assertIsNotNone(R.build_box_payload("add", [0, 0, 300, 50], "甲", "r", 0,
                                                 image_wh=[200, 100])[1])
        self.assertIsNotNone(R.build_box_payload("confirm", [0, 0, 9, 9], "甲", "r", 0)[1])

    def test_revise_payload(self):
        print(f"[3/{N}] revise payload：须 region_id + expected_revision", flush=True)
        body, err = R.build_box_payload("revise", [5, 5, 25, 25], "乙", "框偏", 0, region_id="r1")
        self.assertIsNone(err)
        self.assertEqual((body["region_id"], body["expected_revision"], body["reason"]), ("r1", 0, "框偏"))
        self.assertTrue(body["event_id"])
        self.assertIn("选择", R.build_box_payload("revise", [5, 5, 25, 25], "乙", "框偏", 0)[1])
        self.assertIn("修订号", R.build_box_payload("revise", [5, 5, 25, 25], "乙", "框偏", None,
                                                 region_id="r1")[1])

    def test_box_style_mapping(self):
        print(f"[4/{N}] 叠加颜色映射：候选/人工可训/人工未训/拒绝/争议", flush=True)
        st = R.box_style
        self.assertEqual(st({"train_status": "excluded_candidate", "source": "engine_box"}), ("#F59E0B", "dash"))
        self.assertEqual(st({"train_status": "eligible", "source": "human_box"}), ("#16A34A", "solid"))
        self.assertEqual(st({"train_status": "excluded_disputed", "source": "human_box"}), R.HUMAN_EXCLUDED)
        self.assertEqual(st({"train_status": "excluded_disputed", "source": "engine_box"}), ("#9333EA", "dashdot"))
        self.assertEqual(st({"train_status": "excluded_rejected", "source": "human_box"}), ("#9CA3AF", "dot"))
        self.assertEqual(st({}), ("#F59E0B", "dash"))  # 缺省按候选
        self.assertNotEqual(st({"train_status": "excluded_candidate"}), st({"train_status": "eligible"}))

    def test_to_dict_annotated(self):
        print(f"[5/{N}] to_dict 透出 region_review 派生视图", flush=True)
        from backend.api.schemas import to_dict
        from backend.db.models import Detection
        rv = L.new_review([{"region_id": "r0", "bbox": [0, 0, 8, 8], "source": "engine_box"}])
        rv = L.apply_op(rv, {"op": "confirm", "region_id": "r0", "operator": "甲",
                             "event_id": "e", "expected_revision": 0})
        d = to_dict(Detection(target_type="image", image_path="x.png", n_tiles={"region_review": rv}))
        r0 = d["region_review"]["regions"][0]
        self.assertEqual((r0["train_status"], r0["reviewer"], r0["source"], r0["origin_source"]),
                         ("eligible", "甲", "human_box", "engine_box"))
        self.assertEqual(d["region_review"]["revision"], 1)
        self.assertNotIn("train_status", rv["regions"][0])  # 存储不变
        self.assertIsNone(to_dict(Detection(target_type="image", image_path="x.png", n_tiles={}))["region_review"])


class DialogDrawTest(unittest.TestCase):
    def setUp(self):
        self.c = FakeClient()
        self.d = R.RegionReviewDialog(self.c, 1)
        self.d.canvas.set_image_bytes(_png(100, 50))   # 显示图为原图 1/2
        self.addCleanup(self.d.deleteLater)

    def test_overlay_columns_highlight(self):
        print(f"[6/{N}] 对话框：叠加框/高亮/新列/API 路径", flush=True)
        self.assertTrue(all(p == "/api/detections/1/regions" for p in self.c.paths))
        items = self.d.canvas.region_items()
        self.assertEqual(set(items), {"r0", "r1"})
        r = items["r1"].rect()   # 原图 [50,10,90,50] → 显示 (25,5,20,20)
        self.assertEqual((r.x(), r.y(), r.width(), r.height()), (25.0, 5.0, 20.0, 20.0))
        self.assertEqual(items["r0"].pen().color().name(), "#f59e0b")
        self.assertEqual(self.d.table.item(0, 6).text(), "排除:候选")
        self.assertEqual(self.d.table.item(0, 7).text(), "-")
        self.d.table.selectRow(1)
        items = self.d.canvas.region_items()
        self.assertEqual(items["r1"].pen().width(), 4)
        self.assertEqual(items["r0"].pen().width(), 2)
        self.d.canvas.region_clicked.emit("r0")
        self.assertEqual(self.d.selected_region_id(), "r0")
        self.d.edit_operator.setText("甲")
        self.assertTrue(self.d.apply("confirm"))
        self.assertEqual(self.d.table.item(0, 6).text(), "可训")
        self.assertEqual(self.d.table.item(0, 7).text(), "甲")
        self.assertEqual(self.d.canvas.region_items()["r0"].pen().color().name(), "#16a34a")

    def test_draw_add_and_revise(self):
        print(f"[7/{N}] 对话框：拖框→add / 选中→revise（乐观锁）", flush=True)
        self.d.edit_operator.setText("甲")
        self.d._on_box_drawn([10, 60, 40, 90])
        self.assertFalse(self.d.submit_box("add"))          # 无理由拒绝
        self.assertIn("理由", self.d.lbl_status.text())
        self.d.edit_reason.setText("漏检补框")
        self.assertTrue(self.d.submit_box("add"))
        self.assertEqual(self.c.rv["revision"], 1)
        h0 = self.c.rv["regions"][-1]
        self.assertEqual((h0["region_id"], h0["bbox"], h0["source"]), ("h0", [10, 60, 40, 90], "human_box"))
        self.assertEqual(self.c.rv["events"][-1]["payload"]["reason"], "漏检补框")
        self.assertIsNone(self.d._pending_bbox)
        self.assertEqual(self.d.table.rowCount(), 3)
        # revise：先选框，再拖新框
        self.d.select_region("r1")
        self.assertFalse(self.d.submit_box("revise"))       # 无待提交框
        self.d._on_box_drawn([52, 12, 88, 48])
        self.d.edit_reason.setText("框偏")
        self.assertTrue(self.d.submit_box("revise"))
        r1 = next(r for r in self.c.rv["regions"] if r["region_id"] == "r1")
        self.assertEqual((r1["bbox"], r1["source"], r1["revised_from"]["bbox"]),
                         ([52, 12, 88, 48], "human_box", [50, 10, 90, 50]))
        self.assertEqual(self.c.rv["events"][-1]["payload"]["expected_revision"], 1)
        # 乐观锁：他人抢先修改 → 409 提示并刷新
        self.c.rv = L.apply_op(self.c.rv, {"op": "confirm", "region_id": "r0", "operator": "丙",
                                           "event_id": "x", "expected_revision": 2})
        self.d._on_box_drawn([0, 0, 20, 20])
        self.assertFalse(self.d.submit_box("add"))
        self.assertIn("已被修改", self.d.lbl_status.text())
        self.assertEqual(self.d._review["revision"], 3)
        self.assertTrue(self.d.submit_box("add"))


if __name__ == "__main__":
    unittest.main()
