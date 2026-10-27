"""round2_learning：AC-063 热图复算 UI / AC-019 校准隔离默认 / AC-042 标签时序审计 /
AC-044 标签身份 / AC-052 开放告警处置→学习汇总。

全部 TEMP sqlite / 合成数据 / 离屏 Qt，不触碰生产 DB 与快照。
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from PySide6.QtWidgets import QApplication
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision import localization as L
from backend.db import database
from backend.db.models import Base, Detection, Feedback

_qt = QApplication.instance() or QApplication([])
T0 = datetime(2026, 1, 1, 8, 0, 0)


def _png(w, h):
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QColor, QImage
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor("#808080"))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba)


class _TempDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = create_engine(f"sqlite:///{Path(self.tmp.name) / 'r2.db'}",
                                 connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.eng)
        sl = sessionmaker(bind=self.eng, expire_on_commit=False)
        self.patches = [patch.object(database, "_engine", self.eng),
                        patch.object(database, "_SessionLocal", sl)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.eng.dispose()
        self.tmp.cleanup()


# ───────────────────────── AC-063 ─────────────────────────
class RecomputeClient:
    """regions GET/POST 走内存 localization；heatmap/recompute 返回可配置响应。"""

    def __init__(self, recompute_res):
        regs = [{"region_id": "r0", "bbox": [0, 0, 40, 40], "area": 1600,
                 "source": "engine_box", "score_peak": 0.9}]
        self.rv = L.new_review(regs, "pp-x")
        self.recompute_res = recompute_res
        self.calls = []

    def get(self, path, **kw):
        self.calls.append(("GET", path, None))
        return {"detection_id": 1, "coord_space": [200, 100], **L.annotate_review(self.rv)}

    def post(self, path, json=None, conflict_ok=False, **kw):
        self.calls.append(("POST", path, json))
        if path.endswith("/heatmap/recompute"):
            return self.recompute_res
        self.rv = L.apply_op(self.rv, json, size=[200, 100])
        return {"review": L.annotate_review(self.rv)}


class RecomputeUiTest(unittest.TestCase):
    def test_payload_validation(self):
        print("[AC-063 1/4] build_recompute_payload：None 沿用原参数、越界拒绝", flush=True)
        from ui.widgets import region_review_dialog as R
        self.assertEqual(R.build_recompute_payload(), ({}, ""))
        self.assertEqual(R.build_recompute_payload(0.4, 20, 5),
                         ({"rel_thresh": 0.4, "min_area": 20, "morph_kernel": 5}, ""))
        for args in ((1.0,), (0.0,), (None, 0), (None, 2_000_000), (None, None, 32), (None, None, 0)):
            body, err = R.build_recompute_payload(*args)
            self.assertIsNone(body)
            self.assertTrue(err)

    def _dialog(self, res):
        from ui.widgets import region_review_dialog as R
        c = RecomputeClient(res)
        d = R.RegionReviewDialog(c, 1)
        d.canvas.set_image_bytes(_png(100, 50))
        self.addCleanup(d.deleteLater)
        return c, d

    def test_success_preview_readonly(self):
        print("[AC-063 2/4] 对话框按钮 → POST 复算 → 预览框叠加，区域/修订号不变", flush=True)
        res = {"detection_id": 1, "params": {"rel_thresh": 0.3}, "boxes_image": [[1, 2, 30, 40], [50, 5, 90, 60], [1, 2]],
               "boxes_display": [], "mask_pixels": 1234, "mask_shape": [100, 200], "persisted": False}
        c, d = self._dialog(res)
        rev0, regs0 = c.rv.get("revision", 0), [dict(r) for r in c.rv["regions"]]
        d.spin_rel.setValue(0.3)
        d.btn_recompute.click()
        post = [x for x in c.calls if x[0] == "POST"]
        self.assertEqual(len(post), 1)
        self.assertEqual(post[0][1], "/api/detections/1/heatmap/recompute")
        self.assertEqual(post[0][2], {"rel_thresh": 0.3})   # 0=原值 → 不传 min_area/morph_kernel
        self.assertEqual(len(d.canvas.preview_items()), 2)  # 非 4 元组被丢弃
        self.assertIn("2 框", d.lbl_recompute.text())
        self.assertIn("未落库", d.lbl_recompute.text())
        self.assertEqual(c.rv.get("revision", 0), rev0)
        self.assertEqual(c.rv["regions"], regs0)
        self.assertEqual(set(d.canvas.region_items()), {"r0"})
        d.btn_clear_preview.click()
        self.assertEqual(len(d.canvas.preview_items()), 0)
        self.assertEqual(d.lbl_recompute.text(), "")

    def test_failure_paths(self):
        print("[AC-063 3/4] 409 / None 失败：提示且清空预览", flush=True)
        c, d = self._dialog({"_status": 409, "detail": "未留存原始热图"})
        d.canvas.set_preview_boxes([[0, 0, 5, 5]])
        self.assertIsNone(d.recompute_heatmap())
        self.assertIn("复算失败：未留存原始热图", d.lbl_recompute.text())
        self.assertEqual(len(d.canvas.preview_items()), 0)
        c.recompute_res = None
        self.assertIsNone(d.recompute_heatmap())
        self.assertIn("复算失败", d.lbl_recompute.text())

    def test_invalid_params_no_request(self):
        print("[AC-063 4/4] 参数无效不发请求", flush=True)
        c, d = self._dialog({})
        with patch.object(type(d), "recompute_payload", return_value=(None, "rel_thresh 须在 (0,1) 内")):
            self.assertIsNone(d.recompute_heatmap())
        self.assertFalse([x for x in c.calls if x[0] == "POST"])
        self.assertIn("复算参数无效", d.lbl_recompute.text())


# ───────────────────────── AC-019 ─────────────────────────
class IsolationDefaultTest(unittest.TestCase):
    def test_resolve_isolation(self):
        print("[AC-019 1/1] resolve_isolation：auto 默认 / 小样本回退 / 显式 self|holdout / 未知值 / 确定性", flush=True)
        from algo.eval.offline import resolve_isolation
        iso, cal, fit, info = resolve_isolation({}, 100)
        self.assertEqual((iso, len(fit), len(cal)), ("holdout", 80, 20))
        self.assertEqual(info["requested"], "auto")
        self.assertFalse(set(cal) & set(fit))
        self.assertEqual(sorted(cal + fit), list(range(100)))
        self.assertEqual(resolve_isolation({}, 100)[1], cal)          # 同 seed 确定
        self.assertNotEqual(resolve_isolation({}, 100, seed=7)[1], cal)
        iso, cal, fit, info = resolve_isolation(None, 10)
        self.assertEqual((iso, cal, fit), ("self", list(range(10)), list(range(10))))
        self.assertIn("回退 self", info["fallback_reason"])
        iso, *_, info = resolve_isolation({"isolation": "self"}, 100)
        self.assertEqual((iso, info["fallback_reason"]), ("self", "显式配置 self"))
        iso, cal, fit, _ = resolve_isolation({"isolation": "holdout"}, 10)
        self.assertEqual((iso, len(cal), len(fit)), ("holdout", 4, 6))
        iso, *_, info = resolve_isolation({"isolation": "bogus"}, 100)
        self.assertEqual(iso, "self")
        self.assertIn("未知", info["fallback_reason"])
        # 边界：auto 下 fit 恰好 = holdout_min_fit
        self.assertEqual(resolve_isolation({}, 20)[0], "holdout")      # 20-4=16
        self.assertEqual(resolve_isolation({}, 19)[0], "self")         # 19-4=15


# ───────────────────────── AC-042 ─────────────────────────
class LabelOrderAuditTest(_TempDb):
    def test_pure_audit(self):
        print("[AC-042 1/2] audit_label_order：ok/late/before/out_of_order/unknown/re_detect", flush=True)
        from backend.self_learning.label_audit import audit_label_order
        d = timedelta(days=1)
        ev = [
            {"feedback_id": 1, "detection_id": 1, "image_key": "a", "detected_at": T0, "label_at": T0 + 2 * d},
            {"feedback_id": 2, "detection_id": 2, "image_key": "b", "detected_at": T0 + d, "label_at": T0 + 1.5 * d},
            {"feedback_id": 3, "detection_id": 3, "image_key": "c", "detected_at": T0 + 2 * d, "label_at": T0 + 20 * d},
            {"feedback_id": 4, "detection_id": 4, "image_key": "a", "detected_at": T0 + 3 * d, "label_at": T0 + 2 * d},
            {"feedback_id": 5, "detection_id": 5, "image_key": "d", "detected_at": None, "label_at": T0},
        ]
        r = audit_label_order(reversed(ev), 7)
        st = {i["feedback_id"]: (i["status"], i["seen"], i["trainable"]) for i in r["items"]}
        self.assertEqual(st[1], ("ok", "first", True))
        self.assertEqual(st[2], ("out_of_order", "first", True))
        self.assertEqual(st[3], ("late", "first", False))
        self.assertEqual(st[4], ("before_detection", "re_detect", False))
        self.assertEqual(st[5][0], "unknown_time")
        self.assertFalse(st[5][2])
        self.assertEqual(r["by_seen"], {"first": 4, "re_detect": 1})
        self.assertEqual(sum(r["counts"].values()), 5)

    def test_service_temp_db(self):
        print("[AC-042 2/2] label_order_audit：TEMP sqlite，排除作废反馈，只读", flush=True)
        from backend.self_learning import service as svc
        with database.session_scope() as s:
            for i, (dlag, invalid) in enumerate([(1, False), (10, False), (2, True)]):
                det = Detection(image_path="same.png", category="c", final_score=.1, is_anomaly=False,
                                latency_ms=1, n_tiles={}, created_at=T0 + timedelta(hours=i))
                s.add(det)
                s.flush()
                s.add(Feedback(detection_id=det.id, feedback_type="confirmed", operator_label=0,
                               invalidated=invalid, created_at=T0 + timedelta(hours=i, days=dlag)))
        cfg = Mock()
        cfg.business_budget.return_value = {"label_late_tolerance_days": 7}
        with patch("backend.core.config.get_settings", return_value=cfg):
            r = svc.label_order_audit("c")
            self.assertEqual(svc.label_order_audit("absent")["items"], [])
        self.assertEqual(r["tolerance_days"], 7)
        self.assertEqual(r["counts"], {"ok": 1, "late": 1})
        self.assertEqual(r["by_seen"], {"first": 1, "re_detect": 1})
        with database.session_scope() as s:
            self.assertFalse(any(f.learning_trace for f in s.query(Feedback)))


# ───────────────────────── AC-044 ─────────────────────────
class LabelIdentityTest(unittest.TestCase):
    def test_resolve_labels(self):
        print("[AC-044 1/1] 人工优先/同修订争议/伪标签过期与 teacher 不符/缺 teacher_version", flush=True)
        from backend.self_learning.label_audit import LabelRecord, resolve_labels
        now = T0 + timedelta(days=10)
        P = lambda sid, lab, conf=0.9, tv="t2", vu=None: LabelRecord(
            sid, lab, "pseudo", teacher_version=tv, confidence=conf, valid_until=vu, predicted_at=T0)
        H = lambda sid, lab, rev: LabelRecord(sid, lab, "human", revision=rev, operator="甲", created_at=T0)
        recs = [P("s1", 1), H("s1", 0, 0), H("s1", 1, 2),          # 人工最高修订生效
                H("s2", 0, 1), H("s2", 1, 1),                      # 同修订冲突
                P("s3", 1, vu=T0), P("s3", 0, tv="t1"),            # 过期 + teacher 不符
                P("s4", 1, conf=0.6), P("s4", 0, conf=0.95), P("s4", 1, conf=0.99, vu=T0)]
        r = resolve_labels(recs, now, teacher_version="t2")
        self.assertEqual((r["s1"]["status"], r["s1"]["label"], r["s1"]["source"], r["s1"]["revision"],
                          r["s1"]["replaced_pseudo"]), ("active", 1, "human", 2, 1))
        self.assertEqual((r["s2"]["status"], r["s2"]["usable"]), ("disputed", False))
        self.assertEqual((r["s3"]["status"], r["s3"]["usable"], r["s3"]["n_stale"]), ("stale", False, 2))
        self.assertEqual((r["s4"]["label"], r["s4"]["source"], r["s4"]["n_stale"]), (0, "pseudo", 1))
        self.assertEqual(r["s4"]["record"]["teacher_version"], "t2")
        with self.assertRaises(ValueError):
            LabelRecord("x", 1, "pseudo")
        with self.assertRaises(ValueError):
            LabelRecord("x", 1, "auto")


# ───────────────────────── AC-052 ─────────────────────────
class OpenAlertSummaryTest(_TempDb):
    def test_summary(self):
        print("[AC-052 1/1] /review/open_alerts/summary：状态/原因/学习状态/准入候选计数", flush=True)
        from backend.api.open_alerts import new_item, register
        with database.session_scope() as s:
            fb_ok = Feedback(detection_id=0, feedback_type="confirmed", operator_label=1,
                             learning_trace={"learning_status": "candidate_pending"})
            fb_na = Feedback(detection_id=0, feedback_type="false_positive", operator_label=0,
                             learning_trace={"learning_status": "not_requested"})
            items = [
                {**new_item(3.0, 2.5), "status": "confirmed_defect", "cause": "new_defect",
                 "admission": {"status": "candidate"}, "_fb": fb_ok},
                {**new_item(3.0, 2.5), "status": "confirmed_normal", "cause": "drift", "_fb": fb_na},
                new_item(), None,
            ]
            for it in items:
                fb = it.pop("_fb", None) if it else None
                nt = {"open_alert": True, **({"open_alert_item": it} if it else {})}
                det = Detection(image_path="x.png", category="c", final_score=.1, is_anomaly=False,
                                latency_ms=1, n_tiles=nt)
                s.add(det)
                s.flush()
                if fb is not None:
                    fb.detection_id = det.id
                    s.add(fb)
                    s.flush()
                    det.n_tiles = {**nt, "open_alert_item": {**it, "feedback_id": fb.id}}
            s.add(Detection(image_path="y.png", category="c", final_score=.1, is_anomaly=False,
                            latency_ms=1, n_tiles={}))
            s.add(Detection(image_path="z.png", category="other", final_score=.1, is_anomaly=False,
                            latency_ms=1, n_tiles={"open_alert": True}))
        app = FastAPI()
        router = APIRouter()
        register(router)
        app.include_router(router, prefix="/api")
        with TestClient(app) as c:
            r = c.get("/api/review/open_alerts/summary?category=c")
            self.assertEqual(r.status_code, 200, r.text)
            j = r.json()
            self.assertEqual(j["total"], 4)
            self.assertEqual(j["by_status"], {"confirmed_defect": 1, "confirmed_normal": 1, "queued": 2})
            self.assertEqual(j["by_cause"], {"new_defect": 1, "drift": 1, "unclassified": 2})
            self.assertEqual(j["by_learning_status"],
                             {"candidate_pending": 1, "not_requested": 1, "no_feedback": 2})
            self.assertEqual(j["admission_candidates"], 1)
            self.assertEqual(c.get("/api/review/open_alerts/summary").json()["total"], 5)
            self.assertEqual(c.get("/api/review/open_alerts/1").status_code, 200)   # 路由未被 summary 遮挡


if __name__ == "__main__":
    unittest.main(verbosity=2)
