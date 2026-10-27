"""AC-072 业务预算与保留期可配置（用户定值 2026-10-13）：默认值、类型/范围校验、
TEMP yaml 持久化往返、/system/config 白名单与拒绝原因、设置页离屏保存/恢复默认、
统计页/评估页/候选验收 policy 以配置值为达标线。不触碰生产 DB/快照/配置。"""
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QSpinBox

from backend.api import routes_system as RS
from backend.core import config as C
from ui.pages import eval_page as E
from ui.pages import settings_page as SP
from ui.pages import stats_page as S
from ui.theme import DANGER, SUCCESS, WARNING

_app = QApplication.instance() or QApplication([])
N = 10
_T0 = time.perf_counter()

USER_DEFAULTS = {
    "g2_max_fpr_pct": 5.0, "g2_min_recall_pct": 95.0,
    "g6_max_old_recall_drop_pt": 2.0, "max_gray_ratio_pct": 15.0,
    "raw_hot_days": 90, "raw_cold_months": 12, "heatmap_overlay_days": 30,
    "snapshot_keep_recent": 10, "snapshot_keep_activated": True,
    "evalrun_audit_months": 12, "label_late_tolerance_days": 7,
}


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)

    def __getattr__(self, name):
        # 页面构造期 run_async(self, client.xxx, cb) 只取方法引用；run_async 已被 patch 为空操作
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _tmp_settings(tc, extra_yaml: str = "") -> C.Settings:
    tmp = tempfile.TemporaryDirectory(prefix="aoi_biz_budget_")
    tc.addCleanup(tmp.cleanup)
    p = Path(tmp.name) / "config.yaml"
    p.write_text("system:\n  storage_dir: storage\n" + extra_yaml, encoding="utf-8")
    return C.Settings(p)


class BackendBudgetTest(unittest.TestCase):
    def test_defaults(self):
        _log(1, "默认值 = 用户定值（FPR≤5%/Recall≥95%/回退≤2pt/灰区≤15%，保留期 90d/12m/30d/10 版/12m/7d）")
        s = _tmp_settings(self)
        bb = s.business_budget()
        for k, v in USER_DEFAULTS.items():
            self.assertEqual(bb[k], v, k)
            self.assertIs(type(bb[k]), type(v), k)
        self.assertEqual(bb["feedback_event_wal_retention"], "永久")
        self.assertEqual({k: d for k, (_t, d, *_r) in C.BUSINESS_BUDGET_SPEC.items()},
                         USER_DEFAULTS)

    def test_validation(self):
        _log(2, "校验：越界/NaN/布尔混入/int 字段传 float/字符串一律拒绝并给原因，内存值不变")
        s = _tmp_settings(self)
        bad = [("g2_max_fpr_pct", -1), ("g2_max_fpr_pct", 101), ("g2_min_recall_pct", 100.5),
               ("max_gray_ratio_pct", float("nan")), ("g2_max_fpr_pct", True),
               ("raw_hot_days", 0), ("raw_hot_days", 30.5), ("raw_hot_days", "90"),
               ("snapshot_keep_recent", 0), ("snapshot_keep_activated", 1),
               ("label_late_tolerance_days", -7)]
        for key, val in bad:
            with self.assertRaises(C.ConfigValidationError, msg=f"{key}={val!r}") as cm:
                s.update("business_budget", key, val)
            self.assertIn(f"business_budget.{key}", str(cm.exception))
        with self.assertRaises(C.ConfigValidationError) as cm:
            s.update("business_budget", "g2_max_fpr_pct", 101)
        self.assertIn("超出范围", str(cm.exception))
        self.assertEqual(s.business_budget()["g2_max_fpr_pct"], 5.0)
        # 合法：边界值、float 字段接受 int
        s.update("business_budget", "g2_max_fpr_pct", 0)
        s.update("business_budget", "g2_min_recall_pct", 100.0)
        s.update("business_budget", "snapshot_keep_activated", False)
        bb = s.business_budget()
        self.assertEqual((bb["g2_max_fpr_pct"], bb["g2_min_recall_pct"],
                          bb["snapshot_keep_activated"]), (0.0, 100.0, False))

    def test_persist_roundtrip_and_bad_stored(self):
        _log(3, "持久化往返（TEMP yaml 重读）；手改 yaml 的非法存量值回落默认")
        s = _tmp_settings(self)
        s.update("business_budget", "g2_max_fpr_pct", 3.5)
        s.update("business_budget", "raw_hot_days", 60)
        s.update("business_budget", "label_late_tolerance_days", 14)
        s2 = C.Settings(s.config_path)
        bb = s2.business_budget()
        self.assertEqual((bb["g2_max_fpr_pct"], bb["raw_hot_days"],
                          bb["label_late_tolerance_days"]), (3.5, 60, 14))
        self.assertEqual(bb["g2_min_recall_pct"], 95.0)
        s3 = _tmp_settings(self, "business_budget:\n  g2_max_fpr_pct: 250\n"
                                 "  raw_hot_days: abc\n  snapshot_keep_activated: 1\n")
        bb3 = s3.business_budget()
        self.assertEqual((bb3["g2_max_fpr_pct"], bb3["raw_hot_days"],
                          bb3["snapshot_keep_activated"]), (5.0, 90, True))

    def test_candidate_release_policy(self):
        _log(4, "候选验收 policy 由配置映射（pct/100），满足 candidate_release.evaluate 的 ready 判定")
        s = _tmp_settings(self)
        pol = s.candidate_release_policy(max_predict_ms=200, valid_seconds=3600)
        self.assertEqual(pol, {"max_fpr": 0.05, "min_recall": 0.95, "max_gray": 0.15,
                               "old_recall_drop": 0.02, "max_predict_ms": 200.0,
                               "valid_seconds": 3600.0})
        required = {"max_fpr", "min_recall", "max_gray", "old_recall_drop",
                    "max_predict_ms", "valid_seconds"}
        self.assertEqual(set(pol), required)
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v) and v >= 0
                            for v in pol.values()))
        self.assertTrue(all(0 <= pol[k] <= 1 for k in
                            ("max_fpr", "min_recall", "max_gray", "old_recall_drop")))
        s.update("business_budget", "g2_max_fpr_pct", 1.0)
        self.assertAlmostEqual(
            s.candidate_release_policy(200, 3600)["max_fpr"], 0.01)


class ApiConfigTest(unittest.TestCase):
    def setUp(self):
        self.s = _tmp_settings(self)
        for p in (patch.object(RS, "get_settings", lambda: self.s),
                  patch.object(RS, "log_action", lambda *a, **k: None)):
            p.start()
            self.addCleanup(p.stop)

    def test_get_config_items(self):
        _log(5, "GET /system/config：11 项 business_budget 带 value/default/min/max + 固定 WAL=永久；POST 需 admin")
        data = RS.api_get_config()
        items = {it["key"]: it for it in data["items"]
                 if it["section"] == "business_budget"}
        self.assertEqual(set(items), set(USER_DEFAULTS))
        for k, v in USER_DEFAULTS.items():
            self.assertEqual(items[k]["value"], v)
            self.assertEqual(items[k]["default"], v)
        self.assertEqual((items["g2_max_fpr_pct"]["min"], items["g2_max_fpr_pct"]["max"]),
                         (0.0, 100.0))
        self.assertEqual(items["raw_hot_days"]["type"], "int")
        self.assertEqual(items["snapshot_keep_activated"]["type"], "bool")
        self.assertEqual(data["fixed"]["business_budget.feedback_event_wal_retention"], "永久")
        self.assertEqual(RS.api_system_info.__name__, "api_system_info")
        route = next(r for r in RS.router.routes
                     if getattr(r, "path", "").endswith("/system/config")
                     and "POST" in getattr(r, "methods", set()))
        self.assertTrue(route.dependencies, "POST /system/config 必须挂 require_role")

    def test_update_rejects_with_reason(self):
        _log(6, "POST /system/config：合法项应用+持久化，非法值进 rejected 并回传原因（不再 500）")
        res = RS.api_update_config({
            "business_budget.g2_max_fpr_pct": 4.0,
            "business_budget.max_gray_ratio_pct": 120,
            "business_budget.raw_hot_days": 12.5,
            "business_budget.not_a_key": 1,
        })
        self.assertEqual(res["applied"], ["business_budget.g2_max_fpr_pct"])
        self.assertEqual(set(res["rejected"]), {"business_budget.max_gray_ratio_pct",
                                                "business_budget.raw_hot_days",
                                                "business_budget.not_a_key"})
        self.assertIn("超出范围", res["errors"]["business_budget.max_gray_ratio_pct"])
        self.assertIn("类型错误", res["errors"]["business_budget.raw_hot_days"])
        bb = C.Settings(self.s.config_path).business_budget()
        self.assertEqual((bb["g2_max_fpr_pct"], bb["max_gray_ratio_pct"],
                          bb["raw_hot_days"]), (4.0, 15.0, 90))


class SettingsPageTest(unittest.TestCase):
    def test_edit_save_restore(self):
        _log(7, "设置页离屏：分组“业务预算与保留期”，编辑→保存提交正确类型→恢复默认→保存回默认")
        s = _tmp_settings(self)
        with patch.object(RS, "get_settings", lambda: s):
            data = RS.api_get_config()
        with patch.object(SP, "run_async", lambda *a, **k: None):
            page = SP.SettingsPage(FakeClient())
        page._fill(data)
        titles = [page._form_lay.itemAt(i).widget().title()
                  for i in range(page._form_lay.count())
                  if page._form_lay.itemAt(i).widget() is not None]
        self.assertIn("业务预算与保留期", titles)
        ed = page._editors
        self.assertIsInstance(ed["business_budget.g2_max_fpr_pct"], QDoubleSpinBox)
        self.assertIsInstance(ed["business_budget.raw_hot_days"], QSpinBox)
        self.assertIsInstance(ed["business_budget.snapshot_keep_activated"], QCheckBox)
        self.assertEqual(ed["business_budget.g2_max_fpr_pct"].maximum(), 100.0)
        ed["business_budget.g2_max_fpr_pct"].setValue(150.0)       # 控件层夹到上限
        self.assertEqual(ed["business_budget.g2_max_fpr_pct"].value(), 100.0)
        ed["business_budget.g2_max_fpr_pct"].setValue(3.0)
        ed["business_budget.raw_hot_days"].setValue(45)
        ed["business_budget.snapshot_keep_activated"].setChecked(False)

        captured = {}

        def _capture(_w, fn, cb=None, *a, **k):
            captured["res"] = fn()
            if cb:
                cb(captured["res"])

        client = page._client
        client.save_config = lambda updates: (captured.__setitem__("updates", updates),
                                              RS.api_update_config(updates))[1]
        with patch.object(RS, "get_settings", lambda: s), \
                patch.object(RS, "log_action", lambda *a, **k: None), \
                patch.object(SP, "run_async", _capture), \
                patch.object(SP, "notify", lambda *a, **k: None):
            page._on_save()
            up = captured["updates"]
            self.assertEqual(up["business_budget.g2_max_fpr_pct"], 3.0)
            self.assertEqual(up["business_budget.raw_hot_days"], 45)
            self.assertIs(type(up["business_budget.raw_hot_days"]), int)
            self.assertIs(up["business_budget.snapshot_keep_activated"], False)
            bb_keys = [k for k in up if k.startswith("business_budget.")]
            self.assertTrue(all(k in captured["res"]["applied"] for k in bb_keys),
                            captured["res"])
            bb = C.Settings(s.config_path).business_budget()
            self.assertEqual((bb["g2_max_fpr_pct"], bb["raw_hot_days"],
                              bb["snapshot_keep_activated"]), (3.0, 45, False))
            page.restore_defaults()
            self.assertEqual(ed["business_budget.g2_max_fpr_pct"].value(), 5.0)
            self.assertEqual(ed["business_budget.raw_hot_days"].value(), 90)
            self.assertTrue(ed["business_budget.snapshot_keep_activated"].isChecked())
            page._on_save()
        bb = C.Settings(s.config_path).business_budget()
        for k, v in USER_DEFAULTS.items():
            self.assertEqual(bb[k], v, k)


def _stats_page(get_biz):
    with patch.object(S, "run_async", lambda *a, **k: None):
        return S.StatsPage(FakeClient(), lambda: "", lambda: 200.0,
                           get_biz_budget=get_biz)


class ConsumerTest(unittest.TestCase):
    def test_stats_page_fpr_gray(self):
        _log(8, "统计页：误报率/灰区率按配置上限判超限（默认 5%/15%，改配置后阈值随之变化）")
        biz = {}
        p = _stats_page(lambda: biz)
        q = {"n_detections": 100, "n_reviewed": 100, "coverage": 1.0,
             "true_defect_rate": 0.1, "fp": 6, "tn": 94, "fn": 0, "tp": 0,
             "fp_rate": 0.06, "fn_rate": 0.0}
        p._fill_quality(q)                       # 6% > 默认 5%
        self.assertIn("上限 5%", p.kpi_fp._sub.text())
        self.assertIn("超限", p.kpi_fp._sub.text())
        biz.update(g2_max_fpr_pct=8.0)
        p._fill_quality(q)                       # 6% ≤ 8%
        self.assertIn("上限 8%", p.kpi_fp._sub.text())
        self.assertNotIn("超限", p.kpi_fp._sub.text())
        h = {"scanned": 100, "gray_count": 20, "gray_rate": 0.20,
             "open_alerts": 0, "open_rate": 0.0, "align_warns": 0, "align_rate": 0.0}
        p._fill_health(h)                        # 20% > 默认 15%
        self.assertIn("上限 15%", p.kpi_gray._sub.text())
        self.assertIn("超限", p.kpi_gray._sub.text())
        biz.update(max_gray_ratio_pct=25.0)
        p._fill_health(h)
        self.assertNotIn("超限", p.kpi_gray._sub.text())
        self.assertEqual(p.kpi_gray._value.text(), "20.0%")

    def test_stats_page_fallback(self):
        _log(9, "统计页：未注入/注入非法值时回落 UI 默认（与后端默认一致）")
        self.assertEqual(S.biz_budget_value(None, "g2_max_fpr_pct"), 5.0)
        self.assertEqual(S.biz_budget_value(lambda: {"max_gray_ratio_pct": "x"},
                                            "max_gray_ratio_pct"), 15.0)
        self.assertEqual(S.biz_budget_value(lambda: None, "g2_min_recall_pct"), 95.0)
        for k, v in S.BIZ_BUDGET_DEFAULTS.items():
            self.assertEqual(C.BUSINESS_BUDGET_SPEC[k][1], v, k)
        p = _stats_page(None)
        p._fill_quality({"n_detections": 10, "n_reviewed": 10, "fp": 0, "tn": 10,
                         "fn": 0, "tp": 0, "fp_rate": 0.0, "fn_rate": 0.0,
                         "true_defect_rate": 0.0})
        self.assertIn("上限 5%", p.kpi_fp._sub.text())
        self.assertNotIn("超限", p.kpi_fp._sub.text())

    def test_eval_page_verdict(self):
        _log(10, "评估页：FPR/缺陷召回对照配置达标线（达标/超限/低于下限/无定义）")
        biz = {}
        with patch.object(E, "run_async", lambda *a, **k: None):
            page = E.EvalPage(FakeClient(), lambda: "", lambda: 200.0,
                              get_biz_budget=lambda: biz)
        # fpr=4/100=4%，recall=96/100=96%
        mets = {"tp": 96, "fn": 4, "fp": 4, "tn": 96}
        txt = page._budget_verdict(mets, 96, 96, 4, 4)
        self.assertIn("FPR) ≤5%（达标）", txt)
        self.assertIn("缺陷召回 ≥95%（达标）", txt)
        biz.update(g2_max_fpr_pct=3.0, g2_min_recall_pct=97.0)
        txt = page._budget_verdict(mets, 96, 96, 4, 4)
        self.assertIn("≤3%（超限）", txt)
        self.assertIn("≥97%（低于下限）", txt)
        txt = page._budget_verdict({"tp": 0, "fn": 0, "fp": 0, "tn": 0}, 0, 0, 0, 0)
        self.assertEqual(txt.count("无定义"), 2)
        page._runs = [{"metrics": mets, "latency": {}}]
        page.table.setRowCount(1)
        page.table.setCurrentCell(0, 0)
        page._on_row_selected()
        self.assertIn("业务预算达标线", page.lbl_metrics_detail.text())


if __name__ == "__main__":
    unittest.main()
