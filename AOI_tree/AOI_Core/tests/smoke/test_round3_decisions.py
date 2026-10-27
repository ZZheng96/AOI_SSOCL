"""第三轮"待决策事项"前端化：governance_decisions 默认值/校验、/system/config 冻结 409、
冻结时间服务端维护、依赖锁 warn/block、候选 auto_on_approve、校准隔离透传
resolve_isolation、设置页"待决策事项"分组。全部在 TEMP 配置上运行，不触碰生产 DB/快照/配置。"""
import os
import tempfile
import time
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastapi import HTTPException
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel

from algo.eval.offline import resolve_isolation
from backend.api import routes_candidates as RC
from backend.api import routes_system as RS
from backend.core import config as C
from backend.core import deplock as D
from backend import engine as ENG
from ui.pages import settings_page as SP

_app = QApplication.instance() or QApplication([])
N = 12
_T0 = time.perf_counter()

GOV_DEFAULTS = {
    "gt_threshold_frozen": False, "gt_threshold_frozen_at": "", "gt_dataset_ref": "",
    "ac045_l1_real_training_enabled": False, "ac046_l1_pollution_exit_enabled": False,
    "candidate_release_mode": "manual", "calibration_isolation": "auto",
    "dependency_lock_policy": "warn", "ac062_legacy_artifact_dir": "",
    "sync28_29_accepted": False, "production_acceptance_signed": False,
    "blob_gpu_enabled": False, "approved_extra_good_batches": "",
}


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _tmp_settings(tc, extra_yaml: str = "") -> C.Settings:
    tmp = tempfile.TemporaryDirectory(prefix="aoi_round3_")
    tc.addCleanup(tmp.cleanup)
    p = Path(tmp.name) / "config.yaml"
    p.write_text("system:\n  storage_dir: storage\n" + extra_yaml, encoding="utf-8")
    return C.Settings(p)


def _tmp_lock(tc, text: str) -> Path:
    tmp = tempfile.TemporaryDirectory(prefix="aoi_round3_lock_")
    tc.addCleanup(tmp.cleanup)
    p = Path(tmp.name) / "requirements-lock.txt"
    p.write_text(text, encoding="utf-8")
    return p


class GovernanceConfigTest(unittest.TestCase):
    def test_defaults_and_validation(self):
        _log(1, "governance_decisions 默认值（AC-045/046 默认关、manual/auto/warn）+ 枚举/类型校验")
        s = _tmp_settings(self)
        self.assertEqual(s.governance_decisions(), GOV_DEFAULTS)
        self.assertEqual({k: d for k, (_t, d, _c) in C.GOVERNANCE_DECISIONS_SPEC.items()},
                         GOV_DEFAULTS)
        for key, val in [("candidate_release_mode", "yolo"), ("calibration_isolation", "x"),
                         ("dependency_lock_policy", "ignore"), ("gt_threshold_frozen", 1),
                         ("ac045_l1_real_training_enabled", "true")]:
            with self.assertRaises(C.ConfigValidationError, msg=f"{key}={val!r}"):
                s.update("governance_decisions", key, val)
        with self.assertRaises(C.ConfigValidationError) as cm:
            s.update("governance_decisions", "calibration_isolation", "x")
        self.assertIn("取值非法", str(cm.exception))
        s.update("governance_decisions", "calibration_isolation", "holdout")
        s.update("governance_decisions", "dependency_lock_policy", "block")
        gd = C.Settings(s.config_path).governance_decisions()
        self.assertEqual((gd["calibration_isolation"], gd["dependency_lock_policy"]),
                         ("holdout", "block"))

    def test_bad_stored_fallback(self):
        _log(2, "手改 yaml 非法存量值回落默认")
        s = _tmp_settings(self, "governance_decisions:\n  calibration_isolation: zzz\n"
                                "  gt_threshold_frozen: 1\n  dependency_lock_policy: block\n")
        gd = s.governance_decisions()
        self.assertEqual((gd["calibration_isolation"], gd["gt_threshold_frozen"],
                          gd["dependency_lock_policy"]), ("auto", False, "block"))


class ApiGovernanceTest(unittest.TestCase):
    def setUp(self):
        self.s = _tmp_settings(self)
        for p in (patch.object(RS, "get_settings", lambda: self.s),
                  patch.object(RS, "log_action", lambda *a, **k: None)):
            p.start()
            self.addCleanup(p.stop)

    def _items(self):
        return {f"{it['section']}.{it['key']}": it for it in RS.api_get_config()["items"]}

    def test_get_items(self):
        _log(3, "GET /system/config：11 项待决策带 value/default/impact/readonly/options；未冻结无 locked")
        items = self._items()
        gov = {k.split(".", 1)[1]: v for k, v in items.items()
               if k.startswith("governance_decisions.")}
        self.assertEqual(set(gov), set(GOV_DEFAULTS))
        for k, it in gov.items():
            self.assertEqual(it["value"], GOV_DEFAULTS[k], k)
            self.assertEqual(it["default"], GOV_DEFAULTS[k], k)
            self.assertTrue(it["impact"], k)
            self.assertTrue(it["desc"], k)
        self.assertTrue(gov["gt_threshold_frozen_at"]["readonly"])
        self.assertEqual(gov["calibration_isolation"]["type"], "choice")
        self.assertEqual([o[0] for o in gov["calibration_isolation"]["options"]],
                         ["auto", "self", "holdout"])
        self.assertEqual([o[0] for o in gov["dependency_lock_policy"]["options"]],
                         ["warn", "block"])
        self.assertFalse(any(it.get("locked") for it in items.values()))

    def test_freeze_409(self):
        _log(4, "冻结：时间自动写入；冻结后改阈值整批 409 THRESHOLD_FROZEN，同值回传不触发")
        res = RS.api_update_config({"governance_decisions.gt_threshold_frozen": True,
                                    "governance_decisions.gt_dataset_ref": "gt_v1"})
        self.assertIn("governance_decisions.gt_threshold_frozen_at", res["applied"])
        gd = self.s.governance_decisions()
        self.assertTrue(gd["gt_threshold_frozen"])
        self.assertRegex(gd["gt_threshold_frozen_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        items = self._items()
        for k in C.FROZEN_THRESHOLD_KEYS:
            self.assertTrue(items[f"business_budget.{k}"].get("locked"), k)
        self.assertTrue(items["governance_decisions.gt_dataset_ref"].get("locked"))
        with self.assertRaises(HTTPException) as cm:
            RS.api_update_config({"business_budget.g2_max_fpr_pct": 3.0,
                                  "business_budget.raw_hot_days": 30})
        self.assertEqual(cm.exception.status_code, 409)
        self.assertEqual(cm.exception.detail["reason_code"], "THRESHOLD_FROZEN")
        self.assertEqual(cm.exception.detail["keys"], ["business_budget.g2_max_fpr_pct"])
        self.assertEqual(self.s.business_budget()["raw_hot_days"], 90)  # 整批不写
        with self.assertRaises(HTTPException):
            RS.api_update_config({"governance_decisions.gt_dataset_ref": "gt_v2"})
        # UI 全量回传：值不变 → 不 409；非冻结阈值项照常可改
        frozen_at = gd["gt_threshold_frozen_at"]
        res = RS.api_update_config({"business_budget.g2_max_fpr_pct": 5.0,
                                    "business_budget.raw_hot_days": 30,
                                    "governance_decisions.gt_threshold_frozen": True,
                                    "governance_decisions.gt_threshold_frozen_at": frozen_at})
        self.assertIn("business_budget.raw_hot_days", res["applied"])
        self.assertEqual(res["rejected"], [])
        self.assertEqual(self.s.governance_decisions()["gt_threshold_frozen_at"], frozen_at)

    def test_unfreeze_and_readonly(self):
        _log(5, "同批冻结+定值允许（本次值即冻结值）；解冻同批可改阈值且清空冻结时间；只读项篡改被拒")
        res = RS.api_update_config({"governance_decisions.gt_threshold_frozen": True,
                                    "business_budget.g2_min_recall_pct": 96.0})
        self.assertIn("business_budget.g2_min_recall_pct", res["applied"])
        self.assertTrue(self.s.governance_decisions()["gt_threshold_frozen"])
        with self.assertRaises(HTTPException):
            RS.api_update_config({"business_budget.g2_min_recall_pct": 97.0})
        res = RS.api_update_config({"governance_decisions.gt_threshold_frozen": False,
                                    "business_budget.g2_min_recall_pct": 90.0})
        self.assertIn("business_budget.g2_min_recall_pct", res["applied"])
        gd = self.s.governance_decisions()
        self.assertEqual((gd["gt_threshold_frozen"], gd["gt_threshold_frozen_at"]), (False, ""))
        self.assertEqual(self.s.business_budget()["g2_min_recall_pct"], 90.0)
        res = RS.api_update_config({"governance_decisions.gt_threshold_frozen_at": "2020-01-01"})
        self.assertEqual(res["rejected"], ["governance_decisions.gt_threshold_frozen_at"])
        self.assertIn("只读", res["errors"]["governance_decisions.gt_threshold_frozen_at"])
        res = RS.api_update_config({"governance_decisions.candidate_release_mode": "bad"})
        self.assertIn("取值非法", res["errors"]["governance_decisions.candidate_release_mode"])

    def test_system_info_dependency_lock(self):
        _log(6, "/system/info 依赖锁自检：带 policy；异常不阻断")
        st = RS._dependency_lock_info()
        self.assertEqual(st["policy"], "warn")
        self.assertIn("ok", st)
        with patch.object(D, "lock_status", side_effect=RuntimeError("boom")):
            self.assertEqual(RS._dependency_lock_info(), {"ok": False, "error": "boom"})


class DepLockTest(unittest.TestCase):
    def test_lock_status_and_guard(self):
        _log(7, "依赖锁：+cu126 后缀忽略；版本不一致/缺包 warn 放行、block 抛 DEPENDENCY_LOCK_MISMATCH")
        pyyaml = metadata.version("PyYAML")
        good = _tmp_lock(self, f"# lock\nPyYAML=={pyyaml}+cu126\n")
        self.assertTrue(D.lock_status(good)["ok"])
        self.assertTrue(D.dependency_guard("block", good)["ok"])
        bad = _tmp_lock(self, "PyYAML==0.0.1\nzz-not-installed-pkg==1.0\n")
        st = D.dependency_guard("warn", bad)
        self.assertFalse(st["ok"])
        self.assertEqual(st["mismatches"][0]["package"], "PyYAML")
        self.assertEqual(st["missing"], ["zz-not-installed-pkg"])
        from backend.self_learning.candidate_release import CandidateReleaseError
        with self.assertRaises(CandidateReleaseError) as cm:
            D.dependency_guard("block", bad)
        self.assertEqual(cm.exception.reason_code, "DEPENDENCY_LOCK_MISMATCH")
        self.assertEqual(len(cm.exception.violations), 2)
        nolock = D.lock_status(Path(bad.parent) / "nope.txt")
        self.assertEqual((nolock["ok"], nolock["lock_found"]), (True, False))


class IsolationTest(unittest.TestCase):
    def test_apply_governance_isolation(self):
        _log(8, "校准隔离：auto 不覆盖 yaml；holdout/self 注入 calibration.isolation → resolve_isolation 读到")
        s = _tmp_settings(self)
        with patch.object(C, "get_settings", lambda: s):
            cfg = {"calibration": {"cdf_bins": 64}}
            ENG.apply_governance_isolation(cfg)
            self.assertEqual(cfg, {"calibration": {"cdf_bins": 64}})
            s.update("governance_decisions", "calibration_isolation", "holdout")
            cfg = {}
            ENG.apply_governance_isolation(cfg)
            self.assertEqual(cfg["calibration"]["isolation"], "holdout")
            iso, cal, fit, info = resolve_isolation(cfg["calibration"], 10)
            self.assertEqual((iso, info["requested"]), ("holdout", "holdout"))
            s.update("governance_decisions", "calibration_isolation", "self")
            cfg = {"calibration": {"isolation": "holdout"}}
            ENG.apply_governance_isolation(cfg)
            self.assertEqual(resolve_isolation(cfg["calibration"], 40)[0], "self")


class _FakeRelease:
    def __init__(self):
        self.switched = []

    def decide(self, category, version, approved, report_hash, actor, reason):
        return {"state": "approved" if approved else "rejected"}

    def switch(self, category, version, rollback=False):
        self.switched.append((category, version))
        return {"state": "active", "version": version}


class CandidatePolicyTest(unittest.TestCase):
    def setUp(self):
        self.rel = _FakeRelease()
        self.gov = dict(GOV_DEFAULTS)
        for p in (patch.object(RC, "release", lambda: self.rel),
                  patch.object(RC, "log_action", lambda *a, **k: None),
                  patch.object(RC, "_governance", lambda: self.gov)):
            p.start()
            self.addCleanup(p.stop)
        self.req = RC.DecisionRequest(approved=True, report_hash="h", reason="ok")

    def test_release_mode(self):
        _log(9, "候选发布策略：manual 批准不激活；auto_on_approve 批准后自动激活；拒绝不激活")
        with patch.object(D, "lock_status", lambda path=None: {"ok": True, "mismatches": [],
                                                               "missing": []}):
            out = RC.decision("cat", 3, self.req)
            self.assertNotIn("auto_activate", out)
            self.assertEqual(self.rel.switched, [])
            self.gov["candidate_release_mode"] = "auto_on_approve"
            out = RC.decision("cat", 3, self.req)
            self.assertTrue(out["auto_activate"]["ok"])
            self.assertEqual(self.rel.switched, [("cat", 3)])
            rej = RC.DecisionRequest(approved=False, report_hash="h", reason="no")
            self.assertNotIn("auto_activate", RC.decision("cat", 4, rej))
            self.assertEqual(len(self.rel.switched), 1)

    def test_dependency_policy(self):
        _log(10, "依赖锁策略：block 激活 409 且不切换；auto 激活失败不回滚批准；warn 附告警")
        bad = {"ok": False, "lock_found": True, "lock_path": "x",
               "mismatches": [{"package": "torch", "locked": "2.7.0", "installed": "2.5.0"}],
               "missing": []}
        with patch.object(D, "lock_status", lambda path=None: bad):
            self.gov["dependency_lock_policy"] = "block"
            with self.assertRaises(HTTPException) as cm:
                RC.activate("cat", 3)
            self.assertEqual(cm.exception.status_code, 409)
            self.assertEqual(cm.exception.detail["reason_code"], "DEPENDENCY_LOCK_MISMATCH")
            self.assertEqual(self.rel.switched, [])
            self.gov["candidate_release_mode"] = "auto_on_approve"
            out = RC.decision("cat", 3, self.req)
            self.assertEqual(out["state"], "approved")
            self.assertFalse(out["auto_activate"]["ok"])
            self.assertEqual(out["auto_activate"]["detail"]["reason_code"],
                             "DEPENDENCY_LOCK_MISMATCH")
            self.gov["dependency_lock_policy"] = "warn"
            out = RC.activate("cat", 3)
            self.assertEqual(out["state"], "active")
            self.assertEqual(out["dependency_lock_warning"]["mismatches"][0]["package"], "torch")


class SettingsPageGovTest(unittest.TestCase):
    def _page(self, s):
        with patch.object(RS, "get_settings", lambda: s):
            data = RS.api_get_config()
        with patch.object(SP, "run_async", lambda *a, **k: None):
            page = SP.SettingsPage(FakeClient())
        page._fill(data)
        return page

    def test_group_widgets(self):
        _log(11, "设置页：“待决策事项”分组、下拉框/只读标签/说明-当前值-影响范围、恢复默认不动决策")
        s = _tmp_settings(self)
        page = self._page(s)
        titles = [page._form_lay.itemAt(i).widget().title()
                  for i in range(page._form_lay.count())
                  if page._form_lay.itemAt(i).widget() is not None]
        self.assertIn("待决策事项", titles)
        ed = page._editors
        self.assertIsInstance(ed["governance_decisions.calibration_isolation"], QComboBox)
        self.assertIsInstance(ed["governance_decisions.gt_threshold_frozen"], QCheckBox)
        self.assertNotIn("governance_decisions.gt_threshold_frozen_at", ed)
        ro = page.findChild(QLabel, "ro:governance_decisions.gt_threshold_frozen_at")
        self.assertIsNotNone(ro)
        sub = page.findChild(QLabel, "sub:governance_decisions.dependency_lock_policy")
        for t in ("说明：", "当前值：warn", "默认：warn", "影响范围："):
            self.assertIn(t, sub.text())
        self.assertFalse(any(k.startswith("governance_decisions.") for k in page._defaults))
        cb = ed["governance_decisions.calibration_isolation"]
        cb.setCurrentIndex(cb.findData("holdout"))
        with patch.object(SP, "notify", lambda *a, **k: None):
            page.restore_defaults()
        self.assertEqual(cb.currentData(), "holdout")

    def test_save_and_locked(self):
        _log(12, "设置页：保存下拉/开关值写回；冻结后阈值编辑器禁用；改禁用值触发 409 提示")
        s = _tmp_settings(self)
        page = self._page(s)
        ed = page._editors
        cb = ed["governance_decisions.candidate_release_mode"]
        cb.setCurrentIndex(cb.findData("auto_on_approve"))
        ed["governance_decisions.gt_threshold_frozen"].setChecked(True)
        ed["governance_decisions.ac045_l1_real_training_enabled"].setChecked(True)
        captured, refills = {}, []

        def _capture(_w, fn, cb_=None, *a, **k):
            res = fn()
            if fn == page._client.get_config:
                refills.append(1)
                return
            captured["res"] = res
            if cb_:
                cb_(res)

        page._client.get_config = lambda: None
        page._client.save_config = lambda up: (captured.__setitem__("updates", up),
                                               RS.api_update_config(up))[1]
        with patch.object(RS, "get_settings", lambda: s), \
                patch.object(RS, "log_action", lambda *a, **k: None), \
                patch.object(SP, "run_async", _capture), \
                patch.object(SP, "notify", lambda *a, **k: None):
            page._on_save()
        up = captured["updates"]
        self.assertEqual(up["governance_decisions.candidate_release_mode"], "auto_on_approve")
        self.assertNotIn("governance_decisions.gt_threshold_frozen_at", up)
        self.assertEqual(captured["res"]["rejected"], [])
        self.assertEqual(refills, [1])  # 决策项变化 → 重拉表单
        gd = C.Settings(s.config_path).governance_decisions()
        self.assertEqual((gd["candidate_release_mode"], gd["gt_threshold_frozen"],
                          gd["ac045_l1_real_training_enabled"]),
                         ("auto_on_approve", True, True))
        self.assertTrue(gd["gt_threshold_frozen_at"])
        page = self._page(s)
        ed = page._editors
        for k in C.FROZEN_THRESHOLD_KEYS:
            self.assertFalse(ed[f"business_budget.{k}"].isEnabled(), k)
        self.assertFalse(ed["governance_decisions.gt_dataset_ref"].isEnabled())
        self.assertTrue(ed["business_budget.raw_hot_days"].isEnabled())
        ro = page.findChild(QLabel, "ro:governance_decisions.gt_threshold_frozen_at")
        self.assertEqual(ro.text(), gd["gt_threshold_frozen_at"])
        # 全量回传（含禁用项原值）不触发 409
        with patch.object(RS, "get_settings", lambda: s), \
                patch.object(RS, "log_action", lambda *a, **k: None):
            ups = {k: page._editor_value(w) for k, w in ed.items()}
            self.assertEqual(RS.api_update_config(ups)["rejected"], [])
            fpr = ed["business_budget.g2_max_fpr_pct"]
            fpr.setValue(1.0)  # 程序绕过禁用态 → 后端兜底 409
            ups = {k: page._editor_value(w) for k, w in ed.items()}
            with self.assertRaises(HTTPException) as cm:
                RS.api_update_config(ups)
            self.assertEqual(cm.exception.status_code, 409)
            fpr.setValue(5.0)
            with patch.object(SP, "notify", lambda *a, **k: None):
                page.restore_defaults()
            self.assertEqual(fpr.value(), 5.0)


if __name__ == "__main__":
    unittest.main()
