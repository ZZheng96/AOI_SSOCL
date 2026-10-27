"""第三轮覆盖告警：blob_gpu_enabled 接入（配置/环境变量/槽位 cfg 任一为真，AOI_BLOB_GPU=0 强关）、
批次覆盖只读检查（未覆盖批次 + 域外比例 + 纳入 train/good 建议）、approved_extra_good_batches 登记、
FPR≤预算最大 recall 参考工作点、评估页展示。不触碰生产 DB/快照/配置。"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

import backend.engine as ENG
from algo.eval import coverage as COV
from algo.slots import blob as B
from backend.api import routes_system as RS
from backend.core import config as C
from backend.evaluation import service as SV
from ui.pages import eval_page as E

_app = QApplication.instance() or QApplication([])
N = 8
_T0 = time.perf_counter()


def _log(i, msg):
    print(f"[{i}/{N}] {msg}（+{time.perf_counter() - _T0:.2f}s）", flush=True)


class FakeClient(QObject):
    error_occurred = Signal(str)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _tmp_settings(tc, extra_yaml: str = "") -> C.Settings:
    tmp = tempfile.TemporaryDirectory(prefix="aoi_round3_cov_")
    tc.addCleanup(tmp.cleanup)
    p = Path(tmp.name) / "config.yaml"
    p.write_text("system:\n  storage_dir: storage\n" + extra_yaml, encoding="utf-8")
    return C.Settings(p)


class GovernanceItemsTest(unittest.TestCase):
    def test_defaults_and_api(self):
        _log(1, "新增治理项默认值/类型校验 + /system/config 说明文案")
        s = _tmp_settings(self)
        gd = s.governance_decisions()
        self.assertIs(gd["blob_gpu_enabled"], False)
        self.assertEqual(gd["approved_extra_good_batches"], "")
        with self.assertRaises(C.ConfigValidationError):
            s.update("governance_decisions", "blob_gpu_enabled", 1)
        with self.assertRaises(C.ConfigValidationError):
            s.update("governance_decisions", "approved_extra_good_batches", ["a"])
        s.update("governance_decisions", "approved_extra_good_batches", "Image_2026, t_fov")
        s.update("governance_decisions", "blob_gpu_enabled", True)
        gd = C.Settings(s.config_path).governance_decisions()
        self.assertEqual((gd["blob_gpu_enabled"], gd["approved_extra_good_batches"]),
                         (True, "Image_2026, t_fov"))
        with patch.object(RS, "get_settings", lambda: _tmp_settings(self)):
            items = {f"{it['section']}.{it['key']}": it for it in RS.api_get_config()["items"]}
        g = items["governance_decisions.blob_gpu_enabled"]
        self.assertEqual((g["type"], g["value"]), ("bool", False))
        self.assertIn("316.8ms", g["desc"])
        self.assertIn("181.8ms", g["desc"])
        self.assertIn("AUROC", g["desc"])
        a = items["governance_decisions.approved_extra_good_batches"]
        self.assertEqual((a["type"], a["value"]), ("str", ""))
        self.assertIn("不自动重建", a["desc"])
        self.assertTrue(g["impact"] and a["impact"])


class BlobGpuSwitchTest(unittest.TestCase):
    def setUp(self):
        self._old = B._GPU_OVERRIDE
        self.addCleanup(B.set_gpu_override, self._old)
        p = patch.object(torch.cuda, "is_available", lambda: True)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("AOI_BLOB_GPU", None)

    def _on(self, cfg, env=None):
        envs = {} if env is None else {"AOI_BLOB_GPU": env}
        with patch.dict(os.environ, envs):
            if env is None:
                os.environ.pop("AOI_BLOB_GPU", None)
            return B.BlobSlot(cfg, "cuda")._dev is not None

    def test_or_logic(self):
        _log(2, "blob GPU：治理配置/环境变量/槽位 cfg 任一为真启用；AOI_BLOB_GPU=0 强关；CPU 设备不启用")
        B.set_gpu_override(lambda: False)
        self.assertFalse(self._on({}))
        self.assertTrue(self._on({"gpu": True}))
        self.assertTrue(self._on({}, env="1"))
        B.set_gpu_override(lambda: True)
        self.assertTrue(self._on({}))
        self.assertFalse(self._on({"gpu": True}, env="0"))
        self.assertFalse(B.BlobSlot({}, "cpu")._dev is not None)
        B.set_gpu_override(lambda: 1 / 0)               # 读取异常按关
        self.assertFalse(self._on({}))

    def test_engine_registration_reads_settings(self):
        _log(3, "backend.engine 注册的读取函数取 governance_decisions.blob_gpu_enabled")
        self.assertIs(B._GPU_OVERRIDE, ENG.governance_blob_gpu_enabled)
        s = _tmp_settings(self)
        with patch("backend.core.config.get_settings", lambda: s):
            self.assertFalse(ENG.governance_blob_gpu_enabled())
            self.assertFalse(self._on({}))
            s.update("governance_decisions", "blob_gpu_enabled", True)
            self.assertTrue(ENG.governance_blob_gpu_enabled())
            self.assertTrue(self._on({}))


class CoverageTest(unittest.TestCase):
    TRAIN = [f"D:/data/cat/train/good/{i:04d}.png" for i in range(598, 622)]
    EVAL = (["D:/data/cat/test/good/Image_20260301_001.bmp",
             "D:/data/cat/test/good/Image_20260301_002.bmp",
             "D:/data/cat/test/good/t_fov_03.png",
             "D:/data/cat/test/good/0630.png",
             "D:/data/cat/test/defect/0700.png"])

    def test_infer_batch(self):
        _log(4, "批次推断：子目录优先，否则文件名前缀（Image_2026 / t_fov / numeric）")
        f = COV.infer_batch
        self.assertEqual(f("x/train/good/0598.png"), "numeric")
        self.assertEqual(f("x\\test\\good\\Image_20260301_001.bmp"), "Image_2026")
        self.assertEqual(f("x/test/good/t_fov_03.png"), "t_fov")
        self.assertEqual(f("x/test/good/t_fov3.png"), "t_fov")
        self.assertEqual(f("x/lot_A/img_001.png"), "lot_A")
        self.assertEqual(COV.parse_batch_list(" a, ,b ,"), ["a", "b"])

    def test_batch_coverage(self):
        _log(5, "覆盖检查：未覆盖批次列表 + 域外比例 + 建议文案 + 登记批次不出建议")
        scores = [3.0, 2.5, 4.0, 0.5, 9.0]
        labels = [0, 0, 0, 0, 1]
        r = COV.batch_coverage(self.TRAIN, self.EVAL, scores, labels,
                               train_scores=[0.1, 0.9, 1.2])
        self.assertEqual(r["train_batches"], ["numeric"])
        self.assertEqual(r["uncovered_batches"], ["Image_2026", "t_fov"])
        self.assertEqual(r["n_eval_normal"], 4)                      # 缺陷样本不计
        self.assertEqual(r["out_of_domain_ratio"], 0.75)
        self.assertEqual((r["train_ref"], r["train_ref_source"]), (1.2, "train_max"))
        self.assertTrue(r["per_batch"]["numeric"]["covered"])
        self.assertEqual(r["per_batch"]["Image_2026"]["ood_ratio"], 1.0)
        self.assertIn("将批次 Image_2026 的正常样本纳入 train/good 后重建", r["suggestions"])
        r2 = COV.batch_coverage(self.TRAIN, self.EVAL, scores, labels,
                                train_ref=2.8, approved=["t_fov"])
        self.assertEqual(r2["train_ref_source"], "tau_high(q99)")
        self.assertEqual(r2["out_of_domain_ratio"], 0.5)
        self.assertEqual(r2["suggestions"],
                         ["将批次 Image_2026 的正常样本纳入 train/good 后重建"])
        self.assertTrue(r2["per_batch"]["t_fov"]["approved"])
        r3 = COV.batch_coverage([], self.EVAL, None, labels)
        self.assertIsNone(r3["out_of_domain_ratio"])
        self.assertEqual(r3["uncovered_batches"], [])
        self.assertIsNone(r3["per_batch"]["t_fov"]["covered"])

    def test_max_recall_at_fpr(self):
        _log(6, "FPR≤预算最大 recall 工作点（score>=thr 口径，与暴力枚举一致）")
        rng = np.random.default_rng(0)
        s = np.concatenate([rng.normal(0, 1, 60), rng.normal(1.5, 1, 40)])
        y = np.array([0] * 60 + [1] * 40)
        for mf in (0.0, 0.05, 0.2, 1.0):
            wp = COV.max_recall_at_fpr(s, y, mf)
            best = max(((s[y == 1] >= t).mean() for t in np.unique(s)
                        if (s[y == 0] >= t).mean() <= mf), default=0.0)
            self.assertAlmostEqual(wp["recall"], round(best, 4))
            pred = s >= wp["threshold"]
            self.assertEqual(int((pred & (y == 0)).sum()), wp["fp"])
            self.assertLessEqual(wp["fpr"], mf + 1e-9)
            self.assertEqual(wp["tp"] + wp["fn"], 40)
        self.assertEqual(COV.max_recall_at_fpr([1, 2, 3], [0, 1, 1], 0.0)["threshold"], 2.0)
        self.assertIsNone(COV.max_recall_at_fpr([1, 2], [0, 0], 0.05))

    def test_service_reference_reports(self):
        _log(7, "run_accuracy_eval 参考报告：预算取 g2_max_fpr_pct，train 上沿退回 tau_high，不改 decider")
        s = _tmp_settings(self, "business_budget:\n  g2_max_fpr_pct: 25.0\n"
                                "governance_decisions:\n  approved_extra_good_batches: t_fov\n")
        dec = SimpleNamespace(tau_high=2.8, tau_gray=2.0)
        pipe = SimpleNamespace(decider=dec, _train_normal_paths=self.TRAIN)
        images = [(i, p, l) for i, (p, l) in enumerate(zip(self.EVAL, [0, 0, 0, 0, 1]))]
        sc = np.array([3.0, 2.5, 4.0, 0.5, 9.0])
        lb = np.array([0, 0, 0, 0, 1])
        with patch.object(SV, "get_settings", lambda: s):
            out = SV._reference_reports(pipe, images, sc, lb)
        wp = out["budget_working_point"]
        self.assertEqual((wp["max_fpr"], wp["recall"], wp["fp"]), (0.25, 1.0, 0))
        self.assertEqual(wp["threshold"], 9.0)        # recall 相同取更高阈值（更低 FPR）
        cov = out["batch_coverage"]
        self.assertEqual(cov["uncovered_batches"], ["Image_2026", "t_fov"])
        self.assertEqual(cov["out_of_domain_ratio"], 0.5)
        self.assertEqual(cov["approved_extra_good_batches"], ["t_fov"])
        self.assertEqual((dec.tau_high, dec.tau_gray), (2.8, 2.0))   # 不改生产阈值
        with patch.object(SV, "get_settings", lambda: s):
            out = SV._reference_reports(SimpleNamespace(), images, sc, lb)
        self.assertIsNone(out["batch_coverage"]["out_of_domain_ratio"])


class EvalPageTest(unittest.TestCase):
    def test_eval_page_text(self):
        _log(8, "评估页：业务预算行后展示参考工作点 + 覆盖告警/建议；旧记录不显示")
        with patch.object(E, "run_async", lambda *a, **k: None):
            page = E.EvalPage(FakeClient(), lambda: "", lambda: 200.0,
                              get_biz_budget=lambda: {})
        cov = COV.batch_coverage(CoverageTest.TRAIN, CoverageTest.EVAL,
                                 [3.0, 2.5, 4.0, 0.5, 9.0], [0, 0, 0, 0, 1],
                                 train_ref=2.8, approved=["t_fov"])
        mets = {"tp": 1, "fn": 0, "fp": 3, "tn": 1,
                "budget_working_point": {"recall": 0.8, "fpr": 0.05, "threshold": 1.989,
                                         "max_fpr": 0.05},
                "batch_coverage": cov}
        page._runs = [{"metrics": mets, "latency": {}}]
        page.table.setRowCount(1)
        page.table.setCurrentCell(0, 0)
        page._on_row_selected()
        txt = page.lbl_metrics_detail.text()
        self.assertIn("业务预算达标线", txt)
        self.assertIn("参考工作点（FPR≤5%）：最大缺陷召回 80.0%", txt)
        self.assertIn("阈值 1.989", txt)
        self.assertIn("不修改生产阈值", txt)
        self.assertIn("未覆盖批次 Image_2026、t_fov（已登记）", txt)
        self.assertIn("域外比例 50.0%", txt)
        self.assertIn("建议：将批次 Image_2026 的正常样本纳入 train/good 后重建", txt)
        self.assertLess(txt.index("业务预算达标线"), txt.index("参考工作点"))
        page._runs = [{"metrics": {"tp": 1, "fn": 0, "fp": 0, "tn": 1}, "latency": {}}]
        page._on_row_selected()
        self.assertNotIn("参考工作点", page.lbl_metrics_detail.text())
        self.assertIn("来源未知", page._coverage_text({"batch_coverage": {"train_batches": []}}))


if __name__ == "__main__":
    unittest.main()
