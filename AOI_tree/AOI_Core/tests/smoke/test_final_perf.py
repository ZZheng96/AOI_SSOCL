"""final_perf：§65 遗留（decision_trace 透出/降级分数随图变化）、AC-070 显存上界、AC-071 冷启动预算。

全部 TEMP 构造；替身仅替换算法动作（feedback/预测），不是精度验收。
"""
import json
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from algo.eval.offline import Pipeline
from tests.smoke import test_async_persist as _ap
from tests.smoke import test_isolated_candidate_path as _iso
from tests.smoke import test_snapshot_layout_compat as _compat


def _strip_tests(cls):
    """复用父类 setUp/工具，不重复执行父类测试。"""
    for name in dir(cls):
        if name.startswith("test_"):
            setattr(cls, name, None)
    return cls


# ---------------- §65.2 decision_trace 顶层透出 + 入库不被覆盖 ----------------
class DecisionTraceTest(_strip_tests(type("_AP", (_ap.AsyncPersistTest,), {}))):
    def test_trace_top_level_and_persisted(self):
        from backend.db import database
        from backend.db.models import Detection
        trace = [{"stage": "fusion", "degraded_slots": {"layout": {"layout_degraded": True}}}]
        res = self._result()
        res.extra = {**(res.extra or {}), "decision_trace": trace}
        rgb = self.rng.integers(0, 256, (300, 290, 3), dtype=np.uint8)
        out = self.svc._finish_detect(res, rgb, "img.jpg", "test", time.perf_counter())
        self.assertEqual(out["decision_trace"], trace)
        self.assertEqual(out["n_tiles"]["decision_trace"], trace)
        with database.session_scope() as s:
            row = s.get(Detection, out["detection_id"])
            self.assertEqual(row.n_tiles["decision_trace"], trace)  # 修复前被 [] 覆盖


# ---------------- §65.1 降级快照：不同图 raw 分数不同 ----------------
class DegradedScoresVaryTest(_strip_tests(type("_C", (_compat.SnapshotLayoutCompatTest,), {}))):
    def test_degraded_raw_scores_vary_per_image(self):
        pipe = self._fit_pipe()
        with tempfile.TemporaryDirectory() as tmp:
            self._save(pipe, tmp)
            self._make_legacy(tmp)
            p, _ = self._load(tmp)
        self.assertIn("layout", p.degraded_slots)
        rng = np.random.default_rng(11)
        imgs = [rng.normal(40 + 6 * i, 5 + i, (96, 96, 3)).clip(0, 255).astype(np.uint8)
                for i in range(4)]
        recs = [p._record(_compat._item(im, i)) for i, im in enumerate(imgs)]
        raws = [r["raw_scores"]["blob"] for r in recs]
        self.assertEqual(len(set(np.round(raws, 9))), len(raws), raws)
        for r in recs:
            self.assertIn("layout", r["decision_trace"][0]["degraded_slots"])
        # 同一输入恒定分数属预期（§65 实测 67 张"正常图"为同一 MD5 文件）
        a, b = (p._record(_compat._item(imgs[0], i))["raw_scores"]["blob"] for i in (0, 1))
        self.assertEqual(a, b)


# ---------------- AC-070 显存上界 ----------------
class VramBudgetTest(_strip_tests(type("_I", (_iso.IsolatedCandidateTest,), {}))):
    def _fake_cuda(self, total, peak):
        props = type("P", (), {"total_memory": total})()
        return [patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "get_device_properties", return_value=props),
                patch.object(torch.cuda, "reset_peak_memory_stats"),
                patch.object(torch.cuda, "memory_allocated", return_value=0),
                patch.object(torch.cuda, "max_memory_allocated", side_effect=lambda: peak[0])]

    def test_cpu_degrades_to_skip(self):
        result = self.run_candidate("cpu_skip")
        vram = result["candidate_budget"]["vram"]
        self.assertFalse(vram["enabled"])
        self.assertEqual(vram["skip_reason"], "no_cuda_device")
        self.assertFalse(result["candidate_budget"]["vram_guaranteed"])

    def test_adaptive_limit_and_override(self):
        from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred
        self.engine.device = "cuda"
        peak = [0]
        ps = self._fake_cuda(6 * 1024 ** 3, peak)
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        b = CandidateBudget(self.engine, category="test", event_id="v")
        self.assertEqual(b.vram["limit_bytes"], int(6 * 1024 ** 3 * 0.85))
        self.assertEqual(b.vram["source"], "fraction")
        self.engine.ssocl_cfg["candidate_budget"] = {"vram_fraction": 0.5}
        self.assertEqual(CandidateBudget(self.engine, category="test", event_id="v")
                         .vram["limit_bytes"], 3 * 1024 ** 3)
        self.engine.ssocl_cfg["candidate_budget"] = {"max_vram_bytes": 1000}
        b = CandidateBudget(self.engine, category="test", event_id="v")
        self.assertEqual((b.vram["limit_bytes"], b.vram["source"]), (1000, "max_vram_bytes"))
        peak[0] = 999
        b.check("ok")
        peak[0] = 1001
        with self.assertRaisesRegex(CandidateDeferred, "显存超限") as c:
            b.check("over")
        self.assertEqual(c.exception.state, "deferred")
        self.assertEqual(b.snapshot()["vram"]["peak_bytes"], 1001)
        self.assertEqual(b.snapshot()["vram_guaranteed"], "checkpoint")
        for bad in ({"vram_fraction": 0}, {"vram_fraction": 1.5}, {"max_vram_bytes": -1},
                    {"max_vram_bytes": True}):
            self.engine.ssocl_cfg["candidate_budget"] = bad
            with self.assertRaises(ValueError):
                CandidateBudget(self.engine, category="test", event_id="v")

    def test_vram_over_limit_defers_e2e(self):
        """候选全链路：显存峰值超限 → deferred（可重试），生产 pipe/版本/WAL 不变。"""
        from algo.ssocl.candidate_budget import CandidateDeferred
        peak = [10]
        ps = self._fake_cuda(1 << 30, peak)
        self.engine.ssocl_cfg["candidate_budget"] = {"max_vram_bytes": 100}
        def work(pipe, *a, **kw):
            peak[0] = 500  # 训练步显存峰值
            return self.feedback(pipe, *a, **kw)
        real_budget = self.engine.CandidateBudget
        def budget_cuda(engine, **kw):
            with patch.object(engine, "device", "cuda"):
                return real_budget(engine, **kw)
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(self.engine, "CandidateBudget", side_effect=budget_cuda), \
                patch.object(Pipeline, "feedback", work):
            with self.assertRaisesRegex(CandidateDeferred, "显存超限"):
                self.engine.learn_candidate("test", self.image, verdict="wrong", label=1,
                                            event_id="one", parent_version=1)
        state, ev = self.state()
        ev = json.loads(ev)
        self.assertEqual(state, "deferred")
        self.assertTrue(ev["retryable"])
        self.assertEqual(ev["candidate_budget"]["vram"]["limit_bytes"], 100)
        self.assert_production()
        self.assertFalse(list(self.root.glob(".candidate-*")))

    def test_real_oom_converted_to_deferred(self):
        """OOM 模拟：候选训练步抛 torch.cuda.OutOfMemoryError → deferred，工作态不破坏。"""
        from algo.ssocl.candidate_budget import CandidateDeferred
        def oom(pipe, *a, **kw):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        with patch.object(Pipeline, "feedback", oom), \
                patch.object(torch.cuda, "empty_cache") as ec:
            with self.assertRaisesRegex(CandidateDeferred, "OOM"):
                self.run_candidate()
        ec.assert_called()
        state, ev = self.state()
        ev = json.loads(ev)
        self.assertEqual((state, ev["reason_code"], ev["retryable"]),
                         ("deferred", "CANDIDATE_BUILD_DEFERRED", True))
        self.assert_production()
        self.assertFalse(list(self.root.glob(".candidate-*")))
        # 预测仍可用（工作态未破坏）
        self.assertIs(self.engine._pipe_entry("test")[0], self.pipe)
        # 新 event 可重试成功
        self.assertIn("candidate_budget", self.run_candidate("after_oom"))

    @unittest.skipUnless(torch.cuda.is_available(), "无 CUDA：真实显存路径降级跳过")
    def test_real_cuda_peak_tracked(self):
        from algo.ssocl.candidate_budget import CandidateBudget, CandidateDeferred
        self.engine.device = "cuda"
        base = torch.cuda.memory_allocated()
        self.engine.ssocl_cfg["candidate_budget"] = {"max_vram_bytes": base + 64 * 1024 ** 2}
        b = CandidateBudget(self.engine, category="test", event_id="real")
        x = torch.empty(16 * 1024 ** 2, dtype=torch.float32, device="cuda")  # 64MiB
        b.check("small")
        y = torch.empty(32 * 1024 ** 2, dtype=torch.float32, device="cuda")  # +128MiB
        with self.assertRaises(CandidateDeferred):
            b.check("big")
        self.assertGreaterEqual(b.vram["peak_bytes"], base + 192 * 1024 ** 2)
        del x, y
        torch.cuda.empty_cache()


# ---------------- AC-071 冷启动预算 ----------------
class ColdStartTest(_strip_tests(type("_I2", (_iso.IsolatedCandidateTest,), {}))):
    def _evict(self):
        self.close_engine_pipes()
        self.engine._pipes.clear()

    def test_cold_load_stats_and_warmup(self):
        self._evict()
        self.engine._ensure_loaded("test")
        st = self.engine.cold_start_stats("test")
        self.assertEqual(st["version"], 1)
        self.assertGreaterEqual(st["load_ms"], 0)
        self.assertGreaterEqual(st["replay_ms"], 0)
        calls = []
        def fake_predict(category, rgb, path_hint=None):
            calls.append(category)
            time.sleep(0.02 if len(calls) == 1 else 0.001)  # 首图慢、热态快
        with patch.object(self.engine, "predict_ndarray", side_effect=fake_predict):
            st = self.engine.warmup("test", warm_runs=3)
        self.assertEqual(len(calls), 4)
        self.assertGreater(st["first_infer_ms"], st["warm_ms"])
        self.assertEqual(st["warmup_runs"], 3)
        self.assertTrue(st["within_budget"], st)
        for k in ("load_ms", "replay_ms", "first_infer_ms", "warm_ms"):
            self.assertIn(k, st)

    def test_budget_configurable_and_violations(self):
        self.engine.ssocl_cfg["cold_start_budget"] = {"max_warm_ms": 0, "max_load_ms": 0}
        self._evict()
        self.engine._ensure_loaded("test")
        with patch.object(self.engine, "predict_ndarray",
                          side_effect=lambda *a, **k: time.sleep(0.002)):
            st = self.engine.warmup("test", warm_runs=1)
        self.assertFalse(st["within_budget"])
        self.assertIn("warm_ms", st["violations"])
        self.assertIn("load_ms", st["violations"])
        self.assertEqual(st["budget"]["max_warm_ms"], 0)
        self.engine.ssocl_cfg["cold_start_budget"] = {"max_bogus": 1}
        with self.assertRaisesRegex(ValueError, "未知冷启动预算配置"):
            self.engine._cold_budget()

    def test_load_failure_does_not_switch(self):
        from algo.persist import save_snapshot
        save_snapshot(self.pipe, str(self.root / "v2"), version=2)
        with patch("backend.engine.load_snapshot", create=True), \
                patch("algo.persist.load_snapshot", side_effect=RuntimeError("cold load boom")):
            with self.assertRaisesRegex(RuntimeError, "cold load boom"):
                self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 1)
        self.assertIs(self.engine._pipes["test"][0], self.pipe)
        # 冷载失败：不入缓存、不记录成功统计
        self._evict()
        with patch("algo.persist.load_snapshot", side_effect=RuntimeError("cold load boom")):
            with self.assertRaises(RuntimeError):
                self.engine._ensure_loaded("test")
        self.assertNotIn("test", self.engine._pipes)
        self.assertEqual(self.engine.cold_start_stats("test"), {})
        self.assertEqual(self.engine.current_version("test"), 1)


class UploadRawBytesTest(unittest.TestCase):
    """G1：安全 PNG 上传直接落原始 bytes，像素与 PIL convert('RGB') 逐位一致。"""

    def _run(self, pil_img, raw):
        import io as _io
        from PIL import Image
        from backend.pipeline import service as svc_mod
        captured = {}
        svc = svc_mod.DetectionService.__new__(svc_mod.DetectionService)

        def fake_detect(path, category, target_type=None):
            captured["path"] = Path(path)
            captured["bytes"] = Path(path).read_bytes()
            captured["rgb"] = svc_mod._decode_rgb(path)
            return {}

        svc.detect_image = fake_detect
        with tempfile.TemporaryDirectory() as td:
            reopened = Image.open(_io.BytesIO(raw)) if raw is not None else pil_img
            svc.detect_pil(reopened, "test", save_dir=Path(td), raw=raw)
        return captured

    def _encode(self, img, fmt, **kw):
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, fmt, **kw)
        return buf.getvalue()

    def test_png_rgb_and_gray_raw_written_and_identical(self):
        from PIL import Image
        rng = np.random.default_rng(0)
        for mode, arr in (("RGB", rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)),
                          ("L", rng.integers(0, 256, (64, 80), dtype=np.uint8))):
            img = Image.fromarray(arr, mode)
            raw = self._encode(img, "PNG")
            cap = self._run(None, raw)
            self.assertEqual(cap["bytes"], raw, mode)  # 未重编码
            ref = np.asarray(img.convert("RGB"))
            self.assertTrue(np.array_equal(cap["rgb"], ref), mode)

    def test_unsafe_formats_fall_back_to_reencode(self):
        from PIL import Image
        rng = np.random.default_rng(1)
        rgba = Image.fromarray(rng.integers(0, 256, (32, 40, 4), dtype=np.uint8), "RGBA")
        pal = Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).convert("P")
        jpg_src = Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8))
        for name, img, fmt in (("rgba_png", rgba, "PNG"), ("palette_png", pal, "PNG"),
                               ("jpeg", jpg_src, "JPEG")):
            raw = self._encode(img, fmt)
            cap = self._run(None, raw)
            self.assertNotEqual(cap["bytes"], raw, name)  # 走了重编码
            from io import BytesIO
            ref = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
            self.assertTrue(np.array_equal(cap["rgb"], ref), name)

    def test_no_raw_keeps_legacy_path(self):
        from PIL import Image
        img = Image.fromarray(np.full((16, 16, 3), 7, np.uint8))
        cap = self._run(img, None)
        self.assertTrue(np.array_equal(cap["rgb"], np.asarray(img)))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main(verbosity=2)
