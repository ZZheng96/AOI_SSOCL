"""AC-075/076 模拟：干净离线部署 + DINOv2 权重回退 + sklearn pickle 版本清点。

模拟口径（非真实干净机证据）：
- 每个场景起独立子进程；USERPROFILE/HOME/TORCH_HOME 指向空 TEMP（无用户 hub 缓存），
  HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1，并 patch socket.connect/getaddrinfo 断网。
- 场景①：仅把随包 assets/dinov2（代码+权重）复制到 TEMP，DINOV2_LOCAL_DIR 指向它，CPU 构建
  FrozenDINO 并前向；断言模块从 TEMP 导入、未发起网络。
- 场景②：vendor 目录缺权重 → 回退到 TEMP 内伪造的 hub 缓存（代码+checkpoints 权重），
  特征与场景①逐元素一致（同一份权重）。
- 场景③：vendor 缺权重且无缓存 → 断网下明确 RuntimeError（列出查找路径），不是裸网络异常。
- sklearn：只读加载 defect_clf.pkl 与快照内 pkl（不写），捕获 InconsistentVersionWarning，
  清点 pickle 训练版本 vs 当前环境版本；另对比 requirements-lock.txt。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[2]
VENDOR = CORE.parent / "assets" / "dinov2"
WEIGHT = "dinov2_vits14_pretrain.pth"
RAW_DIR = Path(r"d:\CGAIC\AOI_SSOCL\review\task\final_sim\raw")

_CHILD = r'''
import json, os, socket, sys
NET = []
def _block(*a, **k):
    NET.append(repr(a[:2]))
    raise OSError("simulated offline: network disabled")
socket.socket.connect = _block
socket.getaddrinfo = _block
sys.path.insert(0, os.environ["AOI_CORE"])
out = {"net_attempts": NET}
try:
    import torch
    from algo.backbone import dino
    out["cache_dir"] = dino.DINOV2_CACHE_DIR
    m = dino.FrozenDINO(output_layer=9, grid=4, device="cpu")
    torch.manual_seed(0)
    f = m(torch.rand(1, 3, 56, 56))
    out.update(ok=True, shape=list(f.shape), feat_sum=float(f.double().sum()),
               feat_head=[float(v) for v in f.flatten()[:8]],
               dinov2_file=sys.modules["dinov2"].__file__)
except Exception as e:
    out.update(ok=False, err_type=type(e).__name__, err=str(e)[:600],
               cause=type(e.__cause__).__name__ if e.__cause__ else None)
print("@@RESULT@@" + json.dumps(out, ensure_ascii=False))
'''

_CHILD_PKL = r'''
import glob, json, os, sys, warnings
sys.path.insert(0, os.environ["AOI_CORE"])
import sklearn
from algo.common.safe_pickle import safe_pickle_load, register_pickle_root
from algo.decision import defect_classifier  # noqa: F401 与生产一致：注册 src.* pkl 别名
paths = json.loads(os.environ["PKL_PATHS"])
for p in paths:
    register_pickle_root(os.path.dirname(p))
rows = []
for p in paths:
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            obj = safe_pickle_load(p); err = None
        except Exception as e:
            obj = None; err = f"{type(e).__name__}: {e}"[:300]
    inc = [w for w in ws if type(w.message).__name__ == "InconsistentVersionWarning"]
    rows.append({"path": p, "type": type(obj).__name__ if obj is not None else None,
                 "load_error": err,
                 "inconsistent_version": [{"estimator": getattr(w.message, "estimator_name", None),
                                           "pickled": getattr(w.message, "original_sklearn_version", None),
                                           "current": getattr(w.message, "current_sklearn_version", None)}
                                          for w in inc],
                 "other_warnings": sorted({type(w.message).__name__ for w in ws} - {"InconsistentVersionWarning"})})
print("@@RESULT@@" + json.dumps({"sklearn": sklearn.__version__, "rows": rows}, ensure_ascii=False))
'''


def _run(code, home, extra_env, label):
    env = {k: v for k, v in os.environ.items()
           if k not in ("DINOV2_LOCAL_DIR", "TORCH_HOME", "XDG_CACHE_HOME", "AOI_PICKLE_EXTRA_ROOTS")}
    env.update(USERPROFILE=str(home), HOME=str(home), TORCH_HOME=str(home / ".cache" / "torch"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AOI_CORE=str(CORE),
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra_env)
    print(f"[deploy] 场景 {label} 子进程启动…", flush=True)
    p = subprocess.run([sys.executable, "-c", code], cwd=str(home), env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
    if not line:
        raise AssertionError(f"子进程无结果 rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    res = json.loads(line[-1][len("@@RESULT@@"):])
    res["stdout_tail"] = [l for l in p.stdout.splitlines() if l.startswith("[dino]")]
    print(f"[deploy] 场景 {label} 完成 ok={res.get('ok')}", flush=True)
    return res


def _copy_code(dst):
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VENDOR / "hubconf.py", dst / "hubconf.py")
    shutil.copytree(VENDOR / "dinov2", dst / "dinov2",
                    ignore=shutil.ignore_patterns("__pycache__"))


@unittest.skipUnless((VENDOR / WEIGHT).exists(), "随包 vendor 权重不存在")
class TestFinalSimOfflineDeploy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="aoi_offline_"))
        print(f"[deploy] TEMP={cls.tmp}，复制 vendor 代码与 88MB 权重…", flush=True)
        cls.full = cls.tmp / "pkg_full" / "dinov2"
        _copy_code(cls.full)
        shutil.copy2(VENDOR / WEIGHT, cls.full / WEIGHT)
        cls.noweight = cls.tmp / "pkg_noweight" / "dinov2"
        _copy_code(cls.noweight)
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "ac075_offline_deploy.json").write_text(
            json.dumps(cls.results, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _home(self, name):
        h = self.tmp / name
        h.mkdir(parents=True, exist_ok=True)
        return h

    def test_1_vendor_offline_load(self):
        home = self._home("home_empty1")
        r = _run(_CHILD, home, {"DINOV2_LOCAL_DIR": str(self.full)}, "①vendor")
        self.results["vendor_offline"] = r
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["shape"], [1, 384, 4, 4])
        self.assertEqual(r["net_attempts"], [])
        self.assertTrue(Path(r["dinov2_file"]).resolve().is_relative_to(self.tmp.resolve()), r)
        self.assertTrue(any("随包 vendor 权重" in l for l in r["stdout_tail"]), r)
        self.assertFalse((home / ".cache" / "torch" / "hub" / "checkpoints").exists())

    def test_2_fallback_to_cache(self):
        home = self._home("home_cache")
        cache = home / ".cache" / "torch" / "hub"
        _copy_code(cache / "facebookresearch_dinov2_main")
        (cache / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.full / WEIGHT, cache / "checkpoints" / WEIGHT)
        r = _run(_CHILD, home, {"DINOV2_LOCAL_DIR": str(self.noweight)}, "②回退缓存")
        self.results["fallback_cache"] = r
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["net_attempts"], [])
        self.assertTrue(any("缺权重，回退" in l for l in r["stdout_tail"]), r)
        self.assertTrue(Path(r["cache_dir"]).resolve().is_relative_to(home.resolve()), r)
        ref = self.results.get("vendor_offline") or _run(
            _CHILD, self._home("home_empty1b"), {"DINOV2_LOCAL_DIR": str(self.full)}, "①ref")
        self.assertEqual(r["feat_head"], ref["feat_head"])

    def test_3_missing_weights_explicit_failure(self):
        home = self._home("home_empty3")
        r = _run(_CHILD, home, {"DINOV2_LOCAL_DIR": str(self.noweight)}, "③缺权重")
        self.results["missing_weights"] = r
        self.assertFalse(r["ok"], r)
        self.assertEqual(r["err_type"], "RuntimeError", r)
        self.assertIn("DINOv2 权重缺失", r["err"])
        self.assertIn(str(self.noweight), r["err"])
        self.assertIn("hub 缓存=", r["err"])
        self.assertIsNotNone(r["cause"])


class TestFinalSimSklearnInventory(unittest.TestCase):
    def test_pickle_version_inventory(self):
        import sklearn
        import torch
        pkls = sorted({str(p) for p in [CORE.parent / "assets" / "defect_clf.pkl"] if p.exists()}
                      | {str(p) for p in (CORE / "storage" / "engine" / "snapshots").rglob("*.pkl")})
        self.assertTrue(pkls, "无 pkl 可清点")
        tmp = Path(tempfile.mkdtemp(prefix="aoi_pkl_"))
        try:
            r = _run(_CHILD_PKL, tmp, {"PKL_PATHS": json.dumps(pkls)}, f"sklearn 清点({len(pkls)} 个)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        lock = (CORE / "requirements-lock.txt")
        locked = {}
        if lock.exists():
            for l in lock.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "==" in l and not l.lstrip().startswith("#"):
                    k, v = l.split("==", 1)
                    locked[k.strip().lower()] = v.strip()
        mism = {}
        for name, cur in (("scikit-learn", sklearn.__version__), ("torch", torch.__version__),
                          ("numpy", __import__("numpy").__version__)):
            if name in locked and locked[name] != cur:
                mism[name] = {"locked": locked[name], "env": cur}
        inc = [row for row in r["rows"] if row["inconsistent_version"]]
        summary = {"env": {"python": sys.version.split()[0], "sklearn": r["sklearn"],
                           "torch": torch.__version__},
                   "lock_mismatch": mism, "n_pkl": len(r["rows"]),
                   "n_inconsistent": len(inc),
                   "pickled_versions": sorted({w["pickled"] for row in inc
                                               for w in row["inconsistent_version"] if w["pickled"]}),
                   "load_errors": [row for row in r["rows"] if row["load_error"]],
                   "rows": r["rows"]}
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "ac076_sklearn_inventory.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[deploy] pkl={len(r['rows'])} 版本不一致={len(inc)} "
              f"pickled={summary['pickled_versions']} env={r['sklearn']} lock_mismatch={mism}", flush=True)
        self.assertEqual(summary["load_errors"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
