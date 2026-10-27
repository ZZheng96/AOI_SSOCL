"""AC-079 模拟：打包/数据清理先预览（dry-run）+ 引用完整性（保留回滚链资产）。

- 合成 TEMP storage：快照 v1..v6，激活 v3（meta.parent_version=1），keep=1；
  dry-run 不动任何文件且打印保留原因；真实执行（仅在 TEMP）后回滚目标 v1/v2 与激活 v3 仍在。
- 真实 storage 只跑 dry-run（purge_tmp / purge_heatmaps / snapshots_keep），前后指纹
  （路径+大小+mtime）一致 = 未删除/移动任何用户文件；清单落 raw/ac079_dryrun_inventory.txt。
  不触碰 DB（db_trim/drop_datasource 不在本测试运行）。
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools import cleanup_data as cd

RAW_DIR = Path(r"d:\CGAIC\AOI_SSOCL\review\task\final_sim\raw")


def _fingerprint(*dirs):
    fp = {}
    for d in dirs:
        if d.is_dir():
            for f in d.rglob("*"):
                try:
                    st = f.stat()
                except OSError:
                    continue
                fp[str(f)] = (f.is_dir(), st.st_size if f.is_file() else 0, st.st_mtime_ns)
    return fp


class TestFinalSimCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aoi_cleanup_"))
        snap = self.tmp / "engine" / "snapshots" / "cat"
        for v in range(1, 7):
            (snap / f"v{v}").mkdir(parents=True)
            (snap / f"v{v}" / "x.bin").write_bytes(b"0" * 10)
        (snap / "current.json").write_text(json.dumps({"version": 3}), encoding="utf-8")
        (snap / "v3" / "meta.json").write_text(json.dumps({"parent_version": 1}), encoding="utf-8")
        self.snap = snap
        self.settings = SimpleNamespace(storage_dir=self.tmp, engine_storage=self.tmp / "engine")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_protected_reasons(self):
        r = cd._protected_versions(self.snap, [6, 5, 4, 3, 2, 1], 3, 1)
        self.assertEqual(set(r), {6, 3, 2, 1})
        self.assertIn("激活版父版本（回滚目标）", r[1])
        self.assertIn("激活版前一版（默认回滚目标）", r[2])

    def test_dry_run_moves_nothing_then_real_keeps_rollback_chain(self):
        before = _fingerprint(self.tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cd.snapshots_keep(self.settings, 1, dry=True)
        out = buf.getvalue()
        self.assertEqual(_fingerprint(self.tmp), before)
        self.assertIn("[dry]   归档 cat/v4", out)
        self.assertIn("[dry]   归档 cat/v5", out)
        self.assertNotIn("归档 cat/v1", out)
        self.assertNotIn("归档 cat/v2", out)
        with contextlib.redirect_stdout(io.StringIO()):
            cd.snapshots_keep(self.settings, 1, dry=False)
        left = sorted(p.name for p in self.snap.iterdir() if p.is_dir())
        self.assertEqual(left, ["v1", "v2", "v3", "v6"])
        archived = sorted(p.name.split("_")[1] for p in (self.snap.parent / "_archive").iterdir())
        self.assertEqual(archived, ["v4", "v5"])


class TestFinalSimCleanupRealDryRun(unittest.TestCase):
    def test_real_storage_dry_run_inventory(self):
        settings = cd.get_settings()
        sd = Path(settings.storage_dir)
        targets = [sd / "tmp", sd / "heatmaps", Path(settings.engine_storage) / "snapshots"]
        print("[cleanup] 采集真实 storage 指纹（只读）…", flush=True)
        before = _fingerprint(*targets)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cd.purge_tmp(settings, dry=True)
            cd.purge_heatmaps(settings, 7, dry=True)
            cd.snapshots_keep(settings, 1, dry=True)
        after = _fingerprint(*targets)
        self.assertEqual(after, before, "dry-run 改动了真实 storage")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "ac079_dryrun_inventory.txt").write_text(
            f"storage={sd}\nfiles_fingerprinted={len(before)}（前后一致）\n\n" + buf.getvalue(),
            encoding="utf-8")
        print(f"[cleanup] dry-run 完成，指纹 {len(before)} 项不变", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
