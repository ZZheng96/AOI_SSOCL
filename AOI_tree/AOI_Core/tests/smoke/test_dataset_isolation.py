"""数据导入默认标签与训练/验证/评测分区的轻量隔离回归。"""
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from algo.common.isolation import IsolationViolation, guard_bundle
from backend.core import dataset_adapter, ingest


class DatasetIsolationTest(unittest.TestCase):
    def test_unknown_import_requires_explicit_label(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "plain"
            unknown = root / "batch" / "unclassified.png"
            unknown.parent.mkdir(parents=True)
            unknown.write_bytes(b"unclassified")
            explicit = root / "NG" / "bad.png"
            explicit.parent.mkdir()
            explicit.write_bytes(b"explicit anomaly")
            groups = dataset_adapter.probe_dataset(root)["categories"][0]["groups"]
            self.assertTrue(any(g["unknown"] == 1 for g in groups))
            captured = []
            with patch.object(dataset_adapter, "create_dataset", return_value=1), \
                    patch.object(dataset_adapter, "refresh_dataset_count"), \
                    patch.object(
                        dataset_adapter, "_register_batch",
                        side_effect=lambda records, *a, **kw: (
                            captured.extend(records) or {"imported": len(records)})):
                dataset_adapter.import_adapted(root, "dir_rules")
            self.assertEqual([(r[0], r[3]) for r in captured],
                             [(explicit, "anomaly")])
            captured.clear()
            with patch.object(dataset_adapter, "create_dataset", return_value=1), \
                    patch.object(dataset_adapter, "refresh_dataset_count"), \
                    patch.object(
                        dataset_adapter, "_register_batch",
                        side_effect=lambda records, *a, **kw: (
                            captured.extend(records) or {"imported": len(records)})):
                dataset_adapter.import_adapted(
                    root, "dir_rules", group_overrides={"batch": "anomaly"})
            self.assertIn((unknown, "anomaly"),
                          [(r[0], r[3]) for r in captured])

            records = []
            with patch.object(ingest, "create_dataset", return_value=1), \
                    patch.object(ingest, "refresh_dataset_count"), \
                    patch.object(
                        ingest, "_register_batch",
                        side_effect=lambda batch, *a, **kw: (
                            records.extend(batch) or {"imported": len(batch)})):
                ingest.import_folder(root, copy_to_storage=False)
            self.assertEqual({(r[0].name, r[3]) for r in records}, {
                ("unclassified.png", "unknown"), ("bad.png", "anomaly")})

    def test_guard_rejects_cross_split_content_duplicates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train" / "good" / "a.png"
            train.parent.mkdir(parents=True)
            train.write_bytes(b"normal")
            val = root / "val" / "images" / "b.png"
            val.parent.mkdir(parents=True)
            shutil.copyfile(train, val)
            test = root / "test" / "images" / "c.png"
            test.parent.mkdir(parents=True)
            test.write_bytes(b"different")
            bundle = {"init_normal": [str(train)], "init_defect": [],
                      "val": [(str(val), 0)], "test": [(str(test), 1)]}
            with self.assertRaisesRegex(IsolationViolation, "重复内容"):
                guard_bundle(bundle, {"n_init_defect": 30})
            val.write_bytes(b"val unique")
            self.assertTrue(guard_bundle(bundle, {"n_init_defect": 30}))
            bundle["init_defect"] = [str(test)]
            with self.assertRaisesRegex(IsolationViolation, "重复内容|共用图片"):
                guard_bundle(bundle, {"n_init_defect": 30})


if __name__ == "__main__":
    unittest.main()
