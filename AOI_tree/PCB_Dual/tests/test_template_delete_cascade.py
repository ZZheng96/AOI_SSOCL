"""TemplateStore.delete 级联清理模板专属参数与旧版标定。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.recipe.param_store import ParamStore
from app.template.model import InspectionTemplate
from app.template.store import TemplateStore


class TemplateDeleteCascadeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.calib_dir = root / "calib"
        for target, value in (
            ("app.recipe.param_store.RECIPES_DIR", root / "recipes"),
            ("app.calibration.store.TEMPLATE_CALIB_DIR", self.calib_dir),
        ):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)
        self.store = TemplateStore(root / "templates")
        self.params = ParamStore()
        for tid in ("A", "A__x"):
            self.store.save(InspectionTemplate(id=tid, calibration={"k": 1}))
            self.params.save("template_algorithm", "body_破损", {"t": 1}, template_name=tid)

    def test_delete_removes_own_params_and_calib_only(self):
        self.assertTrue(self.store.delete("A"))
        ta = self.params.root / "template_algorithm"
        left = sorted(p.name for p in ta.glob("*.json"))
        self.assertEqual(left, ["A__x__body_破损.json"])
        self.assertEqual(json.loads((ta / left[0]).read_text(encoding="utf-8"))["template_name"], "A__x")
        self.assertFalse((self.calib_dir / "A.json").exists())
        self.assertTrue((self.calib_dir / "A__x.json").exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.store.delete("nope"))


if __name__ == "__main__":
    unittest.main()
