"""R13：接口限流与显式外部审计锚定的小型独立测试。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.external_anchor import export_anchor
from backend.core.rate_limit import RateLimiter


class R13SecuritySupportTest(unittest.TestCase):
    def test_rate_limiter_window_and_retry(self):
        limiter = RateLimiter(limit=2, window_s=10)
        self.assertEqual(limiter.allow("client", now=0), (True, 0))
        self.assertEqual(limiter.allow("client", now=1), (True, 0))
        allowed, retry = limiter.allow("client", now=2)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry, 8)
        self.assertEqual(limiter.allow("client", now=11), (True, 0))

    def test_anchor_is_explicitly_disabled_without_external_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            result = export_anchor({"verified": True, "entries": []})
        self.assertFalse(result["enabled"])
        self.assertIn("未配置", result["reason"])

    def test_anchor_exports_signed_digest_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "external" / "anchor.json"
            result = export_anchor({"verified": True, "entries": []}, path, "test-secret")
            self.assertTrue(result["enabled"])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["chain_hash"], result["chain_hash"])
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))
            self.assertEqual(saved["algorithm"], "HMAC-SHA256")


if __name__ == "__main__":
    unittest.main(verbosity=2)
