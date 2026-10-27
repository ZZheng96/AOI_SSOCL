"""backbone 等外部大文件哈希缓存：内容替换（size/mtime 变）必须失效。"""
import os
import tempfile
import time
import unittest

from algo import persist


class ExtHashCacheTest(unittest.TestCase):
    def test_cache_hits_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory(prefix="aoi_hash_") as d:
            p = os.path.join(d, "w.pth")
            with open(p, "wb") as f:
                f.write(b"a" * 1000)
            h1 = persist._sha256_file_cached(p)
            self.assertEqual(h1, persist._sha256_file(p))
            self.assertEqual(persist._sha256_file_cached(p), h1)
            time.sleep(0.02)
            with open(p, "wb") as f:
                f.write(b"b" * 1000)  # 同大小不同内容，mtime_ns 变化
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            h2 = persist._sha256_file_cached(p)
            self.assertNotEqual(h1, h2)
            self.assertEqual(h2, persist._sha256_file(p))


if __name__ == "__main__":
    unittest.main()
