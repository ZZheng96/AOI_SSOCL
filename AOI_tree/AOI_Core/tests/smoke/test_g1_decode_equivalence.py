"""G1：service 解码走 cv2 快路径时，像素须与 PIL convert("RGB") 严格一致。"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.pipeline.service import _decode_rgb


def _pil_ref(p):
    return np.array(Image.open(str(p)).convert("RGB"))


class DecodeEquivalence(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="g1dec_中文_"))
        rng = np.random.default_rng(0)
        self.rgb = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _check(self, name, img, **kw):
        p = self.tmp / name
        img.save(p, **kw)
        got = _decode_rgb(p)
        ref = _pil_ref(p)
        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(got.shape, ref.shape)
        self.assertEqual(int(np.abs(got.astype(int) - ref).max()), 0, name)

    def test_jpeg_rgb(self):
        self._check("a.jpg", Image.fromarray(self.rgb), quality=90)

    def test_jpeg_gray(self):
        self._check("g.jpg", Image.fromarray(self.rgb[..., 0]), quality=90)

    def test_jpeg_exif_orientation_ignored(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # 旋转 90°；PIL 不应用，cv2 必须同样忽略
        self._check("o.jpg", Image.fromarray(self.rgb), quality=90,
                    exif=exif.tobytes())

    def test_bmp(self):
        self._check("b.bmp", Image.fromarray(self.rgb))

    def test_png_rgba_fallback(self):
        rgba = np.dstack([self.rgb, self.rgb[..., :1]])
        self._check("r.png", Image.fromarray(rgba, "RGBA"))

    def test_palette_png_fallback(self):
        self._check("p.png", Image.fromarray(self.rgb).convert("P"))

    def test_cmyk_jpeg_fallback(self):
        self._check("c.jpg", Image.fromarray(self.rgb).convert("CMYK"),
                    quality=90)


if __name__ == "__main__":
    unittest.main()
