"""OPT-DECODE：_decode_rgb 快路径（cv2 直出 RGB，含 8 位 PNG）像素一致性与格式回退。

每例同时断言：像素与 PIL convert("RGB") maxdiff=0，且走了预期路径
（fast=cv2 imread 被调用并采纳；fallback=PIL）。
"""
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from backend.pipeline import service as svc


def _pil_ref(p):
    with Image.open(str(p)) as im:
        return np.array(im.convert("RGB"))


def _png_add_chunk(path, ctype, data):
    """在 IHDR 后插入一个辅助块（用于 gAMA 等 PIL 不直接写的块）。"""
    raw = Path(path).read_bytes()
    ihdr_end = 8 + 4 + 4 + 13 + 4
    chunk = struct.pack(">I", len(data)) + ctype + data + struct.pack(
        ">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    Path(path).write_bytes(raw[:ihdr_end] + chunk + raw[ihdr_end:])


class DecodeFastpath(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="optdec_中文路径_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        rng = np.random.default_rng(7)
        self.rgb = rng.integers(0, 256, (61, 83, 3), dtype=np.uint8)

    def _check(self, name, img=None, expect_fast=True, writer=None, **kw):
        p = self.tmp / name
        if writer is not None:
            writer(p)
        else:
            img.save(p, **kw)
        calls = []
        real = svc.imread

        def spy(path, flags):
            out = real(path, flags)
            calls.append(out is not None)
            return out
        with mock.patch.object(svc, "imread", side_effect=spy):
            got = svc._decode_rgb(p)
        ref = _pil_ref(p)
        self.assertEqual(got.dtype, np.uint8, name)
        self.assertEqual(got.shape, ref.shape, name)
        self.assertTrue(got.flags["C_CONTIGUOUS"], name)
        self.assertEqual(int(np.abs(got.astype(np.int16) - ref).max()), 0, name)
        self.assertEqual(bool(calls), expect_fast, f"{name} 路径不符: calls={calls}")
        return got

    # ---- 快路径 ----
    def test_png_rgb8_fast(self):
        self._check("rgb.png", Image.fromarray(self.rgb))

    def test_png_rgb8_max_compress_fast(self):
        self._check("rgb9.png", Image.fromarray(self.rgb), compress_level=9)

    def test_png_gray8_fast(self):
        self._check("g.png", Image.fromarray(self.rgb[..., 1]))

    def test_png_optimize_fast(self):
        self._check("opt.png", Image.fromarray(self.rgb), optimize=True)

    def test_png_gamma_chunk_ignored_both(self):
        def w(p):
            Image.fromarray(self.rgb).save(p)
            _png_add_chunk(p, b"gAMA", struct.pack(">I", 45455))
        self._check("gamma.png", writer=w)

    def test_png_exif_orientation_ignored(self):
        exif = Image.Exif()
        exif[0x0112] = 6
        self._check("ori.png", Image.fromarray(self.rgb), exif=exif.tobytes())

    def test_jpeg_rgb_fast(self):
        self._check("a.jpg", Image.fromarray(self.rgb), quality=92)

    def test_jpeg_gray_fast(self):
        self._check("g.jpg", Image.fromarray(self.rgb[..., 0]), quality=92)

    def test_jpeg_exif_orientation_ignored(self):
        exif = Image.Exif()
        exif[0x0112] = 8
        self._check("o.jpg", Image.fromarray(self.rgb), quality=92, exif=exif.tobytes())

    def test_bmp_fast(self):
        self._check("b.bmp", Image.fromarray(self.rgb))

    def test_chinese_dir_and_name(self):
        sub = self.tmp / "子目录"
        sub.mkdir()
        self._check("子目录/金手指_样本.png", Image.fromarray(self.rgb))

    # ---- 回退 PIL ----
    def test_png_rgba_fallback(self):
        rgba = np.dstack([self.rgb, self.rgb[..., :1]])
        self._check("rgba.png", Image.fromarray(rgba, "RGBA"), expect_fast=False)

    def test_png_palette_fallback(self):
        self._check("p.png", Image.fromarray(self.rgb).convert("P"), expect_fast=False)

    def test_png_rgb_trns_fallback(self):
        self._check("t.png", Image.fromarray(self.rgb), transparency=(1, 2, 3),
                    expect_fast=False)

    def test_png_gray_trns_fallback(self):
        self._check("gt.png", Image.fromarray(self.rgb[..., 0]), transparency=5,
                    expect_fast=False)

    def test_png_16bit_gray_fallback(self):
        g16 = (self.rgb[..., 0].astype(np.uint16) * 257 + 3)
        self._check("g16.png", Image.fromarray(g16), expect_fast=False)

    def test_png_16bit_rgb_fallback(self):
        import cv2
        p = self.tmp / "rgb16.png"

        def w(path):
            ok, buf = cv2.imencode(".png", (self.rgb.astype(np.uint16) * 257 + 1)[..., ::-1])
            self.assertTrue(ok)
            buf.tofile(str(path))
        self._check("rgb16.png", writer=w, expect_fast=False)
        self.assertTrue(p.exists())

    def test_png_1bit_fallback(self):
        self._check("b1.png", Image.fromarray(self.rgb[..., 0] > 128), expect_fast=False)

    def test_apng_multiframe_fallback(self):
        a = Image.fromarray(self.rgb)
        b = Image.fromarray(255 - self.rgb)
        self._check("anim.png", a, save_all=True, append_images=[b], expect_fast=False)

    def test_cmyk_jpeg_fallback(self):
        self._check("c.jpg", Image.fromarray(self.rgb).convert("CMYK"), quality=92,
                    expect_fast=False)

    def test_webp_tiff_fallback(self):
        self._check("w.tif", Image.fromarray(self.rgb), expect_fast=False)

    def test_cv2_failure_falls_back(self):
        p = self.tmp / "x.png"
        Image.fromarray(self.rgb).save(p)
        with mock.patch.object(svc, "imread", return_value=None):
            got = svc._decode_rgb(p)
        self.assertEqual(int(np.abs(got.astype(np.int16) - _pil_ref(p)).max()), 0)

    def test_size_mismatch_falls_back(self):
        p = self.tmp / "y.png"
        Image.fromarray(self.rgb).save(p)
        with mock.patch.object(svc, "imread", return_value=np.zeros((5, 5, 3), np.uint8)):
            got = svc._decode_rgb(p)
        self.assertEqual(got.shape, self.rgb.shape)


if __name__ == "__main__":
    unittest.main()
