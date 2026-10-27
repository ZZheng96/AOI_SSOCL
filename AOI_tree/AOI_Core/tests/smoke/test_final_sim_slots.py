"""AC-060 模拟：七槽编排清单（合成 pipe，只读）。

只验证"槽位清单/能力/健康状态"能如实区分 fast 三槽与七槽基础集合，
不新增编排页；fast 三槽不得被报告为七槽落地。
"""
import unittest
from types import SimpleNamespace

import numpy as np

from algo.fusion.cde_report import SLOT_CAPABILITIES, slot_health

SEVEN = ("sem", "disc", "blob", "trad", "layout", "inp", "tpl")   # AC-060 基础七槽
CANDIDATE = ("color", "open")                                    # 候选/哨兵，不计入七槽
FAST3 = ("sem", "disc", "inp")


def _pipe(names, open_det=False, sanity=None):
    return SimpleNamespace(
        slots={n: object() for n in names}, open_det=object() if open_det else None,
        calibrators={n: object() for n in names},
        slot_sanity=sanity or {n: np.float64(.8) for n in names},
        _cal_normal_hist={n: [0.1] * 5 for n in names},
        weights={n: 1.0 / len(names) for n in names}, handler=None, head_mgr=None,
        cde_report={})


def coverage(health):
    present = set(health["slots"])
    return {"seven_present": sorted(present & set(SEVEN)),
            "seven_missing": sorted(set(SEVEN) - present),
            "is_seven": set(SEVEN) <= present,
            "candidates": sorted(present & set(CANDIDATE))}


class SevenSlotInventoryTest(unittest.TestCase):
    def test_capabilities_cover_seven_and_candidates(self):
        print("[1/3] 能力清单覆盖七槽 + color/open 候选", flush=True)
        for n in SEVEN + CANDIDATE:
            self.assertIn(n, SLOT_CAPABILITIES, n)
            self.assertIn("fit_type", SLOT_CAPABILITIES[n])

    def test_fast3_not_reported_as_seven(self):
        print("[2/3] fast 三槽：清单仅 3 项，缺 4 槽，is_seven=False", flush=True)
        h = slot_health(_pipe(FAST3))
        cov = coverage(h)
        self.assertEqual(cov["seven_present"], sorted(FAST3))
        self.assertEqual(cov["seven_missing"], sorted({"blob", "trad", "layout", "tpl"}))
        self.assertFalse(cov["is_seven"])
        self.assertTrue(all(h["slots"][n]["in_fusion"] for n in FAST3))
        self.assertIn("not", h["slots"]["disc"])            # SLOT_ROLES 职责边界合并

    def test_full_seven_plus_open_sentinel_and_degrade(self):
        print("[3/3] 七槽+open：open 不进融合；反向体检槽标 reverse（降级可见）", flush=True)
        san = {n: np.float64(.8) for n in SEVEN}
        san["tpl"] = np.float64(.4)
        san["layout"] = None
        h = slot_health(_pipe(SEVEN, open_det=True, sanity=san))
        cov = coverage(h)
        self.assertTrue(cov["is_seven"])
        self.assertEqual(cov["candidates"], ["open"])
        self.assertFalse(h["slots"]["open"]["in_fusion"])
        self.assertEqual(h["slots"]["tpl"]["status"], "reverse")
        self.assertEqual(h["slots"]["layout"]["status"], "unverified")
        self.assertAlmostEqual(sum(h["slots"][n]["weight"] for n in SEVEN), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
