"""F1 新入库算法验证：金手指 / 元件本体 / 起皱 实际检测。

数据：
- gold：alg_repo/gold_finger 无自带图 → 用 reference5 UI算法 wjf\金手指\第一对（templ1/test1）
- body：alg_repo/body/pcb_defect/pcb_defect_detector/defect_samples/缺件
- wrinkle：免模板，用 SMT 测试图
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detect.adapters.gold_finger_adapter import GoldFingerAdapter  # noqa: E402
from app.detect.adapters.body_adapter import BodyAdapter  # noqa: E402
from app.detect.adapters.wrinkle_adapter import WrinkleAdapter  # noqa: E402
from app.detect.types import DetectRequest  # noqa: E402
from app.utils.cv_io import imread_unicode  # noqa: E402

UI_ALG = Path(r"d:\CGAIC\reference5\UI算法")
ROOT = Path(__file__).resolve().parent.parent


def make_req(std: str | None, test: str) -> DetectRequest:
    img = imread_unicode(test)
    return DetectRequest(
        image_std=imread_unicode(std) if std else None,
        image_std_raw=imread_unicode(std) if std else None,
        image_test=img,
        image_test_raw=img,
        template_name="verify_new",
    )


def main() -> None:
    # 1) 金手指（templ vs test）
    gf = GoldFingerAdapter()
    pair = UI_ALG / "wjf" / "金手指" / "第一对"
    std, test = pair / "templ1.png", pair / "test1.png"
    if std.exists() and test.exists():
        t0 = time.time()
        r = gf.run("board_板面金手指", make_req(str(std), str(test)), {})
        print(f"[gold] 判定={'NG' if not r.ok else 'OK'} 缺陷={len(r.boxes)} "
              f"({(time.time()-t0)*1000:.0f}ms) {r.message[:80]}")
    else:
        print("[gold] 测试图缺失:", pair)

    # 2) 元件本体缺件（templ vs test）
    body = BodyAdapter()
    sample = (ROOT / "alg_repo" / "body" / "pcb_defect" / "pcb_defect_detector"
              / "defect_samples" / "缺件")
    tpls = sorted((sample / "templ").glob("*")) if (sample / "templ").is_dir() else []
    tests = sorted((sample / "test").glob("*")) if (sample / "test").is_dir() else []
    if tpls and tests:
        t0 = time.time()
        r = body.run("body_缺件", make_req(str(tpls[0]), str(tests[0])), {})
        print(f"[body缺件] 判定={'NG' if not r.ok else 'OK'} 缺陷={len(r.boxes)} "
              f"({(time.time()-t0)*1000:.0f}ms) {r.message[:80]}")
    else:
        print("[body] 缺件样本缺失:", sample)

    # 3) 起皱（免模板，SMT 图）
    wr = WrinkleAdapter()
    test_img = ROOT / "alg_repo" / "smt" / "test_images" / "excess_pair" / "pair4" / "PixPin_2026-07-01_11-21-15.png"
    if test_img.exists():
        t0 = time.time()
        r = wr.run("wrinkle_起皱", make_req(None, str(test_img)), {})
        print(f"[wrinkle] 判定={'NG' if not r.ok else 'OK'} 缺陷={len(r.boxes)} "
              f"({(time.time()-t0)*1000:.0f}ms) {r.message[:80]}")


if __name__ == "__main__":
    main()
