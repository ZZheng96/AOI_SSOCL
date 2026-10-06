"""验证 alg_repo/smt 算法仓库可加载并跑通测试图（OK/NG 配对）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

SMT = Path(__file__).resolve().parent.parent / "alg_repo" / "smt"
sys.path.insert(0, str(SMT / "src"))
sys.path.insert(0, str(SMT / "contract_reference"))

from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg  # noqa: E402


def pad_json_to_frames(pad_json: dict) -> list[list[int]]:
    """labelme 格式 pad.json → [[x, y, w, h], ...]"""
    frames: list[list[int]] = []
    for sh in pad_json.get("shapes", []):
        pts = sh.get("points") or []
        if sh.get("shape_type") == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            frames.append([int(x), int(y), int(w), int(h)])
    return frames


def main() -> None:
    alg = SolderSmtAllAlg()
    pair = SMT / "test_images" / "excess_pair" / "pair4"
    test_ng = cv2.imread(str(pair / "PixPin_2026-07-01_11-21-15.png"))
    test_ok = cv2.imread(str(pair / "PixPin_2026-07-01_11-21-15_OK.png"))
    assert test_ng is not None and test_ok is not None, "测试图缺失"
    pad_frames = pad_json_to_frames(json.loads((pair / "pad.json").read_text(encoding="utf-8")))
    params = json.loads((pair / "params.json").read_text(encoding="utf-8"))
    cfg = alg.default_config()
    cfg.update(params)
    cfg["pad_frames"] = pad_frames
    cfg.update({"detect_excess": True, "detect_insufficient": True,
                "detect_bridge": True, "detect_cold_solder": True,
                "enable_visualization": True})
    print("pad_frames:", pad_frames)
    r_ng = alg.run(test_ng, cfg, original_template_image=test_ok)
    print("NG img  ->", r_ng.code, r_ng.message,
          [(p.label, p.x, p.y, p.width, p.height) for p in r_ng.parts])
    r_ok = alg.run(test_ok, cfg, original_template_image=test_ok)
    print("OK img  ->", r_ok.code, r_ok.message,
          [(p.label, p.x, p.y, p.width, p.height) for p in r_ok.parts])
    assert r_ng.parts, "NG 图应检出缺陷"
    assert not r_ok.parts, "OK 图不应检出缺陷"


if __name__ == "__main__":
    main()
