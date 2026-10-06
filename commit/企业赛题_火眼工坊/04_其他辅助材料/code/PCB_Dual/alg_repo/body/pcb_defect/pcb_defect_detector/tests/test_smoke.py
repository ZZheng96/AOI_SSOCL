from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_defect_detector import PCBDefectDetector, list_algorithms
from pcb_defect_detector.results import Defect, DetectionResult


def _synthetic_pair() -> tuple:
    h, w = 256, 256
    tpl = np.full((h, w, 3), 80, dtype=np.uint8)
    tpl[:, :, 1] = 120
    cv2.rectangle(tpl, (80, 90), (180, 170), (200, 200, 80), -1)
    cv2.putText(tpl, "C1", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
    test = tpl.copy()
    cv2.circle(test, (140, 130), 12, (40, 40, 40), -1)
    return tpl, test


def test_list_algorithms():
    algs = list_algorithms()
    assert isinstance(algs, dict)
    assert len(algs) == 7
    for code in ["component_wrong_part", "component_missing", "component_shift",
                 "component_tombstone", "component_flipped",
                 "component_reverse_polarity", "component_damage"]:
        assert code in algs


def test_pool_loads_all_seven():
    det = PCBDefectDetector()
    avail = det.available_algorithms()
    assert len(avail) == 7, f"期望 7 个算法加载成功，实际 {len(avail)}: {avail}"


def test_detect_returns_result_object():
    det = PCBDefectDetector()
    tpl, test = _synthetic_pair()
    r = det.detect("component_damage", tpl, test)
    assert isinstance(r, DetectionResult)
    assert r.algorithm == "component_damage"
    assert r.status in ("OK", "NG", "ERROR")
    assert isinstance(r.defects, list)
    for d in r.defects:
        assert isinstance(d, Defect)
    assert isinstance(r.cost_time, float)
    # 可视化图应存在且与测试图同尺寸
    if r.output_image is not None:
        assert r.output_image.shape[:2] == test.shape[:2]


def test_detect_all_runs_every_algorithm():
    det = PCBDefectDetector()
    tpl, test = _synthetic_pair()
    results = det.detect_all(tpl, test)
    assert set(results.keys()) == set(list_algorithms().keys())
    for code, r in results.items():
        assert r.algorithm == code
        assert r.status in ("OK", "NG", "ERROR")


def test_unknown_algorithm_returns_error_result():
    det = PCBDefectDetector()
    tpl, test = _synthetic_pair()
    r = det.detect("not_a_real_algorithm", tpl, test)
    assert r.status == "ERROR"
    assert r.has_error


def test_to_dict_is_json_serializable():
    import json
    det = PCBDefectDetector()
    tpl, test = _synthetic_pair()
    r = det.detect("component_missing", tpl, test)
    d = r.to_dict()
    # 不应包含 numpy 数组
    json.dumps(d, ensure_ascii=False)  # 不抛异常即通过


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n全部 {len(funcs)} 项冒烟测试通过。")


if __name__ == "__main__":
    _run_all()
