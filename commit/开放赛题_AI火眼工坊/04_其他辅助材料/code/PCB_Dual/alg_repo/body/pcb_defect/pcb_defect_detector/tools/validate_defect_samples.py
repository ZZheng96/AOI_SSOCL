"""在 defect_samples 数据集上验证各缺陷算法 100% 检出率。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_defect_detector import PCBDefectDetector

SAMPLES_ROOT = ROOT / "defect_samples"
DATASET_MAP = {
    "错件": "component_wrong_part",
    "缺件": "component_missing",
    "移位": "component_shift",
    "破损": "component_damage",
    "极反": "component_reverse_polarity",
}


def evaluate() -> int:
    det = PCBDefectDetector()
    total = 0
    detected = 0
    failures: list[str] = []

    for folder, alg in DATASET_MAP.items():
        d = SAMPLES_ROOT / folder
        test_dir = d / "test"
        templ_dir = d / "templ"
        for test_f in sorted(test_dir.glob("*.png")):
            total += 1
            tpl = templ_dir / test_f.name
            r = det.detect(alg, str(tpl), str(test_f))
            if r.is_ng:
                detected += 1
            else:
                failures.append(f"{folder}/{test_f.name} ({alg}) status={r.status} extra={r.extra}")

    print(f"检出率: {detected}/{total} ({100.0 * detected / max(total, 1):.1f}%)")
    if failures:
        print("漏检:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("全部样本检出 NG，验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
