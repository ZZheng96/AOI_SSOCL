"""TEST-06 运行脚本：AC-062 旧生产 trad 协议八项等价比对（AOI_feature 候选）。

合成图造法同 AOI_Core/tests/smoke/test_final_model_parity.py（含 >max_side 与
<min_side 越界图，否则 preprocess 项恒 False）。输出：
  raw/parity_result.json  八项 checks + diffs + numeric
  raw/group_diffs.json    探针图逐组 max|diff|（差异归因用，不进 harness）
用法：cwd=AOI_Core，py -3.11 <本脚本>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent
AOI_TREE = RAW.parents[3] / "AOI_tree"
for _p in (str(AOI_TREE / "AOI_Core"), str(AOI_TREE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from algo.feature_revision import PARITY_CHECKS          # noqa: E402
from algo.features import registry as freg               # noqa: E402
from algo.features.parity import ProductionCandidate, run_parity  # noqa: E402

from aoi_feature_candidate import AOIFeatureCandidate    # noqa: E402

SEED = 0


def _img(seed, h=64, w=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 40 * np.sin(xx / (3 + seed % 4)) + 30 * np.cos(yy / 5.0)
    g = np.clip(base + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([g, np.roll(g, 1, 0), np.roll(g, 1, 1)], -1)


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print(f"[test06] 构造合成图（train=6, probe=3 含越界 300x280 / 20x40）...", flush=True)
    train = [_img(s) for s in range(6)]
    probe = [_img(100, 300, 280), _img(101, 20, 40), _img(102)]

    print("[test06] 实例化候选适配器（FeatureSet trunk_200 + GroupKNN k=3）...", flush=True)
    cand = AOIFeatureCandidate()

    print("[test06] run_parity 开始（参照=生产 TradSlot 全新实例）...", flush=True)
    ev = run_parity(cand, train, probe, atol=1e-6,
                    lineage={"revision_id": "test06-aoi-feature-local",
                             "seed": SEED,
                             "dataset_split": {"kind": "synthetic",
                                               "train_seeds": list(range(6)),
                                               "probe_seeds": [100, 101, 102]},
                             "model_snapshot": {"category": "none",
                                                "note": "合成图现场拟合，无快照"}})
    (RAW / "parity_result.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[test06] run_parity 完成：status={ev['status']} verdict={ev['verdict']} "
          f"failed={ev['failed_checks']} max_abs_diff={ev['numeric']['max_abs_diff']:.3e}",
          flush=True)

    # 分组归因（诊断用，不篡改为 harness 输入）：候选 vs 生产逐组 max|diff|
    print("[test06] 逐组差异归因（15 组 x 3 探针）...", flush=True)
    ref = ProductionCandidate()
    group_diffs = {}
    for pi, im in enumerate(probe):
        vc, _ = cand.extract(im)
        vr, _ = ref.extract(im)
        per = {}
        for g in freg.TRAD_GROUPS:
            a, b = g["slice"]
            d = float(np.max(np.abs(vc[a:b].astype(np.float64)
                                    - vr[a:b].astype(np.float64))))
            per[g["key"]] = d
        group_diffs[f"probe{pi}_{im.shape[0]}x{im.shape[1]}"] = per
        worst = max(per, key=per.get)
        print(f"  probe{pi} {im.shape[0]}x{im.shape[1]}: worst={worst} "
              f"max|diff|={per[worst]:.3e}", flush=True)
    (RAW / "group_diffs.json").write_text(
        json.dumps(group_diffs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[test06] 全部完成，用时 {time.time() - t0:.1f}s", flush=True)
    for k in PARITY_CHECKS:
        print(f"  {'PASS' if ev['checks'][k] else 'FAIL'}  {k}", flush=True)
    return 0 if ev["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
