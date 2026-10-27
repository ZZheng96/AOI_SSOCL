"""少量 AOI_CV 原图/原始参数/ROI 比对；只写指定证据目录。"""
import argparse
import importlib
import json
import hashlib
import platform
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from app.detect.adapters.path_bootstrap import IsolatedImport
from app.detect.adapters.smt_adapter import SmtAdapter, DETECT_FLAG_BY_ALG, LABEL_KEY_BY_ALG
from app.detect.adapters.shift_adapter import ShiftAdapter
from app.detect.contract import Roi
from app.detect.types import DetectRequest
from app.utils.cv_io import imread_unicode


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compact(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    return value


def read_params(path):
    raw = read_json(path)
    return raw.get("defaultParam", raw)


def rects(path):
    result = []
    for shape in read_json(path)["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        (x1, y1), (x2, y2) = shape["points"]
        result.append([int(min(x1, x2)), int(min(y1, y2)), int(abs(x2-x1)), int(abs(y2-y1))])
    return result


def run(root, output):
    rng = random.Random(20261013)
    records = []
    failures = []
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases = [("bridge_pair", "pair4", "solder_连锡"),
             ("insufficient_pair", "pair4", "solder_少锡"),
             ("cold_solder_pair", "pair1", "solder_虚焊")]
    repo = root / "Dev_solder_smt_Alg"
    for index, (group, pair, alg_id) in enumerate(cases, 1):
        directory = repo / "test_images" / group / pair
        standard = next(directory.glob("*_OK.png"))
        images = sorted(p for p in directory.glob("*.png") if not p.name.endswith("_OK.png"))
        image = rng.choice(images)
        print(f"[real {index}/4] SMT {image}", flush=True)
        params = read_params(directory / "params.json")
        pads = rects(directory / "pad.json")
        test, std = imread_unicode(str(image)), imread_unicode(str(standard))
        with IsolatedImport([repo / "src", repo / "contract_reference"], ["algorithms", "core"]):
            cls = importlib.import_module("algorithms.solder_smt.composite_smt_all.solder_smt_all_alg").SolderSmtAllAlg
            source = cls()
            cfg = dict(source.default_config())
            cfg.update({key: False for key in DETECT_FLAG_BY_ALG.values()})
            cfg.update({"pad_frames": pads, "enable_visualization": True})
            cfg[DETECT_FLAG_BY_ALG[alg_id]] = True
            cfg.update(params)
            cfg["diagnostics"] = True
            original = source.run(test, cfg, original_template_image=std)
        rois = [Roi(id=str(i), kind="smt", layer="smt_pads",
                    shape=dict(zip(("x", "y", "w", "h"), pad))) for i, pad in enumerate(pads)]
        request = DetectRequest(image_test=test, image_std=std,
                                rois=rois, preprocess_meta={"diagnostics": True})
        adapter = SmtAdapter()
        adapted = adapter.run(alg_id, request, params)
        from tests.test_adapter_batch import BatchTests
        calls = [(item, request, params) for item in DETECT_FLAG_BY_ALG]
        BatchTests().equal([adapter.run(*call) for call in calls], adapter.run_many(calls))
        expected = [[int(p.x), int(p.y), int(p.width), int(p.height)] for p in original.parts
                    if p.label == LABEL_KEY_BY_ALG[alg_id]]
        actual = [[p.x, p.y, p.w, p.h] for p in adapted.boxes]
        status = "ERROR" if original.code != 0 else "NG" if expected else "OK"
        sm, am = compact(original.metadata), compact(adapted.metadata)
        common = set(sm) & set(am)
        differences = [key for key in sorted(common) if sm[key] != am[key]]
        source_output = original.metadata["output_image"]
        adapter_output = adapted.metadata["output_image"]
        # AOI_CV renders before inverse resize (source lines 464-476); its parts
        # are already restored. Re-render those unchanged parts on the input.
        normalized_output = source._build_output_image(test, original.parts)
        normalized_equal = np.array_equal(normalized_output, adapter_output)
        geometry_ok = adapter_output.shape == test.shape and all(
            0 <= x < test.shape[1] and 0 <= y < test.shape[0] and w > 0 and h > 0
            and x + w <= test.shape[1] and y + h <= test.shape[0]
            for x, y, w, h in actual)
        for name, array in (("source", source_output), ("adapter", adapter_output),
                            ("normalized_source", normalized_output)):
            np.save(raw_dir / f"smt_{index}_{name}.npy", array, allow_pickle=False)
        contract_pass = (status == adapted.status and expected == actual and
                         normalized_equal and geometry_ok and
                         not set(differences) - {"output_image"})
        if not contract_pass:
            failures.append(f"SMT {index}: status/boxes/normalized output/geometry/metadata")
        records.append({"family": "SMT", "algorithm": alg_id, "image": str(image),
                        "standard": str(standard), "roi_file": str(directory / "pad.json"),
                        "params_file": str(directory / "params.json"), "pads": pads,
                        "source_status": status, "adapter_status": adapted.status,
                        "source_boxes": expected, "adapter_boxes": actual,
                        "status_boxes_equal": status == adapted.status and expected == actual,
                        "raw_equivalence_pass": status == adapted.status and expected == actual and not differences,
                        "normalized_output_equal": normalized_equal, "geometry_pass": geometry_ok,
                        "contract_pass": contract_pass,
                        "normalization_basis": "AOI_CV renders work image before inverse scale; re-render unchanged source parts on original input, no resampling or box fitting",
                        "input_shape": list(test.shape),
                        "metadata_difference_keys": differences,
                        "source_metadata": sm, "adapter_metadata": am,
                        "source_message": original.message, "adapter_message": adapted.message,
                        "boundary": "toe/rim 文件未提供，不能覆盖全部虚焊几何分支；无精度结论"})
        print(f"  status {status}/{adapted.status}; boxes_equal={expected == actual}; metadata_diff={differences}", flush=True)
    repo = root / "Dev_shift_smt_Alg"
    directory = repo / "test_images_shift2" / "113-DY-SMT_capa_balck_Shift-0618"
    image = rng.choice(sorted(directory.glob("*Shift_.png")))
    print(f"[real 4/4] Shift {image}", flush=True)
    params = read_params(directory / "params.json")
    base = rects(directory / "pad.json")[0]
    test = imread_unicode(str(image))
    with IsolatedImport([repo / "src", repo / "contract_reference"], ["algorithms", "core"]):
        cls = importlib.import_module("algorithms.shift_smt.shift_smt_all_alg.shift_smt_all_alg").ShiftSmtAllAlg
        source = cls()
        cfg = dict(source.default_config())
        cfg.update(params)
        cfg["diagnostics"] = True
        cfg["base_frames"] = [base]
        original = source.run(test, cfg, roi_bbox=base)
    adapted = ShiftAdapter().run("shift_移位", DetectRequest(image_test=test,
                 rois=[Roi(id="base", kind="shift", layer="shift", shape=dict(zip(("x", "y", "w", "h"), base)))],
                 preprocess_meta={"diagnostics": True}), params)
    expected = [list(map(int, original.metadata.get("base_box") or base))] if original.metadata.get("is_shifted") else []
    actual = [[p.x, p.y, p.w, p.h] for p in adapted.boxes]
    status = "ERROR" if original.code != 0 else "NG" if expected else "OK"
    sm = compact(original.metadata)
    am = compact(adapted.metadata.get("regions", [{}])[0])
    differences = [key for key in sorted(set(sm) & set(am)) if sm[key] != am[key]]
    records.append({"family": "Shift", "image": str(image), "standard": str(directory / "templ_SMT_capa_balck_3_9.png"),
                    "roi_file": str(directory / "pad.json"), "params_file": str(directory / "params.json"),
                    "source_status": status, "adapter_status": adapted.status,
                    "source_boxes": expected, "adapter_boxes": actual,
                    "status_boxes_equal": status == adapted.status and expected == actual,
                    "metadata_difference_keys": differences, "source_metadata": sm, "adapter_metadata": am})
    print(f"  status {status}/{adapted.status}; boxes_equal={expected == actual}; metadata_diff={differences}", flush=True)
    if status != adapted.status or expected != actual or differences:
        failures.append("Shift source comparison")
    from app.detect.adapters.through_hole_adapter import ThroughHoleAdapter, DEFECT_ID_BY_ALG, _th_runtime
    from app.config import TH_REPO_DIR
    for number in (2, 3):
        standard = TH_REPO_DIR / "detect/input/standard_roi" / f"templ_{number}.png"
        label_file = standard.with_suffix(".json")
        candidates = sorted((TH_REPO_DIR / "detect/input/roi").glob(f"{number}-*.png"))
        image = rng.choice(candidates)
        print(f"[real TH {number}] {image}", flush=True)
        test, std = imread_unicode(str(image)), imread_unicode(str(standard))
        source_cfg = read_json(label_file)
        request = DetectRequest(image_test=test, image_std=std, preprocess_meta={"diagnostics": True})
        adapter = ThroughHoleAdapter()
        calls = [(item, request, {}) for item in DEFECT_ID_BY_ALG]
        scalar = [adapter.run(*call) for call in calls]
        BatchTests().equal(scalar, adapter.run_many(calls))
        source_equal = []
        for item, result in zip(DEFECT_ID_BY_ALG, scalar):
            if result.skipped:
                continue
            with _th_runtime():
                module = importlib.import_module("algorithms.pcb_through_hole.pcb_through_hole_solder_alg")
                cfg = {**source_cfg, "enabled_defects": [DEFECT_ID_BY_ALG[item]], "diagnostics": True}
                original = module.PcbThroughHoleSolderAlg().run(test, cfg, original_template_image=std)
            boxes = [[p.x, p.y, p.width, p.height] for p in original.parts]
            equal = (original.metadata["status"] == result.status and boxes == [[p.x,p.y,p.w,p.h] for p in result.boxes])
            np.testing.assert_array_equal(original.metadata["output_image"], result.diff_image)
            assert equal
            source_equal.append(item)
        records.append({"family": "TH", "image": str(image), "standard": str(standard),
                        "roi_file": str(label_file), "source_config": source_cfg,
                        "source_protocol": "algorithm defaults + paired solder_mode; repository config.yaml not injected",
                        "batch_exact_equal": True, "source_equal_items": source_equal,
                        "statuses": [r.status for r in scalar], "boundary": "no GT; auto solder geometry, no manual ROI fabricated"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    (output / "raw" / "real_cases.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    source_files = [Path(__file__), Path(__file__).with_name("test_pcbdual04.py")]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    (output / "env.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(),
                "opencv": cv2.__version__, "numpy": np.__version__, "seed": 20261013,
                "development_tree": str(Path(__file__).resolve().parents[1]),
                "source_sha256": source_hashes,
                "isolation": "no stores/DB/calibration fallback; original files read-only; execution cwd TEMP"}, indent=2), encoding="utf-8")
    (raw_dir / "comparison_summary.json").write_text(json.dumps({
        "cases": 6, "unexpected_failures": failures,
        "raw_smt_equivalence_pass": sum(r.get("raw_equivalence_pass", False) for r in records if r["family"] == "SMT"),
        "smt_contract_pass": sum(r.get("contract_pass", False) for r in records if r["family"] == "SMT"),
        "raw_equivalence_verdict": "FAIL" if any(not r.get("raw_equivalence_pass", True) for r in records) else "PASS",
        "contract_verdict": "FAIL" if failures else "PASS"}, indent=2), encoding="utf-8")
    print(f"[real] 6 cases finished; unexpected_failures={failures}; raw SMT rendering differences retained; no GT/accuracy claim", flush=True)
    if failures:
        raise AssertionError(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"d:\CGAIC\AOI_CV"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pcb-tuner02-") as directory:
        try:
            os.chdir(directory)
            run(args.root, args.output)
        finally:
            os.chdir(previous)
