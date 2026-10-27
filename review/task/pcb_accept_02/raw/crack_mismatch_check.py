# PCB-ACCEPT-02 补充：Crack pair1 执行复核 + 配准失败错配构造（真实图片构造错配）
# 用稳定版代码只读执行；不写稳定版/开发版任何文件；产物只写 pcb_accept_02/raw。
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

STB = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual")
sys.path.insert(0, str(STB))

from app.detect.adapters.path_bootstrap import IsolatedImport  # noqa: E402
from app.detect.adapters.smt_adapter import SmtAdapter, DETECT_FLAG_BY_ALG  # noqa: E402
from app.detect.adapters.through_hole_adapter import DEFECT_ID_BY_ALG, ThroughHoleAdapter  # noqa: E402
from app.detect.contract import Roi  # noqa: E402
from app.detect.types import DetectRequest  # noqa: E402
from app.utils.cv_io import imread_unicode  # noqa: E402

OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_02\raw")
OUT.mkdir(parents=True, exist_ok=True)
report = {}


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rects(p):
    out = []
    for shape in read_json(p)["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        (x1, y1), (x2, y2) = shape["points"]
        out.append([int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1))])
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def diag_preview(value, depth=0):
    """诊断值压缩预览，npy 只留 shape/dtype/hash。"""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()[:16],
                "nonzero": int(np.count_nonzero(value))}
    if isinstance(value, dict):
        return {str(k): diag_preview(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return {"list_len": len(value), "head": diag_preview(value[:3], depth + 1)}
        return [diag_preview(v, depth + 1) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ── Part 1：Crack pair1 执行复核 ─────────────────────────────
print("[1/3] Crack pair1 源算法执行复核", flush=True)
pair1 = Path(r"d:\CGAIC\AOI_CV\Dev_solder_smt_Alg\test_images\cold_solder_pair\pair1")
test_p = pair1 / "PixPin_2026-07-01_08-41-22.png"
std_p = pair1 / "PixPin_2026-07-01_08-41-56_OK.png"
params = read_json(pair1 / "params.json")
params = params.get("defaultParam", params)
pads = rects(pair1 / "pad.json")
test, std = imread_unicode(str(test_p)), imread_unicode(str(std_p))

repo = STB / "alg_repo" / "smt"
with IsolatedImport([repo / "src", repo / "contract_reference"], ["algorithms", "core"]):
    cls = importlib.import_module("algorithms.solder_smt.composite_smt_all.solder_smt_all_alg").SolderSmtAllAlg
    source = cls()
    cfg = dict(source.default_config())
    cfg.update({key: False for key in DETECT_FLAG_BY_ALG.values()})
    cfg.update({"pad_frames": pads, "enable_visualization": True})
    cfg[DETECT_FLAG_BY_ALG["solder_虚焊"]] = True
    cfg.update(params)
    cfg["diagnostics"] = True
    original = source.run(test, cfg, original_template_image=std)

diag = original.metadata.get("diagnostics") or {}
crack_diag = diag.get("crack")
crack_executed = not (isinstance(crack_diag, str) and crack_diag.startswith("不可用"))
cold_items = [p for p in original.parts]
report["crack_pair1"] = {
    "images": {"test": str(test_p), "std": str(std_p), "test_sha16": sha(test_p), "std_sha16": sha(std_p)},
    "pads": pads, "params_file_sha16": sha(pair1 / "params.json"),
    "code": int(original.code), "message": original.message,
    "crack_diag_executed": crack_executed,
    "crack_diag_preview": diag_preview(crack_diag),
    "diag_keys": sorted(diag.keys()),
    "rim_toe_diag": diag.get("rim_toe") if isinstance(diag.get("rim_toe"), str) else diag_preview(diag.get("rim_toe")),
    "parts_count": len(cold_items),
    "parts": [[int(p.x), int(p.y), int(p.width), int(p.height), p.label] for p in cold_items],
    "verdict": "PASS:Crack分支实际执行且未检出" if (original.code == 0 and crack_executed and not cold_items)
               else ("FAIL:未执行或异常" if not crack_executed or original.code != 0 else "NG检出:需人工确认正例"),
}
print(f"  code={original.code} crack_executed={crack_executed} parts={len(cold_items)} -> {report['crack_pair1']['verdict']}", flush=True)

# 适配器侧对照（同图同参同 ROI）
rois = [Roi(id=str(i), kind="smt", layer="smt_pads", shape=dict(zip(("x", "y", "w", "h"), pad))) for i, pad in enumerate(pads)]
req = DetectRequest(image_test=test, image_std=std, rois=rois, preprocess_meta={"diagnostics": True})
adapted = SmtAdapter().run("solder_虚焊", req, params)
ad_diag = (adapted.metadata or {}).get("diagnostics") or {}
report["crack_pair1"]["adapter"] = {
    "status": adapted.status, "boxes": [[b.x, b.y, b.w, b.h] for b in adapted.boxes],
    "crack_diag_executed": not (isinstance(ad_diag.get("crack"), str) and str(ad_diag.get("crack")).startswith("不可用")),
    "message": adapted.message,
}
print(f"  adapter status={adapted.status} crack_executed={report['crack_pair1']['adapter']['crack_diag_executed']}", flush=True)

# ── Part 2：TH 配准失败错配构造 ──────────────────────────────
print("[2/3] TH 配准失败错配构造（真实图片构造错配）", flush=True)
th_input = STB / "alg_repo" / "through_hole" / "detect" / "input"
mis_std_p = th_input / "standard_roi" / "templ_2.png"   # 单焊点板
mis_test_p = th_input / "roi" / "3-1.png"               # 另一块双焊点板的图
mis_test, mis_std = imread_unicode(str(mis_test_p)), imread_unicode(str(mis_std_p))
req_mis = DetectRequest(image_test=mis_test, image_std=mis_std, preprocess_meta={"diagnostics": True})
th = ThroughHoleAdapter()
mismatched = {}
for alg_id in DEFECT_ID_BY_ALG:
    r = th.run(alg_id, req_mis, {})
    mismatched[alg_id] = {"status": r.status, "ok": r.ok, "skipped": r.skipped,
                          "skip_reason": r.skip_reason, "error_code": r.error_code,
                          "message": r.message, "boxes": len(r.boxes)}
    print(f"  错配 {alg_id}: status={r.status} ok={r.ok} err={r.error_code} msg={str(r.message)[:80]}", flush=True)
fake_ok = [a for a, r in mismatched.items()
           if not r["skipped"] and r["status"] not in ("ERROR", "REVIEW") and r["ok"] and r["boxes"] == 0]
report["th_mismatch"] = {
    "construction": "真实图片构造错配：std=templ_2(单焊点板) vs test=3-1(双焊点板)，非自然产线配准失败样本",
    "std_sha16": sha(mis_std_p), "test_sha16": sha(mis_test_p),
    "results": mismatched,
    "fake_ok_items": fake_ok,
    "verdict": "PASS:全部 fail-closed(ERROR/REVIEW/SKIP)，无伪OK" if not fake_ok else f"FAIL:伪OK {fake_ok}",
}
print(f"  -> {report['th_mismatch']['verdict']}", flush=True)

# ── Part 3：匹配样本配准状态先验 ─────────────────────────────
print("[3/3] TH 匹配样本配准状态先验（templ_2 vs 2-1）", flush=True)
ok_test_p = th_input / "roi" / "2-1.png"
req_ok = DetectRequest(image_test=imread_unicode(str(ok_test_p)), image_std=mis_std,
                       preprocess_meta={"diagnostics": True})
r = th.run("tht_少锡", req_ok, {})
meta = r.metadata or {}
reg_keys = {k: diag_preview(v) for k, v in meta.items() if any(s in str(k).lower() for s in ("align", "reg", "coverage", "match", "shift", "dx", "dy"))}
report["th_matched_registration"] = {
    "pair": "std=templ_2 vs test=2-1（同源板）", "test_sha16": sha(ok_test_p),
    "status": r.status, "message": str(r.message)[:200],
    "registration_keys": reg_keys, "metadata_keys": sorted(meta.keys()),
}
print(f"  status={r.status} reg_keys={list(reg_keys)}", flush=True)

(OUT / "crack_mismatch_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("[done] raw/crack_mismatch_check.json", flush=True)
