# PCB-ACCEPT-01 第三轮：TH 交互模式验收资产与 TEMP 模板准备
# 只写 temp_tree 与 th_assets，不改稳定版/开发版源码与生产数据。
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2

SRC = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual\alg_repo\through_hole\detect\input")
TEMP = Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_01\temp_tree\PCB_Dual")
ASSETS = Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_01\th_assets")

ASSETS.mkdir(parents=True, exist_ok=True)

copies = {
    "standard_roi/templ_2.png": "templ_2.png",
    "standard_roi/templ_3.png": "templ_3.png",
    "roi/2-1.png": "test_2-1_OK.png",
    "roi/2-8.png": "test_2-8_NG.png",
    "roi/3-1.png": "test_3-1_NG.png",
    "roi/3-2.png": "test_3-2_NG.png",
}
manifest = {}
for src_rel, dst_name in copies.items():
    src = SRC / src_rel
    dst = ASSETS / dst_name
    shutil.copy2(src, dst)
    manifest[dst_name] = {
        "from": str(src),
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "shape": list(cv2.imread(str(dst)).shape),
    }

TH_ITEMS = ["tht_孔洞", "tht_少锡", "tht_多锡包锡", "tht_连锡", "tht_不出脚"]
now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def make_template(tid: str, disp: str, std_png: Path, source_name: str) -> None:
    tdir = TEMP / "templates" / tid
    tdir.mkdir(parents=True, exist_ok=True)
    dst_std = tdir / "standard.png"
    shutil.copy2(std_png, dst_std)
    img = cv2.imread(str(dst_std))
    h, w = img.shape[:2]
    items = [
        {
            "algorithm_id": a,
            "enabled": True,
            "params": {},
            "preprocess_recipe": None,
            "region_kind": "th",
        }
        for a in TH_ITEMS
    ]
    tpl = {
        "id": tid,
        "display_name": disp,
        "version": 1,
        "status": "draft",
        "created_at": now,
        "updated_at": now,
        "standard_image": {
            "path": str(dst_std),
            "sha1": hashlib.sha1(dst_std.read_bytes()).hexdigest(),
            "width": int(w),
            "height": int(h),
            "source_name": source_name,
        },
        "category": "插件器件",
        "detection_items": items,
        "algorithms": json.loads(json.dumps(items, ensure_ascii=False)),
        "region_sets": {},
        "calibration": {},
        "gate": {"size_gate_enabled": False, "size_gate_max_diff_px": 10},
        "notes": "PCB-ACCEPT-01 TH 交互验收 TEMP 模板；尺寸门禁关闭（真实 paired 尺寸有差异，由配准处理）",
        "calib_key": tid,
        "engine_mode": "traditional",
        "model_category": "",
        "px_calibration": {},
        "cad_items": [],
        "panels": {},
        "fov_grid": [],
    }
    (tdir / "template.json").write_text(
        json.dumps(tpl, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest[f"template:{tid}"] = {
        "standard": str(dst_std),
        "size": [int(h), int(w)],
        "status": "draft",
    }


make_template("TH_templ2_single", "TH 单焊点(自动/单框)", ASSETS / "templ_2.png", "templ_2.png")
make_template("TH_templ3_dual", "TH 双焊点(多框连锡)", ASSETS / "templ_3.png", "templ_3.png")

(ASSETS / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=1))
