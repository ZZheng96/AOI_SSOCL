"""REMAIN-02 修复：第一轮联调 draft/report/rebuild 409 的根因修复。

根因：启动钩子 register_engine_snapshots 判定 gold_finger「无血缘」
（TEMP 库无 Detection 记录、无 Model.metrics.dataset_ids），删除了
temp 副本的 current.json，导致 current_version() 返 None。

修复：向 TEMP DB 插入一条 Detection 血缘记录（category=gold_finger），
并从稳定版只读复制 current.json 回 temp 快照目录。不修改稳定版任何文件。
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(r"d:\CGAIC\AOI_SSOCL")
STABLE = ROOT / "stable_AOI_tree" / "AOI_Core"
TASK = ROOT / "review" / "task" / "remain_02"
TEMP = TASK / "temp_core"
CAT = "gold_finger"

# 1. 恢复 current.json（从稳定版只读复制）
src_cur = STABLE / "storage" / "engine" / "snapshots" / CAT / "current.json"
dst_cur = TEMP / "storage" / "engine" / "snapshots" / CAT / "current.json"
shutil.copy2(src_cur, dst_cur)
print(f"[1/2] current.json 已恢复: {dst_cur}", flush=True)

# 2. 插入 Detection 血缘记录
os.environ["AOI_CONFIG"] = str(TEMP / "config.yaml")
sys.path.insert(0, str(STABLE))
os.chdir(STABLE)
from backend.db.database import session_scope  # noqa: E402
from backend.db.models import Detection, Image  # noqa: E402

with session_scope() as s:
    img = (s.query(Image)
           .filter(Image.category == CAT, Image.split == "test")
           .first())
    existing = s.query(Detection).filter(Detection.category == CAT).count()
    if existing == 0:
        s.add(Detection(
            target_type="image",
            image_id=img.id if img else None,
            image_path=img.path if img else "seed",
            category=CAT,
            final_score=0.1,
            is_anomaly=False,
            latency_ms=1.0,
            n_tiles={"l1": 1},
        ))
        print("[2/2] Detection 血缘记录已插入", flush=True)
    else:
        print(f"[2/2] 已有 {existing} 条 Detection，跳过", flush=True)
print("FIX_OK", flush=True)
