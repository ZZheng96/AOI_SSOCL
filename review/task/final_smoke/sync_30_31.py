# SYNC-30/31 同步：备份稳定版原文件 -> 复制开发版 -> 哈希核对。仅处理批次登记文件。
import hashlib
import json
import shutil
from pathlib import Path

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core")
STB = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core")
BAK30 = Path(r"d:\CGAIC\AOI_SSOCL\review\sync_backup\SYNC-30")
BAK31 = Path(r"d:\CGAIC\AOI_SSOCL\review\sync_backup\SYNC-31")

SYNC30_DIFF = [
    r"algo\slots\blob.py",
    r"algo\eval\offline.py",
    r"algo\fusion\calibrate.py",
    r"backend\core\config.py",
    r"backend\api\routes_system.py",
    r"backend\api\routes_candidates.py",
    r"backend\engine\__init__.py",
    r"backend\evaluation\service.py",
    r"ui\pages\settings_page.py",
    r"ui\pages\eval_page.py",
    r"tests\smoke\test_candidate_audit_log.py",
]
SYNC30_NEW = [
    r"algo\eval\coverage.py",
    r"backend\core\deplock.py",
    r"tests\smoke\test_round3_speed.py",
    r"tests\smoke\test_round3_decisions.py",
    r"tests\smoke\test_round3_accuracy.py",
    r"tests\smoke\test_round3_coverage.py",
]
SYNC31_DIFF = [
    r"algo\eval\coverage.py",  # SYNC-30 已为 NEW，SYNC-31 再改；备份时稳定版无此文件
    r"backend\api\routes_models.py",
    r"backend\evaluation\service.py",  # 与 SYNC-30 同文件，已在 SYNC-30 备份
    r"backend\api\routes_system.py",  # 同上
]
SYNC31_NEW = [
    r"tests\smoke\test_round4_batch_rebuild.py",
    r"tests\smoke\test_round4_sem_shead.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


report = {"backed_up": [], "copied": [], "hash_mismatch": []}

# 备份（稳定版已存在的 DIFF 文件；同文件只备一次，SYNC-30 优先）
backed = set()
for rel in SYNC30_DIFF:
    src = STB / rel
    dst = BAK30 / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    backed.add(rel)
    report["backed_up"].append(f"SYNC-30: {rel}")
for rel in SYNC31_DIFF:
    if rel in backed:
        continue
    src = STB / rel
    if not src.exists():
        continue  # coverage.py 稳定版原本不存在
    dst = BAK31 / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    backed.add(rel)
    report["backed_up"].append(f"SYNC-31: {rel}")

# 复制（开发版 -> 稳定版），SYNC-31 的覆盖 SYNC-30 的同文件（最终态=开发版）
all_files = list(dict.fromkeys(SYNC30_DIFF + SYNC30_NEW + SYNC31_DIFF + SYNC31_NEW))
for rel in all_files:
    src, dst = DEV / rel, STB / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    report["copied"].append(rel)

# 哈希核对
for rel in all_files:
    if sha(DEV / rel) != sha(STB / rel):
        report["hash_mismatch"].append(rel)

report["ok"] = not report["hash_mismatch"]
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, ensure_ascii=False))
if report["hash_mismatch"]:
    print("MISMATCH:", report["hash_mismatch"])
else:
    print("ALL HASH MATCH:", len(all_files), "files")
