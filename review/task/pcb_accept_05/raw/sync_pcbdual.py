"""PCB-ACCEPT-05 受控同步：备份稳定版将被覆盖文件 → 复制开发版 → 哈希核对。"""
import hashlib, json, shutil
from pathlib import Path

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\PCB_Dual")
STB = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual")
BK = Path(r"d:\CGAIC\AOI_SSOCL\review\sync_backup\PCB-ACCEPT-05")
DIFFS = json.loads(Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_05\raw\tree_diff.json")
                   .read_text(encoding="utf-8"))["report"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


results = []
for rel in DIFFS["diff"]:
    d, s = DEV / rel, STB / rel
    b = BK / rel
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, b)  # 备份稳定版原文件
    shutil.copy2(d, s)
    results.append({"file": rel, "action": "diff_synced",
                    "hash_match": sha(d) == sha(s), "backup": str(b)})
for rel in DIFFS["new_in_dev"]:
    d, s = DEV / rel, STB / rel
    s.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d, s)
    results.append({"file": rel, "action": "new_copied", "hash_match": sha(d) == sha(s)})

ok = all(r["hash_match"] for r in results)
(BK / "sync_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
print(f"synced={len(results)} all_hash_match={ok}")
print(f"backup_dir={BK}")
