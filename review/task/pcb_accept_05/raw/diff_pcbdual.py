"""PCB-ACCEPT-05 前置：开发版 vs 稳定版 PCB_Dual 只读差异清单（冻结双方哈希）。

排除生产 data/DB/模板/参数/快照/缓存目录；只读，不修改任何文件。
"""
import hashlib, json, os
from pathlib import Path

DEV = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\PCB_Dual")
STB = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual")
OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_05\raw")
OUT.mkdir(parents=True, exist_ok=True)

# 排除：生产数据/DB/模板/参数/快照/缓存
EXCLUDE_DIRS = {"__pycache__", ".git", "data", "database", "db", "cache", "caches",
                "snapshots", "snapshot", "templates", "template", "params", "param_lib",
                "logs", "log", "output", "outputs", "tmp", "temp", "test_images"}
EXCLUDE_EXT = {".pyc", ".db", ".sqlite", ".sqlite3", ".log", ".pkl", ".npy"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def scan(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d.lower() not in EXCLUDE_DIRS]
        for f in fn:
            p = Path(dp) / f
            if p.suffix.lower() in EXCLUDE_EXT:
                continue
            rel = p.relative_to(root).as_posix()
            out[rel] = {"sha256": sha(p), "size": p.stat().st_size}
    return out


dev, stb = scan(DEV), scan(STB)
diff = sorted(k for k in dev.keys() & stb.keys() if dev[k]["sha256"] != stb[k]["sha256"])
new = sorted(k for k in dev.keys() - stb.keys())
gone = sorted(k for k in stb.keys() - dev.keys())
same = sorted(k for k in dev.keys() & stb.keys() if dev[k]["sha256"] == stb[k]["sha256"])

report = {
    "dev_root": str(DEV), "stable_root": str(STB),
    "counts": {"dev_files": len(dev), "stable_files": len(stb),
               "identical": len(same), "diff": len(diff), "new_in_dev": len(new),
               "only_in_stable": len(gone)},
    "diff": diff, "new_in_dev": new, "only_in_stable": gone,
    "excluded_dirs": sorted(EXCLUDE_DIRS),
}
(OUT / "tree_diff.json").write_text(json.dumps(
    {"report": report, "dev": dev, "stable": stb}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print(json.dumps(report["counts"], ensure_ascii=False))
print("--- DIFF ---")
[print(" ", k) for k in diff]
print("--- NEW(dev only) ---")
[print(" ", k) for k in new]
print("--- ONLY-STABLE ---")
[print(" ", k) for k in gone]
