# AOI_Core 双树只读比对：列 DIFF/NEW/稳定版独有，排除数据/缓存目录。
import hashlib
import json
import os
import sys

DEV = r"d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core"
STB = r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "data", "storage", "outputs", "logs", "exports", "heatmaps",
    "snapshots", "models", "cache", "tmp", "temp", "node_modules",
}
EXCLUDE_EXT = {".pyc", ".pyo", ".log", ".db", ".sqlite", ".sqlite3"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def walk(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                continue
            fp = os.path.join(dirpath, f)
            rel = os.path.relpath(fp, root)
            out[rel] = fp
    return out


dev, stb = walk(DEV), walk(STB)
diff, new, only_stb = [], [], []
for rel in sorted(dev):
    if rel not in stb:
        new.append(rel)
    elif sha(dev[rel]) != sha(stb[rel]):
        diff.append(rel)
for rel in sorted(stb):
    if rel not in dev:
        only_stb.append(rel)

print(f"counts dev={len(dev)} stb={len(stb)} DIFF={len(diff)} NEW={len(new)} STB_ONLY={len(only_stb)}")
print(json.dumps({"DIFF": diff, "NEW": new, "STB_ONLY": only_stb}, ensure_ascii=False, indent=1))
