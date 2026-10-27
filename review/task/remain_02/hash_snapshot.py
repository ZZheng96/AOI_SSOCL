"""REMAIN-02：对指定文件/目录做 SHA256 快照，输出 JSON（隔离证据用）。

用法: py -3.11 hash_snapshot.py <out.json> <label1=path1> [label2=path2 ...]
目录递归哈希所有文件（相对路径+哈希）。
"""
import hashlib
import json
import sys
from pathlib import Path


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(p: Path):
    if p.is_file():
        return {"type": "file", "sha256": hash_file(p)}
    if p.is_dir():
        items = {}
        for f in sorted(p.rglob("*")):
            if f.is_file():
                items[str(f.relative_to(p)).replace("\\", "/")] = hash_file(f)
        return {"type": "dir", "n_files": len(items), "files": items}
    return {"type": "missing"}


out = sys.argv[1]
result = {}
for arg in sys.argv[2:]:
    label, _, path = arg.partition("=")
    result[label] = {"path": path, **hash_path(Path(path))}
Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                     encoding="utf-8")
print(f"HASH_SNAPSHOT -> {out} ({len(result)} 项)", flush=True)
