"""REMAIN-02：TEMP DB 表级快照（绕过服务进程只读 SQLite，WAL 下安全）。

用法: py -3.11 db_dump.py <db_path> <out.json>
导出 images/models/eval_runs 全量行 + operation_logs 的 action 序列，
整体 SHA256，用于 draft/report/rebuild 前后对比。
（operation_logs 预期会增——它是审计轨迹；业务表不变才是隔离证据。）
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

db_path, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
dump = {}
for table, cols in (
        ("images", "id,path,category,split,label,source,width,height,sha1"),
        ("models", "id,name,category,version,path,format,metrics,is_active,origin"),
        ("eval_runs", "id,name,run_type,category,dataset_desc,n_images"),
        ("operation_logs", "id,user,action,detail")):
    try:
        rows = [dict(r) for r in con.execute(
            f"SELECT {cols} FROM {table} ORDER BY id")]
    except sqlite3.Error as e:
        rows = f"ERROR: {e}"
    dump[table] = rows
con.close()
blob = json.dumps(dump, ensure_ascii=False, sort_keys=True, default=str)
result = {"db": db_path, "sha256": hashlib.sha256(blob.encode()).hexdigest(),
          "counts": {t: (len(r) if isinstance(r, list) else r)
                     for t, r in dump.items()},
          "tables": dump}
Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                default=str), encoding="utf-8")
print(f"DB_DUMP -> {out} sha256={result['sha256']} counts={result['counts']}",
      flush=True)
