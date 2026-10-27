import json, sqlite3, pathlib
t = pathlib.Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_01\temp_tree\PCB_Dual")
tj = json.loads((t / "templates" / "PixPin_2026-07-01_11-21-15_OK" / "template.json").read_text(encoding="utf-8"))
print("TEMP template status:", tj.get("status"))
db = t / "storage" / "aoi.db"
print("TEMP db exists:", db.exists(), round(db.stat().st_size / 1024, 1), "KB")
if db.exists():
    c = sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True)
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for tb in tables:
        try:
            print(" ", tb, c.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0])
        except Exception as e:
            print(" ", tb, "err", e)
    c.close()
s = pathlib.Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual\templates\PixPin_2026-07-01_11-21-15_OK\template.json")
print("stable original status:", json.loads(s.read_text(encoding="utf-8")).get("status"))
