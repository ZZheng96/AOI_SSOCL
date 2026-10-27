import sqlite3, json
db = r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core\database\aoi.db"
c = sqlite3.connect("file:" + db + "?mode=ro", uri=True)
cols = [r[1] for r in c.execute("PRAGMA table_info(images)")]
print("images cols:", cols)
q = "SELECT category, COUNT(*) FROM images GROUP BY category"
print(c.execute(q).fetchall())
for col in ("label", "is_defect", "split", "source", "target_type"):
    if col in cols:
        print(col, c.execute(f"SELECT {col}, COUNT(*) FROM images GROUP BY {col}").fetchall()[:20])
# feedback / detection 判定分布
try:
    print("feedback:", c.execute("SELECT verdict, label, COUNT(*) FROM feedback GROUP BY verdict, label").fetchall())
except Exception as e:
    print("feedback err", e)
try:
    print("detections by decision:", c.execute("SELECT decision, COUNT(*) FROM detections GROUP BY decision").fetchall())
except Exception as e:
    print("det err", e)
c.close()
